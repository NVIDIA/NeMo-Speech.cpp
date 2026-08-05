#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified NeMo → GGUF converter for nemo-speech.

Handles both architectures:
    * FastConformer-CTC (e.g. nvidia/parakeet-ctc-1.1b)
    * FastConformer-RNNT cache-aware (e.g. nvidia/nemotron-speech-streaming-en-0.6b)

Output convention (matches what src/asr/* C++ code expects):

    * `general.architecture = "asr"`
    * Metadata namespace `asr.*` (encoder / preprocessor / ctc / rnnt / tokenizer)
    * `asr.head_type ∈ {"ctc", "rnnt", "tdt"}` discriminator
    * Conv weights emitted in their **native 3D** shapes (no squeeze) — our
      runtime's Conv1D module reshapes lazily in build_graph.
    * RNNT LSTM tensors **renamed** from PyTorch's `weight_ih_l0` / `bias_ih_l0`
      style to `ih_l0.weight` / `ih_l0.bias`, so the runtime's `Linear` module
      (which assumes `{name}.weight` / `{name}.bias`) can own them directly.
    * Positional encoding pre-computed analytically (no need to copy the
      checkpoint buffer).

Usage:
    python3 convert_model.py <input.nemo|hf_repo_id> --outfile <output.gguf>

If `--head-type` is omitted, the head is auto-detected from the model_config.yaml
inside the .nemo (presence of `joint` block ⇒ RNNT, `labels` only ⇒ CTC).
"""
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from gguf import GGMLQuantizationType, GGUFWriter

from .quantization import DEPLOYMENT_WEIGHT_TYPES
from .quantization import K_QUANTS as _K_QUANTS
from .quantization import QK_K as _QK_K
from .quantization import quantize as _quantize
from .source import extract_archive, find_checkpoint_files
from .source import load_state_dict as _load_state_dict

ARCH = "asr"

# ---------------------------------------------------------------------------
# Weight-type plumbing
# ---------------------------------------------------------------------------

# CLI flag → (per-Linear quant type, default non-Linear dtype, file_type meta).
# file_type values match `gguf.LlamaFileType` (a.k.a. `general.file_type`).
WEIGHT_TYPES = DEPLOYMENT_WEIGHT_TYPES

# ---------------------------------------------------------------------------
# Tensor → emission dtype dispatch
# ---------------------------------------------------------------------------

# Linear-layer weights eligible for quantization. After the LSTM rename in
# remap_for_rnnt, the dec_rnn pattern is normalized.
_LINEAR_WEIGHT_PATTERNS = [
    re.compile(r"^encoder\.layers\.\d+\.self_attn\.linear_(q|k|v|pos|out)\.weight$"),
    re.compile(r"^encoder\.layers\.\d+\.feed_forward[12]\.linear[12]\.weight$"),
    re.compile(r"^encoder\.pre_encode\.out\.weight$"),
    re.compile(r"^decoder\.prediction\.dec_rnn\.lstm\.(ih|hh)_l\d+\.weight$"),
    re.compile(r"^joint\.enc\.weight$"),
    re.compile(r"^joint\.pred\.weight$"),
    re.compile(r"^joint\.joint_net\.2\.weight$"),
    # ConformerConv pointwise convs are 1×1 — mathematically Linear. We squeeze
    # the k=1 dim in reshape_for_emission so the Q8_0/k-quant kernels see a
    # 2D (out, in) tensor; the runtime detects 2D stored shape and dispatches
    # the matmul through ggml_mul_mat (see Conv1D::build_graph).
    re.compile(r"^encoder\.layers\.\d+\.conv\.pointwise_conv[12]\.weight$"),
]


def _is_linear_weight(name: str) -> bool:
    return any(p.match(name) for p in _LINEAR_WEIGHT_PATTERNS)


# Depthwise Conv1D and subsampling Conv2D weights remain F16. Pointwise
# Conformer convolutions are handled as quantized Linear weights above.
_CONV_WEIGHT_PATTERN = re.compile(
    r"^(encoder\.layers\.\d+\.conv\.depthwise_conv" r"|encoder\.pre_encode\.conv\.\d+)\.weight$"
)
# Pointwise convs are treated as Linear (quantized); squeeze their k=1 dim
# before emission so the matrix is 2D.
_POINTWISE_PATTERN = re.compile(r"^encoder\.layers\.\d+\.conv\.pointwise_conv[12]\.weight$")
# Embedding (RNNT predictor): get_rows is dtype-agnostic, so F16 is safe.
_EMBED_WEIGHT_PATTERN = re.compile(r"^decoder\.prediction\.embed\.weight$")

# Encoder Q8 matrices can optionally be serialized in the exact two-plane
# layout consumed by the CUDA kernels: all int8 quants first, followed by all
# FP16 block scales. The byte count and GGUF logical type remain Q8_0; a model
# metadata key tells the runtime that encoder Q8 tensors use this layout.
_PLANAR_Q8_WEIGHT_PATTERN = re.compile(
    r"^(encoder\.pre_encode\.out\.weight"
    r"|encoder\.layers\.\d+\.feed_forward[12]\.linear[12]\.weight"
    r"|encoder\.layers\.\d+\.self_attn\.linear_(pos|out)\.weight"
    r"|encoder\.layers\.\d+\.conv\.pointwise_conv[12]\.weight)$"
)
_QKV_Q8_WEIGHT_PATTERN = re.compile(r"^encoder\.layers\.\d+\.self_attn\.linear_(q|k|v)\.weight$")


def _is_conv_weight(name: str) -> bool:
    return bool(_CONV_WEIGHT_PATTERN.match(name))


def _is_embed_weight(name: str) -> bool:
    return bool(_EMBED_WEIGHT_PATTERN.match(name))


def _is_planar_q8_weight(name: str) -> bool:
    return bool(_PLANAR_Q8_WEIGHT_PATTERN.match(name) or _QKV_Q8_WEIGHT_PATTERN.match(name))


def _q8_0_to_tensor_planar(packed: np.ndarray, n_per_row: int) -> np.ndarray:
    """block_q8_0 rows -> [all tensor qs][all tensor FP16 scales]."""
    if n_per_row % 32 != 0:
        raise ValueError(f"planar Q8 row width must be divisible by 32, got {n_per_row}")
    row_bytes = (n_per_row // 32) * 34
    raw = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1, row_bytes)
    blocks = raw.reshape(raw.shape[0], n_per_row // 32, 34)
    out = np.empty(raw.size, dtype=np.uint8)
    qs_bytes = raw.shape[0] * n_per_row
    out[:qs_bytes] = blocks[:, :, 2:34].reshape(-1)
    out[qs_bytes:] = blocks[:, :, 0:2].reshape(-1)
    return out.reshape(packed.shape)


def _pick_dtype(
    name: str, shape: tuple, linear_qtype: GGMLQuantizationType, default_qtype: GGMLQuantizationType
) -> tuple[GGMLQuantizationType, Optional[str]]:
    """Return (chosen GGMLQuantizationType, fallback reason or None).

    Dispatch by tensor class:
      * Linear weights         → user-selected `--weight-type` (linear_qtype).
      * Conv weights, embed    → F16 (regardless of --weight-type) — saves
                                  ~50% disk vs F32 with no runtime cost.
      * Everything else        → F32 (default_qtype) — norms, biases,
                                  pos_bias, pe table, etc. The runtime keeps
                                  these in F32 in memory.
    """
    if _is_linear_weight(name):
        inner = int(shape[-1]) if len(shape) >= 1 else 1
        if linear_qtype in _K_QUANTS and inner % _QK_K != 0:
            return GGMLQuantizationType.F16, "block alignment"
        if (
            linear_qtype in (GGMLQuantizationType.NVFP4, GGMLQuantizationType.MXFP4)
            and inner % 64 != 0
        ):
            return GGMLQuantizationType.F16, "block alignment"
        return linear_qtype, None
    if _is_conv_weight(name) or _is_embed_weight(name):
        return GGMLQuantizationType.F16, None
    return default_qtype, None


# Architecture / discriminator
KEY_ARCH = "general.architecture"
KEY_NAME = "general.name"
KEY_HEAD_TYPE = f"{ARCH}.head_type"

# Encoder hparams
KEY_ENC_D_MODEL = f"{ARCH}.encoder.d_model"
KEY_ENC_N_LAYERS = f"{ARCH}.encoder.n_layers"
KEY_ENC_N_HEADS = f"{ARCH}.encoder.n_heads"
KEY_ENC_D_FF = f"{ARCH}.encoder.d_ff"
KEY_ENC_CONV_KERNEL = f"{ARCH}.encoder.conv_kernel_size"
KEY_ENC_SUBSAMPLE = f"{ARCH}.encoder.subsampling_factor"
KEY_ENC_SUBSAMPLE_CONV_CHANNELS = f"{ARCH}.encoder.subsampling_conv_channels"
KEY_ENC_FEAT_IN = f"{ARCH}.encoder.feat_in"
KEY_ENC_XSCALING = f"{ARCH}.encoder.xscaling"
KEY_ENC_USE_BIAS = f"{ARCH}.encoder.use_bias"
KEY_ENC_Q8_LAYOUT = f"{ARCH}.encoder.q8_layout"
KEY_ENC_QKV_Q8_LAYOUT = f"{ARCH}.encoder.qkv_q8_layout"
KEY_ENC_PE_MAX_LEN = f"{ARCH}.encoder.pos_emb_max_len"
# Conformer block variants — these differ between parakeet and nemotron and
# need to match the C++ encoder code path that loads the model.
KEY_ENC_CONV_NORM = f"{ARCH}.encoder.conv_norm"  # "batch_norm" | "layer_norm"
KEY_ENC_CONV_CONTEXT = f"{ARCH}.encoder.conv_context"  # "symmetric" | "causal"
KEY_ENC_ATT_CONTEXT_STYLE = f"{ARCH}.encoder.att_context_style"  # "regular" | "chunked_limited"

# Cache-aware streaming params (RNNT only). For CTC these are 0.
KEY_ENC_CACHE_SUPPORTED = f"{ARCH}.encoder.cache_supported"
KEY_ENC_TRAIN_LEFT_CTX = f"{ARCH}.encoder.train_left_ctx"
KEY_ENC_TRAIN_RIGHT_CTX = f"{ARCH}.encoder.train_right_ctx"
KEY_ENC_OFFLINE_LEFT_CTX = f"{ARCH}.encoder.offline_left_ctx"
KEY_ENC_OFFLINE_RIGHT_CTX = f"{ARCH}.encoder.offline_right_ctx"

# Preprocessor hparams (FE)
KEY_FE_SAMPLE_RATE = f"{ARCH}.preprocessor.sample_rate"
KEY_FE_WINDOW_SIZE = f"{ARCH}.preprocessor.window_size"
KEY_FE_WINDOW_STRIDE = f"{ARCH}.preprocessor.window_stride"
KEY_FE_N_FFT = f"{ARCH}.preprocessor.n_fft"
KEY_FE_N_MELS = f"{ARCH}.preprocessor.features"
KEY_FE_NORMALIZE = f"{ARCH}.preprocessor.normalize"
KEY_FE_PREEMPH = f"{ARCH}.preprocessor.preemph"
KEY_FE_DITHER = f"{ARCH}.preprocessor.dither"
KEY_FE_STFT_CENTER_WINDOW = f"{ARCH}.preprocessor.stft_center_window"
KEY_FE_HANN_PERIODIC = f"{ARCH}.preprocessor.hann_periodic"
KEY_FE_MASK_INVALID_FRAMES = f"{ARCH}.preprocessor.mask_invalid_frames"

# CTC head
KEY_CTC_NUM_CLASSES = f"{ARCH}.ctc.num_classes"
KEY_CTC_BLANK_ID = f"{ARCH}.ctc.blank_id"

# RNNT head
KEY_RNNT_VOCAB_SIZE = f"{ARCH}.rnnt.vocab_size"
KEY_RNNT_BLANK_ID = f"{ARCH}.rnnt.blank_id"
KEY_RNNT_PRED_EMBED_DIM = f"{ARCH}.rnnt.pred_embed_dim"
KEY_RNNT_PRED_HIDDEN = f"{ARCH}.rnnt.pred_hidden"
KEY_RNNT_PRED_NUM_LAYERS = f"{ARCH}.rnnt.pred_num_layers"
KEY_RNNT_JOINT_DIM = f"{ARCH}.rnnt.joint_dim"
KEY_RNNT_MAX_SYMBOLS_PER_STEP = f"{ARCH}.rnnt.max_symbols_per_step"
# Language-ID prompt conditioning (EncDecRNNTBPEModelWithPrompt, e.g.
# nemotron-3.5). num_prompts is the one-hot width; the dictionary maps a
# language tag to its prompt slot, emitted as a string array of "lang:idx".
KEY_RNNT_NUM_PROMPTS = f"{ARCH}.rnnt.num_prompts"
KEY_RNNT_PROMPT_DICT = f"{ARCH}.rnnt.prompt_dictionary"
KEY_TDT_DURATIONS = f"{ARCH}.tdt.durations"

# Tokenizer
KEY_TOK_TYPE = f"{ARCH}.tokenizer.type"
KEY_TOK_VOCAB = f"{ARCH}.tokenizer.vocab"
KEY_TOK_MODEL = f"{ARCH}.tokenizer.spm_model"  # base64 SentencePiece model proto


# ===========================================================================
# .nemo unpacking
# ===========================================================================


extract_nemo = extract_archive
find_files = find_checkpoint_files
load_state_dict = _load_state_dict


def detect_head_type(cfg: dict, sd: dict) -> str:
    """Return 'ctc', 'rnnt', or 'tdt' from config + state_dict.

    Decisions:
        * model_config has a `joint` block ⇒ RNNT.
        * Otherwise, state_dict has joint/lstm tensors ⇒ RNNT.
        * Else ⇒ CTC.
    """
    if "joint" in cfg:
        decoding = cfg.get("decoding", {}) or {}
        durations = decoding.get("durations")
        num_extra = (cfg.get("joint", {}) or {}).get("num_extra_outputs", 0)
        if durations or int(num_extra or 0) > 0:
            return "tdt"
        return "rnnt"
    for k in sd.keys():
        if k.startswith("joint.") or "lstm" in k:
            return "rnnt"
    return "ctc"


# ===========================================================================
# Positional encoding (analytical, matches NeMo's RelPositionalEncoding)
# ===========================================================================


def build_pe(d_model: int, max_len: int) -> np.ndarray:
    positions = torch.arange(max_len - 1, -max_len, -1, dtype=torch.float32).unsqueeze(1)
    pe = torch.zeros(positions.size(0), d_model, dtype=torch.float32)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe.numpy()


# ===========================================================================
# Tensor name + shape remapping
# ===========================================================================

# Tensors to drop entirely regardless of head type.
SKIP_PREFIXES = (
    "preprocessor.",  # We rebuild mel filterbank C++-side from metadata.
    "spec_augmentation.",
    "ctc_loss.",
    "wer.",
    "loss.",
    "_target_",
    "transcriber.",  # Some older RNNT checkpoints have this — unused.
)


def remap_for_ctc(name: str) -> Optional[str]:
    """CTC remapping. We keep encoder.* + decoder.decoder_layers.0.{weight,bias}."""
    if name.startswith(SKIP_PREFIXES):
        return None
    if name.startswith("joint."):
        return None
    if name.endswith(".num_batches_tracked"):
        return None
    if name.endswith(".inv_freq"):
        return None
    if name == "encoder.pos_enc.pe":
        return None  # rebuilt analytically
    return name


# Rename PyTorch LSTM tensor names into a form compatible with our runtime's
# Linear module (`{name}.weight` / `{name}.bias`).
#
#   decoder.prediction.dec_rnn.lstm.weight_ih_l0  →
#                              ↘ decoder.prediction.dec_rnn.lstm.ih_l0.weight
#   decoder.prediction.dec_rnn.lstm.bias_hh_l1    →
#                              ↘ decoder.prediction.dec_rnn.lstm.hh_l1.bias
_LSTM_RE = re.compile(r"(.*\.lstm)\.(weight|bias)_(ih|hh)_l(\d+)$")


def remap_for_rnnt(name: str) -> Optional[str]:
    if name.startswith(SKIP_PREFIXES):
        return None
    if name.endswith(".num_batches_tracked"):
        return None
    if name.endswith(".inv_freq"):
        return None
    if name == "encoder.pos_enc.pe":
        return None
    m = _LSTM_RE.match(name)
    if m:
        # `{lstm_prefix}.{ih|hh}_l{n}.{weight|bias}`
        prefix, weight_or_bias, ih_hh, layer_n = m.groups()
        return f"{prefix}.{ih_hh}_l{layer_n}.{weight_or_bias}"
    return name


def reshape_for_emission(name: str, arr: np.ndarray) -> np.ndarray:
    """Per-tensor shape adjustments before emission.

    Gut check: PyTorch state_dict 1D conv weights are (out, in, k); after
    `GGUFWriter.add_tensor` stores them, the gguf reader sees the dims
    reversed to [k, in, out] — which is exactly what our runtime's
    `Conv1D::define_tensors` registers (kernel, in, out). So passing the
    PyTorch-native shape straight through is correct; no transpose needed.
    Same for 4D Conv2D weights.

    Exception: pointwise convs (k=1) get the trailing k dim squeezed so
    their weight matrix is 2D (out, in). This serves two purposes:
      * Q8_0 / k-quant block sizes apply along the last numpy dim; with
        k=1 as the last dim, no quantization fits (1 not divisible by 32).
        After squeeze, the last dim is `in` (1024 or 2048), which divides.
      * The runtime's `Conv1D::build_graph` detects 2D stored weights and
        dispatches to `ggml_mul_mat` (with an input permute) — strictly
        cheaper than going through `ggml_conv_1d` for k=1.
    """
    if _POINTWISE_PATTERN.match(name) and arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(axis=-1)
    return arr


# ===========================================================================
# Conversion driver
# ===========================================================================


def convert(
    nemo_path: Path,
    out_path: Path,
    head_override: Optional[str],
    weight_type: str = "q8_0",
    q8_layout: str = "block",
) -> None:
    if weight_type not in WEIGHT_TYPES:
        raise ValueError(
            f"unknown --weight-type {weight_type!r}; " f"choose from {sorted(WEIGHT_TYPES)}"
        )
    linear_qtype, default_qtype, file_type_value = WEIGHT_TYPES[weight_type]
    if q8_layout not in ("block", "planar"):
        raise ValueError(f"unknown Q8 layout {q8_layout!r}; choose block or planar")
    if q8_layout == "planar" and weight_type != "q8_0":
        raise ValueError("--q8-layout planar requires --weight-type q8_0")
    print(
        f"[convert] weight_type = {weight_type} "
        f"(linear={linear_qtype.name}, default={default_qtype.name})"
    )

    print(
        f"[convert] {'using unpacked checkpoint' if nemo_path.is_dir() else 'unpacking'} {nemo_path}"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = nemo_path if nemo_path.is_dir() else Path(tmp)
        if not nemo_path.is_dir():
            extract_nemo(nemo_path, root)
        files = find_files(root)
        if files["config"] is None or files["weights"] is None:
            raise RuntimeError(f"Missing artifacts in {nemo_path}: {files}")
        print(f"[convert]   config    = {files['config']}")
        print(f"[convert]   weights   = {files['weights']}")
        print(f"[convert]   tokenizer = {files['tokenizer']}")

        # Force UTF-8: NeMo model_config.yaml can carry non-ASCII bytes, and on
        # Windows open() defaults to the locale codepage (cp1252), which raises
        # UnicodeDecodeError. The TTS/codec converters already pin utf-8.
        with open(files["config"], "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        sd = load_state_dict(files["weights"])
        print(f"[convert] state_dict has {len(sd)} tensors")

        head_type = head_override or detect_head_type(cfg, sd)
        print(f"[convert] head_type = {head_type}")

        # ---- Encoder hparams ----
        enc_cfg = cfg.get("encoder", {})
        if "params" in enc_cfg:
            enc_cfg = enc_cfg["params"]
        d_model = int(enc_cfg["d_model"])
        n_layers = int(enc_cfg["n_layers"])
        n_heads = int(enc_cfg["n_heads"])
        ff_exp = int(enc_cfg.get("ff_expansion_factor", 4))
        d_ff = d_model * ff_exp
        conv_kernel = int(enc_cfg["conv_kernel_size"])
        subsample = int(enc_cfg.get("subsampling_factor", 8))
        subsample_conv_channels = int(enc_cfg.get("subsampling_conv_channels", 256))
        feat_in = int(enc_cfg["feat_in"])
        xscaling = bool(enc_cfg.get("xscaling", True))
        use_bias = bool(enc_cfg.get("use_bias", True))
        pe_max_len = int(enc_cfg.get("pos_emb_max_len", 5000))

        # `att_context_size` is either a single `[left, right]` pair (offline /
        # full-attention models like parakeet-ctc) or a list of pairs
        # (cache-aware-trained models like nemotron, which support multiple
        # (left, right) chunk modes at inference time).
        att_ctx_raw = enc_cfg.get("att_context_size", [-1, -1])
        if (
            isinstance(att_ctx_raw, list)
            and len(att_ctx_raw) > 0
            and isinstance(att_ctx_raw[0], list)
        ):
            # List of pairs — NeMo's active/default attention preset is the
            # first entry. Preserve that for full-utterance inference; the
            # cache-aware runtime can still select another right context.
            pairs = att_ctx_raw
            default_context = pairs[0]
            train_left_ctx, train_right_ctx = int(default_context[0]), int(default_context[1])
            offline_left_ctx, offline_right_ctx = train_left_ctx, train_right_ctx
            cache_supported = True
        elif isinstance(att_ctx_raw, list) and len(att_ctx_raw) >= 2:
            offline_left_ctx = int(att_ctx_raw[0])
            offline_right_ctx = int(att_ctx_raw[1])
            train_left_ctx = int(att_ctx_raw[0]) if att_ctx_raw[0] != -1 else 0
            train_right_ctx = int(att_ctx_raw[1]) if att_ctx_raw[1] != -1 else 0
            cache_supported = att_ctx_raw[0] != -1
        else:
            train_left_ctx = 0
            train_right_ctx = 0
            offline_left_ctx = -1
            offline_right_ctx = -1
            cache_supported = False

        conv_norm = str(enc_cfg.get("conv_norm_type", "batch_norm"))
        conv_context = str(enc_cfg.get("conv_context_size", "symmetric"))
        att_context_style = str(enc_cfg.get("att_context_style", "regular"))
        cache_supported = cache_supported and att_context_style != "regular"

        # ---- Preprocessor hparams ----
        pp_cfg = cfg.get("preprocessor", {})
        if "params" in pp_cfg:
            pp_cfg = pp_cfg["params"]

        # ---- Head config ----
        if head_type == "ctc":
            dec_cfg = cfg.get("decoder", {})
            if "params" in dec_cfg:
                dec_cfg = dec_cfg["params"]
            num_classes = int(dec_cfg.get("num_classes", -1))
            if num_classes <= 0:
                vocab_list = dec_cfg.get("vocabulary", [])
                num_classes = len(vocab_list) if vocab_list else 0
            blank_id = num_classes  # NeMo CTC convention
        else:  # rnnt / tdt
            joint_cfg = cfg.get("joint", {})
            if "params" in joint_cfg:
                joint_cfg = joint_cfg["params"]
            pred_cfg = cfg.get("decoder", {})
            if "params" in pred_cfg:
                pred_cfg = pred_cfg["params"]
            # vocab_size in NeMo RNNT excludes blank; we add 1 for it.
            vocab_n = int(pred_cfg.get("vocab_size", joint_cfg.get("num_classes", 0)))
            if vocab_n <= 0:
                # Fallback: derive from tokenizer pieces, set later.
                vocab_n = 1024
            vocab_size = vocab_n + 1
            blank_id = vocab_size - 1
            pred_embed_dim = int(pred_cfg.get("prednet", {}).get("pred_hidden", 640))
            pred_hidden = pred_embed_dim
            pred_num_layers = int(pred_cfg.get("prednet", {}).get("pred_rnn_layers", 2))
            joint_dim = int(joint_cfg.get("jointnet", {}).get("joint_hidden", 640))
            greedy_cfg = (cfg.get("decoding", {}) or {}).get("greedy", {}) or {}
            max_symbols_per_step = int(
                greedy_cfg.get("max_symbols", greedy_cfg.get("max_symbols_per_step", 10)) or 10
            )
            # Language-ID prompt conditioning lives under model_defaults for
            # prompt-conditioned multilingual models (nemotron-3.5). Absent =>
            # plain RNNT (num_prompts stays 0, no prompt metadata emitted).
            md_cfg = cfg.get("model_defaults", {}) or {}
            num_prompts = int(md_cfg.get("num_prompts", 0) or 0)
            prompt_dictionary = dict(md_cfg.get("prompt_dictionary", {}) or {})
            tdt_durations = []
            if head_type == "tdt":
                tdt_durations = list(
                    (cfg.get("decoding", {}) or {}).get("durations")
                    or md_cfg.get("tdt_durations")
                    or []
                )
                if not tdt_durations:
                    raise ValueError("TDT model has no decoding.durations metadata")

        print(
            f"[convert] d_model={d_model} n_layers={n_layers} n_heads={n_heads} "
            f"d_ff={d_ff} conv_kernel={conv_kernel} subsample={subsample} "
            f"feat_in={feat_in} xscaling={xscaling}"
        )
        if head_type in ("rnnt", "tdt"):
            print(
                f"[convert] rnnt vocab_size={vocab_size} blank_id={blank_id} "
                f"pred_hidden={pred_hidden} joint_dim={joint_dim} "
                f"max_symbols_per_step={max_symbols_per_step}"
            )
            print(
                f"[convert] cache_supported={cache_supported} "
                f"train_left_ctx={train_left_ctx} train_right_ctx={train_right_ctx}"
            )

        # ---- Initialize GGUF writer ----
        gw = GGUFWriter(str(out_path), arch=ARCH)
        gw.add_architecture()
        gw.add_string(KEY_NAME, cfg.get("name", out_path.stem))
        gw.add_string(KEY_HEAD_TYPE, head_type)
        gw.add_uint32("general.file_type", file_type_value)

        gw.add_uint32(KEY_ENC_D_MODEL, d_model)
        gw.add_uint32(KEY_ENC_N_LAYERS, n_layers)
        gw.add_uint32(KEY_ENC_N_HEADS, n_heads)
        gw.add_uint32(KEY_ENC_D_FF, d_ff)
        gw.add_uint32(KEY_ENC_CONV_KERNEL, conv_kernel)
        gw.add_uint32(KEY_ENC_SUBSAMPLE, subsample)
        gw.add_uint32(KEY_ENC_SUBSAMPLE_CONV_CHANNELS, subsample_conv_channels)
        gw.add_uint32(KEY_ENC_FEAT_IN, feat_in)
        gw.add_bool(KEY_ENC_XSCALING, xscaling)
        gw.add_bool(KEY_ENC_USE_BIAS, use_bias)
        gw.add_string(
            KEY_ENC_Q8_LAYOUT, "tensor_planar_v1" if q8_layout == "planar" else "block_q8_0"
        )
        gw.add_string(
            KEY_ENC_QKV_Q8_LAYOUT,
            "tensor_planar_v1" if q8_layout == "planar" else "block_q8_0",
        )
        gw.add_uint32(KEY_ENC_PE_MAX_LEN, pe_max_len)
        gw.add_bool(KEY_ENC_CACHE_SUPPORTED, cache_supported)
        gw.add_uint32(KEY_ENC_TRAIN_LEFT_CTX, train_left_ctx)
        gw.add_uint32(KEY_ENC_TRAIN_RIGHT_CTX, train_right_ctx)
        gw.add_int32(KEY_ENC_OFFLINE_LEFT_CTX, offline_left_ctx)
        gw.add_int32(KEY_ENC_OFFLINE_RIGHT_CTX, offline_right_ctx)
        gw.add_string(KEY_ENC_CONV_NORM, conv_norm)
        gw.add_string(KEY_ENC_CONV_CONTEXT, conv_context)
        gw.add_string(KEY_ENC_ATT_CONTEXT_STYLE, att_context_style)

        gw.add_uint32(KEY_FE_SAMPLE_RATE, int(pp_cfg.get("sample_rate", 16000)))
        gw.add_float32(KEY_FE_WINDOW_SIZE, float(pp_cfg.get("window_size", 0.025)))
        gw.add_float32(KEY_FE_WINDOW_STRIDE, float(pp_cfg.get("window_stride", 0.01)))
        gw.add_uint32(KEY_FE_N_FFT, int(pp_cfg.get("n_fft", 512)))
        gw.add_uint32(KEY_FE_N_MELS, int(pp_cfg.get("features", feat_in)))
        gw.add_string(KEY_FE_NORMALIZE, str(pp_cfg.get("normalize", "per_feature")))
        gw.add_float32(KEY_FE_PREEMPH, float(pp_cfg.get("preemph", 0.97) or 0.0))
        gw.add_float32(KEY_FE_DITHER, float(pp_cfg.get("dither", 1e-5)))
        # NeMo FilterbankFeatures / the Transformers Nemotron processor use a
        # symmetric Hann centered inside n_fft. center=True emits one trailing
        # padded frame, which their attention mask marks invalid. Serialize all
        # three choices so the runtime never has to guess frontend geometry.
        gw.add_bool(KEY_FE_STFT_CENTER_WINDOW, True)
        gw.add_bool(KEY_FE_HANN_PERIODIC, False)
        gw.add_bool(KEY_FE_MASK_INVALID_FRAMES, True)

        if head_type == "ctc":
            gw.add_uint32(KEY_CTC_NUM_CLASSES, num_classes)
            gw.add_uint32(KEY_CTC_BLANK_ID, blank_id)
        else:
            gw.add_uint32(KEY_RNNT_VOCAB_SIZE, vocab_size)
            gw.add_uint32(KEY_RNNT_BLANK_ID, blank_id)
            gw.add_uint32(KEY_RNNT_PRED_EMBED_DIM, pred_embed_dim)
            gw.add_uint32(KEY_RNNT_PRED_HIDDEN, pred_hidden)
            gw.add_uint32(KEY_RNNT_PRED_NUM_LAYERS, pred_num_layers)
            gw.add_uint32(KEY_RNNT_JOINT_DIM, joint_dim)
            gw.add_uint32(KEY_RNNT_MAX_SYMBOLS_PER_STEP, max_symbols_per_step)
            gw.add_uint32(KEY_RNNT_NUM_PROMPTS, num_prompts)
            if head_type == "tdt":
                gw.add_array(KEY_TDT_DURATIONS, [int(d) for d in tdt_durations])
            if prompt_dictionary:
                gw.add_array(
                    KEY_RNNT_PROMPT_DICT,
                    [f"{k}:{int(v)}" for k, v in prompt_dictionary.items()],
                )
                print(
                    f"[convert] prompt-conditioned: num_prompts={num_prompts} "
                    f"languages={len(prompt_dictionary)}"
                )

        # ---- Tokenizer ----
        if files["tokenizer"] is not None:
            import base64

            import sentencepiece as spm

            proto_bytes = open(files["tokenizer"], "rb").read()
            sp = spm.SentencePieceProcessor()
            sp.LoadFromSerializedProto(proto_bytes)
            vocab = [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]
            gw.add_string(KEY_TOK_TYPE, "sentencepiece_bpe")
            gw.add_array(KEY_TOK_VOCAB, vocab)
            # Embed the full SentencePiece model (base64) so the C++ runtime can
            # tokenize arbitrary text exactly as the model does - needed for RNNT
            # word boosting (context biasing). base64 keeps it a valid UTF-8 GGUF
            # string value.
            gw.add_string(KEY_TOK_MODEL, base64.b64encode(proto_bytes).decode("ascii"))
        else:
            gw.add_string(KEY_TOK_TYPE, "none")

        # ---- Positional encoding (analytical) ----
        # Always emit as F32; it's a relatively small lookup table that the
        # encoder slices by position at runtime.
        pe = build_pe(d_model, pe_max_len)
        gw.add_tensor("encoder.pos_enc.pe", pe, raw_dtype=GGMLQuantizationType.F32)

        # ---- Mel filterbank (Slaney-normalized) ----
        # Match NeMo's Slaney-normalized filterbank. The runtime's analytical
        # fallback uses unit-peak triangles, which can shift WER on noisy audio.
        fb_sr = int(pp_cfg.get("sample_rate", 16000))
        fb_n_fft = int(pp_cfg.get("n_fft", 512))
        fb_n_mels = int(pp_cfg.get("features", feat_in))
        try:
            import librosa  # type: ignore

            fb = librosa.filters.mel(
                sr=fb_sr,
                n_fft=fb_n_fft,
                n_mels=fb_n_mels,
                fmin=0.0,
                fmax=fb_sr / 2.0,
                norm="slaney",
                htk=False,
            ).astype(np.float32)
        except ImportError:
            print(
                "WARN: librosa not installed; skipping asr.preprocessor.fb. "
                "Runtime will fall back to unit-peak triangles (lossy on noisy audio).",
                file=sys.stderr,
            )
            fb = None
        if fb is not None:
            gw.add_tensor("preprocessor.fb", fb, raw_dtype=GGMLQuantizationType.F32)

        # ---- Walk state_dict and emit tensors ----
        remap = remap_for_ctc if head_type == "ctc" else remap_for_rnnt
        emitted, skipped = 0, 0
        skipped_keys: list[str] = []
        # Tally of how many tensors got each output dtype, for the post-run
        # summary.
        dtype_counts: dict[str, int] = {}
        for src_name, tensor in sd.items():
            dst_name = remap(src_name)
            if dst_name is None:
                skipped += 1
                skipped_keys.append(src_name)
                continue
            arr = tensor.detach().cpu().float().numpy()
            arr = reshape_for_emission(dst_name, arr)

            chosen_qtype, fallback_reason = _pick_dtype(
                dst_name, arr.shape, linear_qtype, default_qtype
            )
            if fallback_reason is not None:
                print(f"[convert] kept fp16 ({fallback_reason}): {dst_name}")

            if chosen_qtype in (GGMLQuantizationType.F32,):
                gw.add_tensor(dst_name, arr.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
            elif chosen_qtype == GGMLQuantizationType.F16:
                gw.add_tensor(dst_name, arr.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
            else:
                # BF16, Q8_0, Q4_K, Q5_K, Q6_K — go through the (possibly
                # ctypes-backed) quantizer, which returns a packed uint8 byte
                # buffer matching the layout ggml expects on-disk.
                packed = _quantize(arr, chosen_qtype)
                if (
                    q8_layout == "planar"
                    and chosen_qtype == GGMLQuantizationType.Q8_0
                    and _is_planar_q8_weight(dst_name)
                ):
                    packed = _q8_0_to_tensor_planar(packed, int(arr.shape[-1]))
                # `packed` is a uint8 byte-shape array (e.g. Q8_0 with row=1024
                # → 1088 bytes/row). gguf's add_tensor_info converts byte-shape
                # back to logical shape via quant_shape_from_byte_shape when
                # tensor.dtype is uint8 + raw_dtype is set. Leave raw_shape
                # unset so the byte shape from `packed.shape` flows through.
                gw.add_tensor(dst_name, packed, raw_dtype=chosen_qtype)

            dtype_counts[chosen_qtype.name] = dtype_counts.get(chosen_qtype.name, 0) + 1
            emitted += 1

        print(f"[convert] emitted {emitted} tensors, skipped {skipped}")
        print(
            f"[convert] dtype tally: "
            + ", ".join(f"{k}={v}" for k, v in sorted(dtype_counts.items()))
        )
        if skipped:
            print(f"[convert] skipped (first 10): {skipped_keys[:10]}")

        gw.write_header_to_file()
        gw.write_kv_data_to_file()
        gw.write_tensors_to_file()
        gw.close()
        print(f"[convert] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
