#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a NeMo streaming Sortformer diarization checkpoint to GGUF.

Target model: nvidia/diar_streaming_sortformer_4spk-v2 (SortformerEncLabelModel):
  * `encoder.*`      — 17-layer NEST Fast-Conformer (d512, 8x dw_striding) —
                       tensor names identical to the ASR encoder, so the C++
                       FastConformerEncoder loads them unchanged.
  * `transformer_encoder.*` — 18-layer post-LN Transformer (d192, inner 768)
                       → emitted as `transformer.*`.
  * `sortformer_modules.*`  — encoder_proj (512→192) + the sigmoid head
                       (first_hidden_to_hidden 192→192, single_hidden_to_spks
                       192→4) → emitted as `encoder_proj.*` / `head.*`.
                       `hidden_to_spks` (2*192→4) is unused at inference and
                       skipped.
  * `preprocessor.featurizer.fb` — the trained mel filterbank; emitted
                       verbatim as `preprocessor.fb` (no librosa rebuild).

Usage:
    python3 convert_model.py <src.nemo|hf-repo-id> --outfile <out.gguf> \
        [--outtype f32|f16|bf16|q8_0|...]

Weight-type default is f32 (bit-tight parity with NeMo); fp16/q8_0 are for
deployment once parity is established.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from gguf import GGMLQuantizationType, GGUFWriter

from .asr import build_pe, find_files, load_state_dict
from .quantization import DEPLOYMENT_WEIGHT_TYPES, K_QUANTS, QK_K, quantize
from .source import extracted_checkpoint

ARCH = "sortformer"

# All-F32 emission for parity work, on top of the shared deployment types.
WEIGHT_TYPES = dict(DEPLOYMENT_WEIGHT_TYPES)
WEIGHT_TYPES["f32"] = (GGMLQuantizationType.F32, GGMLQuantizationType.F32, 0)  # ALL_F32

# ---------------------------------------------------------------------------
# Metadata keys
# ---------------------------------------------------------------------------

KEY_ARCH = "general.architecture"
KEY_NAME = "general.name"

# NEST encoder — same key layout as asr.encoder.* so config parsing stays
# uniform between the ASR and diarizer loaders.
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
KEY_ENC_PE_MAX_LEN = f"{ARCH}.encoder.pos_emb_max_len"
KEY_ENC_CONV_NORM = f"{ARCH}.encoder.conv_norm"
KEY_ENC_CONV_CONTEXT = f"{ARCH}.encoder.conv_context"
KEY_ENC_ATT_CONTEXT_STYLE = f"{ARCH}.encoder.att_context_style"

# Transformer head
KEY_TF_N_LAYERS = f"{ARCH}.transformer.n_layers"
KEY_TF_HIDDEN = f"{ARCH}.transformer.hidden_size"
KEY_TF_INNER = f"{ARCH}.transformer.inner_size"
KEY_TF_N_HEADS = f"{ARCH}.transformer.n_heads"
KEY_TF_PRE_LN = f"{ARCH}.transformer.pre_ln"

KEY_NUM_SPEAKERS = f"{ARCH}.num_speakers"

# Preprocessor (FE)
KEY_FE_SAMPLE_RATE = f"{ARCH}.preprocessor.sample_rate"
KEY_FE_WINDOW_SIZE = f"{ARCH}.preprocessor.window_size"
KEY_FE_WINDOW_STRIDE = f"{ARCH}.preprocessor.window_stride"
KEY_FE_N_FFT = f"{ARCH}.preprocessor.n_fft"
KEY_FE_N_MELS = f"{ARCH}.preprocessor.features"
KEY_FE_NORMALIZE = f"{ARCH}.preprocessor.normalize"
KEY_FE_PREEMPH = f"{ARCH}.preprocessor.preemph"
KEY_FE_DITHER = f"{ARCH}.preprocessor.dither"
KEY_FE_LOG_ZERO_GUARD = f"{ARCH}.preprocessor.log_zero_guard"

# AOSC scoring constants (model-tied; the compression algorithm must use the
# values the checkpoint was trained/tuned with).
KEY_SC_SIL_FRAMES_PER_SPK = f"{ARCH}.scoring.spkcache_sil_frames_per_spk"
KEY_SC_PRED_THRESHOLD = f"{ARCH}.scoring.pred_score_threshold"
KEY_SC_BOOST_LATEST = f"{ARCH}.scoring.scores_boost_latest"
KEY_SC_SIL_THRESHOLD = f"{ARCH}.scoring.sil_threshold"
KEY_SC_STRONG_BOOST_RATE = f"{ARCH}.scoring.strong_boost_rate"
KEY_SC_WEAK_BOOST_RATE = f"{ARCH}.scoring.weak_boost_rate"
KEY_SC_MIN_POS_RATE = f"{ARCH}.scoring.min_pos_scores_rate"

# Streaming geometry as configured in the checkpoint - PROVENANCE ONLY (the
# training-time values, e.g. 188/0/188). The runtime never reads these keys;
# its geometry comes from DiarGeometry/DiarConfig presets.
KEY_ST_SPKCACHE_LEN = f"{ARCH}.streaming.spkcache_len"
KEY_ST_FIFO_LEN = f"{ARCH}.streaming.fifo_len"
KEY_ST_CHUNK_LEN = f"{ARCH}.streaming.chunk_len"
KEY_ST_UPDATE_PERIOD = f"{ARCH}.streaming.spkcache_update_period"
KEY_ST_CHUNK_LC = f"{ARCH}.streaming.chunk_left_context"
KEY_ST_CHUNK_RC = f"{ARCH}.streaming.chunk_right_context"

# NeMo SortformerModules constructor defaults, used when the checkpoint's
# model_config omits the scoring block (the released 4spk-v2 does).
_SCORING_DEFAULTS = {
    "spkcache_sil_frames_per_spk": 3,
    "pred_score_threshold": 0.25,
    "scores_boost_latest": 0.05,
    "sil_threshold": 0.2,
    "strong_boost_rate": 0.75,
    "weak_boost_rate": 1.5,
    "min_pos_scores_rate": 0.5,
}

# ---------------------------------------------------------------------------
# Tensor name remapping
# ---------------------------------------------------------------------------

_SKIP_EXACT = {
    "encoder.pos_enc.pe",  # rebuilt analytically
    "preprocessor.featurizer.fb",  # emitted manually as preprocessor.fb
    "preprocessor.featurizer.window",  # FE builds the hann window from config
    "sortformer_modules.hidden_to_spks.weight",  # unused at inference
    "sortformer_modules.hidden_to_spks.bias",
}
_SKIP_PREFIXES = ("spec_augmentation.", "loss.")


def remap(name: str) -> Optional[str]:
    if name in _SKIP_EXACT or name.startswith(_SKIP_PREFIXES):
        return None
    if name.endswith(".num_batches_tracked"):
        return None
    if name.startswith("sortformer_modules.encoder_proj."):
        return name.replace("sortformer_modules.", "", 1)
    if name.startswith("sortformer_modules."):
        # first_hidden_to_hidden / single_hidden_to_spks
        return name.replace("sortformer_modules.", "head.", 1)
    if name.startswith("transformer_encoder."):
        return name.replace("transformer_encoder.", "transformer.", 1)
    if name.startswith("encoder."):
        return name
    print(f"[convert] WARNING: unrecognized tensor skipped: {name}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Dtype dispatch shared with ASR, extended for the Sortformer names.
# ---------------------------------------------------------------------------

_LINEAR_WEIGHT_PATTERNS = [
    re.compile(r"^encoder\.layers\.\d+\.self_attn\.linear_(q|k|v|pos|out)\.weight$"),
    re.compile(r"^encoder\.layers\.\d+\.feed_forward[12]\.linear[12]\.weight$"),
    re.compile(r"^encoder\.pre_encode\.out\.weight$"),
    re.compile(r"^encoder\.layers\.\d+\.conv\.pointwise_conv[12]\.weight$"),
    re.compile(r"^encoder_proj\.weight$"),
    re.compile(
        r"^transformer\.layers\.\d+\."
        r"(first_sub_layer\.(query_net|key_net|value_net|out_projection)"
        r"|second_sub_layer\.dense_(in|out))\.weight$"
    ),
    # head.* weights are tiny (192x192 / 4x192): always F32, not listed here.
]
_CONV_WEIGHT_PATTERN = re.compile(
    r"^(encoder\.layers\.\d+\.conv\.depthwise_conv" r"|encoder\.pre_encode\.conv\.\d+)\.weight$"
)
_POINTWISE_PATTERN = re.compile(r"^encoder\.layers\.\d+\.conv\.pointwise_conv[12]\.weight$")


def _pick_dtype(
    name: str, shape: tuple, linear_qtype: GGMLQuantizationType, default_qtype: GGMLQuantizationType
) -> tuple[GGMLQuantizationType, Optional[str]]:
    if any(p.match(name) for p in _LINEAR_WEIGHT_PATTERNS):
        inner = int(shape[-1]) if len(shape) >= 1 else 1
        if linear_qtype in K_QUANTS and inner % QK_K != 0:
            return GGMLQuantizationType.F16, "block alignment"
        if (
            linear_qtype in (GGMLQuantizationType.NVFP4, GGMLQuantizationType.MXFP4)
            and inner % 64 != 0
        ):
            return GGMLQuantizationType.F16, "block alignment"
        if linear_qtype == GGMLQuantizationType.Q8_0 and inner % 32 != 0:
            return GGMLQuantizationType.F16, "block alignment"
        return linear_qtype, None
    if _CONV_WEIGHT_PATTERN.match(name):
        # ggml's CPU im2col path asserts F16 conv kernels; keep F16 even in
        # --weight-type f32 mode (the runtime upcasts activations, parity
        # impact is below the test tolerance — same as the ASR GGUFs).
        return GGMLQuantizationType.F16, None
    return default_qtype, None


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------


def convert(nemo_path: Path, out_path: Path, weight_type: str) -> None:
    if weight_type not in WEIGHT_TYPES:
        raise ValueError(
            f"unknown --weight-type {weight_type!r}; choose from {sorted(WEIGHT_TYPES)}"
        )
    linear_qtype, default_qtype, file_type_value = WEIGHT_TYPES[weight_type]
    print(f"[convert] weight_type = {weight_type} (linear={linear_qtype.name})")

    print(f"[convert] reading {nemo_path}")
    with extracted_checkpoint(nemo_path, "sortformer-convert-") as tmp:
        files = find_files(tmp)
        if files["config"] is None or files["weights"] is None:
            raise RuntimeError(f"Missing artifacts in {nemo_path}: {files}")

        with open(files["config"], "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sd = load_state_dict(files["weights"])
        print(f"[convert] state_dict has {len(sd)} tensors")

        enc_cfg = cfg["encoder"]
        pp_cfg = cfg["preprocessor"]
        tf_cfg = cfg["transformer_encoder"]
        sm_cfg = cfg.get("sortformer_modules", {}) or {}

        d_model = int(enc_cfg["d_model"])
        n_layers = int(enc_cfg["n_layers"])
        d_ff = d_model * int(enc_cfg.get("ff_expansion_factor", 4))
        pe_max_len = int(enc_cfg.get("pos_emb_max_len", 5000))
        feat_in = int(enc_cfg["feat_in"])
        num_speakers = int(cfg.get("max_num_of_spks", sm_cfg.get("num_spks", 4)))

        if str(enc_cfg.get("self_attention_model", "rel_pos")) != "rel_pos":
            raise RuntimeError("only rel_pos NEST encoders are supported")

        print(
            f"[convert] encoder d_model={d_model} n_layers={n_layers} "
            f"n_heads={enc_cfg['n_heads']} d_ff={d_ff} feat_in={feat_in}; "
            f"transformer n_layers={tf_cfg['num_layers']} hidden={tf_cfg['hidden_size']}; "
            f"num_speakers={num_speakers}"
        )

        gw = GGUFWriter(str(out_path), arch=ARCH)
        gw.add_architecture()
        gw.add_string(KEY_NAME, cfg.get("name", out_path.stem))
        gw.add_uint32("general.file_type", file_type_value)

        gw.add_uint32(KEY_ENC_D_MODEL, d_model)
        gw.add_uint32(KEY_ENC_N_LAYERS, n_layers)
        gw.add_uint32(KEY_ENC_N_HEADS, int(enc_cfg["n_heads"]))
        gw.add_uint32(KEY_ENC_D_FF, d_ff)
        gw.add_uint32(KEY_ENC_CONV_KERNEL, int(enc_cfg["conv_kernel_size"]))
        gw.add_uint32(KEY_ENC_SUBSAMPLE, int(enc_cfg.get("subsampling_factor", 8)))
        gw.add_uint32(
            KEY_ENC_SUBSAMPLE_CONV_CHANNELS, int(enc_cfg.get("subsampling_conv_channels", 256))
        )
        gw.add_uint32(KEY_ENC_FEAT_IN, feat_in)
        gw.add_bool(KEY_ENC_XSCALING, bool(enc_cfg.get("xscaling", True)))
        gw.add_bool(KEY_ENC_USE_BIAS, bool(enc_cfg.get("use_bias", True)))
        gw.add_uint32(KEY_ENC_PE_MAX_LEN, pe_max_len)
        gw.add_string(KEY_ENC_CONV_NORM, str(enc_cfg.get("conv_norm_type", "batch_norm")))
        gw.add_string(KEY_ENC_CONV_CONTEXT, "symmetric")
        gw.add_string(KEY_ENC_ATT_CONTEXT_STYLE, str(enc_cfg.get("att_context_style", "regular")))

        gw.add_uint32(KEY_TF_N_LAYERS, int(tf_cfg["num_layers"]))
        gw.add_uint32(KEY_TF_HIDDEN, int(tf_cfg["hidden_size"]))
        gw.add_uint32(KEY_TF_INNER, int(tf_cfg["inner_size"]))
        gw.add_uint32(KEY_TF_N_HEADS, int(tf_cfg["num_attention_heads"]))
        gw.add_bool(KEY_TF_PRE_LN, bool(tf_cfg.get("pre_ln", False)))
        if str(tf_cfg.get("hidden_act", "relu")) != "relu":
            raise RuntimeError("only relu transformer FF activation is supported")

        gw.add_uint32(KEY_NUM_SPEAKERS, num_speakers)

        gw.add_uint32(KEY_FE_SAMPLE_RATE, int(pp_cfg.get("sample_rate", 16000)))
        gw.add_float32(KEY_FE_WINDOW_SIZE, float(pp_cfg.get("window_size", 0.025)))
        gw.add_float32(KEY_FE_WINDOW_STRIDE, float(pp_cfg.get("window_stride", 0.01)))
        gw.add_uint32(KEY_FE_N_FFT, int(pp_cfg.get("n_fft", 512)))
        gw.add_uint32(KEY_FE_N_MELS, int(pp_cfg.get("features", feat_in)))
        gw.add_string(KEY_FE_NORMALIZE, str(pp_cfg.get("normalize", "NA")))
        gw.add_float32(KEY_FE_PREEMPH, float(pp_cfg.get("preemph", 0.97) or 0.0))
        gw.add_float32(KEY_FE_DITHER, float(pp_cfg.get("dither", 1e-5)))
        # NeMo FilterbankFeatures default (log_zero_guard_type="add") is 2**-24;
        # the released checkpoint config doesn't override it. Emitting it keeps
        # the C++ FE's log floor identical to training (silence bins hit it).
        gw.add_float32(KEY_FE_LOG_ZERO_GUARD, float(pp_cfg.get("log_zero_guard_value", 2.0**-24)))

        for cfg_key, default in _SCORING_DEFAULTS.items():
            val = sm_cfg.get(cfg_key, default)
            gguf_key = f"{ARCH}.scoring.{cfg_key}"
            if isinstance(default, int):
                gw.add_uint32(gguf_key, int(val))
            else:
                gw.add_float32(gguf_key, float(val))

        gw.add_uint32(KEY_ST_SPKCACHE_LEN, int(sm_cfg.get("spkcache_len", 188)))
        gw.add_uint32(KEY_ST_FIFO_LEN, int(sm_cfg.get("fifo_len", 0)))
        gw.add_uint32(KEY_ST_CHUNK_LEN, int(sm_cfg.get("chunk_len", 188)))
        gw.add_uint32(KEY_ST_UPDATE_PERIOD, int(sm_cfg.get("spkcache_update_period", 188)))
        gw.add_uint32(KEY_ST_CHUNK_LC, int(sm_cfg.get("chunk_left_context", 1)))
        gw.add_uint32(KEY_ST_CHUNK_RC, int(sm_cfg.get("chunk_right_context", 1)))

        # ---- Analytical rel-pos PE table ----
        pe = build_pe(d_model, pe_max_len)
        gw.add_tensor("encoder.pos_enc.pe", pe, raw_dtype=GGMLQuantizationType.F32)

        # ---- Trained mel filterbank, verbatim from the checkpoint ----
        fb = sd["preprocessor.featurizer.fb"].detach().cpu().float().numpy()
        fb = np.ascontiguousarray(np.squeeze(fb, axis=0))  # (1, n_mels, n_freq) → (n_mels, n_freq)
        gw.add_tensor("preprocessor.fb", fb, raw_dtype=GGMLQuantizationType.F32)
        print(f"[convert] preprocessor.fb from checkpoint, shape {fb.shape}")

        # ---- Walk state_dict ----
        emitted, skipped = 0, 0
        dtype_counts: dict[str, int] = {}
        for src_name, tensor in sd.items():
            dst_name = remap(src_name)
            if dst_name is None:
                skipped += 1
                continue
            arr = tensor.detach().cpu().float().numpy()
            if _POINTWISE_PATTERN.match(dst_name) and arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr.squeeze(axis=-1)

            chosen_qtype, fallback_reason = _pick_dtype(
                dst_name, arr.shape, linear_qtype, default_qtype
            )
            if fallback_reason is not None:
                print(f"[convert] kept fp16 ({fallback_reason}): {dst_name}")

            if chosen_qtype == GGMLQuantizationType.F32:
                gw.add_tensor(dst_name, arr.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
            elif chosen_qtype == GGMLQuantizationType.F16:
                gw.add_tensor(dst_name, arr.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
            else:
                packed = quantize(arr, chosen_qtype)
                gw.add_tensor(dst_name, packed, raw_dtype=chosen_qtype)
            dtype_counts[chosen_qtype.name] = dtype_counts.get(chosen_qtype.name, 0) + 1
            emitted += 1

        print(f"[convert] emitted {emitted} tensors, skipped {skipped}")
        print(
            "[convert] dtype tally: "
            + ", ".join(f"{k}={v}" for k, v in sorted(dtype_counts.items()))
        )

        gw.write_header_to_file()
        gw.write_kv_data_to_file()
        gw.write_tensors_to_file()
        gw.close()
        print(f"[convert] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
