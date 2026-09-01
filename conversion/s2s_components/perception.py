#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert VoiceChat perception and RNNT tensors to one GGUF artifact.\n\nThe output uses the ASR tensor schema and includes the S2S projection layer.\n"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFWriter
from safetensors import safe_open

# Share tensor naming and quantization policy with the root converter.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from conversion.asr import (  # type: ignore
    _K_QUANTS,
    _LSTM_RE,
    _QK_K,
    ARCH,
    KEY_ARCH,
    KEY_ENC_ATT_CONTEXT_STYLE,
    KEY_ENC_CACHE_SUPPORTED,
    KEY_ENC_CONV_CONTEXT,
    KEY_ENC_CONV_KERNEL,
    KEY_ENC_CONV_NORM,
    KEY_ENC_D_FF,
    KEY_ENC_D_MODEL,
    KEY_ENC_FEAT_IN,
    KEY_ENC_N_HEADS,
    KEY_ENC_N_LAYERS,
    KEY_ENC_PE_MAX_LEN,
    KEY_ENC_SUBSAMPLE,
    KEY_ENC_SUBSAMPLE_CONV_CHANNELS,
    KEY_ENC_TRAIN_LEFT_CTX,
    KEY_ENC_TRAIN_RIGHT_CTX,
    KEY_ENC_USE_BIAS,
    KEY_ENC_XSCALING,
    KEY_FE_N_FFT,
    KEY_FE_N_MELS,
    KEY_FE_NORMALIZE,
    KEY_FE_PREEMPH,
    KEY_FE_SAMPLE_RATE,
    KEY_FE_WINDOW_SIZE,
    KEY_FE_WINDOW_STRIDE,
    KEY_HEAD_TYPE,
    KEY_NAME,
    KEY_RNNT_BLANK_ID,
    KEY_RNNT_JOINT_DIM,
    KEY_RNNT_MAX_SYMBOLS_PER_STEP,
    KEY_RNNT_PRED_EMBED_DIM,
    KEY_RNNT_PRED_HIDDEN,
    KEY_RNNT_PRED_NUM_LAYERS,
    KEY_RNNT_VOCAB_SIZE,
    KEY_TOK_TYPE,
    KEY_TOK_VOCAB,
    WEIGHT_TYPES,
    _pick_dtype,
    _quantize,
    build_pe,
    reshape_for_emission,
)

# ----------------------------------------------------------------------------
# S2S-specific metadata keys (under `s2s.perception.*`)
# ----------------------------------------------------------------------------
S2S_PROJ_OUT_DIM = "s2s.perception.proj.out_dim"  # = LLM hidden size (4480)
S2S_PROJ_IN_DIM = "s2s.perception.proj.in_dim"  # = encoder d_model (1024)

# ----------------------------------------------------------------------------
# The runtime recomputes the window. The trained mel basis is retained as
# ``asr.preprocessor.fb``.
_SKIP_PREFIXES = ("preprocessor.featurizer.window",)  # recomputed in C++
_SKIP_SUFFIXES = (
    ".num_batches_tracked",
    ".inv_freq",
)
_SUBSAMPLING_CONV_PREFIX = "encoder.pre_encode.conv."


def _preserve_subsampling_precision(
    name: str,
    values: np.ndarray,
    qtype: GGMLQuantizationType,
) -> tuple[np.ndarray, GGMLQuantizationType]:
    """Keep the BF16 perception stem when the rest of the model is quantized."""
    if not name.startswith(_SUBSAMPLING_CONV_PREFIX):
        return values, qtype

    rounded = torch.from_numpy(np.ascontiguousarray(values)).to(torch.bfloat16).float().numpy()
    storage = GGMLQuantizationType.BF16 if name.endswith(".weight") else GGMLQuantizationType.F32
    return rounded, storage


def _remap(name: str) -> str | None:
    """Map VoiceChat checkpoint tensor names to the runtime GGUF schema.

    The S2S model's RNNT prefixes are `rnnt_decoder.` and `rnnt_joint.`, vs
    Nemotron/Parakeet's `decoder.` and `joint.`. After this rename, the LSTM
    name normalization (weight_ih_l0 → ih_l0.weight) is identical.
    """
    if name.startswith(_SKIP_PREFIXES) or name.endswith(_SKIP_SUFFIXES):
        return None
    if name == "encoder.pos_enc.pe":
        return None  # rebuilt analytically below
    if name.startswith("rnnt_decoder."):
        name = "decoder." + name[len("rnnt_decoder.") :]
    elif name.startswith("rnnt_joint."):
        name = "joint." + name[len("rnnt_joint.") :]
    # PyTorch-native LSTM names → `{ih|hh}_l{n}.{weight|bias}`
    m = _LSTM_RE.match(name)
    if m:
        prefix, weight_or_bias, ih_hh, layer_n = m.groups()
        return f"{prefix}.{ih_hh}_l{layer_n}.{weight_or_bias}"
    return name


# ----------------------------------------------------------------------------
# Encoder geometry inference from the perception tensors.
# ----------------------------------------------------------------------------
def _infer_encoder_geometry(
    perception_keys: list[str], shape_of: dict[str, tuple[int, ...]]
) -> dict:
    """Read encoder geometry off the tensor shapes — no external config needed."""
    layer_idxs = sorted(
        {int(k.split(".")[2]) for k in perception_keys if k.startswith("encoder.layers.")}
    )
    n_layers = max(layer_idxs) + 1

    # d_model: ne[0] of any layer's norm_self_att.weight
    norm_name = "encoder.layers.0.norm_self_att.weight"
    d_model = shape_of[norm_name][0]

    # d_ff: feed_forward1.linear1.weight is (d_ff, d_model)
    d_ff = shape_of["encoder.layers.0.feed_forward1.linear1.weight"][0]

    # conv_kernel: depthwise_conv.weight is (d_model, 1, k)
    conv_kernel = shape_of["encoder.layers.0.conv.depthwise_conv.weight"][2]

    # Self-attn n_heads: linear_q.weight is (d_model, d_model). Standard NeMo
    # uses head_dim = d_model // n_heads; n_heads isn't in the safetensors
    # directly, so we infer from pos_bias_u shape which is (n_heads, d_k).
    pos_bias_u = shape_of["encoder.layers.0.self_attn.pos_bias_u"]
    n_heads = pos_bias_u[0]
    d_k = pos_bias_u[1]
    assert n_heads * d_k == d_model, f"n_heads*d_k != d_model: {n_heads}*{d_k} vs {d_model}"

    # pre_encode.out.weight is (d_model, subsample_out_channels * subsample_freq)
    pre_out = shape_of["encoder.pre_encode.out.weight"]
    pre_out_in = pre_out[1]  # = subsampling_conv_channels * subsample_freq_out
    # First SubSampling conv: ne is (out=256, in=1, kH=3, kW=3) → subsampling_conv_channels=256
    pre_conv0 = shape_of["encoder.pre_encode.conv.0.weight"]
    subsampling_conv_channels = pre_conv0[0]
    # subsample_freq_out: depends on feat_in and subsampling_factor. NeMo default is
    # feat_in=80 / 8 = 10 → pre_out_in = 256 * 10 = 2560; for our config 4352 = 256 * 17
    # which implies feat_in=128 (matches edge model bundle).
    subsample_freq_out = pre_out_in // subsampling_conv_channels

    # NeMo subsampling factor is fixed at 8 for FastConformer 8x; reverse-derive
    # feat_in from subsample_freq_out (apply ceil-div by 2 three times back).
    # Try both symmetric (`(len+1)//2`) and causal (`len//2 + 1`) subsampling
    # formulas. The causal form is used with cache-aware streaming when
    # conv_context=Causal.
    def _unsubsample(out_len: int, n_stages: int = 3) -> tuple[int, str]:
        for feat_in in (80, 128):
            v_sym = feat_in
            v_cau = feat_in
            for _ in range(n_stages):
                v_sym = (v_sym + 1) // 2
                v_cau = v_cau // 2 + 1
            if v_sym == out_len:
                return feat_in, "symmetric"
            if v_cau == out_len:
                return feat_in, "causal"
        raise ValueError(f"can't infer feat_in from pre_out_in={out_len}")

    feat_in, conv_context_detected = _unsubsample(subsample_freq_out)

    # Detect use_bias by checking whether linear_q has a bias tensor.
    use_bias = "encoder.layers.0.self_attn.linear_q.bias" in perception_keys
    # ConformerConv norm flavor: NeMo names it `batch_norm.*` even when it's LN.
    # The disambiguator is whether `running_mean` exists.
    conv_norm = (
        "batch_norm"
        if "encoder.layers.0.conv.batch_norm.running_mean" in perception_keys
        else "layer_norm"
    )

    return {
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "conv_kernel_size": conv_kernel,
        "subsampling_factor": 8,
        "subsampling_conv_channels": subsampling_conv_channels,
        "feat_in": feat_in,
        "pos_emb_max_len": 5000,
        "xscaling": False,
        "use_bias": use_bias,
        "conv_norm": conv_norm,
        # Detected from the subsampling math — causal subsampling is preserved
        # in the encoder graph regardless of cache mode. S2S consumes the
        # encoder *offline* (CUDA-graph captures a fixed-size padded buffer per
        # perception call; no cache state crosses calls), so we set
        # `cache_supported=False` even though the model was trained cache-
        # aware. Causal conv/conv_context stays, so offline outputs match the
        # training-time computation.
        "conv_context": conv_context_detected,
        "att_context_style": "regular",
        "cache_supported": False,
        "train_left_ctx": 0,
        "train_right_ctx": 0,
        # The S2S perception model was trained with `att_context_size=[70, 0]`
        # — strictly causal self-attention. nemotron-edge's offline encoder
        # respects these limits when `asr.encoder.offline_{left,right}_ctx`
        # are emitted (sentinel -1 = unlimited; absent key = unlimited).
        # Bundle config.json: model.stt.model.perception.encoder.att_context_size
        # = [70, 0].
        "offline_left_ctx": 70,
        "offline_right_ctx": 0,
    }


def _infer_rnnt_geometry(rnnt_shape_of: dict[str, tuple[int, ...]]) -> dict:
    # `decoder.prediction.embed.weight` is (vocab_size, pred_embed_dim)
    embed = rnnt_shape_of["decoder.prediction.embed.weight"]
    vocab_size, pred_embed_dim = embed
    # Post-remap names: bias_hh_l0 → hh_l0.bias, weight_ih_l0 → ih_l0.weight
    bias_hh = rnnt_shape_of["decoder.prediction.dec_rnn.lstm.hh_l0.bias"]
    pred_hidden = bias_hh[0] // 4
    pred_num_layers = sum(
        1
        for k in rnnt_shape_of
        if re.match(r"decoder\.prediction\.dec_rnn\.lstm\.ih_l\d+\.weight$", k)
    )
    # joint_dim: enc.weight is (joint_dim, d_model_enc)
    joint_dim = rnnt_shape_of["joint.enc.weight"][0]
    # blank_id is last token by NeMo convention
    blank_id = vocab_size - 1
    return {
        "vocab_size": vocab_size,
        "blank_id": blank_id,
        "pred_embed_dim": pred_embed_dim,
        "pred_hidden": pred_hidden,
        "pred_num_layers": pred_num_layers,
        "joint_dim": joint_dim,
        "max_symbols_per_step": 10,
    }


def _shape_dict(
    path: Path, prefix: str = "", required_name_prefix: str = ""
) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    with safe_open(str(path), framework="pt") as f:
        for source_name in f.keys():
            if not source_name.startswith(prefix):
                continue
            name = source_name[len(prefix) :]
            if required_name_prefix and not name.startswith(required_name_prefix):
                continue
            out[name] = tuple(f.get_slice(source_name).get_shape())
    return out


def _safetensors_tensors(path: Path, prefix: str = "", required_name_prefix: str = ""):
    """Iterator of (name, np.ndarray) over every tensor in a safetensors file."""
    with safe_open(str(path), framework="pt") as f:
        for source_name in f.keys():
            if not source_name.startswith(prefix):
                continue
            name = source_name[len(prefix) :]
            if required_name_prefix and not name.startswith(required_name_prefix):
                continue
            yield name, f.get_tensor(source_name).detach().cpu().float().numpy()


def convert(bundle_dir: Path, out_path: Path, weight_type: str) -> None:
    if weight_type not in WEIGHT_TYPES:
        raise ValueError(f"--weight-type must be one of {sorted(WEIGHT_TYPES)}")
    linear_qtype, default_qtype, file_type_value = WEIGHT_TYPES[weight_type]
    print(
        f"[convert-s2s] weight_type={weight_type} "
        f"(linear={linear_qtype.name}, default={default_qtype.name})"
    )

    perception_path = bundle_dir / "perception.safetensors"
    rnnt_path = bundle_dir / "rnnt-asr.safetensors"
    perception_prefix = ""
    rnnt_prefix = ""
    rnnt_required_prefix = ""
    if not perception_path.is_file() or not rnnt_path.is_file():
        monolithic = bundle_dir / "model.safetensors"
        if not monolithic.is_file():
            raise FileNotFoundError(
                f"expected split perception/RNNT files or public checkpoint {monolithic}"
            )
        perception_path = monolithic
        rnnt_path = monolithic
        perception_prefix = "stt_model.perception."
        rnnt_prefix = "stt_model."
        rnnt_required_prefix = "rnnt_"

    perception_shapes = _shape_dict(perception_path, perception_prefix)
    rnnt_shapes_raw = _shape_dict(rnnt_path, rnnt_prefix, rnnt_required_prefix)
    if not perception_shapes or not rnnt_shapes_raw:
        raise RuntimeError("VoiceChat checkpoint is missing perception or RNNT tensors")
    # Apply the same remap to rnnt shape dict so downstream geometry detection works on the renamed keys.
    rnnt_shapes = {_remap(k): v for k, v in rnnt_shapes_raw.items() if _remap(k) is not None}

    enc = _infer_encoder_geometry(list(perception_shapes), perception_shapes)
    rnnt = _infer_rnnt_geometry(rnnt_shapes)
    proj_shape = perception_shapes["proj.weight"]
    proj_out_dim, proj_in_dim = proj_shape
    print(
        f"[convert-s2s] encoder: d_model={enc['d_model']} n_layers={enc['n_layers']} "
        f"n_heads={enc['n_heads']} d_ff={enc['d_ff']} k={enc['conv_kernel_size']} "
        f"feat_in={enc['feat_in']} subsample={enc['subsampling_factor']}"
    )
    print(
        f"[convert-s2s] rnnt: vocab={rnnt['vocab_size']} blank={rnnt['blank_id']} "
        f"pred_hidden={rnnt['pred_hidden']} pred_layers={rnnt['pred_num_layers']} "
        f"joint_dim={rnnt['joint_dim']}"
    )
    print(f"[convert-s2s] proj: {proj_in_dim} → {proj_out_dim} (LLM hidden)")

    gw = GGUFWriter(str(out_path), "asr")
    gw.add_string(KEY_ARCH, "asr")
    gw.add_string(KEY_NAME, "s2s-perception")
    gw.add_string(KEY_HEAD_TYPE, "rnnt")
    gw.add_uint32("general.file_type", file_type_value)

    # Encoder hparams
    gw.add_uint32(KEY_ENC_D_MODEL, enc["d_model"])
    gw.add_uint32(KEY_ENC_N_LAYERS, enc["n_layers"])
    gw.add_uint32(KEY_ENC_N_HEADS, enc["n_heads"])
    gw.add_uint32(KEY_ENC_D_FF, enc["d_ff"])
    gw.add_uint32(KEY_ENC_CONV_KERNEL, enc["conv_kernel_size"])
    gw.add_uint32(KEY_ENC_SUBSAMPLE, enc["subsampling_factor"])
    gw.add_uint32(KEY_ENC_SUBSAMPLE_CONV_CHANNELS, enc["subsampling_conv_channels"])
    gw.add_uint32(KEY_ENC_FEAT_IN, enc["feat_in"])
    gw.add_uint32(KEY_ENC_PE_MAX_LEN, enc["pos_emb_max_len"])
    gw.add_bool(KEY_ENC_XSCALING, enc["xscaling"])
    gw.add_bool(KEY_ENC_USE_BIAS, enc["use_bias"])
    gw.add_string(KEY_ENC_CONV_NORM, enc["conv_norm"])
    gw.add_string(KEY_ENC_CONV_CONTEXT, enc["conv_context"])
    gw.add_string(KEY_ENC_ATT_CONTEXT_STYLE, enc["att_context_style"])
    gw.add_bool(KEY_ENC_CACHE_SUPPORTED, enc["cache_supported"])
    gw.add_uint32(KEY_ENC_TRAIN_LEFT_CTX, enc["train_left_ctx"])
    gw.add_uint32(KEY_ENC_TRAIN_RIGHT_CTX, enc["train_right_ctx"])
    # Offline (non-cache-aware) attention context. -1 = unlimited (no mask).
    # The runtime builds an additive attention mask when either value is >= 0.
    gw.add_int32("asr.encoder.offline_left_ctx", int(enc["offline_left_ctx"]))
    gw.add_int32("asr.encoder.offline_right_ctx", int(enc["offline_right_ctx"]))

    # Preprocessor defaults; the configured runtime feature extractor may
    # override them.
    gw.add_uint32(KEY_FE_SAMPLE_RATE, 16000)
    gw.add_float32(KEY_FE_WINDOW_SIZE, 0.025)
    gw.add_float32(KEY_FE_WINDOW_STRIDE, 0.01)
    gw.add_uint32(KEY_FE_N_FFT, 512)
    gw.add_uint32(KEY_FE_N_MELS, enc["feat_in"])
    gw.add_float32(KEY_FE_PREEMPH, 0.97)
    gw.add_string(KEY_FE_NORMALIZE, "per_feature")

    # RNNT hparams
    gw.add_uint32(KEY_RNNT_VOCAB_SIZE, rnnt["vocab_size"])
    gw.add_uint32(KEY_RNNT_BLANK_ID, rnnt["blank_id"])
    gw.add_uint32(KEY_RNNT_PRED_EMBED_DIM, rnnt["pred_embed_dim"])
    gw.add_uint32(KEY_RNNT_PRED_HIDDEN, rnnt["pred_hidden"])
    gw.add_uint32(KEY_RNNT_PRED_NUM_LAYERS, rnnt["pred_num_layers"])
    gw.add_uint32(KEY_RNNT_JOINT_DIM, rnnt["joint_dim"])
    gw.add_uint32(KEY_RNNT_MAX_SYMBOLS_PER_STEP, rnnt["max_symbols_per_step"])

    # S2S-specific projection layer
    gw.add_uint32(S2S_PROJ_OUT_DIM, proj_out_dim)
    gw.add_uint32(S2S_PROJ_IN_DIM, proj_in_dim)

    # Tokenizer — vocab not shipped with this model bundle (S2S decodes via
    # its own tokenizer ./rnnt_tokenizer/). Mark it as 'none'; the runtime
    # consumer wires the tokenizer separately.
    gw.add_string(KEY_TOK_TYPE, "none")

    # Positional encoding (analytical — matches what convert_nemo_to_gguf does).
    pe = build_pe(enc["d_model"], enc["pos_emb_max_len"])
    gw.add_tensor("encoder.pos_enc.pe", pe, raw_dtype=GGMLQuantizationType.F32)

    # NeMo's `librosa.filters.mel(norm="slaney")` filterbank, baked into the
    # bundle as `preprocessor.featurizer.fb`. nemotron-edge's C++
    # MelSpectrogramExtractor::set_mel_basis consumes this when present;
    # otherwise it falls back to auto-generated unit-peak triangles (the
    # legacy behaviour for older nemotron-asr GGUFs).
    with safe_open(str(perception_path), framework="pt") as _f:
        fb_name = perception_prefix + "preprocessor.featurizer.fb"
        if fb_name in _f.keys():
            fb = _f.get_tensor(fb_name).to("cpu").float().numpy()
            # The safetensors tensor ships as (1, n_mels, n_bins) — strip the
            # leading 1. Final layout: (n_mels, n_bins) row-major.
            if fb.ndim == 3 and fb.shape[0] == 1:
                fb = fb[0]
            gw.add_tensor(
                "asr.preprocessor.fb", np.ascontiguousarray(fb), raw_dtype=GGMLQuantizationType.F32
            )
            print(f"[convert-s2s] emitted asr.preprocessor.fb (slaney) " f"shape={fb.shape}")

    # ----- Walk both safetensors files and emit -----
    emitted = 0
    skipped: list[str] = []
    dtype_counts: dict[str, int] = {}
    tensor_groups = (
        (perception_path, perception_prefix, ""),
        (rnnt_path, rnnt_prefix, rnnt_required_prefix),
    )
    for src_path, prefix, required_prefix in tensor_groups:
        for src_name, arr in _safetensors_tensors(src_path, prefix, required_prefix):
            dst_name = _remap(src_name)
            if dst_name is None:
                skipped.append(src_name)
                continue
            arr = reshape_for_emission(dst_name, arr)
            chosen_qtype, fallback_reason = _pick_dtype(
                dst_name, arr.shape, linear_qtype, default_qtype
            )
            if dst_name.startswith(_SUBSAMPLING_CONV_PREFIX):
                # VoiceChat runs its perception encoder in BF16. Keep the
                # subsampling convolutions at that precision even when the
                # remaining linear weights are quantized; F16 changes both
                # the kernels and their lowered activation inputs.
                arr, chosen_qtype = _preserve_subsampling_precision(dst_name, arr, chosen_qtype)
                fallback_reason = None
            if fallback_reason is not None:
                print(f"[convert-s2s] kept fp16 ({fallback_reason}): {dst_name}")

            if chosen_qtype == GGMLQuantizationType.F32:
                gw.add_tensor(dst_name, arr.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
            elif chosen_qtype == GGMLQuantizationType.F16:
                gw.add_tensor(dst_name, arr.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
            else:
                packed = _quantize(arr, chosen_qtype)
                gw.add_tensor(dst_name, packed, raw_dtype=chosen_qtype)
            dtype_counts[chosen_qtype.name] = dtype_counts.get(chosen_qtype.name, 0) + 1
            emitted += 1

    print(f"[convert-s2s] emitted {emitted} tensors, skipped {len(skipped)}")
    print(
        f"[convert-s2s] dtype tally: "
        + ", ".join(f"{k}={v}" for k, v in sorted(dtype_counts.items()))
    )
    if skipped:
        print(f"[convert-s2s] skipped (first 10): {skipped[:10]}")

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    print(f"[convert-s2s] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "bundle_dir",
        type=Path,
        help="VoiceChat repository directory",
    )
    p.add_argument("out", type=Path, help="output .gguf path")
    p.add_argument(
        "--weight-type",
        choices=sorted(WEIGHT_TYPES),
        default="bf16",
        help="dtype/quant for Linear weight tensors (default: bf16)",
    )
    args = p.parse_args()
    convert(args.bundle_dir, args.out, args.weight_type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
