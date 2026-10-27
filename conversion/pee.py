# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a Parallel Expert Encoder (PEE) bundle to GGUF.

Target: the Transformer-AED + 8-speaker Sortformer PEE
(``ParallelExpertEncoderPT`` bundles with ``asr_encoder_type: transformer``). Two experts run on the same 128-bin mel:

  asr   : FeatureStacking(x8) -> LayerNorm -> N x [pre-LN RoPE Transformer block]
          -> LayerNorm                                   (d_model 1280, 32 layers)
  diar  : same module family                             (d_model 512, 31 layers)
          -> encoder_proj 512->192 -> subpixel_upsample (x8, 10 ms grid)
          -> relu -> Linear -> relu -> Linear(8) -> sigmoid
  fusion: avg-pool preds to the 80 ms ASR grid -> threshold -> LayerNorm(8)
          -> @ sinusoid kernel (8, d_model) -> + LayerNorm(d_model)(asr states)

GGUF layout (arch "pee"):
  asr.*           encoder.asr_encoder.*            (state-dict names kept)
  diar.*          encoder.diarization_model.encoder.*
  diar.head.*     encoder.diarization_model.sortformer_modules.{encoder_proj,
                  first_hidden_to_hidden,single_hidden_to_spks,subpixel_upsample}
  fusion.asr_norm / fusion.diar_norm / fusion.diar_kernel (analytical, x spk_kernel_scale)
  fusion.proj.*   optional SALM modality projection d_model -> LLM hidden (pee.fusion.proj_dim)
  preprocessor.fb  trained mel filterbank

Sources: a PEE ``.nemo`` bundle (or its extracted directory), or a speechlm2 SALM
checkpoint directory (``config.json`` with ``pe_encoder_config`` + ``*.safetensors``
holding ``perception.*``), in which case ``perception.proj`` is exported too.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from gguf import GGMLQuantizationType, GGUFWriter

from .quantization import DEPLOYMENT_WEIGHT_TYPES, K_QUANTS, QK_K, quantize
from .source import (
    extracted_checkpoint,
    find_checkpoint_files,
    is_salm_checkpoint_dir,
    load_state_dict,
)

ARCH = "pee"

WEIGHT_TYPES = dict(DEPLOYMENT_WEIGHT_TYPES)
WEIGHT_TYPES["f32"] = (GGMLQuantizationType.F32, GGMLQuantizationType.F32, 0)

KEY_NAME = "general.name"

# Per-branch RoPE transformer hparams: pee.<branch>.<field>
BRANCH_FIELDS = (
    "feat_in",
    "d_model",
    "n_heads",
    "n_layers",
    "ff_hidden",
    "subsampling_factor",
    "rotary_dims",
    "qk_norm",
    "qkv_bias",
    "pre_block_norm",
)

KEY_NUM_SPEAKERS = f"{ARCH}.diar.num_speakers"
KEY_DIAR_TF_D_MODEL = f"{ARCH}.diar.tf_d_model"
KEY_DIAR_UPSAMPLE = f"{ARCH}.diar.upsample_factor"
KEY_DIAR_OUT_SUBSAMPLE = f"{ARCH}.diar.output_subsampling_factor"
KEY_FUSION_THRESHOLD = f"{ARCH}.fusion.speaker_activity_threshold"
KEY_FUSION_MODE = f"{ARCH}.fusion.speaker_feature_mode"
KEY_FUSION_PROJ_DIM = f"{ARCH}.fusion.proj_dim"
KEY_ASR_NORMALIZE = f"{ARCH}.asr.normalize_type"
KEY_DIAR_NORMALIZE = f"{ARCH}.diar.normalize_type"
# Long-form windowing (provenance; the runtime owns its own geometry).
KEY_ONLINE_LEN = f"{ARCH}.online.inference_length"
KEY_ONLINE_LC = f"{ARCH}.online.chunk_left_context"
KEY_ONLINE_RC = f"{ARCH}.online.chunk_right_context"
KEY_FE_SAMPLE_RATE = f"{ARCH}.preprocessor.sample_rate"
KEY_FE_WINDOW_SIZE = f"{ARCH}.preprocessor.window_size"
KEY_FE_WINDOW_STRIDE = f"{ARCH}.preprocessor.window_stride"
KEY_FE_N_FFT = f"{ARCH}.preprocessor.n_fft"
KEY_FE_N_MELS = f"{ARCH}.preprocessor.features"
KEY_FE_DITHER = f"{ARCH}.preprocessor.dither"

_ASR_PREFIX = "encoder.asr_encoder."
_DIAR_ENC_PREFIX = "encoder.diarization_model.encoder."
_DIAR_HEAD_PREFIX = "encoder.diarization_model.sortformer_modules."
_HEAD_KEEP = (
    "encoder_proj",
    "first_hidden_to_hidden",
    "single_hidden_to_spks",
    "subpixel_upsample",
    "learnable_sil_emb",
)

# Linear weights eligible for quantization (2-D projection matrices).
_LINEAR_WEIGHT_PATTERNS = [
    re.compile(r"^(asr|diar)\.layers\.\d+\.attn\.(w_qkv|out_proj)\.weight$"),
    re.compile(r"^(asr|diar)\.layers\.\d+\.ffn\.net\.(0|3)\.weight$"),
    re.compile(r"^(asr|diar)\.pre_encode\.proj\.weight$"),
    re.compile(
        r"^diar\.head\.(encoder_proj|first_hidden_to_hidden|single_hidden_to_spks)\.weight$"
    ),
    re.compile(r"^fusion\.proj\.weight$"),
]
_CONV_WEIGHT_PATTERN = re.compile(r"^diar\.head\.subpixel_upsample\.weight$")


def remap(name: str) -> Optional[str]:
    if name.startswith(_ASR_PREFIX):
        return "asr." + name[len(_ASR_PREFIX) :]
    if name.startswith(_DIAR_ENC_PREFIX):
        return "diar." + name[len(_DIAR_ENC_PREFIX) :]
    if name.startswith(_DIAR_HEAD_PREFIX):
        rest = name[len(_DIAR_HEAD_PREFIX) :]
        if rest.split(".")[0] in _HEAD_KEEP:
            return "diar.head." + rest
        return None  # hidden_to_spks (unused at inference)
    if name in ("encoder.asr_norm.weight", "encoder.asr_norm.bias"):
        return name.replace("encoder.asr_norm.", "fusion.asr_norm.")
    if name in ("encoder.diar_norm.weight", "encoder.diar_norm.bias"):
        return name.replace("encoder.diar_norm.", "fusion.diar_norm.")
    if name.startswith("encoder.diarization_model.preprocessor.") or name.startswith(
        "preprocessor."
    ):
        return None  # fb emitted manually; window rebuilt from config
    if name in ("proj.weight", "proj.bias"):
        return "fusion." + name  # SALM modality projection (d_model -> LLM hidden)
    print(f"[convert] WARNING: unrecognized tensor skipped: {name}", file=sys.stderr)
    return None


def _pick_dtype(
    name: str, shape: tuple, linear_qtype: GGMLQuantizationType, default_qtype: GGMLQuantizationType
) -> tuple[GGMLQuantizationType, Optional[str]]:
    if any(p.match(name) for p in _LINEAR_WEIGHT_PATTERNS):
        inner = int(shape[-1])
        if linear_qtype in K_QUANTS and inner % QK_K != 0:
            return GGMLQuantizationType.F16, "block alignment"
        if linear_qtype in (GGMLQuantizationType.NVFP4, GGMLQuantizationType.MXFP4) and inner % 64:
            return GGMLQuantizationType.F16, "block alignment"
        if linear_qtype == GGMLQuantizationType.Q8_0 and inner % 32:
            return GGMLQuantizationType.F16, "block alignment"
        return linear_qtype, None
    if _CONV_WEIGHT_PATTERN.match(name):
        return GGMLQuantizationType.F16, None  # ggml conv kernels are F16
    if name == "diar.head.learnable_sil_emb":
        return GGMLQuantizationType.F32, None  # read on the host (speaker-cache silence rows)
    return default_qtype, None


def sinusoid_kernel(n_spk: int, d_model: int) -> np.ndarray:
    """Mirror of ParallelExpertEncoder._build_sinusoid_position_encoding."""
    position = np.arange(n_spk, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))
    pe = np.zeros((n_spk, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def _branch_hparams(enc_cfg: dict) -> dict:
    d_model = int(enc_cfg["d_model"])
    n_heads = int(enc_cfg["n_heads"])
    if str(enc_cfg.get("self_attention_model", "rel_pos")) != "rope":
        raise RuntimeError("PEE branches must use self_attention_model=rope")
    if str(enc_cfg.get("subsampling", "feature_stacking")) != "feature_stacking":
        raise RuntimeError("PEE branches must use feature_stacking subsampling")
    if str(enc_cfg.get("attn_mode", "full")) != "full":
        raise RuntimeError("only attn_mode=full is supported")
    if bool(enc_cfg.get("xscaling", False)):
        raise RuntimeError("xscaling is not supported")
    d_k = d_model // n_heads
    rotary_dims = int(d_k * float(enc_cfg.get("rotary_fraction", 1.0)))
    if rotary_dims < 2 or rotary_dims % 2:
        raise RuntimeError(f"bad rotary dims {rotary_dims}")
    if float(enc_cfg.get("rope_base", 10000.0)) != 10000.0:
        raise RuntimeError("only rope_base=10000 is supported")
    return {
        "feat_in": int(enc_cfg["feat_in"]),
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": int(enc_cfg["n_layers"]),
        "ff_hidden": int(float(enc_cfg.get("ff_expansion", 4.0)) * d_model),
        "subsampling_factor": int(enc_cfg.get("subsampling_factor", 8)),
        "rotary_dims": rotary_dims,
        "qk_norm": bool(enc_cfg.get("qk_norm", False)),
        "qkv_bias": bool(enc_cfg.get("qkv_bias", False)),
        "pre_block_norm": bool(enc_cfg.get("pre_block_norm", True)),
    }


def _add_branch(gw: GGUFWriter, branch: str, hp: dict) -> None:
    for field in BRANCH_FIELDS:
        key = f"{ARCH}.{branch}.{field}"
        val = hp[field]
        if isinstance(val, bool):
            gw.add_bool(key, val)
        else:
            gw.add_uint32(key, int(val))


_SALM_PREFIX = "perception."


def _load_salm_perception(path: Path) -> tuple[dict, dict]:
    """PEE config and ``perception.*`` tensors (prefix stripped) from a SALM checkpoint dir."""
    from safetensors import safe_open

    with (path / "config.json").open("r", encoding="utf-8") as f:
        cfg = dict(json.load(f)["pe_encoder_config"])
    sd: dict = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(_SALM_PREFIX) and "_extra_state" not in key:
                    sd[key[len(_SALM_PREFIX) :]] = f.get_tensor(key)
    if not sd:
        raise RuntimeError(f"no perception.* tensors found in {path}")
    return cfg, sd


def convert(nemo_path: Path, out_path: Path, weight_type: str) -> None:
    if weight_type not in WEIGHT_TYPES:
        raise ValueError(
            f"unknown --weight-type {weight_type!r}; choose from {sorted(WEIGHT_TYPES)}"
        )
    linear_qtype, default_qtype, file_type_value = WEIGHT_TYPES[weight_type]
    print(f"[convert] weight_type = {weight_type} (linear={linear_qtype.name})")

    if is_salm_checkpoint_dir(nemo_path):
        cfg, sd = _load_salm_perception(nemo_path)
        print(f"[convert] SALM checkpoint: {len(sd)} perception tensors")
        _write_gguf(cfg, sd, out_path, linear_qtype, default_qtype, file_type_value)
        return

    with extracted_checkpoint(nemo_path, "pee-convert-") as tmp:
        files = find_checkpoint_files(tmp)
        if files["config"] is None or files["weights"] is None:
            raise RuntimeError(f"Missing artifacts in {nemo_path}: {files}")
        with open(files["config"], "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sd = load_state_dict(files["weights"], weights_only=True)
        print(f"[convert] state_dict has {len(sd)} tensors")
        _write_gguf(cfg, sd, out_path, linear_qtype, default_qtype, file_type_value)


def _write_gguf(
    cfg: dict,
    sd: dict,
    out_path: Path,
    linear_qtype: GGMLQuantizationType,
    default_qtype: GGMLQuantizationType,
    file_type_value: int,
) -> None:
    if str(cfg.get("asr_encoder_type", "")) != "transformer":
        raise RuntimeError("only asr_encoder_type=transformer PEE bundles are supported")
    asr_hp = _branch_hparams(cfg["asr_encoder_cfg"])
    dcfg = cfg["diarization_model_cfg"]
    diar_hp = _branch_hparams(dcfg["encoder"])
    sm = dcfg["sortformer_modules"]
    n_spk = int(sm.get("num_spks", dcfg.get("max_num_of_spks", 8)))
    tf_d_model = int(sm.get("tf_d_model", 192))
    if not bool(dcfg.get("high_resolution", False)):
        raise RuntimeError("expected a high_resolution Sortformer (subpixel upsample)")
    if "encoder.diarization_model.transformer_encoder.layers.0.layer_norm_1.weight" in sd:
        raise RuntimeError("Sortformer with a post-LN transformer stage is not supported here")
    upsample = diar_hp["subsampling_factor"]
    if asr_hp["subsampling_factor"] != diar_hp["subsampling_factor"]:
        raise RuntimeError("ASR and diar subsampling factors differ")
    if asr_hp["feat_in"] != diar_hp["feat_in"]:
        raise RuntimeError("ASR and diar feat_in differ")

    print(
        f"[convert] asr d_model={asr_hp['d_model']} L={asr_hp['n_layers']} H={asr_hp['n_heads']} "
        f"rot={asr_hp['rotary_dims']} qk_norm={asr_hp['qk_norm']}; diar d_model={diar_hp['d_model']} "
        f"L={diar_hp['n_layers']} H={diar_hp['n_heads']} rot={diar_hp['rotary_dims']}; "
        f"n_spk={n_spk} tf_d_model={tf_d_model} upsample={upsample}"
    )

    gw = GGUFWriter(str(out_path), arch=ARCH)
    gw.add_architecture()
    gw.add_string(KEY_NAME, str(cfg.get("name") or out_path.stem))
    gw.add_uint32("general.file_type", file_type_value)
    _add_branch(gw, "asr", asr_hp)
    _add_branch(gw, "diar", diar_hp)
    gw.add_uint32(KEY_NUM_SPEAKERS, n_spk)
    gw.add_uint32(KEY_DIAR_TF_D_MODEL, tf_d_model)
    gw.add_uint32(KEY_DIAR_UPSAMPLE, upsample)
    gw.add_uint32(KEY_DIAR_OUT_SUBSAMPLE, asr_hp["subsampling_factor"])
    gw.add_float32(KEY_FUSION_THRESHOLD, float(cfg.get("speaker_activity_threshold", 0.5)))
    gw.add_string(KEY_FUSION_MODE, str(cfg.get("speaker_feature_mode") or "thresholded"))
    gw.add_string(KEY_ASR_NORMALIZE, str(cfg.get("asr_normalize_type") or "per_feature"))
    gw.add_string(KEY_DIAR_NORMALIZE, str(cfg.get("diar_normalize_type") or "NA"))
    gw.add_uint32(KEY_ONLINE_LEN, int(cfg.get("online_inference_length") or 500))
    gw.add_uint32(KEY_ONLINE_LC, int(cfg.get("chunk_left_context") or 50))
    gw.add_uint32(KEY_ONLINE_RC, int(cfg.get("chunk_right_context") or 50))

    pp = dcfg["preprocessor"]
    gw.add_uint32(KEY_FE_SAMPLE_RATE, int(pp.get("sample_rate", 16000)))
    gw.add_float32(KEY_FE_WINDOW_SIZE, float(pp.get("window_size", 0.025)))
    gw.add_float32(KEY_FE_WINDOW_STRIDE, float(pp.get("window_stride", 0.01)))
    gw.add_uint32(KEY_FE_N_FFT, int(pp.get("n_fft", 512)))
    gw.add_uint32(KEY_FE_N_MELS, int(pp.get("features", asr_hp["feat_in"])))
    gw.add_float32(KEY_FE_DITHER, float(pp.get("dither", 1e-5)))

    # Analytical speaker kernel (scaled: fusion adds spk_kernel_scale * infusion) and
    # the trained filterbank. A SALM export carries the perception preprocessor's
    # filterbank as well as the diarizer's; both branches see one mel, so they agree.
    kernel_scale = float(cfg.get("spk_kernel_scale") or 1.0)
    gw.add_tensor(
        "fusion.diar_kernel",
        sinusoid_kernel(n_spk, asr_hp["d_model"]) * np.float32(kernel_scale),
        raw_dtype=GGMLQuantizationType.F32,
    )
    fb_keys = [
        k
        for k in (
            "preprocessor.featurizer.fb",
            "encoder.diarization_model.preprocessor.featurizer.fb",
        )
        if k in sd
    ]
    if not fb_keys:
        raise RuntimeError("checkpoint has no mel filterbank tensor")
    fbs = [
        np.ascontiguousarray(np.squeeze(sd[k].detach().cpu().float().numpy(), axis=0))
        for k in fb_keys
    ]
    if len(fbs) == 2 and not np.allclose(fbs[0], fbs[1], atol=1e-6):
        raise RuntimeError("perception and diarizer mel filterbanks differ")
    gw.add_tensor("preprocessor.fb", fbs[0], raw_dtype=GGMLQuantizationType.F32)
    proj_dim = 0
    if "proj.weight" in sd:
        proj_dim = int(sd["proj.weight"].shape[0])
        if int(sd["proj.weight"].shape[1]) != asr_hp["d_model"]:
            raise RuntimeError("fusion.proj input dim does not match the ASR d_model")
        gw.add_uint32(KEY_FUSION_PROJ_DIM, proj_dim)
        print(f"[convert] SALM projection: {asr_hp['d_model']} -> {proj_dim}")

    emitted, skipped = 0, 0
    dtype_counts: dict[str, int] = {}
    for src_name, tensor in sd.items():
        dst_name = remap(src_name)
        if dst_name is None:
            skipped += 1
            continue
        arr = tensor.detach().cpu().float().numpy()
        chosen, why = _pick_dtype(dst_name, arr.shape, linear_qtype, default_qtype)
        if why is not None:
            print(f"[convert] kept fp16 ({why}): {dst_name}")
        if chosen == GGMLQuantizationType.F32:
            gw.add_tensor(dst_name, arr.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
        elif chosen == GGMLQuantizationType.F16:
            gw.add_tensor(dst_name, arr.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
        else:
            gw.add_tensor(dst_name, quantize(arr, chosen), raw_dtype=chosen)
        dtype_counts[chosen.name] = dtype_counts.get(chosen.name, 0) + 1
        emitted += 1

    print(f"[convert] emitted {emitted} tensors, skipped {skipped}")
    print(
        "[convert] dtype tally: " + ", ".join(f"{k}={v}" for k, v in sorted(dtype_counts.items()))
    )
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    print(f"[convert] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
