#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert EarTTS side-network weights to a single GGUF.

The EarTTS backbone (Gemma3 decoder) already runs in llama.cpp via the
existing `model.safetensors → .gguf` converter shipped with the bundle.
This script handles everything else — the `total_emb.*` input pipeline and
the `sampler.*` MaskGIT/MoG generator — which previously sat in the
`eartts_torch/` PyTorch package.

GGUF schema (single file, `general.architecture = "s2s_eartts_side"`):

KV metadata (all values inferred from the safetensors + config.json):
    eartts_side.hidden_size                u32
    eartts_side.latent_size                u32
    eartts_side.codebook_size              u32
    eartts_side.num_quantizers             u32
    eartts_side.intermediate_size          u32
    eartts_side.mog_num_predictions        u32
    eartts_side.mog_num_layers             u32
    eartts_side.mog_low_rank               u32
    eartts_side.mog_min_log_std            f32
    eartts_side.mog_eps                    f32
    eartts_side.noise_scale                f32
    eartts_side.top_p_or_k                 f32
    eartts_side.num_iter                   u32
    eartts_side.exponent                   f32

    eartts_side.char.hidden_size           u32   (== hidden_size today)
    eartts_side.char.intermediate_size     u32
    eartts_side.char.num_attention_heads   u32
    eartts_side.char.num_kv_heads          u32
    eartts_side.char.head_dim              u32
    eartts_side.char.num_hidden_layers     u32
    eartts_side.char.rope_theta            f32
    eartts_side.char.rms_norm_eps          f32   (default 1e-6)
    eartts_side.char.attn_logit_softcap    f32   (default 0; gemma3 uses 50)
    eartts_side.char.max_char_len          u32
    eartts_side.char.char_vocab_size       u32
    eartts_side.char.subword_vocab_size    u32

    eartts_side.use_subword_flag_emb       bool
    eartts_side.use_bos_eos_emb            bool
    eartts_side.use_gated_fusion           bool
    eartts_side.use_audio_prompt_proj      bool
    eartts_side.guidance_scale             f32

Tensors (names mirror eartts_torch state-dict 1:1 after stripping the
"model.total_emb." / "model.sampler." prefixes):

    # ---- total_emb / input embedding ----
    bos_emb                          (hidden)
    null_emb                         (hidden)
    audio_prompt_projection_W        (hidden, hidden)   [optional]
    rvq_embs.{0..Q-1}.weight         (codebook+1, latent)
    embed_code.weight                (hidden, latent)
    embed_subword.embed_subwords.weight        (subword_vocab+1, max_char_len)
    embed_subword.embed_subwords_mask.weight   (subword_vocab+1, max_char_len)
    embed_subword.embed_tokens.weight          (char_vocab+1, char_hidden)
    embed_subword.proj_embedding.weight        (out, hidden)
    embed_subword.backbone.encoder.norm.weight (char_hidden)
    embed_subword.backbone.encoder.layers.0.{
        pre_self_attn_layernorm.weight,
        post_self_attn_layernorm.weight,
        pre_feedforward_layernorm.weight,
        post_feedforward_layernorm.weight,
        self_attn.{q,k,v,o}_proj.weight,
        mlp.{gate,up,down}_proj.weight,
    }
    subword_flag_emb.cont_emb.weight  (2, hidden)            [opt]
    subword_flag_emb.is_continuation  (subword_vocab+1)      i32 [opt]
    bos_eos_emb.special_emb.weight    (3, hidden)            [opt]
    bos_eos_emb.special_flags         (subword_vocab+1)      i32 [opt]
    gated_fusion.audio_proj.weight    (hidden, hidden)       [opt]
    gated_fusion.audio_proj.bias      (hidden)               [opt]
    gated_fusion.text_proj.weight     (hidden, hidden)       [opt]
    gated_fusion.text_proj.bias       (hidden)               [opt]
    gated_fusion.gate_sigmoid         (hidden)   f32         [opt]
    gated_fusion.residual_scale_sigmoid scalar  f32          [opt]
    gated_fusion.final_norm.weight    (hidden)               [opt]

    # ---- sampler ----
    sampler.embed_code.weight         (hidden, latent)
    sampler.rvq_embs                  (Q, codebook, latent)
    sampler.mog_head.proj_logits.weight   (num_predictions, hidden)
    sampler.mog_head.proj_mus.weight      (num_predictions * low_rank, hidden)
    sampler.mog_head.proj_logs.weight     (1, hidden)
    sampler.mog_head.proj_else.weight     (latent, hidden)
    sampler.mog_head.low_mat              (num_predictions, latent, low_rank)
    sampler.mog_head.final_norm.weight    (hidden)
    sampler.mog_head.mlp_stack.{0..L-1}.{
        pre_norm.weight, post_norm.weight,
        mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight,
    }

Usage:
    python scripts/convert_eartts_side_to_gguf.py \
        /path/to/eartts_vllm \
        /tmp/eartts_side.gguf \
        --weight-type q8_0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open
from voicechat_source import build_character_tables, build_eartts_config, is_hf_voicechat_source

ARCH = "s2s_eartts_side"


WEIGHT_TYPES = {
    "fp32": (
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F32,
        gguf.LlamaFileType.ALL_F32,
    ),
    "fp16": (
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.F32,
        gguf.LlamaFileType.MOSTLY_F16,
    ),
    "bf16": (
        gguf.GGMLQuantizationType.BF16,
        gguf.GGMLQuantizationType.F32,
        gguf.LlamaFileType.MOSTLY_BF16,
    ),
    "q4_k": (
        gguf.GGMLQuantizationType.Q4_K,
        gguf.GGMLQuantizationType.F32,
        gguf.LlamaFileType.MOSTLY_Q4_K_M,
    ),
    "q5_k": (
        gguf.GGMLQuantizationType.Q5_K,
        gguf.GGMLQuantizationType.F32,
        gguf.LlamaFileType.MOSTLY_Q5_K_M,
    ),
    "q6_k": (
        gguf.GGMLQuantizationType.Q6_K,
        gguf.GGMLQuantizationType.F32,
        gguf.LlamaFileType.MOSTLY_Q6_K,
    ),
    "q8_0": (
        gguf.GGMLQuantizationType.Q8_0,
        gguf.GGMLQuantizationType.F32,
        gguf.LlamaFileType.MOSTLY_Q8_0,
    ),
}


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().contiguous().cpu().float().numpy()


def _quantize(arr: np.ndarray, qt: gguf.GGMLQuantizationType) -> np.ndarray:
    if qt == gguf.GGMLQuantizationType.F32:
        return arr
    if qt == gguf.GGMLQuantizationType.F16:
        return arr.astype(np.float16)
    return gguf.quants.quantize(arr, qt)


def _load_legacy_inputs(bundle_dir: Path) -> tuple[dict, dict, dict]:
    checkpoint = bundle_dir / "model.safetensors"
    with safe_open(str(checkpoint), framework="pt", device="cpu") as tensors:
        total_emb = {
            key.removeprefix("model.total_emb."): tensors.get_tensor(key)
            for key in tensors.keys()
            if key.startswith("model.total_emb.")
        }
        sampler = {
            key.removeprefix("model.sampler."): tensors.get_tensor(key)
            for key in tensors.keys()
            if key.startswith("model.sampler.")
        }
    cfg = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
    return cfg, total_emb, sampler


def _load_public_inputs(source: Path, tokenizer_json: Path) -> tuple[dict, dict, dict]:
    checkpoint = source / "model.safetensors"
    prefix = "tts_model.tts_model."
    cfg = build_eartts_config(source, tokenizer_json)
    total_emb: dict[str, torch.Tensor] = {}
    sampler: dict[str, torch.Tensor] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())

        def read(relative: str) -> torch.Tensor:
            name = prefix + relative
            if name not in keys:
                raise KeyError(f"VoiceChat checkpoint is missing {name}")
            return tensors.get_tensor(name)

        relatives = [
            "bos_emb",
            "null_emb",
            "embed_code.weight",
            "embed_subword.embed_tokens.weight",
            "embed_subword.proj_embedding.weight",
        ]
        if cfg.get("use_audio_prompt_frozen_projection", False):
            relatives.append("audio_prompt_projection_W")
        if cfg.get("use_gated_fusion_for_text_audio", False):
            relatives.extend(
                [
                    "gated_fusion_audio_text.audio_proj.bias",
                    "gated_fusion_audio_text.audio_proj.weight",
                    "gated_fusion_audio_text.final_norm.weight",
                    "gated_fusion_audio_text.gate",
                    "gated_fusion_audio_text.residual_scale",
                    "gated_fusion_audio_text.text_proj.bias",
                    "gated_fusion_audio_text.text_proj.weight",
                ]
            )
        for relative in relatives:
            total_emb[relative] = read(relative)

        for source_prefix in ("embed_subword.backbone.",):
            for name in sorted(key for key in keys if key.startswith(prefix + source_prefix)):
                total_emb[name.removeprefix(prefix)] = tensors.get_tensor(name)

        for group in ("bos_eos_emb", "subword_flag_emb"):
            source_prefix = prefix + "embed_subword." + group + "."
            for name in sorted(key for key in keys if key.startswith(source_prefix)):
                relative = group + "." + name.removeprefix(source_prefix)
                if relative.endswith("pad_tensor"):
                    continue
                total_emb[relative] = tensors.get_tensor(name)

        rvq = read("rvq_embs")
        zero_row = torch.zeros((1, rvq.shape[2]), dtype=rvq.dtype, device=rvq.device)
        for index in range(rvq.shape[0]):
            total_emb[f"rvq_embs.{index}.weight"] = torch.cat((rvq[index], zero_row), dim=0)

        sampler["rvq_embs"] = rvq
        sampler["embed_code.weight"] = total_emb["embed_code.weight"]
        for name in sorted(key for key in keys if key.startswith(prefix + "mog_head.")):
            sampler[name.removeprefix(prefix)] = tensors.get_tensor(name)

    char_ids, char_mask, _ = build_character_tables(tokenizer_json)
    total_emb["embed_subword.embed_subwords.weight"] = torch.from_numpy(char_ids)
    total_emb["embed_subword.embed_subwords_mask.weight"] = torch.from_numpy(char_mask)
    return cfg, total_emb, sampler


def _add_linear(gw, name: str, w: torch.Tensor, qtype) -> None:
    arr = _to_np(w)
    qt = qtype
    # 1-D tensors (norm weights, biases, scalars) get the default qtype path
    # so they stay in F32; multi-dim Linear weights get the requested type.
    if arr.ndim < 2:
        qt = gguf.GGMLQuantizationType.F32
        arr = arr.astype(np.float32)
        gw.add_tensor(name, arr, raw_dtype=qt)
        return
    gw.add_tensor(name, _quantize(arr, qt), raw_dtype=qt)


def convert(
    bundle_dir: Path, out_path: Path, weight_type: str, tokenizer_json: Path | None = None
) -> None:
    if weight_type not in WEIGHT_TYPES:
        raise ValueError(f"--weight-type must be one of {sorted(WEIGHT_TYPES)}")
    linear_qtype, _, file_type = WEIGHT_TYPES[weight_type]
    print(f"[convert-eartts] weight_type={weight_type} (linear={linear_qtype.name})")

    if is_hf_voicechat_source(bundle_dir):
        if tokenizer_json is None or not tokenizer_json.is_file():
            raise FileNotFoundError("the public VoiceChat checkpoint requires --tokenizer-json")
        cfg, total_emb, sampler = _load_public_inputs(bundle_dir, tokenizer_json)
    else:
        cfg, total_emb, sampler = _load_legacy_inputs(bundle_dir)
    if not total_emb or not sampler:
        raise RuntimeError("expected model.total_emb.* and model.sampler.* in model.safetensors")

    char_cfg = cfg["emb_backbone_config"]["encoder"]
    n_layers_char = int(char_cfg["num_hidden_layers"])

    gw = gguf.GGUFWriter(str(out_path), arch=ARCH)
    gw.add_file_type(file_type)

    # ---------- KV metadata ----------
    gw.add_uint32(f"{ARCH}.hidden_size", int(cfg["hidden_size"]))
    gw.add_uint32(f"{ARCH}.latent_size", int(cfg["latent_size"]))
    gw.add_uint32(f"{ARCH}.codebook_size", int(cfg["codebook_size"]))
    gw.add_uint32(f"{ARCH}.num_quantizers", int(cfg["num_quantizers"]))
    gw.add_uint32(f"{ARCH}.intermediate_size", int(cfg["intermediate_size"]))
    gw.add_uint32(f"{ARCH}.mog_num_predictions", int(cfg["mog_num_predictions"]))
    gw.add_uint32(f"{ARCH}.mog_num_layers", int(cfg["mog_num_layers"]))
    gw.add_uint32(f"{ARCH}.mog_low_rank", int(cfg["mog_low_rank"]))
    gw.add_float32(f"{ARCH}.mog_min_log_std", float(cfg["mog_min_log_std"]))
    gw.add_float32(f"{ARCH}.mog_eps", float(cfg["mog_eps"]))
    gw.add_float32(f"{ARCH}.noise_scale", float(cfg["noise_scale"]))
    gw.add_float32(f"{ARCH}.top_p_or_k", float(cfg["top_p_or_k"]))
    gw.add_uint32(f"{ARCH}.num_iter", int(cfg["num_iter"]))
    gw.add_float32(f"{ARCH}.exponent", float(cfg["exponent"]))
    gw.add_float32(f"{ARCH}.guidance_scale", float(cfg.get("guidance_scale", 0.0)))

    gw.add_uint32(f"{ARCH}.char.hidden_size", int(char_cfg["hidden_size"]))
    gw.add_uint32(f"{ARCH}.char.intermediate_size", int(char_cfg["intermediate_size"]))
    gw.add_uint32(f"{ARCH}.char.num_attention_heads", int(char_cfg["num_attention_heads"]))
    gw.add_uint32(f"{ARCH}.char.num_kv_heads", int(char_cfg["num_key_value_heads"]))
    gw.add_uint32(f"{ARCH}.char.head_dim", int(char_cfg["head_dim"]))
    gw.add_uint32(f"{ARCH}.char.num_hidden_layers", n_layers_char)
    gw.add_float32(f"{ARCH}.char.rope_theta", float(char_cfg.get("rope_theta", 10000.0)))
    gw.add_float32(f"{ARCH}.char.rms_norm_eps", float(char_cfg.get("rms_norm_eps", 1e-6)))
    # T5Gemma attention scales by `query_pre_attn_scalar ** -0.5` (NOT
    # `1/sqrt(head_dim)`) and applies a tanh logit softcap of 50.0. Both
    # default to the T5GemmaConfig defaults when not overridden.
    gw.add_float32(
        f"{ARCH}.char.query_pre_attn_scalar", float(char_cfg.get("query_pre_attn_scalar", 256.0))
    )
    gw.add_float32(
        f"{ARCH}.char.attn_logit_softcap", float(char_cfg.get("attn_logit_softcapping", 50.0))
    )
    gw.add_uint32(f"{ARCH}.char.max_char_len", int(cfg["max_char_len"]))
    gw.add_uint32(f"{ARCH}.char.char_vocab_size", int(cfg["emb_char_vocab_size"]))
    gw.add_uint32(f"{ARCH}.char.subword_vocab_size", int(cfg["emb_vocab_size"]))

    gw.add_bool(f"{ARCH}.use_subword_flag_emb", bool(cfg.get("use_subword_flag_emb", False)))
    gw.add_bool(f"{ARCH}.use_bos_eos_emb", bool(cfg.get("use_bos_eos_emb", False)))
    gw.add_bool(f"{ARCH}.use_gated_fusion", bool(cfg.get("use_gated_fusion_for_text_audio", False)))
    gw.add_bool(
        f"{ARCH}.use_audio_prompt_proj", bool(cfg.get("use_audio_prompt_frozen_projection", False))
    )

    # ---------- total_emb tensors ----------
    _add_linear(gw, "bos_emb", total_emb["bos_emb"], linear_qtype)
    _add_linear(gw, "null_emb", total_emb["null_emb"], linear_qtype)
    if cfg.get("use_audio_prompt_frozen_projection", False):
        _add_linear(
            gw, "audio_prompt_projection_W", total_emb["audio_prompt_projection_W"], linear_qtype
        )

    for i in range(int(cfg["num_quantizers"])):
        _add_linear(gw, f"rvq_embs.{i}.weight", total_emb[f"rvq_embs.{i}.weight"], linear_qtype)
    _add_linear(gw, "embed_code.weight", total_emb["embed_code.weight"], linear_qtype)

    # Char encoder. The two subword→char lookup tables hold integer-valued
    # floats (rounded back at runtime) with an inner dim of `max_char_len=76`
    # which is not a multiple of 32 — keep as F32 since they're not weights
    # we'd benefit from quantizing anyway.
    _add_linear(
        gw,
        "char.embed_subwords.weight",
        total_emb["embed_subword.embed_subwords.weight"],
        gguf.GGMLQuantizationType.F32,
    )
    _add_linear(
        gw,
        "char.embed_subwords_mask.weight",
        total_emb["embed_subword.embed_subwords_mask.weight"],
        gguf.GGMLQuantizationType.F32,
    )
    _add_linear(
        gw, "char.embed_tokens.weight", total_emb["embed_subword.embed_tokens.weight"], linear_qtype
    )
    _add_linear(
        gw,
        "char.proj_embedding.weight",
        total_emb["embed_subword.proj_embedding.weight"],
        linear_qtype,
    )
    _add_linear(
        gw,
        "char.encoder.norm.weight",
        total_emb["embed_subword.backbone.encoder.norm.weight"],
        linear_qtype,
    )
    for L in range(n_layers_char):
        p = f"embed_subword.backbone.encoder.layers.{L}"
        out_p = f"char.encoder.layers.{L}"
        for ln in (
            "pre_self_attn_layernorm",
            "post_self_attn_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            _add_linear(gw, f"{out_p}.{ln}.weight", total_emb[f"{p}.{ln}.weight"], linear_qtype)
        for sub in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _add_linear(
                gw,
                f"{out_p}.self_attn.{sub}.weight",
                total_emb[f"{p}.self_attn.{sub}.weight"],
                linear_qtype,
            )
        for sub in ("gate_proj", "up_proj", "down_proj"):
            _add_linear(
                gw, f"{out_p}.mlp.{sub}.weight", total_emb[f"{p}.mlp.{sub}.weight"], linear_qtype
            )

    # Optional flag/BOS-EOS/gated-fusion tables.
    if cfg.get("use_subword_flag_emb", False):
        _add_linear(
            gw,
            "subword_flag_emb.cont_emb.weight",
            total_emb["subword_flag_emb.cont_emb.weight"],
            linear_qtype,
        )
        cont = (
            total_emb["subword_flag_emb.is_continuation"].to(torch.int32).contiguous().cpu().numpy()
        )
        gw.add_tensor(
            "subword_flag_emb.is_continuation", cont, raw_dtype=gguf.GGMLQuantizationType.I32
        )
    if cfg.get("use_bos_eos_emb", False):
        _add_linear(
            gw,
            "bos_eos_emb.special_emb.weight",
            total_emb["bos_eos_emb.special_emb.weight"],
            linear_qtype,
        )
        sf = total_emb["bos_eos_emb.special_flags"].to(torch.int32).contiguous().cpu().numpy()
        gw.add_tensor("bos_eos_emb.special_flags", sf, raw_dtype=gguf.GGMLQuantizationType.I32)
    if cfg.get("use_gated_fusion_for_text_audio", False):
        _add_linear(
            gw,
            "gated_fusion.audio_proj.weight",
            total_emb["gated_fusion_audio_text.audio_proj.weight"],
            linear_qtype,
        )
        _add_linear(
            gw,
            "gated_fusion.audio_proj.bias",
            total_emb["gated_fusion_audio_text.audio_proj.bias"],
            linear_qtype,
        )
        _add_linear(
            gw,
            "gated_fusion.text_proj.weight",
            total_emb["gated_fusion_audio_text.text_proj.weight"],
            linear_qtype,
        )
        _add_linear(
            gw,
            "gated_fusion.text_proj.bias",
            total_emb["gated_fusion_audio_text.text_proj.bias"],
            linear_qtype,
        )
        _add_linear(
            gw,
            "gated_fusion.final_norm.weight",
            total_emb["gated_fusion_audio_text.final_norm.weight"],
            linear_qtype,
        )
        # Pre-fold sigmoids so the runtime doesn't need to recompute them.
        gate = torch.sigmoid(total_emb["gated_fusion_audio_text.gate"].float())
        res = torch.sigmoid(total_emb["gated_fusion_audio_text.residual_scale"].float())
        gw.add_tensor(
            "gated_fusion.gate_sigmoid", _to_np(gate), raw_dtype=gguf.GGMLQuantizationType.F32
        )
        # residual_scale is a scalar (). Store as shape (1,).
        gw.add_tensor(
            "gated_fusion.residual_scale_sigmoid",
            np.array([float(res)], dtype=np.float32),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )

    # ---------- sampler tensors ----------
    _add_linear(gw, "sampler.embed_code.weight", sampler["embed_code.weight"], linear_qtype)
    _add_linear(gw, "sampler.rvq_embs", sampler["rvq_embs"], linear_qtype)

    _add_linear(
        gw,
        "sampler.mog_head.proj_logits.weight",
        sampler["mog_head.proj_logits.weight"],
        linear_qtype,
    )
    _add_linear(
        gw, "sampler.mog_head.proj_mus.weight", sampler["mog_head.proj_mus.weight"], linear_qtype
    )
    _add_linear(
        gw, "sampler.mog_head.proj_logs.weight", sampler["mog_head.proj_logs.weight"], linear_qtype
    )
    _add_linear(
        gw, "sampler.mog_head.proj_else.weight", sampler["mog_head.proj_else.weight"], linear_qtype
    )
    _add_linear(gw, "sampler.mog_head.low_mat", sampler["mog_head.low_mat"], linear_qtype)

    # mlp_stack: in PyTorch nn.Sequential the last element is the final RMSNorm
    # ("mlp_stack.{L}.weight" where L == mog_num_layers).
    L = int(cfg["mog_num_layers"])
    for i in range(L):
        for ln in ("pre_norm", "post_norm"):
            _add_linear(
                gw,
                f"sampler.mog_head.mlp_stack.{i}.{ln}.weight",
                sampler[f"mog_head.mlp_stack.{i}.{ln}.weight"],
                linear_qtype,
            )
        for sub in ("gate_proj", "up_proj", "down_proj"):
            _add_linear(
                gw,
                f"sampler.mog_head.mlp_stack.{i}.mlp.{sub}.weight",
                sampler[f"mog_head.mlp_stack.{i}.mlp.{sub}.weight"],
                linear_qtype,
            )
    _add_linear(
        gw,
        "sampler.mog_head.final_norm.weight",
        sampler[f"mog_head.mlp_stack.{L}.weight"],
        linear_qtype,
    )

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    sz_mb = out_path.stat().st_size / 1e6
    print(f"[convert-eartts] wrote {out_path} ({sz_mb:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "bundle_dir",
        type=Path,
        help="EarTTS bundle directory (contains config.json + model.safetensors)",
    )
    ap.add_argument("out", type=Path)
    ap.add_argument("--weight-type", default="q8_0", choices=list(WEIGHT_TYPES.keys()))
    ap.add_argument(
        "--tokenizer-json",
        type=Path,
        help="Nano tokenizer.json used to reconstruct character lookup tables",
    )
    args = ap.parse_args()
    convert(args.bundle_dir, args.out, args.weight_type, args.tokenizer_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
