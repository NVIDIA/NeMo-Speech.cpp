#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert S2S LLM-side auxiliary weights to GGUF.

These two tensors live outside the LLM backbone (llama.cpp ingests the
backbone) and currently get loaded as raw safetensors at runtime in
checkpoint_utils/load_utils.py. They become part of the LLM-side GGML
component:

  function_head.weight  : (n_function_classes, n_embd) -- single Linear head
                          run on the LLM hidden state to pick a "function"
                          token (tool-calling).
  embed_tokens.weight   : (n_vocab, n_embd) -- token embedding lookup used
                          when the S2S orchestrator needs to feed text tokens
                          back as embeddings (BOS, pad, last-text-token, ...).

Both are large (131072 × 4480 ≈ 2.3 GB f32). Default --weight-type is q8_0
which knocks the GGUF down to ~600 MB per tensor; pass --weight-type fp16 or
bf16 for closer parity at the cost of size.

GGUF schema (single file, `general.architecture = "s2s_llm_aux"`):

    s2s_llm_aux.hidden_size            (u32)
    s2s_llm_aux.vocab_size             (u32)
    s2s_llm_aux.function_head.out_dim  (u32 -- usually == vocab_size)
    s2s_llm_aux.user_channel_weight    (f32)
    s2s_llm_aux.text_channel_weight    (f32)
    s2s_llm_aux.function_channel_weight (f32)
    function_head.weight               (out_dim, hidden) Q8_0 / F16 / etc.
    embed_tokens.weight                (vocab,   hidden) Q8_0 / F16 / etc.

Usage:
    python scripts/convert_llm_aux_to_gguf.py \
        /path/to/model_repo/1 \
        /path/to/out/llm_aux.gguf \
        --weight-type q8_0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open
from voicechat_source import load_llm_channel_weights

ARCH = "s2s_llm_aux"
KEY_HIDDEN = f"{ARCH}.hidden_size"
KEY_VOCAB = f"{ARCH}.vocab_size"
KEY_FN_OUT = f"{ARCH}.function_head.out_dim"
KEY_USER_WEIGHT = f"{ARCH}.user_channel_weight"
KEY_TEXT_WEIGHT = f"{ARCH}.text_channel_weight"
KEY_FUNCTION_WEIGHT = f"{ARCH}.function_channel_weight"


WEIGHT_TYPES = {
    "bf16": (
        gguf.GGMLQuantizationType.BF16,
        gguf.GGMLQuantizationType.F32,
        gguf.LlamaFileType.MOSTLY_BF16,
    ),
    "fp16": (
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.F32,
        gguf.LlamaFileType.MOSTLY_F16,
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


def _quantize(t: torch.Tensor, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
    """Quantize a 2-D float tensor to the requested GGML type."""
    source = t.detach().cpu()
    # gguf-py's generic BF16 path promotes every fp32 word to uint64 while
    # rounding.  A 131072 x 4480 table then needs several temporary gigabytes
    # and can be killed despite ample steady-state memory.  PyTorch performs
    # the same round-to-nearest-even conversion with a single 2-byte output
    # buffer; expose those bytes in gguf's expected packed shape.
    if qtype == gguf.GGMLQuantizationType.BF16:
        packed = source.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().view(np.uint8)
        return packed.reshape(*source.shape[:-1], source.shape[-1] * 2)
    arr = source.float().numpy()
    if qtype == gguf.GGMLQuantizationType.F32:
        return arr
    if qtype == gguf.GGMLQuantizationType.F16:
        return arr.astype(np.float16)
    # Q4_K / Q5_K / Q6_K / Q8_0 — gguf.quants handles row-wise blocking.
    # The current embedding table contains a small set of effectively-zero
    # rows (~1e-37).  Quantized block scales are fp16, so these values cannot
    # be represented and gguf's reciprocal-scale calculation overflows.
    # Canonicalize only those unrepresentable rows to exact zero.
    row_max = np.max(np.abs(arr), axis=-1)
    tiny_rows = row_max < 1e-30
    if np.any(tiny_rows):
        print(f"[convert-llm-aux] zeroing {int(tiny_rows.sum())} unrepresentable tiny rows")
        arr[tiny_rows] = 0.0
    return gguf.quants.quantize(arr, qtype)


def convert(bundle_dir: Path, out_path: Path, weight_type: str) -> None:
    if weight_type not in WEIGHT_TYPES:
        raise ValueError(f"--weight-type must be one of {sorted(WEIGHT_TYPES)}")
    linear_qtype, default_qtype, file_type = WEIGHT_TYPES[weight_type]
    print(
        f"[convert-llm-aux] weight_type={weight_type} "
        f"(linear={linear_qtype.name}, default={default_qtype.name})"
    )

    # Older bundles split both auxiliary tensors out at the version root.
    # In the current VoiceChat bundle the function head remains inside the
    # Nano-v2 checkpoint as ``stt_model.function_head.weight`` while the
    # embeddings are still provided separately.  Accept both layouts so one
    # converter works for old and current model repositories.
    fn_path = bundle_dir / "function_head.safetensors"
    nano_path = bundle_dir / "nano-v2-vllm" / "model.safetensors"
    emb_path = bundle_dir / "embeddings.safetensors"
    public_path = bundle_dir / "model.safetensors"

    if fn_path.is_file():
        fn_source = (fn_path, "function_head.weight")
        print(f"[convert-llm-aux] function head: {fn_path}")
    elif nano_path.is_file():
        fn_source = (nano_path, "stt_model.function_head.weight")
        print(f"[convert-llm-aux] function head: {nano_path}:stt_model.function_head.weight")
    elif public_path.is_file():
        fn_source = (public_path, "stt_model.function_head.weight")
        print(f"[convert-llm-aux] function head: {public_path}:stt_model.function_head.weight")
    else:
        raise FileNotFoundError(
            f"Neither {fn_path} nor current-layout checkpoint {nano_path} exists"
        )

    if emb_path.is_file():
        emb_source = (emb_path, "embed_tokens.weight")
        print(f"[convert-llm-aux] embeddings: {emb_path}")
    elif nano_path.is_file():
        emb_source = (nano_path, "stt_model.embed_tokens.weight")
        print(f"[convert-llm-aux] embeddings: {nano_path}:stt_model.embed_tokens.weight")
    elif public_path.is_file():
        emb_source = (public_path, "stt_model.embed_tokens.weight")
        print(f"[convert-llm-aux] embeddings: {public_path}:stt_model.embed_tokens.weight")
    else:
        raise FileNotFoundError(emb_path)

    def tensor_shape(source: tuple[Path, str]) -> tuple[int, ...]:
        path, name = source
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            return tuple(tensors.get_slice(name).get_shape())

    out_dim, hidden = tensor_shape(fn_source)
    vocab, hidden_em = tensor_shape(emb_source)
    if hidden != hidden_em:
        raise ValueError(f"hidden mismatch: function_head={hidden} vs embed_tokens={hidden_em}")

    print(f"[convert-llm-aux] hidden={hidden} vocab={vocab} fn_out={out_dim}")
    channel_weights = load_llm_channel_weights(bundle_dir)
    print(
        "[convert-llm-aux] channel weights: "
        f"user={channel_weights['user']} text={channel_weights['text']} "
        f"function={channel_weights['function']}"
    )

    # Spill packed tensors as they are added so only one source table and one
    # converted table are resident at a time.
    gw = gguf.GGUFWriter(str(out_path), arch=ARCH, use_temp_file=True)
    gw.add_file_type(file_type)
    gw.add_uint32(KEY_HIDDEN, hidden)
    gw.add_uint32(KEY_VOCAB, vocab)
    gw.add_uint32(KEY_FN_OUT, out_dim)
    gw.add_float32(KEY_USER_WEIGHT, channel_weights["user"])
    gw.add_float32(KEY_TEXT_WEIGHT, channel_weights["text"])
    gw.add_float32(KEY_FUNCTION_WEIGHT, channel_weights["function"])

    for output_name, (path, tensor_name) in (
        ("function_head.weight", fn_source),
        ("embed_tokens.weight", emb_source),
    ):
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            source_tensor = tensors.get_tensor(tensor_name)
        packed = _quantize(source_tensor, linear_qtype)
        gw.add_tensor(output_name, packed, raw_dtype=linear_qtype)
        del packed, source_tensor

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    sz_mb = out_path.stat().st_size / 1e6
    print(f"[convert-llm-aux] wrote {out_path} ({sz_mb:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--weight-type", default="q8_0", choices=list(WEIGHT_TYPES.keys()))
    args = ap.parse_args()
    convert(args.bundle_dir, args.out, args.weight_type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
