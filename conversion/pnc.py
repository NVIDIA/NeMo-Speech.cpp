#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a NeMo BERT punctuation+capitalization (.nemo) model to GGUF.

    python3 convert_model.py <pnc.nemo> --outfile <out.gguf> [--outtype q8_0]

Emits arch="pnc": a BERT encoder + two token-classification heads (punct, capit)
under the `pnc.*` namespace, plus the WordPiece vocab and label sets.
"""

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFWriter

from .quantization import quantize as _quantize
from .source import extract_archive

ARCH = "pnc"

WEIGHT_TYPES = {
    "bf16": (GGMLQuantizationType.BF16, 32),
    "fp16": (GGMLQuantizationType.F16, 1),
    "q8_0": (GGMLQuantizationType.Q8_0, 7),
}


# --- tensor rename: HF BERT (under bert_model.) + 2 heads -> pnc.* ---
def remap(name: str) -> Optional[str]:
    m = "bert_model."
    if name.startswith(m):
        s = name[len(m) :]
        if s == "embeddings.word_embeddings.weight":
            return "pnc.token_embd.weight"
        if s == "embeddings.position_embeddings.weight":
            return "pnc.pos_embd.weight"
        if s == "embeddings.token_type_embeddings.weight":
            return "pnc.type_embd.weight"
        if s.startswith("embeddings.LayerNorm."):
            return "pnc.embd_norm." + s.split(".")[-1]
        if s.startswith("encoder.layer."):
            parts = s.split(".")
            i = parts[2]
            tail = ".".join(parts[3:])
            sub = {
                "attention.self.query.weight": "attn_q.weight",
                "attention.self.query.bias": "attn_q.bias",
                "attention.self.key.weight": "attn_k.weight",
                "attention.self.key.bias": "attn_k.bias",
                "attention.self.value.weight": "attn_v.weight",
                "attention.self.value.bias": "attn_v.bias",
                "attention.output.dense.weight": "attn_out.weight",
                "attention.output.dense.bias": "attn_out.bias",
                "attention.output.LayerNorm.weight": "attn_norm.weight",
                "attention.output.LayerNorm.bias": "attn_norm.bias",
                "intermediate.dense.weight": "ffn_up.weight",
                "intermediate.dense.bias": "ffn_up.bias",
                "output.dense.weight": "ffn_down.weight",
                "output.dense.bias": "ffn_down.bias",
                "output.LayerNorm.weight": "ffn_norm.weight",
                "output.LayerNorm.bias": "ffn_norm.bias",
            }.get(tail)
            return f"pnc.blk.{i}.{sub}" if sub else None
        return None  # pooler, position_ids
    if name.startswith("punct_classifier.mlp.layer0."):
        return "pnc.punct_head." + name.split(".")[-1]
    if name.startswith("capit_classifier.mlp.layer0."):
        return "pnc.capit_head." + name.split(".")[-1]
    return None


# attn/ffn projection weights are quantized; embeddings -> F16; heads/norms/biases -> F32.
def pick_dtype(dst: str, linear_qtype: GGMLQuantizationType) -> GGMLQuantizationType:
    base = dst.rsplit(".", 1)[0]
    if dst.endswith(".weight") and base.split(".")[-1] in (
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_out",
        "ffn_up",
        "ffn_down",
    ):
        return linear_qtype
    if dst in ("pnc.token_embd.weight", "pnc.pos_embd.weight", "pnc.type_embd.weight"):
        return GGMLQuantizationType.F16
    return GGMLQuantizationType.F32


def convert(
    source: Path, out_path: Path, weight_type: str = "q8_0", max_seq_length: int = 128
) -> None:
    linear_qtype, file_type = WEIGHT_TYPES[weight_type]

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        extract_archive(source, tmp)
        files = {p.name: p for p in tmp.rglob("*")}

        def find(suffix):
            for n, p in files.items():
                if n.endswith(suffix):
                    return p
            return None

        vocab_path = find("vocab.txt")
        punct_path = find("punct_label_ids.csv")
        capit_path = find("capit_label_ids.csv")
        enc_cfg_path = find("encoder_config.json")
        ckpt_path = find("model_weights.ckpt")
        if not all([vocab_path, punct_path, capit_path, enc_cfg_path, ckpt_path]):
            raise RuntimeError("missing expected PnC artifacts in .nemo")

        import json

        # Pin utf-8: Path.read_text() defaults to the locale codepage on Windows
        # (cp1252), which fails on non-ASCII vocab/config bytes.
        cfg = json.loads(enc_cfg_path.read_text(encoding="utf-8"))
        vocab = [ln.rstrip("\n") for ln in vocab_path.read_text(encoding="utf-8").splitlines()]
        punct_labels = [
            ln.strip()
            for ln in punct_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() != ""
        ]
        capit_labels = [
            ln.strip()
            for ln in capit_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() != ""
        ]

        def tok_id(t):
            return vocab.index(t) if t in vocab else 0

        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        print(
            f"[convert] BERT h={cfg['hidden_size']} L={cfg['num_hidden_layers']} "
            f"heads={cfg['num_attention_heads']} ff={cfg['intermediate_size']} "
            f"vocab={len(vocab)} punct={punct_labels} capit={capit_labels}"
        )

        gw = GGUFWriter(str(out_path), arch=ARCH)
        gw.add_architecture()
        gw.add_string("general.name", out_path.stem)
        gw.add_uint32("general.file_type", file_type)
        gw.add_uint32("pnc.hidden_size", int(cfg["hidden_size"]))
        gw.add_uint32("pnc.n_layers", int(cfg["num_hidden_layers"]))
        gw.add_uint32("pnc.n_heads", int(cfg["num_attention_heads"]))
        gw.add_uint32("pnc.intermediate_size", int(cfg["intermediate_size"]))
        gw.add_uint32("pnc.max_position_embeddings", int(cfg["max_position_embeddings"]))
        gw.add_uint32("pnc.type_vocab_size", int(cfg.get("type_vocab_size", 2)))
        gw.add_float32("pnc.layer_norm_eps", float(cfg.get("layer_norm_eps", 1e-12)))
        gw.add_uint32("pnc.max_seq_length", int(max_seq_length))
        gw.add_array("pnc.punct.labels", punct_labels)
        gw.add_array("pnc.capit.labels", capit_labels)
        gw.add_array("pnc.tokenizer.vocab", vocab)
        gw.add_uint32("pnc.tokenizer.cls_id", tok_id("[CLS]"))
        gw.add_uint32("pnc.tokenizer.sep_id", tok_id("[SEP]"))
        gw.add_uint32("pnc.tokenizer.pad_id", tok_id("[PAD]"))
        gw.add_uint32("pnc.tokenizer.unk_id", tok_id("[UNK]"))

        emitted, skipped = 0, 0
        tally: dict[str, int] = {}
        for src, tensor in sd.items():
            dst = remap(src)
            if dst is None:
                skipped += 1
                continue
            arr = tensor.detach().cpu().float().numpy()
            qt = pick_dtype(dst, linear_qtype)
            if qt == GGMLQuantizationType.F32:
                gw.add_tensor(dst, arr.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
            elif qt == GGMLQuantizationType.F16:
                gw.add_tensor(dst, arr.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
            else:
                gw.add_tensor(dst, _quantize(arr, qt), raw_dtype=qt)
            tally[qt.name] = tally.get(qt.name, 0) + 1
            emitted += 1

        print(
            f"[convert] emitted {emitted}, skipped {skipped}; dtypes "
            + ", ".join(f"{k}={v}" for k, v in sorted(tally.items()))
        )
        gw.write_header_to_file()
        gw.write_kv_data_to_file()
        gw.write_tensors_to_file()
        gw.close()
        print(f"[convert] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
