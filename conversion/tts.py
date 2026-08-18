#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert NVIDIA NeMo MagpieTTS checkpoints to GGUF.

This converter targets the public MagpieTTS multilingual 357M checkpoint:
https://huggingface.co/nvidia/magpie_tts_multilingual_357m

The generated GGUF stores the autoregressive MagpieTTS model weights and
metadata needed by the GGML example. The NanoCodec vocoder referenced by the
NeMo config is not bundled in the Magpie .nemo archive and is not converted
here.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch

from .source import extract_archive, find_checkpoint_files, load_state_dict, read_checkpoint_config

SPECIAL_AUDIO_TOKENS = 8
SPEAKER_NAMES = ["John", "Sofia", "Aria", "Jason", "Leo"]


def extract_nemo(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path.is_dir():
        return path, None

    tmp = tempfile.TemporaryDirectory(prefix="magpietts-nemo-")
    out = Path(tmp.name)
    extract_archive(path, out, basenames={"model_config.yaml", "model_weights.ckpt"})
    return out, tmp


def read_config(root: Path) -> dict[str, Any]:
    return read_checkpoint_config(root)


def read_state_dict(root: Path) -> dict[str, torch.Tensor]:
    checkpoint = find_checkpoint_files(root)["weights"]
    if checkpoint is None:
        raise RuntimeError(f"checkpoint contains no model_weights.ckpt: {root}")
    return load_state_dict(checkpoint, weights_only=True)


def cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def add_i32(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    writer.add_int32(key, int(value))


def add_f32(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    writer.add_float32(key, float(value))


def add_bool(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    writer.add_bool(key, bool(value))


def add_i32_array(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    if value is None:
        return
    items = [int(x) for x in value]
    if items:
        writer.add_array(key, items)


def add_metadata(
    writer: gguf.GGUFWriter, cfg: dict[str, Any], sd: dict[str, torch.Tensor]
) -> dict[str, Any]:
    audio_vocab = int(sd["audio_embeddings.0.weight"].shape[0])
    codebook_size = audio_vocab - SPECIAL_AUDIO_TOKENS
    text_vocab = int(sd["text_embedding.weight"].shape[0])
    baked_t = int(sd["_baked_embedding_T"].item())
    baked_d = int(sd["_baked_embedding_D"].item())
    baked_lens = [int(x) for x in sd["baked_context_embedding_len"].tolist()]

    encoder = cfg["encoder"]
    decoder = cfg["decoder"]
    lt_hidden = int(cfg.get("local_transformer_hidden_dim", 256))
    frame_stacking = int(cfg.get("frame_stacking_factor", 1))
    n_stacked_codebooks = int(
        len([k for k in sd if k.startswith("audio_embeddings.") and k.endswith(".weight")])
    )
    if frame_stacking < 1 or n_stacked_codebooks == 0 or n_stacked_codebooks % frame_stacking:
        raise ValueError(
            "audio embedding count must be a positive multiple of frame_stacking_factor: "
            f"embeddings={n_stacked_codebooks} frame_stacking_factor={frame_stacking}"
        )
    n_codebooks = n_stacked_codebooks // frame_stacking
    expected_logits = n_stacked_codebooks * audio_vocab
    if int(sd["final_proj.weight"].shape[0]) != expected_logits:
        raise ValueError(
            "final_proj rows do not match stacked audio layout: "
            f"rows={sd['final_proj.weight'].shape[0]} expected={expected_logits}"
        )
    n_lt_heads = len(
        [
            k
            for k in sd
            if k.startswith("local_transformer_out_projections.") and k.endswith(".weight")
        ]
    )
    if n_lt_heads != n_stacked_codebooks:
        raise ValueError(
            "local transformer output heads do not match audio embeddings: "
            f"heads={n_lt_heads} embeddings={n_stacked_codebooks}"
        )

    inf = cfg.get("inference_parameters", {})

    summary: dict[str, Any] = {
        "architecture": "magpietts",
        "model_type": cfg.get("model_type"),
        "nemo_target": cfg.get("target"),
        "nemo_version": cfg.get("nemo_version"),
        "codec_model": cfg.get("codecmodel_path"),
        "text_vocab_size": text_vocab,
        "audio_codebooks": n_codebooks,
        "stacked_audio_codebooks": n_stacked_codebooks,
        "audio_codebook_size": codebook_size,
        "audio_vocab_size": audio_vocab,
        "frame_stacking_factor": frame_stacking,
        "embedding_dim": int(cfg.get("embedding_dim", decoder["d_model"])),
        "encoder_layers": int(encoder["n_layers"]),
        "decoder_layers": int(decoder["n_layers"]),
        "context_length": int(decoder["max_length_causal_mask"]),
        "speaker_names": SPEAKER_NAMES,
        "baked_context_length": baked_t,
        "baked_context_dim": baked_d,
        "baked_context_lens": baked_lens,
    }

    writer.add_name("NVIDIA MagpieTTS multilingual 357M")
    writer.add_description(
        "MagpieTTS autoregressive codec-token generator converted from NeMo to GGUF"
    )
    writer.add_string("magpietts.nemo_target", str(cfg.get("target", "")))
    writer.add_string("magpietts.nemo_version", str(cfg.get("nemo_version", "")))
    writer.add_string("magpietts.model_type", str(cfg.get("model_type", "")))
    writer.add_string("magpietts.codec_model", str(cfg.get("codecmodel_path", "")))
    writer.add_string("magpietts.config_json", json.dumps(cfg, ensure_ascii=False, sort_keys=True))
    writer.add_array("magpietts.speaker_names", SPEAKER_NAMES)
    writer.add_array("magpietts.baked_context_lens", baked_lens)

    add_i32(writer, "magpietts.text_vocab_size", text_vocab)
    add_i32(writer, "magpietts.audio_codebooks", n_codebooks)
    add_i32(writer, "magpietts.stacked_audio_codebooks", n_stacked_codebooks)
    add_i32(writer, "magpietts.audio_codebook_size", codebook_size)
    add_i32(writer, "magpietts.audio_vocab_size", audio_vocab)
    add_i32(writer, "magpietts.audio_bos_id", codebook_size + 0)
    add_i32(writer, "magpietts.audio_eos_id", codebook_size + 1)
    add_i32(writer, "magpietts.context_audio_bos_id", codebook_size + 2)
    add_i32(writer, "magpietts.context_audio_eos_id", codebook_size + 3)
    add_i32(writer, "magpietts.mask_token_id", codebook_size + 4)
    add_i32(writer, "magpietts.frame_stacking_factor", frame_stacking)
    add_i32(writer, "magpietts.embedding_dim", summary["embedding_dim"])
    add_i32(writer, "magpietts.ffn_dim", int(decoder["d_ffn"]))
    add_i32(writer, "magpietts.context_length", summary["context_length"])
    add_i32(writer, "magpietts.encoder.layers", int(encoder["n_layers"]))
    add_i32(writer, "magpietts.encoder.heads", int(encoder["sa_n_heads"]))
    add_i32(writer, "magpietts.encoder.kernel_size", int(encoder["kernel_size"]))
    add_i32(writer, "magpietts.decoder.layers", int(decoder["n_layers"]))
    add_i32(writer, "magpietts.decoder.heads", int(decoder["sa_n_heads"]))
    add_i32(writer, "magpietts.decoder.cross_heads", int(decoder["xa_n_heads"]))
    add_i32(writer, "magpietts.decoder.cross_head_dim", int(decoder["xa_d_head"]))
    add_i32(writer, "magpietts.decoder.kernel_size", int(decoder["kernel_size"]))
    add_i32(
        writer, "magpietts.local_transformer.layers", int(cfg.get("local_transformer_n_layers", 1))
    )
    add_i32(
        writer, "magpietts.local_transformer.heads", int(cfg.get("local_transformer_n_heads", 1))
    )
    add_i32(writer, "magpietts.local_transformer.hidden_dim", lt_hidden)
    add_i32(
        writer,
        "magpietts.local_transformer.context_length",
        int(sd["local_transformer.position_embeddings.weight"].shape[0]),
    )
    add_i32(writer, "magpietts.baked_context_length", baked_t)
    add_i32(writer, "magpietts.baked_context_dim", baked_d)
    add_i32(writer, "magpietts.baked_speakers", int(sd["baked_context_embedding.weight"].shape[0]))
    add_i32(writer, "magpietts.inference.max_decoder_steps", inf.get("max_decoder_steps", 500))
    add_i32(writer, "magpietts.inference.topk", inf.get("topk", 80))
    add_i32(writer, "magpietts.inference.min_generated_frames", inf.get("min_generated_frames", 4))
    add_f32(writer, "magpietts.inference.temperature", inf.get("temperature", 0.6))
    add_f32(writer, "magpietts.inference.cfg_scale", inf.get("cfg_scale", 2.5))
    add_bool(
        writer,
        "magpietts.inference.apply_attention_prior",
        inf.get("apply_attention_prior", False),
    )
    add_f32(
        writer,
        "magpietts.inference.attention_prior_epsilon",
        inf.get("attention_prior_epsilon", 0.1),
    )
    add_i32(
        writer,
        "magpietts.inference.attention_prior_lookahead_window",
        inf.get("attention_prior_lookahead_window", 5),
    )
    add_i32(
        writer,
        "magpietts.inference.start_prior_after_n_audio_steps",
        inf.get("start_prior_after_n_audio_steps", 0),
    )
    add_i32(
        writer,
        "magpietts.inference.attention_prior_advance_threshold",
        inf.get("attention_prior_advance_threshold", 8),
    )
    add_i32(
        writer,
        "magpietts.inference.attention_prior_decay_threshold",
        inf.get("attention_prior_decay_threshold", 10),
    )
    add_i32_array(
        writer,
        "magpietts.inference.estimate_alignment_from_layers",
        inf.get("estimate_alignment_from_layers"),
    )
    add_i32_array(
        writer,
        "magpietts.inference.apply_prior_to_layers",
        inf.get("apply_prior_to_layers"),
    )

    return summary


def should_store_f32(name: str, tensor: torch.Tensor) -> bool:
    if not tensor.is_floating_point():
        return False
    if tensor.ndim <= 1:
        return True
    return name.endswith(".bias")


def tensor_to_numpy(name: str, tensor: torch.Tensor, outtype: str) -> np.ndarray:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.is_floating_point():
        if outtype == "f16" and not should_store_f32(name, tensor):
            tensor = tensor.to(torch.float16)
        else:
            tensor = tensor.to(torch.float32)
    elif tensor.dtype == torch.int64:
        tensor = tensor.to(torch.int32)
    return tensor.numpy()


def add_tensor(writer: gguf.GGUFWriter, name: str, tensor: torch.Tensor, outtype: str) -> None:
    writer.add_tensor(name, tensor_to_numpy(name, tensor, outtype))


def is_local_transformer_tensor(name: str) -> bool:
    return name.startswith(
        (
            "audio_embeddings.",
            "local_transformer.",
            "local_transformer_in_projection.",
            "local_transformer_out_projections.",
        )
    )


def add_tensors(
    writer: gguf.GGUFWriter,
    sd: dict[str, torch.Tensor],
    outtype: str,
    local_transformer_outtype: str | None,
) -> tuple[int, list[str]]:
    n_written = 0
    skipped: list[str] = []

    for name, tensor in sd.items():
        if name.endswith(".causal_mask"):
            skipped.append(name)
            continue

        tensor_outtype = (
            local_transformer_outtype
            if local_transformer_outtype and is_local_transformer_tensor(name)
            else outtype
        )

        if name.endswith(".pos_ff.proj.conv.weight") or name.endswith(".pos_ff.o_net.conv.weight"):
            if tensor.ndim != 3:
                raise ValueError(
                    f"expected Conv1d weight to be rank 3: {name} {tuple(tensor.shape)}"
                )
            kernel = int(tensor.shape[2])
            if kernel == 1:
                add_tensor(writer, name + ".k0", tensor[:, :, 0], tensor_outtype)
                n_written += 1
            else:
                for k in range(kernel):
                    add_tensor(writer, f"{name}.k{k}", tensor[:, :, k], tensor_outtype)
                    n_written += 1
            continue

        add_tensor(writer, name, tensor, tensor_outtype)
        n_written += 1

    return n_written, skipped


def convert(
    source: Path,
    output: Path,
    outtype: str = "f16",
    metadata_json: Path | None = None,
    local_transformer_outtype: str | None = None,
) -> None:
    root, tmp = extract_nemo(source)
    try:
        cfg = read_config(root)
        sd = read_state_dict(root)

        if cfg.get("target") != "nemo.collections.tts.models.magpietts.MagpieTTSModel":
            raise RuntimeError(f"unsupported target: {cfg.get('target')}")
        # v2602 records this in the config while v2607 retains the baked
        # tensor but omits the legacy config flag.
        if cfg.get("model_type") != "decoder_ce" or "baked_context_embedding.weight" not in sd:
            raise RuntimeError(
                "this GGML example expects MagpieTTS decoder_ce with baked context embeddings"
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        writer = gguf.GGUFWriter(output, "magpietts")
        summary = add_metadata(writer, cfg, sd)
        n_written, skipped = add_tensors(writer, sd, outtype, local_transformer_outtype)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        summary["tensors_written"] = n_written
        summary["tensors_skipped"] = skipped
        summary["output"] = str(output)
        summary["local_transformer_outtype"] = local_transformer_outtype or outtype
        if metadata_json:
            metadata_json.parent.mkdir(parents=True, exist_ok=True)
            metadata_json.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

        print(f"wrote {output}")
        print(f"stored {n_written} tensors; skipped {len(skipped)} deterministic causal masks")
    finally:
        if tmp is not None:
            tmp.cleanup()
