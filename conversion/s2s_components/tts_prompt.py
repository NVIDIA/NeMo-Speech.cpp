#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the VoiceChat speaker prompt to its runtime GGUF artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter
from safetensors import safe_open


def _load_legacy(path: Path) -> tuple[np.ndarray, ...]:
    prompt = torch.load(path, map_location="cpu", weights_only=True)
    code = prompt["code"][0].to(torch.int32).numpy()
    subword_ids = prompt["subword_ids"][0].to(torch.int32).numpy()
    subword_mask = prompt["subword_mask"][0].to(torch.float32).numpy()
    non_prompt_mask = prompt["non_prompt_mask"][0].to(torch.float32).numpy()
    latent = prompt.get("audio_prompt_latent")
    if latent is not None:
        latent = latent.squeeze(0).to(torch.float32).numpy()
    return code, subword_ids, subword_mask, non_prompt_mask, latent


def _token_id(tokenizer_json: Path, token: str) -> int:
    tokenizer = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocab = tokenizer.get("model", {}).get("vocab", {})
    if token not in vocab:
        raise KeyError(f"{tokenizer_json} has no {token!r} token")
    return int(vocab[token])


def _load_public(source: Path, tokenizer_json: Path) -> tuple[np.ndarray, ...]:
    checkpoint = source / "model.safetensors"
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    speaker = str(config.get("model", {}).get("inference_speaker_name", "Aria"))
    with safe_open(str(checkpoint), framework="pt", device="cpu") as tensors:
        latent = tensors.get_tensor(f"tts_model.audio_prompt_latents.{speaker}").squeeze(0)
        control_codes = tensors.get_tensor("tts_model._control_codes")
        silence = tensors.get_tensor("tts_model.codec_silence_tokens")

    latent_np = latent.to(torch.float32).numpy()
    silence_np = silence.to(torch.int32).numpy()
    length = int(latent_np.shape[0])
    if length < 2 or silence_np.ndim != 1:
        raise ValueError("invalid VoiceChat speaker prompt tensors")
    mask_code = int(control_codes.min().item())
    code = np.broadcast_to(silence_np, (length, silence_np.size)).copy()
    code[0, :] = mask_code
    code[-1, :] = mask_code

    pad_id = _token_id(tokenizer_json, "<SPECIAL_12>")
    eos_id = _token_id(tokenizer_json, "</s>")
    subword_ids = np.full(length, pad_id, dtype=np.int32)
    subword_ids[-1] = eos_id
    subword_mask = np.zeros(length, dtype=np.float32)
    subword_mask[-2:] = 1.0
    non_prompt_mask = np.zeros(length, dtype=np.float32)
    non_prompt_mask[-1] = 1.0
    return code, subword_ids, subword_mask, non_prompt_mask, latent_np


def convert(source: Path, destination: Path, tokenizer_json: Path | None = None) -> None:
    if source.is_dir():
        if tokenizer_json is None or not tokenizer_json.is_file():
            raise FileNotFoundError("the public VoiceChat checkpoint requires --tokenizer-json")
        values = _load_public(source, tokenizer_json)
    else:
        values = _load_legacy(source)
    code, subword_ids, subword_mask, non_prompt_mask, audio_prompt_latent = values
    length, quantizers = code.shape

    if audio_prompt_latent is not None:
        if audio_prompt_latent.ndim == 1:
            audio_prompt_latent = np.broadcast_to(
                audio_prompt_latent, (length, audio_prompt_latent.shape[0])
            ).copy()
        if audio_prompt_latent.shape[0] == 1:
            audio_prompt_latent = np.broadcast_to(
                audio_prompt_latent, (length, audio_prompt_latent.shape[1])
            ).copy()
        if audio_prompt_latent.shape[0] != length:
            raise ValueError(
                f"audio prompt length {audio_prompt_latent.shape[0]} != prompt length {length}"
            )

    writer = GGUFWriter(str(destination), arch="s2s_tts_prompt")
    writer.add_uint32("s2s_tts_prompt.length", length)
    writer.add_uint32("s2s_tts_prompt.num_quantizers", quantizers)
    if audio_prompt_latent is not None:
        writer.add_uint32("s2s_tts_prompt.hidden_size", audio_prompt_latent.shape[1])
    writer.add_tensor("tts_prompt.code", np.ascontiguousarray(code, dtype=np.int32))
    writer.add_tensor("tts_prompt.subword_ids", np.ascontiguousarray(subword_ids, dtype=np.int32))
    writer.add_tensor(
        "tts_prompt.subword_mask", np.ascontiguousarray(subword_mask, dtype=np.float32)
    )
    writer.add_tensor(
        "tts_prompt.non_prompt_mask", np.ascontiguousarray(non_prompt_mask, dtype=np.float32)
    )
    if audio_prompt_latent is not None:
        writer.add_tensor(
            "tts_prompt.audio_prompt_latent",
            np.ascontiguousarray(audio_prompt_latent, dtype=np.float32),
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    hidden = f" H={audio_prompt_latent.shape[1]}" if audio_prompt_latent is not None else ""
    print(f"wrote {destination}: T={length} Q={quantizers}{hidden}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="VoiceChat repository or legacy prompt file")
    parser.add_argument("destination", type=Path, help="output GGUF path")
    parser.add_argument("--tokenizer-json", type=Path, help="Nano tokenizer.json")
    args = parser.parse_args()
    convert(args.source, args.destination, args.tokenizer_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
