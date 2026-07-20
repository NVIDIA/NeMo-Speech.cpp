#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokenize text for the GGML MagpieTTS example.

The C++ GGML binary intentionally accepts token IDs instead of embedding NeMo's
language-specific tokenizers. This helper uses NeMo's tokenizer config from the
Magpie .nemo archive or an extracted archive directory and prints one tokenized
chunk per line.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import yaml
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import setup_tokenizers
from nemo.collections.tts.parts.utils.tts_dataset_utils import chunk_and_tokenize_text_by_sentence
from omegaconf import OmegaConf

MIN_TOKENS_EXCLUDING_FIRST_AND_EOS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", type=Path, help="Magpie .nemo archive or extracted archive directory"
    )
    parser.add_argument("text", help="Text to tokenize")
    parser.add_argument(
        "--language", default="en", help="Language code, e.g. en/de/es/fr/it/vi/zh/hi/ja"
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output file")
    parser.add_argument(
        "--json", action="store_true", help="Write JSON instead of plain token lines"
    )
    return parser.parse_args()


def extract_meta(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path.is_dir():
        return path, None

    tmp = tempfile.TemporaryDirectory(prefix="magpietts-tokenizer-")
    root = Path(tmp.name)
    with tarfile.open(path, "r:*") as tar:
        members = [
            m
            for m in tar.getmembers()
            if m.name.endswith("model_config.yaml")
            or m.name.endswith(".dict")
            or "heteronym" in m.name
            or m.name.endswith(".txt")
        ]
        tar.extractall(root, members=members)
    return root, tmp


def read_config(root: Path) -> dict[str, Any]:
    cfg_path = root / "model_config.yaml"
    if not cfg_path.exists():
        cfg_path = root / "." / "model_config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_nemo_paths(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {k: resolve_nemo_paths(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_nemo_paths(v, root) for v in value]
    if isinstance(value, str) and value.startswith("nemo:"):
        rel = value.split(":", 1)[1]
        for candidate in (root / rel, root / "." / rel):
            if candidate.exists():
                return str(candidate)
    return value


def get_tokenizer_for_language(language: str, available_tokenizers: list[str]) -> str:
    language_tokenizer_map = {
        "en": ["english_phoneme", "english"],
        "de": ["german_phoneme", "german"],
        "es": ["spanish_phoneme", "spanish"],
        "fr": ["french_chartokenizer", "french"],
        "it": ["italian_phoneme", "italian"],
        "vi": ["vietnamese_phoneme", "vietnamese"],
        "zh": ["mandarin_phoneme", "mandarin", "chinese"],
        "hi": ["hindi_chartokenizer", "hindi"],
        "ja": ["japanese_phoneme", "japanese"],
    }

    for candidate in language_tokenizer_map.get(language, []):
        if candidate in available_tokenizers:
            return candidate

    if available_tokenizers:
        return available_tokenizers[0]

    raise ValueError("No text tokenizers are available in the model config")


def pad_short_tokens_before_eos(tokens: list[int], eos_id: int, pad_id: int) -> list[int]:
    if len(tokens) < 2 or tokens[-1] != eos_id:
        return tokens

    tokens_excluding_first_and_eos = len(tokens) - 2
    if tokens_excluding_first_and_eos >= MIN_TOKENS_EXCLUDING_FIRST_AND_EOS:
        return tokens

    pad_count = MIN_TOKENS_EXCLUDING_FIRST_AND_EOS - tokens_excluding_first_and_eos
    return tokens[:-1] + ([pad_id] * pad_count) + tokens[-1:]


def main() -> None:
    args = parse_args()
    root, tmp = extract_meta(args.model)
    try:
        cfg = resolve_nemo_paths(read_config(root), root)
        tokenizer_cfg = OmegaConf.create(cfg["text_tokenizers"])
        tokenizer = setup_tokenizers(tokenizer_cfg, mode="test")
        tokenizer_name = get_tokenizer_for_language(
            args.language, list(tokenizer.tokenizers.keys())
        )
        eos_id = len(tokenizer.tokens) + 1
        pad_id = tokenizer.tokenizer_pad_ids[tokenizer_name]

        chunks, lens, texts = chunk_and_tokenize_text_by_sentence(
            text=args.text,
            tokenizer_name=tokenizer_name,
            text_tokenizer=tokenizer,
            eos_token_id=eos_id,
        )
        rows = [
            pad_short_tokens_before_eos(chunk[:length].cpu().tolist(), eos_id, pad_id)
            for chunk, length in zip(chunks, lens, strict=True)
        ]

        if args.json:
            rendered = (
                json.dumps(
                    {
                        "language": args.language,
                        "tokenizer": tokenizer_name,
                        "eos_id": eos_id,
                        "chunks": [
                            {"text": text, "tokens": tokens, "length": len(tokens)}
                            for text, tokens in zip(texts, rows)
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
        else:
            rendered = "\n".join(",".join(str(x) for x in row) for row in rows) + "\n"

        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
