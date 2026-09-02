#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert supported speech and translation checkpoints to runtime GGUF files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from conversion import ARCHITECTURES, ConversionRequest, convert_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local checkpoint/directory or Hugging Face repository ID")
    parser.add_argument(
        "--outfile",
        "-o",
        type=Path,
        required=True,
        help="output GGUF path, or output artifact directory for S2S",
    )
    parser.add_argument(
        "--architecture",
        choices=("auto", *ARCHITECTURES),
        default="auto",
        help="override checkpoint architecture detection",
    )
    parser.add_argument(
        "--outtype",
        default="auto",
        help="output weight type; supported values depend on the architecture",
    )
    parser.add_argument("--revision", help="Hugging Face branch, tag, or commit")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub",
        help="Hugging Face download cache",
    )
    parser.add_argument(
        "--head-type", choices=("ctc", "rnnt", "tdt"), help="override ASR head detection"
    )
    parser.add_argument(
        "--q8-layout",
        choices=("block", "planar"),
        default="block",
        help="ASR Q8 storage layout",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=128, help="PnC inference window length"
    )
    parser.add_argument("--metadata-json", type=Path, help="write converter metadata JSON")
    parser.add_argument(
        "--local-transformer-outtype",
        choices=("f16", "f32"),
        help="override MagpieTTS local-transformer and audio-embedding storage type",
    )
    parser.add_argument("--silero-version", default="6.2.0", help="Silero VAD package version")
    parser.add_argument(
        "--from-whisper-ggml", type=Path, help="convert a whisper.cpp Silero VAD file"
    )
    parser.add_argument(
        "--llama-cpp",
        type=Path,
        help="llama.cpp checkout used for S2S and NMT conversion (default: ./llama.cpp)",
    )
    parser.add_argument(
        "--llama-quantize",
        type=Path,
        help="advanced: use this llama-quantize binary instead of automatic provisioning",
    )
    parser.add_argument(
        "--no-build-quantizer",
        action="store_true",
        help="disable the automatic S2S llama-quantize build",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing S2S bundle artifacts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the S2S conversion plan without writing artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = ConversionRequest(
        source=args.source,
        outfile=args.outfile,
        architecture=args.architecture,
        outtype=args.outtype,
        revision=args.revision,
        cache_dir=args.cache_dir,
        head_type=args.head_type,
        q8_layout=args.q8_layout,
        max_seq_length=args.max_seq_length,
        metadata_json=args.metadata_json,
        local_transformer_outtype=args.local_transformer_outtype,
        silero_version=args.silero_version,
        from_whisper_ggml=args.from_whisper_ggml,
        llama_cpp=args.llama_cpp,
        llama_quantize=args.llama_quantize,
        auto_build_quantizer=not args.no_build_quantizer,
        force=args.force,
        dry_run=args.dry_run,
    )
    try:
        architecture = convert_model(request)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"convert_model.py: error: {error}", file=sys.stderr)
        return 1
    if not args.dry_run:
        print(f"converted {architecture} model to {args.outfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
