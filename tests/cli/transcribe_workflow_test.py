#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import pathlib
import re
import shutil
import subprocess
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    args = parse_args()
    source = pathlib.Path(args.audio)
    with tempfile.TemporaryDirectory(prefix="nemo-speech-batch-cli-") as temporary:
        root = pathlib.Path(temporary)
        inputs = root / "inputs"
        outputs = root / "outputs"
        (inputs / "nested").mkdir(parents=True)
        shutil.copy2(source, inputs / "first.wav")
        shutil.copy2(source, inputs / "nested" / "second.wav")
        shutil.copy2(source, inputs / "nested" / "third.wav")

        existing = outputs / "nested" / "second.srt"
        existing.parent.mkdir(parents=True)
        existing.write_text("do not replace\n", encoding="utf-8")
        result = subprocess.run(
            [
                args.binary,
                "transcribe",
                str(inputs),
                "--model",
                args.model,
                "--recursive",
                "--output-dir",
                str(outputs),
                "--concurrency",
                "2",
                "--format",
                "srt",
                "--no-warmup",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
        )
        require(result.returncode == 1, "one existing output must produce a partial failure")
        require(result.stdout == "", "directory results must be written to the output directory")
        require("3 files (1 failed)" in result.stderr, "deterministic directory summary")
        require(existing.read_text(encoding="utf-8") == "do not replace\n", "existing output")
        for path in (outputs / "first.srt", outputs / "nested" / "third.srt"):
            text = path.read_text(encoding="utf-8")
            require(text.startswith("1\n"), f"SRT sequence number: {path}")
            require(
                re.search(r"\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}", text) is not None,
                f"SRT timestamp: {path}",
            )
            require(len(text.splitlines()) >= 4, f"SRT transcript: {path}")


if __name__ == "__main__":
    main()
