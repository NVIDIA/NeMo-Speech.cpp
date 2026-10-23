#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check release archives before they are published.

  check_release.py archives DIR --version VERSION
                                          Check every archive in DIR: checksum, name, layout,
                                          and required files.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys
import tarfile
import tempfile
import zipfile

ARCHIVE = re.compile(
    r"^nemo-speech-(?P<version>nightly|[0-9][0-9A-Za-z.+-]*)-(?P<os>linux|macos|windows)-"
    r"(?P<arch>x86_64|aarch64)-(?P<backend>cpu|vulkan|metal|cuda|cuda12|cuda13)"
    r"\.(?P<ext>tar\.gz|zip)$"
)


def extract(archive: pathlib.Path, dest: pathlib.Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        if not hasattr(tarfile, "data_filter"):
            raise SystemExit(
                "error: safe tar extraction needs Python 3.12 or a release with tarfile.data_filter"
            )
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")


def check_archives(directory: pathlib.Path, version: str) -> list[str]:
    failures = []
    archives = sorted(p for p in directory.iterdir() if p.name.endswith((".tar.gz", ".zip")))
    if not archives:
        return [f"{directory}: no archives"]
    for archive in archives:
        match = ARCHIVE.match(archive.name)
        if not match:
            failures.append(f"{archive.name}: unexpected archive name")
            continue
        if match["version"] != version:
            failures.append(f"{archive.name}: version is not {version}")
        expected_ext = "zip" if match["os"] == "windows" else "tar.gz"
        if match["ext"] != expected_ext:
            failures.append(f"{archive.name}: {match['os']} archives must be .{expected_ext}")
        checksum = archive.with_name(archive.name + ".sha256")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if not checksum.is_file():
            failures.append(f"{archive.name}: missing .sha256")
        elif checksum.read_text().split() != [digest, archive.name]:
            failures.append(f"{archive.name}: .sha256 does not match")

        package = archive.name[: -len("." + match["ext"])]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            extract(archive, tmp_path)
            entries = list(tmp_path.iterdir())
            if [e.name for e in entries] != [package] or not entries[0].is_dir():
                failures.append(f"{archive.name}: must contain exactly one directory, {package}/")
                continue
            root = entries[0]
            exe = "nemo-speech.exe" if match["os"] == "windows" else "nemo-speech"
            for required in (
                f"bin/{exe}",
                "share/licenses/nemo-speech/LICENSE",
                "share/licenses/nemo-speech/THIRD_PARTY_NOTICES.md",
            ):
                if not (root / required).is_file():
                    failures.append(f"{archive.name}: missing {required}")
        print(f"checked {archive.name}", flush=True)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    archives = sub.add_parser("archives")
    archives.add_argument("directory", type=pathlib.Path)
    archives.add_argument("--version", required=True)
    args = parser.parse_args()

    failures = check_archives(args.directory, args.version)
    for failure in failures:
        print(f"error: {failure}", file=sys.stderr)
    if failures:
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
