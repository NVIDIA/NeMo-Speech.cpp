#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check or add the project license header to tracked first-party source files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

COPYRIGHT = (
    "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved."
)
LICENSE = "SPDX-License-Identifier: Apache-2.0"
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".map",
    ".proto",
    ".py",
    ".sh",
    ".ps1",
    ".cmake",
}
DERIVED_PREFIX = "riva/runtime/ggml/"
DERIVED_FILES = {
    "riva/asr/encoder/fastconformer.cpp",
    "riva/asr/encoder/fastconformer.h",
    "riva/asr/encoder/rel_pos_attention.cpp",
    "riva/asr/encoder/rel_pos_attention.h",
}


def repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return Path(result.stdout.strip())


def tracked_source_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    paths: list[Path] = []
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0]
        if mode == b"160000":
            continue
        relative = Path(raw_path.decode("utf-8"))
        if is_source_file(relative):
            paths.append(relative)
    return sorted(paths)


def is_source_file(path: Path) -> bool:
    name = path.name
    return (
        path.suffix.lower() in SOURCE_SUFFIXES
        or name == "CMakeLists.txt"
        or name == "Dockerfile"
        or name.startswith("Dockerfile.")
    )


def comment_prefix(path: Path) -> str:
    if path.suffix.lower() == ".map":
        return "/*"
    if path.suffix.lower() in {".py", ".sh", ".ps1", ".cmake"}:
        return "#"
    if path.name == "CMakeLists.txt" or path.name.startswith("Dockerfile"):
        return "#"
    return "//"


def expected_header(path: Path) -> str:
    prefix = comment_prefix(path)
    if prefix == "/*":
        lines = [f"/* {COPYRIGHT}", f" * {LICENSE}", " */"]
    else:
        lines = [f"{prefix} {COPYRIGHT}", f"{prefix} {LICENSE}"]

    path_string = path.as_posix()
    if path_string.startswith(DERIVED_PREFIX) or path_string in DERIVED_FILES:
        if prefix == "/*":
            lines.extend(
                [
                    "/* Portions derived from parakeet.cpp:",
                    " * Copyright (c) 2025 Jason Ni",
                    " * Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.",
                    " */",
                ]
            )
        else:
            lines.extend(
                [
                    f"{prefix}",
                    f"{prefix} Portions derived from parakeet.cpp:",
                    f"{prefix} Copyright (c) 2025 Jason Ni",
                    f"{prefix} Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.",
                ]
            )
    return "\n".join(lines) + "\n"


def has_header(path: Path, text: str) -> bool:
    first_lines = "\n".join(text.splitlines()[:12])
    if COPYRIGHT not in first_lines or LICENSE not in first_lines:
        return False
    path_string = path.as_posix()
    if path_string.startswith(DERIVED_PREFIX) or path_string in DERIVED_FILES:
        return (
            "Copyright (c) 2025 Jason Ni" in first_lines
            and "Licensed under the MIT License" in first_lines
        )
    return True


def add_header(path: Path, text: str) -> str:
    lines = text.splitlines(keepends=True)
    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    header = expected_header(path)

    # Replace an existing NVIDIA SPDX block instead of duplicating it.
    probe_end = min(len(lines), insert_at + 10)
    existing = "".join(lines[insert_at:probe_end])
    if "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION" in existing:
        while insert_at < len(lines) and (
            "SPDX-FileCopyrightText:" in lines[insert_at]
            or "SPDX-License-Identifier:" in lines[insert_at]
        ):
            del lines[insert_at]

    if insert_at and lines[0] and not lines[0].endswith("\n"):
        lines[0] += "\n"
    lines.insert(insert_at, header)
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix", action="store_true", help="insert or replace missing project headers"
    )
    args = parser.parse_args()

    root = repository_root()
    failures: list[Path] = []
    for relative in tracked_source_files(root):
        path = root / relative
        text = path.read_text(encoding="utf-8")
        if has_header(relative, text):
            continue
        failures.append(relative)
        if args.fix:
            path.write_text(add_header(relative, text), encoding="utf-8")

    if failures:
        action = "updated" if args.fix else "missing license header"
        for path in failures:
            print(f"{action}: {path}")
        return 0 if args.fix else 1

    print("All tracked first-party source files have the required license header.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
