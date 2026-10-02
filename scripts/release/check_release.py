#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check release archives before they are published.

  check_release.py isa PATH...            Reject AVX-512/AMX code in x86_64 binaries under PATH.
  check_release.py archives DIR --version VERSION
                                          Check every archive in DIR: checksum, name, layout,
                                          required files, and the x86_64 instruction baseline.

x86_64 archives target x86-64-v3 (AVX2, FMA, F16C, BMI2). ELF, PE, and Mach-O files are
disassembled with llvm-objdump when available, otherwise with objdump.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ARCHIVE = re.compile(
    r"^nemo-speech-(?P<version>nightly|[0-9][0-9A-Za-z.+-]*)-(?P<os>linux|macos|windows)-"
    r"(?P<arch>x86_64|aarch64)-(?P<backend>cpu|vulkan|metal|cuda|cuda12|cuda13)"
    r"\.(?P<ext>tar\.gz|zip)$"
)

# x86-64-v3 has no EVEX (AVX-512) instructions, and in 64-bit code a leading 0x62 opcode byte
# after legacy prefixes is always EVEX, so the encoding catches every AVX-512 instruction
# whatever its registers. The mnemonics below add the VEX-encoded extensions beyond v3
# (AVX-VNNI, AMX) and AVX-512 forms reported by name. Both disassemblers put a tab before the
# mnemonic; GNU objdump continues a long instruction's bytes on lines without one, which are
# not instructions.
INSTRUCTION = re.compile(r"^\s*[0-9a-f]+:\s+((?:[0-9a-f]{2} )+)\s*\t(\S.*)$")
LEGACY_PREFIXES = {"26", "2e", "36", "3e", "64", "65", "66", "67", "f0", "f2", "f3"}

# AT&T syntax, as printed by objdump and llvm-objdump.
BEYOND_X86_64_V3 = re.compile(
    r"%zmm\d+|%[xy]mm(?:1[6-9]|2\d|3[01])\b|%k[0-7]\b|\{%k[0-7]\}"
    r"|\b(?:vpternlog[dq]|vmovdqu(?:8|16|32|64)|vmovdqa(?:32|64)|vperm[it]2\w*|vpcompress\w*"
    r"|vpexpand\w*|vfixupimm\w*|vgetexp\w*|vgetmant\w*|vrndscale\w*|vreduce\w*|vscalef\w*"
    r"|vpmovm2\w*|vpbroadcastm\w*|vpdpbusds?|vpdpwssds?|vdpbf16ps|vcvtne2ps2bf16"
    r"|kmov[bwdq]|kand\w*|kor\w*|kxor\w*|knot\w*|ktest\w*|kshift\w*|kunpck\w*"
    r"|tdp\w+|tileload\w*|tilestored|tilezero|ldtilecfg|sttilecfg|tilerelease)\b"
)


def binary_arch(path: pathlib.Path) -> str | None:
    """Return 'x86_64', 'aarch64', 'other', or None for files that are not executables."""
    try:
        with path.open("rb") as f:
            head = f.read(64)
            if head[:4] == b"\x7fELF":
                machine = struct.unpack_from("<H", head, 18)[0]
                return {62: "x86_64", 183: "aarch64"}.get(machine, "other")
            if head[:2] == b"MZ" and len(head) >= 64:
                f.seek(struct.unpack_from("<I", head, 60)[0])
                pe = f.read(6)
                if pe[:4] != b"PE\0\0":
                    return None
                machine = struct.unpack_from("<H", pe, 4)[0]
                return {0x8664: "x86_64", 0xAA64: "aarch64"}.get(machine, "other")
            if head[:4] == b"\xcf\xfa\xed\xfe":
                cputype = struct.unpack_from("<I", head, 4)[0]
                return {0x01000007: "x86_64", 0x0100000C: "aarch64"}.get(cputype, "other")
            if head[:4] == b"\xca\xfe\xba\xbe":
                return "other"  # universal binaries are not produced
    except OSError:
        return None
    return None


def disassembler() -> str:
    for tool in ("llvm-objdump", "objdump"):
        if shutil.which(tool):
            return tool
    raise SystemExit("error: llvm-objdump or objdump is required")


def beyond_v3(disassembly: str) -> list[str]:
    """Return the instructions beyond x86-64-v3 in llvm-objdump or objdump output."""
    lines = [m for m in map(INSTRUCTION.match, disassembly.splitlines()) if m]
    # Bytes the disassembler cannot decode are data in a code section, such as a jump table:
    # llvm-objdump prints <unknown>, GNU objdump (bad). Data next to them can decode as a
    # valid-looking instruction, so ignore hits adjacent to undecodable bytes; real AVX-512
    # code never appears only there.
    data = [m.group(2).startswith(("(bad)", "<unknown>")) for m in lines]
    hits = []
    for i, m in enumerate(lines):
        if data[i] or (i > 0 and data[i - 1]) or (i + 1 < len(lines) and data[i + 1]):
            continue
        opcode = next((b for b in m.group(1).split() if b not in LEGACY_PREFIXES), "")
        if opcode == "62" or BEYOND_X86_64_V3.search(m.group(2)):
            hits.append(m.group(0).strip())
    return hits


def scan_isa(paths: list[pathlib.Path]) -> list[str]:
    tool = disassembler()
    failures = []
    for root in paths:
        files = (
            [root]
            if root.is_file()
            else sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink())
        )
        for path in files:
            arch = binary_arch(path)
            if arch == "other":
                failures.append(f"{path}: unsupported binary format or architecture")
            if arch != "x86_64":
                continue
            result = subprocess.run(
                [tool, "-d", str(path)],
                capture_output=True,
                text=True,
                errors="replace",
            )
            if result.returncode != 0 or "file format not recognized" in result.stderr:
                failures.append(f"{path}: {tool} could not disassemble it")
                continue
            hits = beyond_v3(result.stdout)
            if hits:
                failures.append(
                    f"{path}: {len(hits)} instructions beyond x86-64-v3, first: {hits[0]}"
                )
    return failures


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
            if match["arch"] == "x86_64":
                failures += [f"{archive.name}: {f}" for f in scan_isa([root])]
        print(f"checked {archive.name}", flush=True)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    isa = sub.add_parser("isa")
    isa.add_argument("paths", nargs="+", type=pathlib.Path)
    archives = sub.add_parser("archives")
    archives.add_argument("directory", type=pathlib.Path)
    archives.add_argument("--version", required=True)
    args = parser.parse_args()

    if args.command == "isa":
        failures = scan_isa(args.paths)
    else:
        failures = check_archives(args.directory, args.version)
    for failure in failures:
        print(f"error: {failure}", file=sys.stderr)
    if failures:
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
