#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise install.ps1 against local, checksummed release fixtures on Windows."""

from __future__ import annotations

import hashlib
import http.server
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def archive(version: str, arch: str, binary: Path) -> tuple[str, bytes, str]:
    name = f"nemo-speech-{version}-windows-{arch}-cpu.zip"
    with tempfile.TemporaryDirectory(prefix="nemo-speech-windows-archive-") as temporary:
        root = Path(temporary) / "nemo-speech"
        (root / "bin").mkdir(parents=True)
        shutil.copy2(binary, root / "bin" / "nemo-speech.exe")
        for library in binary.parent.glob("*.dll"):
            shutil.copy2(library, root / "bin" / library.name)
        license_path = root / "share" / "licenses" / "nemo-speech" / "LICENSE"
        license_path.parent.mkdir(parents=True)
        license_path.write_text("test license\n", encoding="utf-8")
        output = Path(temporary) / name
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in root.rglob("*"):
                if path.is_file():
                    bundle.write(path, path.relative_to(root.parent))
        contents = output.read_bytes()
    return name, contents, hashlib.sha256(contents).hexdigest()


def source_repository(root: Path, version: str, binary: Path) -> Path:
    repository = root / "source-repository"
    scripts = repository / "scripts" / "windows"
    scripts.mkdir(parents=True)
    shutil.copy2(binary, repository / "nemo-speech.exe")
    (repository / "LICENSE").write_text("test license\n", encoding="utf-8")
    (repository / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.26)
project(installer_source_fixture NONE)
install(PROGRAMS nemo-speech.exe DESTINATION bin)
install(FILES LICENSE DESTINATION share/licenses/nemo-speech)
""",
        encoding="utf-8",
    )
    (scripts / "build.ps1").write_text(
        """[CmdletBinding()]
param(
    [string]$Backend,
    [string]$Profile,
    [string]$BuildDir,
    [string]$Config,
    [string]$CudaArch,
    [string]$VcpkgRoot,
    [string]$VcpkgTriplet,
    [switch]$Grpc,
    [switch]$Nmt,
    [switch]$Flashlight,
    [switch]$TtsJa,
    [switch]$TtsZh,
    [switch]$AsrOnly,
    [switch]$Http,
    [switch]$HttpTls
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
& cmake -S $root -B $BuildDir -G Ninja
if ($LASTEXITCODE -ne 0) { throw 'fixture configure failed' }
& cmake --build $BuildDir
if ($LASTEXITCODE -ne 0) { throw 'fixture build failed' }
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "installer@test.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Installer Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-m", "source fixture"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(["git", "-C", str(repository), "branch", "-M", "main"], check=True)
    subprocess.run(["git", "-C", str(repository), "tag", f"v{version}"], check=True)
    return repository


def main() -> None:
    if os.name != "nt":
        raise RuntimeError("install_ps1_test.py is a Windows-only test")
    powershell = sys.argv[1]
    binary = Path(sys.argv[2]).resolve()
    source_root = Path(__file__).resolve().parents[2]
    installer = source_root / "scripts" / "install.ps1"
    arch = "aarch64" if platform.machine().lower() in {"aarch64", "arm64"} else "x86_64"
    releases: dict[str, bytes] = {}
    for version in ("1.2.3", "1.2.4", "1.2.5", "nightly"):
        name, contents, checksum = archive(version, arch, binary)
        tag = "nightly" if version == "nightly" else f"v{version}"
        releases[f"/releases/download/{tag}/{name}"] = contents
        releases[f"/releases/download/{tag}/{name}.sha256"] = (
            ("0" * 64 if version == "1.2.5" else checksum) + f"  {name}\n"
        ).encode()
    requests: dict[str, int] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            requests[self.path] = requests.get(self.path, 0) + 1
            body = (
                b"NEMO_SPEECH_VERSION: 1.2.3\n"
                if self.path == "/VERSION"
                else releases.get(self.path)
            )
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="nemo-speech-ps-installer-") as temporary:
            root = Path(temporary)
            local_app_data = root / "local-app-data"
            local_app_data.mkdir()
            prefix = root / "install"
            source = source_repository(root, "1.2.6", binary)
            environment = os.environ.copy()
            environment["LOCALAPPDATA"] = str(local_app_data)
            environment["NEMO_SPEECH_RELEASE_BASE_URL"] = (
                f"http://127.0.0.1:{server.server_port}/releases"
            )
            environment["NEMO_SPEECH_SOURCE_URL"] = str(source)
            environment["NEMO_SPEECH_VERSION_URL"] = (
                f"http://127.0.0.1:{server.server_port}/VERSION"
            )

            def run(*arguments: str, ok: bool = True) -> subprocess.CompletedProcess[str]:
                result = subprocess.run(
                    [
                        powershell,
                        "-NoLogo",
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(installer),
                        *arguments,
                    ],
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                if ok and result.returncode != 0:
                    raise RuntimeError(f"installer failed ({result.returncode}):\n{result.stdout}")
                if not ok and result.returncode == 0:
                    raise RuntimeError(f"installer unexpectedly succeeded:\n{result.stdout}")
                return result

            remote_source = "https://github.com/NVIDIA/NeMo-Speech.cpp.git"
            environment["NEMO_SPEECH_SOURCE_URL"] = remote_source
            environment.pop("NEMO_SPEECH_SOURCE_REF", None)
            remote_plan = run("-Source", "-Backend", "cpu", "-DryRun")
            require(f"{remote_source}#main" in remote_plan.stdout, "remote source ref")
            environment["NEMO_SPEECH_SOURCE_REF"] = "review-test"
            override_plan = run("-Source", "-Backend", "cpu", "-DryRun")
            require(f"{remote_source}#review-test" in override_plan.stdout, "source ref override")
            environment["NEMO_SPEECH_SOURCE_URL"] = str(source)
            environment.pop("NEMO_SPEECH_SOURCE_REF")

            run("-Prefix", str(prefix), "-Backend", "cpu", "-NoModifyPath")
            metadata = prefix / ".nemo-speech-install"
            require(
                metadata.read_text(encoding="utf-8") == "1.2.3 windows " + arch + " cpu", "latest"
            )
            require((prefix / "bin" / "nemo-speech.exe").is_file(), "binary")
            require(
                (prefix / "share" / "licenses" / "nemo-speech" / "LICENSE").is_file(),
                "licenses",
            )

            archive_path = next(
                path for path in releases if "/v1.2.3/" in path and not path.endswith(".sha256")
            )
            before = requests.get(archive_path, 0)
            result = run(
                "-Prefix", str(prefix), "-Version", "1.2.3", "-Backend", "cpu", "-NoModifyPath"
            )
            require("Already installed" in result.stdout, "idempotence")
            require(requests.get(archive_path, 0) == before, "idempotent install redownloaded")

            run("-Prefix", str(prefix), "-Version", "1.2.4", "-Backend", "cpu", "-NoModifyPath")
            require(metadata.read_text(encoding="utf-8").startswith("1.2.4 "), "upgrade")
            failed = run(
                "-Prefix",
                str(prefix),
                "-Version",
                "1.2.5",
                "-Backend",
                "cpu",
                "-NoModifyPath",
                ok=False,
            )
            require("SHA-256 verification failed" in failed.stdout, "checksum failure")
            require(metadata.read_text(encoding="utf-8").startswith("1.2.4 "), "rollback")

            source_prefix = root / "source-install"
            fallback = run(
                "-Prefix",
                str(source_prefix),
                "-Version",
                "1.2.6",
                "-Backend",
                "cpu",
                "-NoModifyPath",
            )
            require("building from source" in fallback.stdout.lower(), "source fallback message")
            require((source_prefix / "bin" / "nemo-speech.exe").is_file(), "source binary")
            require(
                (source_prefix / ".nemo-speech-install").read_text(encoding="utf-8")
                == f"1.2.6 windows {arch} cpu source:v1.2.6 profile:server",
                "source metadata",
            )
            repeated = run(
                "-Prefix",
                str(source_prefix),
                "-Version",
                "1.2.6",
                "-Backend",
                "cpu",
                "-NoModifyPath",
            )
            require("Already installed from source" in repeated.stdout, "source idempotence")

            name, contents, checksum = archive("1.2.6", arch, binary)
            releases[f"/releases/download/v1.2.6/{name}"] = contents
            releases[f"/releases/download/v1.2.6/{name}.sha256"] = f"{checksum}  {name}\n".encode()
            run(
                "-Prefix",
                str(source_prefix),
                "-Version",
                "1.2.6",
                "-Backend",
                "cpu",
                "-NoModifyPath",
            )
            require(
                (source_prefix / ".nemo-speech-install").read_text(encoding="utf-8")
                == f"1.2.6 windows {arch} cpu",
                "published binary did not replace source installation",
            )

            binary_only = root / "binary-only"
            failed_binary_only = run(
                "-Prefix",
                str(binary_only),
                "-Version",
                "1.2.7",
                "-Backend",
                "cpu",
                "-BinaryOnly",
                "-NoModifyPath",
                ok=False,
            )
            require("unavailable" in failed_binary_only.stdout.lower(), "binary-only failure")
            require(not binary_only.exists(), "binary-only failure changed the prefix")

            nightly = root / "nightly"
            run("-Prefix", str(nightly), "-Channel", "nightly", "-Backend", "cpu", "-NoModifyPath")
            require(
                (nightly / ".nemo-speech-install").read_text().startswith("nightly "), "nightly"
            )

            dry = root / "dry-run"
            run(
                "-Prefix",
                str(dry),
                "-Version",
                "9.9.9",
                "-Backend",
                "cpu",
                "-NoModifyPath",
                "-DryRun",
            )
            require(not dry.exists(), "dry run changed the filesystem")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
