#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import http.server
import io
import json
import os
import platform
import subprocess
import tarfile
import tempfile
import threading
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def archive(version: str, arch: str) -> tuple[str, bytes, str]:
    os_name = "macos" if platform.system() == "Darwin" else "linux"
    name = f"nemo-speech-{version}-{os_name}-{arch}-cpu.tar.gz"
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        files = {
            "nemo-speech/bin/nemo-speech": (
                f"#!/bin/sh\necho 'nemo-speech {version}'\n".encode(),
                0o755,
            ),
            "nemo-speech/share/licenses/nemo-speech/LICENSE": (
                b"test license\n",
                0o644,
            ),
        }
        for path, (contents, mode) in files.items():
            info = tarfile.TarInfo(path)
            info.size = len(contents)
            info.mode = mode
            bundle.addfile(info, io.BytesIO(contents))
    contents = output.getvalue()
    return name, contents, hashlib.sha256(contents).hexdigest()


def source_repository(root: Path, version: str) -> Path:
    repository = root / "source-repository"
    (repository / "scripts").mkdir(parents=True)
    (repository / "LICENSE").write_text("test license\n", encoding="utf-8")
    executable = repository / "nemo-speech"
    executable.write_text(f"#!/bin/sh\necho 'nemo-speech {version} (source)'\n", encoding="utf-8")
    executable.chmod(0o755)
    (repository / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.26)
project(installer_source_fixture NONE)
install(PROGRAMS nemo-speech DESTINATION bin)
install(FILES LICENSE DESTINATION share/licenses/nemo-speech)
""",
        encoding="utf-8",
    )
    presets = {
        "version": 6,
        "configurePresets": [
            {
                "name": "cpu-server",
                "generator": "Ninja",
                "binaryDir": "${sourceDir}/build/cpu-server",
            }
        ],
        "buildPresets": [{"name": "cpu-server", "configurePreset": "cpu-server"}],
    }
    (repository / "CMakePresets.json").write_text(json.dumps(presets), encoding="utf-8")
    configure = repository / "scripts" / "configure.sh"
    configure.write_text('#!/bin/sh\nset -eu\ncmake --preset "$1"\n', encoding="utf-8")
    configure.chmod(0o755)
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
    source_root = Path(__file__).resolve().parents[2]
    installer = source_root / "scripts" / "install.sh"
    os_name = "macos" if platform.system() == "Darwin" else "linux"
    arch = "aarch64" if platform.machine().lower() in {"aarch64", "arm64"} else "x86_64"
    releases = {}
    for version in ("1.2.3", "1.2.4", "1.2.5", "nightly"):
        name, contents, checksum = archive(version, arch)
        tag = "nightly" if version == "nightly" else f"v{version}"
        releases[f"/releases/download/{tag}/{name}"] = contents
        releases[f"/releases/download/{tag}/{name}.sha256"] = (
            ("0" * 64 if version == "1.2.5" else checksum) + f"  {name}\n"
        ).encode()
    requests: dict[str, int] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            requests[self.path] = requests.get(self.path, 0) + 1
            if self.path == "/releases/latest":
                self.send_response(302)
                self.send_header("Location", "/releases/tag/v1.2.3")
                self.end_headers()
                return
            if self.path == "/releases/tag/v1.2.3":
                body = b"latest\n"
            else:
                body = releases.get(self.path)
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
        with tempfile.TemporaryDirectory(prefix="nemo-speech-installer-") as temporary:
            root = Path(temporary)
            home = root / "home"
            home.mkdir()
            prefix = root / "install"
            source = source_repository(root, "1.2.6")
            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "SHELL": "/bin/bash",
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "NEMO_SPEECH_RELEASE_BASE_URL": (
                        f"http://127.0.0.1:{server.server_port}/releases"
                    ),
                    "NEMO_SPEECH_SOURCE_URL": str(source),
                }
            )

            def run(*arguments: str, ok: bool = True) -> subprocess.CompletedProcess[str]:
                result = subprocess.run(
                    ["sh", str(installer), *arguments],
                    env=env,
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

            run("--prefix", str(prefix), "--backend", "cpu")
            binary = prefix / "bin" / "nemo-speech"
            require(binary.exists(), "latest release was not installed")
            require("1.2.3" in subprocess.check_output([binary, "--version"], text=True), "version")
            require((home / ".local/bin/nemo-speech").is_symlink(), "CLI link")
            require("NeMo-Speech.cpp" in (home / ".bashrc").read_text(), "PATH update")
            require(
                (prefix / "share/licenses/nemo-speech/LICENSE").exists(),
                "license files were not preserved",
            )

            archive_path = next(
                path for path in releases if "/v1.2.3/" in path and not path.endswith(".sha256")
            )
            before = requests.get(archive_path, 0)
            result = run("--prefix", str(prefix), "--version", "1.2.3", "--backend", "cpu")
            require("Already installed" in result.stdout, "idempotent install was not detected")
            require(requests.get(archive_path, 0) == before, "idempotent install redownloaded")

            rc_before = (home / ".bashrc").read_text()
            run(
                "--prefix",
                str(prefix),
                "--version",
                "1.2.4",
                "--backend",
                "cpu",
                "--no-modify-path",
            )
            require("1.2.4" in subprocess.check_output([binary, "--version"], text=True), "upgrade")
            require(
                (home / ".bashrc").read_text() == rc_before,
                "--no-modify-path changed shell configuration",
            )

            nightly_prefix = root / "nightly"
            run(
                "--prefix",
                str(nightly_prefix),
                "--channel",
                "nightly",
                "--backend",
                "cpu",
                "--no-modify-path",
            )
            require(
                "nightly"
                in subprocess.check_output(
                    [nightly_prefix / "bin" / "nemo-speech", "--version"], text=True
                ),
                "nightly channel",
            )

            failed = run(
                "--prefix",
                str(prefix),
                "--version",
                "1.2.5",
                "--backend",
                "cpu",
                "--no-modify-path",
                ok=False,
            )
            require("SHA-256 verification failed" in failed.stdout, "checksum failure message")
            require(
                "1.2.4" in subprocess.check_output([binary, "--version"], text=True),
                "checksum failure damaged the existing installation",
            )

            source_prefix = root / "source-install"
            fallback = run(
                "--prefix",
                str(source_prefix),
                "--version",
                "1.2.6",
                "--backend",
                "cpu",
                "--no-modify-path",
            )
            source_binary = source_prefix / "bin" / "nemo-speech"
            require("building from source" in fallback.stdout.lower(), "source fallback message")
            require(source_binary.is_file(), "source fallback did not install the CLI")
            require(
                "1.2.6" in subprocess.check_output([source_binary, "--version"], text=True),
                "source fallback version",
            )
            require(
                (source_prefix / ".nemo-speech-install").read_text().strip()
                == f"1.2.6 {os_name} {arch} cpu source:v1.2.6 profile:speech-server",
                "source installation metadata",
            )
            repeated = run(
                "--prefix",
                str(source_prefix),
                "--version",
                "1.2.6",
                "--backend",
                "cpu",
                "--no-modify-path",
            )
            require(
                "Already installed from source" in repeated.stdout,
                "source install was rebuilt unnecessarily",
            )

            name, contents, checksum = archive("1.2.6", arch)
            releases[f"/releases/download/v1.2.6/{name}"] = contents
            releases[f"/releases/download/v1.2.6/{name}.sha256"] = f"{checksum}  {name}\n".encode()
            run(
                "--prefix",
                str(source_prefix),
                "--version",
                "1.2.6",
                "--backend",
                "cpu",
                "--no-modify-path",
            )
            require(
                (source_prefix / ".nemo-speech-install").read_text().strip()
                == f"1.2.6 {os_name} {arch} cpu",
                "published binary did not replace the source installation",
            )
            require(
                "(source)" not in subprocess.check_output([source_binary, "--version"], text=True),
                "source binary remained after the release was published",
            )

            binary_only_prefix = root / "binary-only"
            binary_only = run(
                "--prefix",
                str(binary_only_prefix),
                "--version",
                "1.2.7",
                "--backend",
                "cpu",
                "--binary-only",
                "--no-modify-path",
                ok=False,
            )
            require("unavailable" in binary_only.stdout.lower(), "binary-only failure message")
            require(not binary_only_prefix.exists(), "binary-only failure changed the prefix")

            dry_prefix = root / "dry-run"
            run(
                "--prefix",
                str(dry_prefix),
                "--version",
                "9.9.9",
                "--backend",
                "cpu",
                "--dry-run",
            )
            require(not dry_prefix.exists(), "dry run changed the filesystem")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
