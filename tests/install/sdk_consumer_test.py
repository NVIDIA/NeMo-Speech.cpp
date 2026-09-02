#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import subprocess
import sys
import tempfile


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def cache_value(cache: pathlib.Path, name: str) -> str:
    prefix = f"{name}:"
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1]
    return ""


def cache_bool(cache: pathlib.Path, name: str) -> bool:
    return cache_value(cache, name).upper() in {"1", "ON", "TRUE", "YES", "Y"}


def main() -> None:
    build = pathlib.Path(sys.argv[1]).resolve()
    source = pathlib.Path(sys.argv[2]).resolve()
    cmake = sys.argv[3]
    multi_config = bool(cache_value(build / "CMakeCache.txt", "CMAKE_CONFIGURATION_TYPES"))
    config = os.environ.get("CTEST_CONFIGURATION_TYPE", "Release") if multi_config else ""

    with tempfile.TemporaryDirectory(prefix="nemo-speech-installed-sdk-") as temporary:
        root = pathlib.Path(temporary)
        install_prefix = root / "install"
        prefix = root / "relocated"
        consumer = root / "consumer"
        install_command = [cmake, "--install", str(build), "--prefix", str(install_prefix)]
        if config:
            install_command.extend(("--config", config))
        run(*install_command)
        install_prefix.rename(prefix)
        cache = build / "CMakeCache.txt"
        expected = {
            "ASR": cache_bool(cache, "NEMO_SPEECH_BUILD_ASR"),
            "Diarization": cache_bool(cache, "NEMO_SPEECH_BUILD_DIAR"),
            "NMT": cache_bool(cache, "NEMO_SPEECH_BUILD_NMT"),
            "TTS": cache_bool(cache, "NEMO_SPEECH_BUILD_TTS"),
        }
        suffix = ".exe" if os.name == "nt" else ""
        configure_command = [
            cmake,
            "-S",
            str(source / "tests/install/consumer"),
            "-B",
            str(consumer),
            f"-DCMAKE_PREFIX_PATH={prefix}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        configure_command.extend(
            f"-DEXPECT_NEMO_SPEECH_{component}={'ON' if enabled else 'OFF'}"
            for component, enabled in expected.items()
        )
        run(*configure_command)
        # The consumer uses CMake's default generator, which can be multi-config
        # (Visual Studio) even when the SDK build tree is single-config Ninja.
        consumer_multi_config = bool(
            cache_value(consumer / "CMakeCache.txt", "CMAKE_CONFIGURATION_TYPES")
        )
        consumer_config = "Release" if consumer_multi_config else ""
        build_command = [cmake, "--build", str(consumer)]
        if consumer_config:
            build_command.extend(("--config", consumer_config))
        run(*build_command)

        consumer_binary = consumer / consumer_config / f"installed_consumer{suffix}"
        consumer_environment = os.environ.copy()
        if os.name == "nt":
            consumer_environment["PATH"] = (
                str(prefix / "bin") + os.pathsep + consumer_environment.get("PATH", "")
            )
        subprocess.run([str(consumer_binary)], check=True, env=consumer_environment)

        include = prefix / "include"
        assert not (include / "ggml.h").exists(), "internal ggml headers leaked into SDK"
        assert not (include / "llama.h").exists(), "internal llama headers leaked into SDK"
        assert not (prefix / "bin" / "convert_hf_to_gguf.py").exists()

        if cache_bool(cache, "GGML_BLAS"):
            if os.name == "nt":
                blas_libraries = list((prefix / "bin").glob("*ggml-blas*.dll"))
            elif sys.platform == "darwin":
                blas_libraries = list((prefix / "lib").glob("libggml-blas*.dylib"))
            else:
                blas_libraries = list((prefix / "lib").glob("libggml-blas.so*"))
            assert blas_libraries, "installed runtime is missing the ggml BLAS backend"


if __name__ == "__main__":
    main()
