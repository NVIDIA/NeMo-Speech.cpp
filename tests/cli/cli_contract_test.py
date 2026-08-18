#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
import socket
import subprocess
import sys
import tempfile
import threading
import time


def run(binary: str, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [binary, *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def expect_json_error(result: subprocess.CompletedProcess, exit_code: int, error_type: str) -> dict:
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert result.stdout == "", f"error polluted stdout: {result.stdout!r}"
    error = json.loads(result.stderr)["error"]
    assert error["exit_code"] == exit_code
    assert error["type"] == error_type
    assert error["message"]
    return error


def main() -> None:
    binary = sys.argv[1]
    expect_live = "--expect-live" in sys.argv[2:]
    with tempfile.TemporaryDirectory(prefix="nemo-speech-cli-contract-") as temporary:
        help_result = run(binary, "--help")
        assert help_result.returncode == 0, help_result.stderr
        assert "Usage:" in help_result.stdout

        model_help = run(binary, "model", "--help")
        assert model_help.returncode == 0, model_help.stderr
        assert "info FILE" in model_help.stdout
        assert "list" in model_help.stdout
        assert "pull REPO" in model_help.stdout

        expect_json_error(run(binary, "--json", "not-a-command"), 2, "invalid_argument")
        expect_json_error(
            run(binary, "model", "info", "not-installed.gguf", "--json"),
            3,
            "missing_model",
        )
        expect_json_error(
            run(binary, "--json", "--quiet", "--verbose", "doctor"),
            2,
            "invalid_argument",
        )

        translation = run(
            binary,
            "translate",
            "--from",
            "en",
            "--to",
            "es",
            "hello",
            "--json",
        )
        if translation.returncode != 0:
            error = json.loads(translation.stderr)["error"]
            assert error["type"] in ("missing_model", "unsupported_feature"), error
            assert error["exit_code"] in (3, 4), error

        serve_help = run(binary, "serve", "--help")
        if "--cors-origin" in serve_help.stdout:
            with socket.socket() as stalled_server:
                stalled_server.bind(("127.0.0.1", 0))
                stalled_server.listen()
                release_connection = threading.Event()

                def stall_response() -> None:
                    connection, _ = stalled_server.accept()
                    with connection:
                        release_connection.wait(5)

                server_thread = threading.Thread(target=stall_response, daemon=True)
                server_thread.start()
                started = time.monotonic()
                health = run(
                    binary,
                    "health",
                    "--url",
                    f"http://127.0.0.1:{stalled_server.getsockname()[1]}/ready",
                    "--timeout",
                    "3",
                    "--wait",
                    "1",
                    "--quiet",
                )
                elapsed = time.monotonic() - started
                release_connection.set()
                server_thread.join(timeout=2)
                assert health.returncode == 1, health.stdout + health.stderr
                assert elapsed < 2, f"health deadline exceeded: {elapsed:.2f}s"

            invalid_config = pathlib.Path(temporary) / "invalid-server.yaml"
            invalid_config.write_text("http:\n  port: 0\n", encoding="utf-8")
            expect_json_error(
                run(binary, "--json", "serve", "--config", str(invalid_config)),
                2,
                "invalid_argument",
            )
            expect_json_error(
                run(binary, "--json", "serve", "--cors-origin", "bad\r\nheader"),
                2,
                "invalid_argument",
            )
            expect_json_error(
                run(binary, "--json", "serve", "--no-ui", "--open"),
                2,
                "invalid_argument",
            )
            expect_json_error(
                run(binary, "--json", "serve", "--log-format", "xml"),
                2,
                "invalid_argument",
            )
            missing_arguments = []
            missing_paths = []
            for option, name in (
                ("--asr-model", "asr.gguf"),
                ("--nmt-model", "nmt.gguf"),
                ("--tts-model", "tts.gguf"),
                ("--codec-model", "codec.gguf"),
                ("--tokenizer-dir", "tokenizer"),
            ):
                if option in serve_help.stdout:
                    path = pathlib.Path(temporary) / "missing" / name
                    missing_arguments.extend((option, str(path)))
                    missing_paths.append(str(path))
            if len(missing_paths) >= 2:
                error = expect_json_error(
                    run(binary, "--json", "serve", *missing_arguments, "--no-warmup"),
                    3,
                    "missing_model",
                )
                for path in missing_paths:
                    assert path in error["message"], error["message"]

        transcribe_help = run(binary, "transcribe", "--help")
        if transcribe_help.returncode == 0:
            assert "--backend" in transcribe_help.stdout
            assert ("--live" in transcribe_help.stdout) == expect_live
            assert "session" not in transcribe_help.stderr
            lifecycle = run(binary, "transcribe")
            assert lifecycle.returncode == 2, lifecycle.stdout + lifecycle.stderr
            assert "[nemo-speech] transcribe session started" in lifecycle.stderr
            assert "[nemo-speech] transcribe session failed (exit code 2)" in lifecycle.stderr
            quiet_lifecycle = run(binary, "--quiet", "transcribe")
            assert "session" not in quiet_lifecycle.stderr
            expect_json_error(run(binary, "--json", "transcribe"), 2, "invalid_argument")
            if expect_live:
                expect_json_error(
                    run(binary, "--json", "transcribe", "--live", "recording.wav"),
                    2,
                    "invalid_argument",
                )
                expect_json_error(
                    run(binary, "--json", "transcribe", "--live", "--output-dir", temporary),
                    2,
                    "invalid_argument",
                )

        synthesize_help = run(binary, "synthesize", "--help")
        if synthesize_help.returncode == 0:
            assert "metal" in synthesize_help.stdout
            assert "vulkan" in synthesize_help.stdout

        doctor = run(binary, "doctor", "--json")
        assert doctor.returncode in (0, 1), doctor.stdout + doctor.stderr
        assert json.loads(doctor.stdout)["version"]


if __name__ == "__main__":
    main()
