# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import http.client
import io
import json
import os
import signal
import socket
import struct
import subprocess
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nemotron_mock_worker import MockWorker


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def open_speech(port, text, response_format="pcm", seed=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
    payload = {
        "model": "nemotron-tts",
        "input": text,
        "voice": "english_sample",
        "language": "en-US",
        "response_format": response_format,
    }
    if seed is not None:
        payload["seed"] = seed
    connection.request(
        "POST",
        "/v1/audio/speech",
        json.dumps(payload),
        {"Content-Type": "application/json"},
    )
    return connection, connection.getresponse()


def speech(port, text, response_format="pcm", seed=None):
    connection, response = open_speech(port, text, response_format, seed=seed)
    try:
        return response.status, response.read(), dict(response.getheaders())
    finally:
        response.close()
        connection.close()


def failed_stream(port, text, require_no_audio=False):
    connection, response = open_speech(port, text)
    try:
        try:
            payload = response.read()
        except http.client.IncompleteRead as error:
            if require_no_audio:
                require(not error.partial, "invalid F32 frame published partial PCM audio")
            return
        require(
            response.status >= 400 and payload,
            "worker failure became a successfully completed audio response",
        )
    finally:
        response.close()
        connection.close()


def run(binary, float_audio=False, runtime_seed=None):
    with tempfile.TemporaryDirectory(prefix="nemo-native-http-") as temporary:
        folder = Path(temporary)
        worker = MockWorker(folder / "worker.sock", float_audio=float_audio)
        port = free_port()
        with (folder / "server.log").open("w+") as log:
            process = subprocess.Popen(
                [
                    str(binary),
                    "serve",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--tts-backend",
                    "nemotron",
                    "--native-socket",
                    worker.path,
                    "--http-threads",
                    "8",
                    "--no-warmup",
                ]
                + ([] if runtime_seed is None else ["--tts.seed", str(runtime_seed)]),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("server exited during startup")
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                    try:
                        connection.request("GET", "/ready")
                        response = connection.getresponse()
                        response.read()
                        if response.status == 200:
                            break
                    except OSError:
                        time.sleep(0.05)
                    finally:
                        connection.close()
                else:
                    raise RuntimeError("server did not become ready")
                require(worker.ready.is_set(), "server skipped bridge handshake")

                connection, response = open_speech(port, "stream")
                try:
                    require(response.status == 200, "streaming request failed")
                    require(
                        response.getheader("Content-Type", "").startswith("audio/pcm"),
                        "PCM content type",
                    )
                    require(
                        response.getheader("x-audio-sample-rate") == "22050",
                        "missing PCM sample rate",
                    )
                    first = response.read(512)
                    require(len(first) == 512, "missing first PCM chunk")
                    require(not worker.completed, "worker completed before first audio arrived")
                    worker.release_stream.set()
                    rest = response.read()
                    require(len(first + rest) == 1024, "stream PCM was truncated or duplicated")
                    if float_audio:
                        expected = struct.pack("<512h", *([-16384] * 256 + [16384] * 256))
                        require(first + rest == expected, "F32 conversion changed PCM16 bytes")
                finally:
                    worker.release_stream.set()
                    response.close()
                    connection.close()

                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(speech, port, text) for text in ("parallel-A", "parallel-B")
                    ]
                    results = [future.result(timeout=10) for future in futures]
                require(
                    worker.parallel_ready.is_set(), "requests did not overlap inside the worker"
                )
                require(
                    all(status == 200 and len(pcm) == 1024 for status, pcm, _ in results),
                    "concurrent requests lost audio",
                )

                status, payload, _ = speech(port, "wave", "wav")
                require(status == 200, "WAV request failed")
                with wave.open(io.BytesIO(payload), "rb") as audio:
                    require(
                        (
                            audio.getnchannels(),
                            audio.getsampwidth(),
                            audio.getframerate(),
                            audio.getnframes(),
                        )
                        == (1, 2, 22050, 512),
                        "WAV metadata or sample extent mismatch",
                    )

                status, payload, _ = speech(port, "failure", "wav")
                require(status >= 400, "worker preparation failure became HTTP success")
                require(b"injected preparation failure" in payload, "worker error message was lost")
                failed_stream(port, "failure")
                require(
                    speech(port, "recovery")[0] == 200, "request failure poisoned other requests"
                )

                connection, response = open_speech(port, "cancel")
                require(
                    response.status == 200 and len(response.read(512)) == 512,
                    "cancellation request lacked audio",
                )
                response.close()
                connection.close()
                require(worker.aborted.wait(5), "HTTP disconnect did not abort backend request")
                require(
                    speech(port, "after-cancel")[0] == 200,
                    "cancellation poisoned subsequent requests",
                )
                require(
                    speech(port, "seed-override", seed=11)[0] == 200,
                    "explicit request seed failed",
                )

                failed_stream(port, "malformed", require_no_audio=float_audio)
                require(not worker.errors, "mock protocol errors: " + repr(worker.errors))
                require(
                    all(
                        request["voice"] == "english_sample"
                        and request["language"] == "en-US"
                        and request["seed"]
                        == (
                            11
                            if request["text"] == "seed-override"
                            else (0 if runtime_seed is None else runtime_seed)
                        )
                        for request in worker.submissions
                    ),
                    "request metadata changed",
                )
            except BaseException:
                log.flush()
                log.seek(0)
                print(log.read())
                raise
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                worker.close()
        print(
            f"PASS ({'F32' if float_audio else 'PCM16'} frames): "
            "streaming, concurrency, WAV, errors, cancellation, malformed PCM, runtime seed"
        )


def reject_runtime_options(binary):
    cases = (
        ("--tts.steps", "64", "Nemotron requires steps=512"),
        ("--tts.top-k", "10", "Nemotron requires top_k=80"),
        ("--tts.temperature", "0.5", "Nemotron requires temperature=0.7"),
        ("--tts.cfg-scale", "1.0", "Nemotron does not support cfg_scale"),
    )
    with tempfile.TemporaryDirectory(prefix="nemo-native-runtime-") as temporary:
        for option, value, error in cases:
            result = subprocess.run(
                [
                    str(binary),
                    "serve",
                    "--tts-backend",
                    "nemotron",
                    "--native-socket",
                    str(Path(temporary) / "absent.sock"),
                    "--no-warmup",
                    option,
                    value,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            )
            require(
                result.returncode != 0 and error in result.stdout,
                f"{option}={value} did not reject the incompatible runtime setting: {result.stdout}",
            )
    print("PASS: incompatible CLI steps, top-k, temperature, and CFG rejected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    arguments = parser.parse_args()
    binary = arguments.binary.resolve(strict=True)
    run(binary, runtime_seed=7)
    run(binary, float_audio=True)
    reject_runtime_options(binary)
