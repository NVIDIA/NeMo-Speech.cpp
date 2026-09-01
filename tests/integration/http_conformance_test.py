#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--vad-model")
    parser.add_argument("--diar-model")
    parser.add_argument("--pnc-model")
    parser.add_argument("--itn-model-dir")
    parser.add_argument("--nmt-model")
    parser.add_argument("--tts-model")
    parser.add_argument("--codec-model")
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--node")
    parser.add_argument("--openai-js-package")
    parser.add_argument("--openai-js-example")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--tts-preempt", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def wait_for_log_entry(log_file, start: int, marker: bytes, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        size = os.fstat(log_file.fileno()).st_size
        if size > start and marker in os.pread(log_file.fileno(), size - start, start):
            return True
        time.sleep(0.05)
    return False


def test_javascript_client(args: argparse.Namespace, base: str, api_key: str) -> None:
    javascript = (args.node, args.openai_js_package, args.openai_js_example)
    if not any(javascript):
        return
    if not all(javascript):
        raise RuntimeError(
            "JavaScript conformance requires --node, --openai-js-package, and "
            "--openai-js-example together"
        )
    package = Path(args.openai_js_package).resolve()
    if not (package / "package.json").is_file():
        raise RuntimeError(f"OpenAI JavaScript package is invalid: {package}")
    with tempfile.TemporaryDirectory(prefix="nemo-speech-openai-js-") as temporary:
        root = Path(temporary)
        sdk = root / "node_modules" / "openai"
        sdk.parent.mkdir(parents=True)
        shutil.copytree(package, sdk)
        example = root / "openai_client.mjs"
        shutil.copy2(args.openai_js_example, example)
        environment = os.environ.copy()
        environment["OPENAI_BASE_URL"] = f"{base}/v1"
        environment["NEMO_SPEECH_API_KEY"] = api_key
        result = subprocess.run(
            [args.node, str(example), args.audio],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=args.timeout,
        )
        require(bool(result.stdout.strip()), "OpenAI JavaScript client returned no transcript")


def main() -> None:
    try:
        import httpx
        from openai import OpenAI
        from websockets.exceptions import InvalidStatus
        from websockets.sync.client import connect
    except ImportError as error:
        raise RuntimeError(
            "HTTP conformance requires the openai, httpx, and websockets Python packages"
        ) from error

    args = parse_args()
    port = free_port()
    api_key = "nemo-speech-conformance-key"
    base = f"http://127.0.0.1:{port}"
    command = [
        args.binary,
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--threads",
        "2",
        "--api-key",
        api_key,
        "--access-log",
        "--log-format",
        "json",
        "--no-warmup",
        "--asr-model",
        args.asr_model,
    ]
    for option, value in (
        ("--vad-model", args.vad_model),
        ("--diar-model", args.diar_model),
        ("--pnc-model", args.pnc_model),
        ("--itn-model-dir", args.itn_model_dir),
        ("--nmt-model", args.nmt_model),
        ("--tts-model", args.tts_model),
        ("--codec-model", args.codec_model),
        ("--tokenizer-dir", args.tokenizer_dir),
    ):
        if value:
            command.extend((option, value))
    if args.tts_preempt:
        command.extend(("--tts.preempt", "--verbose"))

    server_log = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    client = httpx.Client(timeout=60, trust_env=False)
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                server_log.flush()
                server_log.seek(0)
                output = server_log.read()
                raise RuntimeError(
                    f"server exited during startup ({process.returncode}):\n{output}"
                )
            try:
                if client.get(f"{base}/ready").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("server did not become ready")

        playground = client.get(f"{base}/")
        require(playground.status_code == 200, "browser playground is not available")
        require(
            playground.headers.get("content-type", "").startswith("text/html"),
            "browser playground content type",
        )
        require(
            "default-src 'self'" in playground.headers.get("content-security-policy", ""),
            "browser playground is missing its content security policy",
        )
        require(
            playground.headers.get("x-content-type-options") == "nosniff",
            "browser playground is missing nosniff protection",
        )
        require(bool(playground.headers.get("x-request-id")), "request ID header")
        require("NeMo-Speech.cpp" in playground.text, "browser playground body")
        require('id="server-status"' in playground.text, "browser playground status panel")
        require('id="asr-speakers"' in playground.text, "speaker-tagged ASR control")
        require('data-source="microphone"' in playground.text, "speech translation microphone")
        require('data-source="text"' in playground.text, "speech translation text input")
        require('id="tts-voice"' in playground.text, "TTS voice selector")
        require('id="tts-lang"' in playground.text, "TTS language selector")
        require("ondrop" in playground.text, "browser playground drag-and-drop support")
        require('value="text"' in playground.text, "browser playground plain-text output")
        require(
            re.search(r"(?:src|href)\s*=\s*['\"]https?://", playground.text, re.IGNORECASE) is None,
            "browser playground loads a remote asset",
        )
        require(
            not any(
                marker in playground.text.lower()
                for marker in ("google-analytics", "googletagmanager", "segment.io")
            ),
            "browser playground contains analytics",
        )

        unauthorized = client.get(f"{base}/v1/models")
        require(unauthorized.status_code == 401, "protected route accepted no API key")
        require(unauthorized.json()["error"]["type"] == "invalid_request_error", "error envelope")
        headers = {"Authorization": f"Bearer {api_key}"}
        missing_route = client.get(f"{base}/v1/not-a-route", headers=headers)
        require(missing_route.status_code == 404, "unknown API route status")
        require(
            missing_route.json()["error"]["type"] == "invalid_request_error",
            "unknown API route error envelope",
        )
        preflight = client.options(
            f"{base}/v1/audio/transcriptions",
            headers={"Origin": "https://client.example"},
        )
        require(preflight.status_code == 204, "CORS preflight")
        require(
            "access-control-allow-origin" not in preflight.headers,
            "cross-origin browser access must be opt-in",
        )

        sdk = OpenAI(
            api_key=api_key,
            base_url=f"{base}/v1",
            http_client=httpx.Client(timeout=120, trust_env=False),
        )
        models = sdk.models.list()
        require(all(getattr(model, "device", "") for model in models.data), "model device status")
        capabilities = {getattr(model, "capability", "") for model in models.data}
        require("transcription" in capabilities, "model inventory omitted transcription")
        asr_id = next(
            model.id for model in models.data if getattr(model, "capability", "") == "transcription"
        )
        with Path(args.audio).open("rb") as audio:
            transcript = sdk.audio.transcriptions.create(
                model=asr_id,
                file=audio,
                response_format="verbose_json",
            )
        require(bool(transcript.text.strip()), "OpenAI client returned an empty transcript")
        if args.diar_model:
            with Path(args.audio).open("rb") as audio:
                response = client.post(
                    f"{base}/v1/audio/transcriptions",
                    headers=headers,
                    data={"response_format": "verbose_json", "diarization": "true"},
                    files={"file": (Path(args.audio).name, audio, "audio/wav")},
                )
            require(response.status_code == 200, f"speaker-tagged ASR failed: {response.text}")
            words = response.json().get("words", [])
            require(words and any(word.get("speaker", 0) > 0 for word in words), "ASR speaker tags")
        test_javascript_client(args, base, api_key)

        for response_format in ("srt", "vtt"):
            with Path(args.audio).open("rb") as audio:
                subtitle = client.post(
                    f"{base}/v1/audio/transcriptions",
                    headers=headers,
                    data={"model": asr_id, "response_format": response_format},
                    files={"file": (Path(args.audio).name, audio, "audio/wav")},
                )
            require(
                subtitle.status_code == 200,
                f"{response_format} transcription failed: {subtitle.text}",
            )
            if response_format == "srt":
                require(subtitle.text.startswith("1\n"), "SRT sequence number")
                require(
                    re.search(r"\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}", subtitle.text)
                    is not None,
                    "SRT timestamps",
                )
            else:
                require(subtitle.text.startswith("WEBVTT\n\n"), "WebVTT header")
                require(
                    re.search(r"\d\d:\d\d:\d\d\.\d{3} --> \d\d:\d\d:\d\d\.\d{3}", subtitle.text)
                    is not None,
                    "WebVTT timestamps",
                )
            blocks = subtitle.text.strip().split("\n\n")
            if response_format == "vtt":
                blocks = blocks[1:]
            text_lines = []
            for block in blocks:
                lines = block.splitlines()
                cue_text = lines[2:] if response_format == "srt" else lines[1:]
                require(1 <= len(cue_text) <= 2, "subtitle cue line count")
                text_lines.extend(cue_text)
            require(text_lines, "subtitle cue text")
            require(max(map(len, text_lines)) <= 42, "subtitle line length")

        with wave.open(args.audio, "rb") as audio:
            require(audio.getnchannels() == 1, "WebSocket fixture must be mono")
            require(audio.getsampwidth() == 2, "WebSocket fixture must be PCM16")
            sample_rate = audio.getframerate()
            pcm = audio.readframes(audio.getnframes())
        events = []
        websocket_url = f"ws://127.0.0.1:{port}/v1/audio/transcriptions/realtime?api_key={api_key}"
        with connect(
            websocket_url,
            open_timeout=30,
            close_timeout=10,
            ping_interval=None,
        ) as websocket:
            events.append(json.loads(websocket.recv()))
            websocket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "sample_rate": sample_rate,
                            "word_timestamps": True,
                            "speaker_diarization": bool(args.diar_model),
                        },
                    }
                )
            )
            websocket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm[: min(len(pcm), 640)]).decode("ascii"),
                    }
                )
            )
            websocket.send(json.dumps({"type": "input_audio_buffer.clear"}))
            chunk = max(2, sample_rate * 2 // 5)
            for offset in range(0, len(pcm), chunk):
                websocket.send(pcm[offset : offset + chunk])
            websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            while True:
                event = json.loads(websocket.recv())
                events.append(event)
                if event.get("type") == "input_audio_buffer.committed":
                    break
        completed = [
            event for event in events if event.get("type", "").endswith("transcription.completed")
        ]
        require(completed and completed[-1].get("transcript", "").strip(), "WebSocket final")
        require("words" in completed[-1], "WebSocket word timestamps")
        if args.diar_model:
            require(
                any(word.get("speaker", 0) > 0 for word in completed[-1]["words"]),
                "WebSocket speaker tags",
            )
        require(
            any(event.get("type") == "input_audio_buffer.cleared" for event in events),
            "WebSocket clear acknowledgement",
        )

        with connect(
            websocket_url,
            open_timeout=30,
            close_timeout=10,
            ping_interval=None,
        ) as websocket:
            json.loads(websocket.recv())
            websocket.send(json.dumps({"type": "input_audio_buffer.append", "audio": "A"}))
            error = json.loads(websocket.recv())
            require(error.get("type") == "error", "truncated Base64 was accepted")

        try:
            with connect(
                f"ws://127.0.0.1:{port}/v1/audio/transcriptions/realtime",
                open_timeout=30,
                close_timeout=10,
                ping_interval=None,
            ) as unauthorized_socket:
                unauthorized_socket.recv()
            raise RuntimeError("realtime WebSocket accepted no API key")
        except InvalidStatus as error:
            require(error.response.status_code == 401, "realtime WebSocket authorization status")

        if args.diar_model:
            with Path(args.audio).open("rb") as audio:
                response = client.post(
                    f"{base}/v1/audio/diarizations",
                    headers=headers,
                    files={"file": (Path(args.audio).name, audio, "audio/wav")},
                )
            require(response.status_code == 200, f"diarization failed: {response.text}")
            require(isinstance(response.json().get("segments"), list), "diarization response")

        if args.nmt_model:
            response = client.post(
                f"{base}/v1/translations",
                headers=headers,
                json={
                    "input": "Speech runs locally.",
                    "source_language": "en-US",
                    "target_language": "es-US",
                },
            )
            require(response.status_code == 200, f"text translation failed: {response.text}")
            require(response.json()["translations"][0]["text"], "empty translation")
            with Path(args.audio).open("rb") as audio:
                response = client.post(
                    f"{base}/v1/audio/translations",
                    headers=headers,
                    data={"target_language": "es-US", "response_format": "verbose_json"},
                    files={"file": (Path(args.audio).name, audio, "audio/wav")},
                )
            require(response.status_code == 200, f"speech translation failed: {response.text}")
            require(response.json().get("text"), "empty speech translation")

        if args.tts_model and args.codec_model and args.tokenizer_dir:
            speech_model = next(
                model for model in models.data if getattr(model, "capability", "") == "speech"
            )
            require(getattr(speech_model, "voices", []), "TTS model inventory voices")
            require(getattr(speech_model, "languages", []), "TTS model inventory languages")
            voice = speech_model.voices[0]
            speech = sdk.audio.speech.create(
                model=speech_model.id,
                voice=voice,
                input="Hello from NeMo-Speech.cpp.",
                response_format="wav",
            )
            require(speech.content.startswith(b"RIFF"), "OpenAI speech response is not WAV")
            response = client.post(
                f"{base}/v1/audio/speech",
                headers=headers,
                json={"input": "Invalid sample rate.", "voice": voice, "sample_rate": 96001},
            )
            require(response.status_code == 400, "invalid TTS sample rate was accepted")
            if args.tts_preempt:

                def synthesize_long_text() -> tuple[int, bytes]:
                    with httpx.Client(timeout=args.timeout, trust_env=False) as concurrent_client:
                        response = concurrent_client.post(
                            f"{base}/v1/audio/speech",
                            headers=headers,
                            json={"input": "NeMo-Speech.cpp runs locally. " * 100, "voice": voice},
                        )
                        return response.status_code, response.content

                first_result: list[tuple[int, bytes]] = []

                def collect_first() -> None:
                    first_result.append(synthesize_long_text())

                admission_log_offset = os.fstat(server_log.fileno()).st_size
                first = threading.Thread(target=collect_first)
                first.start()
                require(
                    wait_for_log_entry(server_log, admission_log_offset, b"riva_tts encoding", 5),
                    "long TTS request was not admitted",
                )
                with httpx.Client(timeout=args.timeout, trust_env=False) as concurrent_client:
                    latest = concurrent_client.post(
                        f"{base}/v1/audio/speech",
                        headers=headers,
                        json={"input": "Newest synthesis request.", "voice": voice},
                    )
                first.join(args.timeout)
                require(not first.is_alive(), "preempted TTS request did not finish")
                require(
                    first_result and first_result[0][0] == 409,
                    "older TTS request was not canceled",
                )
                require(latest.status_code == 200, f"newest TTS request failed: {latest.text}")
                require(latest.content.startswith(b"RIFF"), "newest TTS response is not WAV")
            if args.nmt_model:
                with Path(args.audio).open("rb") as audio:
                    response = client.post(
                        f"{base}/v1/audio/speech/translations",
                        headers=headers,
                        data={"target_language": "es-US", "response_format": "wav"},
                        files={"file": (Path(args.audio).name, audio, "audio/wav")},
                    )
                require(response.status_code == 200, f"speech-to-speech failed: {response.text}")
                require(
                    response.content.startswith(b"RIFF"), "speech-to-speech response is not WAV"
                )
    finally:
        client.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        server_log.flush()
        server_log.seek(0)
        output = server_log.read()
        server_log.close()
        if sys.exc_info()[0] is None:
            access_events = []
            for line in output.splitlines():
                if not line.startswith("{"):
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "http.request":
                    access_events.append(event)
            require(access_events, "structured HTTP access log")
            require(
                all("?" not in event.get("path", "") for event in access_events),
                "access log exposed a query string",
            )
        if process.returncode not in (0, -signal.SIGTERM):
            if sys.exc_info()[0] is None:
                raise RuntimeError(f"server exited with {process.returncode}:\n{output}")


if __name__ == "__main__":
    main()
