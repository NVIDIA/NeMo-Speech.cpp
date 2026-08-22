#!/usr/bin/env python3
"""Record microphone audio and send it to a persistent NeMo-Speech.cpp HTTP server."""

from __future__ import annotations

import argparse
import io
import json
import queue
import sys
import time
import wave
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
import sounddevice as sd

SAMPLE_RATE = 16_000
CHANNELS = 1
MIN_DURATION_SECONDS = 0.08
SILENCE_RMS_THRESHOLD = 1e-5


@dataclass(slots=True)
class RecordedAudio:
    samples: np.ndarray
    duration_seconds: float
    rms: float
    capture_ms: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8081/v1/audio/transcriptions")
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--device", help="Input-device index or a case-insensitive part of its name."
    )
    parser.add_argument("--list-devices", action="store_true", help="List input devices and exit.")
    parser.add_argument(
        "--show-words", action="store_true", help="Print word timestamps when present."
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    return parser


def input_devices() -> list[tuple[int, dict[str, Any]]]:
    return [
        (index, dict(device))
        for index, device in enumerate(sd.query_devices())
        if int(device["max_input_channels"]) > 0
    ]


def list_devices() -> None:
    for index, device in input_devices():
        print(f"{index}: {device['name']} ({device['max_input_channels']} input channel(s))")


def resolve_device(selector: str | None) -> tuple[int | None, str]:
    if selector is None:
        return None, "default"
    devices = input_devices()
    try:
        index = int(selector)
    except ValueError:
        matches = [(i, d) for i, d in devices if selector.casefold() in str(d["name"]).casefold()]
        if not matches:
            raise ValueError(f"No input device contains {selector!r}. Use --list-devices.")
        if len(matches) > 1:
            choices = ", ".join(f"{i}: {d['name']}" for i, d in matches)
            raise ValueError(
                f"More than one input device matches {selector!r}: {choices}. Use its index."
            )
        index, device = matches[0]
        return index, str(device["name"])
    for available_index, device in devices:
        if available_index == index:
            return index, str(device["name"])
    raise ValueError(f"Input device index {index} is unavailable. Use --list-devices.")


def calculate_rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0


def record_until_enter(device: int | None) -> RecordedAudio:
    blocks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(
        input_data: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags
    ) -> None:
        del frames, time_info
        if status:
            print(f"Microphone warning: {status}", file=sys.stderr)
        blocks.put(input_data[:, 0].copy())

    input("Press Enter to start recording...")
    started = time.perf_counter()
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", device=device, callback=callback
    ):
        input("Recording. Press Enter again to stop...")
    capture_ms = (time.perf_counter() - started) * 1000
    captured = [blocks.get_nowait() for _ in range(blocks.qsize())]
    samples = (
        np.ascontiguousarray(np.concatenate(captured), dtype=np.float32)
        if captured
        else np.empty(0, dtype=np.float32)
    )
    return RecordedAudio(samples, samples.size / SAMPLE_RATE, calculate_rms(samples), capture_ms)


def encode_wav(samples: np.ndarray) -> tuple[bytes, float]:
    started = time.perf_counter()
    safe = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(safe))) if safe.size else 0.0
    if peak > 1.0:
        safe = safe / peak
    pcm16 = np.clip(safe * 32767.0, -32768, 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm16.tobytes())
    return output.getvalue(), (time.perf_counter() - started) * 1000


def ready_url(transcription_url: str) -> str:
    parts = urlsplit(transcription_url)
    return urlunsplit((parts.scheme, parts.netloc, "/ready", "", ""))


def check_server(url: str, timeout: float) -> None:
    response = requests.get(ready_url(url), timeout=min(timeout, 5.0))
    response.raise_for_status()


def request_transcription(
    session: requests.Session, url: str, wav_data: bytes, language: str, timeout: float
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = session.post(
        url,
        files={"file": ("microphone.wav", wav_data, "audio/wav")},
        data={"model": "default", "language": language, "response_format": "verbose_json"},
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    try:
        return response.json(), elapsed_ms
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Server response was not JSON: {response.text}") from error


def result_words(payload: dict[str, Any]) -> list[dict[str, Any]]:
    words = payload.get("words")
    if isinstance(words, list):
        return [word for word in words if isinstance(word, dict)]
    collected: list[dict[str, Any]] = []
    for segment in payload.get("segments", []):
        if isinstance(segment, dict) and isinstance(segment.get("words"), list):
            collected.extend(word for word in segment["words"] if isinstance(word, dict))
    return collected


def print_result(
    audio: RecordedAudio,
    preparation_ms: float,
    request_ms: float,
    payload: dict[str, Any],
    show_words: bool,
) -> None:
    text = str(payload.get("text", "")).strip()
    print("\n" + "=" * 64)
    print(f'Text: "{text}"')
    print("\nTimings:")
    print(f"  Capture:               {audio.capture_ms:.2f} ms")
    print(f"  In-memory preparation: {preparation_ms:.2f} ms")
    print(f"  HTTP + inference:      {request_ms:.2f} ms")
    print(f"  Total request:         {request_ms:.2f} ms")
    if audio.duration_seconds and request_ms:
        rtf = (request_ms / 1000) / audio.duration_seconds
        print(f"  RTF:                   {rtf:.4f}")
        print(f"  Speed:                 {1 / rtf:.3f}x realtime")
    if payload.get("duration") is not None:
        print(f"  Server duration:       {float(payload['duration']):.3f} s")
    if payload.get("language"):
        print(f"  Returned language:     {payload['language']}")
    if show_words:
        words = result_words(payload)
        if words:
            print("\nWords:")
            for word in words:
                token = str(word.get("word", word.get("text", ""))).strip()
                start, end = word.get("start"), word.get("end")
                timing = (
                    f"{float(start):.3f}–{float(end):.3f}"
                    if start is not None and end is not None
                    else "unknown time"
                )
                confidence = (
                    f" | confidence={word['confidence']}"
                    if word.get("confidence") is not None
                    else ""
                )
                print(f"  {timing} {token}{confidence}")
        else:
            print("\nWords: not returned by the server.")
    print("=" * 64 + "\n")


def main() -> int:
    args = build_parser().parse_args()
    if args.list_devices:
        list_devices()
        return 0
    try:
        device, device_name = resolve_device(args.device)
        check_server(args.url, args.timeout)
    except (ValueError, requests.RequestException) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("NeMo-Speech.cpp — HTTP microphone client")
    print(f"Server: {args.url}")
    print(f"Language: {args.language}")
    print(f"Microphone: {device_name}")
    print("The model remains loaded in the separate persistent server.")
    print("Press Ctrl+C to exit.\n")
    session = requests.Session()
    try:
        while True:
            audio = record_until_enter(device)
            print(f"\nCaptured duration: {audio.duration_seconds:.3f} s")
            print(f"RMS: {audio.rms:.8f}")
            if audio.duration_seconds < MIN_DURATION_SECONDS:
                print("Audio is too short; ignored.\n")
                continue
            if audio.rms < SILENCE_RMS_THRESHOLD:
                print("Silence detected; ignored.\n")
                continue
            wav_data, preparation_ms = encode_wav(audio.samples)
            try:
                payload, request_ms = request_transcription(
                    session, args.url, wav_data, args.language, args.timeout
                )
                print_result(audio, preparation_ms, request_ms, payload, args.show_words)
            except requests.HTTPError as error:
                body = error.response.text if error.response is not None else str(error)
                print(f"HTTP ERROR: {body}\n", file=sys.stderr)
            except (RuntimeError, requests.RequestException) as error:
                print(f"ERROR: {error}\n", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
