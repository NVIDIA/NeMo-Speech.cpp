#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI smoke test with real models: ASR, diarization, and a TTS round trip.

Models are pulled by the CLI from the indexed Hugging Face repos into
NEMO_SPEECH_MODEL_DIR, so a warm cache makes this cheap. Text comparisons use a
similarity ratio rather than exact match so backend numerics cannot flake it.
"""
import argparse
import difflib
import json
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
import wave

JFK_TEXT = (
    "and so my fellow americans ask not what your country can do for you "
    "ask what you can do for your country"
)
TTS_TEXT = "The quick brown fox jumps over the lazy dog."


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip()


def similarity(actual: str, expected: str) -> float:
    return difflib.SequenceMatcher(None, normalize(actual), normalize(expected)).ratio()


def run(binary: str, *args: str) -> str:
    command = [binary, *args]
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise SystemExit(f"command failed with exit code {result.returncode}")
    return result.stdout


def check(condition: bool, message: str) -> None:
    print(("[PASS] " if condition else "[FAIL] ") + message, flush=True)
    if not condition:
        raise SystemExit(1)


def read_wav_stats(path: pathlib.Path) -> tuple[float, int, int]:
    with wave.open(str(path)) as audio:
        frames = audio.getnframes()
        samples = struct.unpack(f"<{frames}h", audio.readframes(frames))
        seconds = frames / audio.getframerate()
    return seconds, len(set(samples)), max(abs(s) for s in samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--skip-tts", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="nemo-speech-smoke-") as temporary:
        work = pathlib.Path(temporary)

        text = run(args.binary, "transcribe", args.audio, "--backend", args.backend)
        ratio = similarity(text, JFK_TEXT)
        print(f"asr transcript: {text.strip()!r} (similarity {ratio:.2f})")
        check(ratio >= 0.9, "ASR offline transcript matches the reference")

        streamed = run(
            args.binary,
            "transcribe",
            args.audio,
            "--backend",
            args.backend,
            "--stream",
            "--endpointing",
        )
        ratio = similarity(streamed, JFK_TEXT)
        print(f"asr streaming transcript: {streamed.strip()!r} (similarity {ratio:.2f})")
        check(ratio >= 0.8, "ASR streaming transcript with endpointing matches the reference")

        diar = json.loads(
            run(args.binary, "diarize", args.audio, "--backend", args.backend, "--format", "json")
        )
        segments = diar.get("segments", [])
        speech = sum(s["end"] - s["start"] for s in segments)
        speakers = {s["speaker"] for s in segments}
        print(f"diarization: {len(segments)} segments, {speech:.1f}s speech, speakers {speakers}")
        check(len(segments) >= 1 and speech >= 8.0, "diarization covers the utterance")
        check(len(speakers) == 1, "single-speaker audio yields one speaker")

        if args.skip_tts:
            return

        wav = work / "tts.wav"
        run(args.binary, "synthesize", TTS_TEXT, "--backend", args.backend, "--output", str(wav))
        seconds, distinct, peak = read_wav_stats(wav)
        print(f"tts: {seconds:.2f}s, {distinct} distinct sample values, peak {peak}")
        check(0.5 <= seconds <= 6.0, "TTS output has a plausible duration")
        # An all-NaN decode clamps to a single constant value (see #19).
        check(distinct > 100 and peak < 32767, "TTS output is real audio, not a constant")

        round_trip = run(args.binary, "transcribe", str(wav), "--backend", args.backend)
        ratio = similarity(round_trip, TTS_TEXT)
        print(f"tts round trip: {round_trip.strip()!r} (similarity {ratio:.2f})")
        check(ratio >= 0.8, "TTS output transcribes back to the input text")


if __name__ == "__main__":
    main()
