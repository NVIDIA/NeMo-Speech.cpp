#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Client for the NeMo Speech transcription API, built on the official `openai` SDK.

Commands:
  models      list the served models                      GET  /v1/models
  transcribe  speech to text                              POST /v1/audio/transcriptions
  translate   speech translation                          POST /v1/audio/translations
  chat        speech-LLM transcription with an instruction POST /v1/chat/completions (input_audio part)
  batch       a JSONL manifest through transcribe or chat, N requests at a time
  bench       latency and throughput at fixed numbers of requests in flight (a concurrency sweep)

Every request is made by the openai SDK, so anything this client sends an OpenAI client can send. Server
extensions (for example `diarization` on transcriptions, `language` on chat) travel in the SDK's `extra_body`.
Value rules (ranges, field combinations, audio formats) belong to the server: the client sends what it is given
and reports the server's answer, so the server's own errors are what you see.

`--stats` on transcribe, translate and chat prints the request's latency, audio duration and RTFx to stderr; batch
and bench report them over all requests.

Exit status: 0 success, 1 a request (or a batch row) failed, 2 usage error.
"""

import argparse
import base64
import contextlib
import itertools
import json
import os
import sys
import threading
import time
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

try:  # the HTTP library of the openai SDK: httpx2 from SDK 3, httpx before
    import httpx2 as httpx
except ImportError:
    import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
STATS_HELP = "print latency, audio duration and RTFx to stderr"
TEXT_FORMATS = ("text", "srt", "vtt")  # response formats whose body is the transcript itself
# What bench picks from a directory.
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".webm")


class UsageError(Exception):
    """A problem with the command line or the inputs; exit status 2."""


class RequestFailed(Exception):
    """The server refused the request or could not be reached; exit status 1."""

    def __init__(self, message, status=None, code=None, param=None):
        super().__init__(message)
        self.status, self.code, self.param = status, code, param

    def as_dict(self):
        return {"status": self.status, "code": self.code, "param": self.param, "message": str(self)}


# ---------------------------------------------------------------------------------------------------------------
# Requests


def make_client(args):
    api_key = (
        args.api_key or os.environ.get("NEMO_SPEECH_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )
    # The SDK requires a key; a server without authentication ignores the placeholder.
    return openai.OpenAI(
        base_url=args.base_url,
        api_key=api_key or "not-used",
        timeout=args.timeout,
        max_retries=args.max_retries,
    )


@contextlib.contextmanager
def api_errors():
    """SDK errors as RequestFailed with the server's message, status, code and param."""
    try:
        yield
    except openai.APIStatusError as e:
        error = {}
        try:
            error = e.response.json().get("error") or {}
        except ValueError:
            pass
        request = (
            e.response.request
        )  # without an error body, say which endpoint answered (404: not served)
        message = (
            error.get("message")
            or e.response.text
            or f"{e.response.reason_phrase or 'no body'} ({request.method} {request.url.path})"
        )
        raise RequestFailed(message, e.status_code, error.get("code"), error.get("param")) from None
    except openai.APITimeoutError:
        raise RequestFailed("request timed out") from None
    except openai.APIConnectionError as e:
        raise RequestFailed(f"cannot reach the server: {e.__cause__ or e}") from None


def call(create, **kwargs):
    """Run an SDK `with_raw_response.create` call. Returns (body, latency): the parsed JSON object, or the text for
    non-JSON responses (text, srt, vtt)."""
    started = time.monotonic()
    with api_errors():
        raw = create(**kwargs)
    latency = time.monotonic() - started
    response = raw.http_response
    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json(), latency
    return response.text, latency


def resolve_model(client, args):
    """The --model value, else the served model GET /v1/models reports (a transcription model first)."""
    if args.model:
        return args.model
    try:
        data = client.models.with_raw_response.list().http_response.json().get("data") or []
    except openai.APIStatusError as e:
        raise RequestFailed(f"GET /v1/models failed: HTTP {e.status_code}", e.status_code) from None
    except openai.APIConnectionError as e:
        raise RequestFailed(f"cannot reach the server: {e.__cause__ or e}") from None
    if not data:
        raise RequestFailed("the server lists no models; pass --model")
    preferred = [m for m in data if m.get("capability") == "transcription"]
    return (preferred or data)[0]["id"]


def parse_extras(pairs):
    """--extra key=value pairs; a value that parses as JSON is sent as that JSON value."""
    out = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise UsageError(f"--extra expects key=value, got {pair!r}")
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
    return out


def form_fields(extra):
    """Extra fields of a multipart request: form fields are text, so a JSON value is sent as its JSON text."""
    return {k: v if isinstance(v, str) else json.dumps(v) for k, v in extra.items()}


def audio_file(path):
    path = Path(path)
    if not path.is_file():
        raise UsageError(f"no such audio file: {path}")
    return path


def audio_seconds(path):
    """The duration a WAV header states, else None (other formats: the duration the server reports is used)."""
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate() if w.getframerate() else None
    except (wave.Error, EOFError, OSError):
        return None


def transcription_request(client, model, path, opts):
    """POST /v1/audio/transcriptions. `opts`: language, prompt, response_format, timestamps, diarize,
    max_speakers, temperature, extra."""
    response_format = opts.get("response_format")
    if not response_format and (opts.get("timestamps") or opts.get("diarize")):
        response_format = "verbose_json"  # timestamps and speakers come in verbose_json
    kwargs = {"model": model, "file": (path.name, path.read_bytes())}
    for key in ("language", "prompt", "temperature"):
        if opts.get(key) is not None:
            kwargs[key] = opts[key]
    if response_format:
        kwargs["response_format"] = response_format
    if opts.get("timestamps"):
        kwargs["timestamp_granularities"] = list(opts["timestamps"])
    extra = {}
    if opts.get("diarize"):
        extra["diarization"] = "true"
    if opts.get("max_speakers") is not None:
        extra["max_speaker_count"] = str(opts["max_speakers"])
    extra.update(form_fields(opts.get("extra") or {}))
    if extra:
        kwargs["extra_body"] = extra
    return call(client.audio.transcriptions.with_raw_response.create, **kwargs)


def translation_request(client, model, path, opts):
    """POST /v1/audio/translations. `opts`: prompt, response_format, temperature, language, target_language,
    extra (language and target_language are server extensions)."""
    kwargs = {"model": model, "file": (path.name, path.read_bytes())}
    for key in ("prompt", "response_format", "temperature"):
        if opts.get(key) is not None:
            kwargs[key] = opts[key]
    extra = {k: opts[k] for k in ("language", "target_language") if opts.get(k)}
    extra.update(form_fields(opts.get("extra") or {}))
    if extra:
        kwargs["extra_body"] = extra
    return call(client.audio.translations.with_raw_response.create, **kwargs)


def chat_kwargs(model, path, opts):
    """The SDK arguments of a chat request with one user message: an optional text part (the instruction) and the
    audio as an input_audio part. `opts`: instruction, audio_format, language, timestamps, seed, temperature, extra.
    """
    audio_format = opts.get("audio_format") or path.suffix.lstrip(".").lower()
    if not audio_format:
        raise UsageError(f"{path}: no file extension to name the audio format; pass --audio-format")
    content = []
    if opts.get("instruction"):
        content.append({"type": "text", "text": opts["instruction"]})
    content.append(
        {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                "format": audio_format,
            },
        }
    )
    kwargs = {"model": model, "messages": [{"role": "user", "content": content}]}
    for key in ("seed", "temperature"):
        if opts.get(key) is not None:
            kwargs[key] = opts[key]
    extra = {}
    if opts.get("language"):
        extra["language"] = opts["language"]
    if opts.get("timestamps"):
        extra["timestamp_granularities"] = list(opts["timestamps"])
    extra.update(opts.get("extra") or {})
    if extra:
        kwargs["extra_body"] = extra
    return kwargs


def chat_request(client, model, path, opts):
    """POST /v1/chat/completions (see chat_kwargs)."""
    return call(client.chat.completions.with_raw_response.create, **chat_kwargs(model, path, opts))


def encoded_chat_body(model, path, opts):
    """The JSON body chat_request would send, encoded once. Bench sends it as is: the SDK encodes every request's
    JSON (with the base64 audio) in Python again, which would cap the request rate one client can reach.
    """
    kwargs = chat_kwargs(model, path, opts)
    body = {k: v for k, v in kwargs.items() if k != "extra_body"}
    body.update(kwargs.get("extra_body") or {})
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()


def encoded_chat_request(client, body):
    """POST /v1/chat/completions with an encoded body, through the SDK's request path (client.post). Returns
    (body, latency)."""
    started = time.monotonic()
    with api_errors():
        response = client.post(
            "/chat/completions",
            cast_to=httpx.Response,
            content=body,
            options={"headers": {"Content-Type": "application/json"}},
        )
    return response.json(), time.monotonic() - started


# ---------------------------------------------------------------------------------------------------------------
# Results


def summarize(body, mode):
    """The transcript, the timed words and the duration (seconds, when the server reports one) of a response."""
    if isinstance(body, str):
        return {"text": body, "words": [], "duration": None}
    if mode == "chat":
        details = body.get("transcription") or {}
        choices = body.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content") or ""
        return {
            "text": text,
            "words": details.get("words") or [],
            "duration": details.get("duration"),
        }
    return {
        "text": body.get("text", ""),
        "words": body.get("words") or [],
        "duration": body.get("duration"),
    }


def render(body, mode, args):
    """What a single-file command prints: the full JSON with --json, else the transcript (and words with
    --show-words)."""
    if args.json:
        return body if isinstance(body, str) else json.dumps(body, indent=2, ensure_ascii=False)
    result = summarize(body, mode)
    lines = [result["text"].rstrip("\n")]
    if args.show_words:
        for w in result["words"]:
            speaker = f" {w['speaker']}" if w.get("speaker") else ""
            lines.append(
                f"  [{w.get('start', 0):9.3f} - {w.get('end', 0):9.3f}]{speaker} {w.get('word', '')}"
            )
    return "\n".join(lines)


def request_stats(body, mode, latency, path):
    """One request's numbers: latency, audio seconds (the WAV header, else the duration the server reports), RTFx,
    completion tokens (when the server reports usage), finish reason."""
    seconds = audio_seconds(path) or summarize(body, mode)["duration"]
    usage = body.get("usage") if isinstance(body, dict) else None
    choices = body.get("choices") if isinstance(body, dict) else None
    return {
        "latency_ms": round(latency * 1000, 1),
        "audio_seconds": seconds,
        "rtfx": round(seconds / latency, 1) if seconds and latency > 0 else None,
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "finish_reason": (choices or [{}])[0].get("finish_reason"),
    }


def format_stats(stats):
    """The --stats line of a single request."""
    parts = [f"latency {stats['latency_ms']:.0f} ms"]
    parts.append(f"audio {stats['audio_seconds']:.1f} s" if stats["audio_seconds"] else "audio n/a")
    if stats["rtfx"] is not None:
        parts.append(f"rtfx {stats['rtfx']:.1f}")
    if stats["completion_tokens"]:
        parts.append(f"tokens {stats['completion_tokens']}")
    return "stats: " + " | ".join(parts)


def percentile(values, q):
    """The q-quantile (0..1) by linear interpolation between the measured values; never past the maximum."""
    v = sorted(values)
    position = (len(v) - 1) * q
    low = int(position)
    high = min(low + 1, len(v) - 1)
    return v[low] + (v[high] - v[low]) * (position - low)


def emit(text, output):
    if output:
        Path(output).write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
    else:
        print(text)


# ---------------------------------------------------------------------------------------------------------------
# Batch


def manifest_rows(path):
    """(line number, row) for every non-blank line; a row that is not a JSON object is a usage error."""
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                raise UsageError(f"{path}:{number}: invalid JSON ({e})") from None
            if not isinstance(row, dict) or not isinstance(row.get("audio_filepath"), str):
                raise UsageError(f"{path}:{number}: each row needs an audio_filepath string")
            yield number, row


def run_row(client, model, mode, base_dir, number, row, opts, include_response):
    out = {"line": number, "input": row}
    path = Path(row["audio_filepath"]).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    row_opts = dict(opts)
    for key in ("language", "prompt", "instruction"):  # per-row values override the command line
        if row.get(key):
            row_opts[key] = row[key]
    try:
        if not path.is_file():
            raise RequestFailed(f"no such audio file: {path}")
        request = chat_request if mode == "chat" else transcription_request
        body, latency = request(client, model, path, row_opts)
        stats = request_stats(body, mode, latency, path)
        out.update(status="ok", latency_ms=stats["latency_ms"], **summarize(body, mode))
        if stats["audio_seconds"]:
            out["audio_seconds"] = stats["audio_seconds"]
        if include_response:
            out["response"] = body
    except (RequestFailed, UsageError) as e:
        failure = e.as_dict() if isinstance(e, RequestFailed) else {"message": str(e)}
        out.update(status="error", error=failure)
    return out


def run_batch(client, model, args, opts):
    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    if output == manifest:
        raise UsageError("the output must not be the manifest")
    if output.exists() and not args.overwrite:
        raise UsageError(f"{output} exists; pass --overwrite to replace it")
    warm = list(itertools.islice(manifest_rows(manifest), args.warmup))
    if warm:  # the first rows again and again, not timed and not written
        with ThreadPoolExecutor(args.workers) as pool:
            for i in range(args.warmup):
                pool.submit(
                    run_row,
                    client,
                    model,
                    args.mode,
                    manifest.parent,
                    *warm[i % len(warm)],
                    opts,
                    False,
                )
    rows = manifest_rows(manifest)
    started = time.monotonic()
    counts, latencies, audio_total = {"ok": 0, "error": 0}, [], 0.0
    lock = threading.Lock()
    with open(output, "w", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        pending = deque()

        def submit():
            item = next(rows, None)
            if item is not None:
                pending.append(
                    pool.submit(
                        run_row,
                        client,
                        model,
                        args.mode,
                        manifest.parent,
                        *item,
                        opts,
                        args.include_response,
                    )
                )

        for _ in range(args.workers * 2):  # bounded look-ahead keeps memory flat on long manifests
            submit()
        while pending:
            result = pending.popleft().result()  # results are written in manifest order
            submit()
            with lock:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                counts[result["status"]] += 1
                if result["status"] == "ok":
                    latencies.append(result["latency_ms"])
                    audio_total += result.get("audio_seconds") or 0.0
            if result["status"] == "error":
                print(f"line {result['line']}: {result['error']['message']}", file=sys.stderr)
    wall = time.monotonic() - started
    summary = {"rows": counts["ok"] + counts["error"], **counts, "wall_seconds": round(wall, 3)}
    if latencies:
        summary.update(
            latency_p50_ms=round(percentile(latencies, 0.5)),
            latency_p90_ms=round(percentile(latencies, 0.9)),
            latency_p99_ms=round(percentile(latencies, 0.99)),
            latency_max_ms=round(max(latencies)),
        )
    if audio_total > 0:  # WAV headers, else the durations the server reported (verbose_json, chat)
        summary.update(audio_seconds=round(audio_total, 3), rtfx=round(audio_total / wall, 1))
    print(json.dumps(summary), file=sys.stderr)
    if args.stats_json:
        Path(args.stats_json).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 1 if counts["error"] else 0


# ---------------------------------------------------------------------------------------------------------------
# Bench

BENCH_COLUMNS = (
    "conc", "requests", "errors", "audio_s", "wall_s", "rtfx", "latency_p50_ms", "latency_p90_ms",
    "latency_p99_ms", "finish_length",
)  # fmt: skip


def bench_inputs(paths):
    """The audio files of the command line; a directory gives its audio files in name order."""
    files = []
    for name in paths:
        path = Path(name)
        if path.is_dir():
            files += sorted(
                f for f in path.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_SUFFIXES
            )
        elif path.is_file():
            files.append(path)
        else:
            raise UsageError(f"no such audio file or directory: {path}")
    if not files:
        raise UsageError("no audio files to send")
    return files


def bench_sender(client, model, mode, files, opts):
    """A function that sends one file and returns its result. Chat bodies are encoded once per file (this also
    reads every file, so no disk read lands in a timed window)."""
    encoded = {f: encoded_chat_body(model, f, opts) for f in files} if mode == "chat" else {}
    for f in files:
        f.read_bytes()

    def send(path):
        try:
            if mode == "chat":
                body, latency = encoded_chat_request(client, encoded[path])
            else:
                body, latency = transcription_request(client, model, path, opts)
            return {"audio": str(path), "error": None, **request_stats(body, mode, latency, path)}
        except (RequestFailed, UsageError) as e:
            return {"audio": str(path), "error": str(e)}

    return send


def bench_wave(send, files, count, conc, start):
    """`count` requests, `conc` in flight at a time (each finished request starts the next), clips round robin
    from `start`. Returns (wall seconds from the first send to the last response, the per-request results).
    """
    clips = [files[(start + i) % len(files)] for i in range(count)]
    started = time.monotonic()
    with ThreadPoolExecutor(conc) as pool:
        results = list(pool.map(send, clips))
    return time.monotonic() - started, results


def bench_point(conc, wall, results):
    ok = [r for r in results if r["error"] is None]
    latencies = [r["latency_ms"] for r in ok]
    known = all(r["audio_seconds"] for r in ok)  # RTFx only when every duration is known
    audio = sum(r["audio_seconds"] for r in ok) if ok and known else None
    q = lambda values, p: round(percentile(values, p)) if values else None  # noqa: E731
    return {
        "conc": conc,
        "requests": len(results),
        "errors": len(results) - len(ok),
        "audio_s": None if audio is None else round(audio, 1),
        "wall_s": round(wall, 3),
        "rtfx": None if audio is None else round(audio / wall, 1),
        "latency_p50_ms": q(latencies, 0.5),
        "latency_p90_ms": q(latencies, 0.9),
        "latency_p99_ms": q(latencies, 0.99),
        "finish_length": sum(r["finish_reason"] == "length" for r in ok),
    }


def run_bench(client, model, args, opts):
    files = bench_inputs(args.audio)
    send = bench_sender(client, model, args.mode, files, opts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.warmup:
        bench_wave(send, files, args.warmup, min(args.warmup, max(args.conc)), -args.warmup)
    tsv = out / "points.tsv"
    if not tsv.exists():
        tsv.write_text("\t".join(BENCH_COLUMNS) + "\n", encoding="utf-8")
    cell = lambda v: "-" if v is None else str(v)  # noqa: E731
    widths = [max(len(c), 7) for c in BENCH_COLUMNS]
    print("  ".join(c.ljust(w) for c, w in zip(BENCH_COLUMNS, widths)), flush=True)
    failed, offset = False, 0
    for conc in args.conc:
        count = max(2 * conc, args.min_requests)  # at least two waves of `conc`
        wall, results = bench_wave(send, files, count, conc, offset)
        offset += count
        point = bench_point(conc, wall, results)
        failed |= point["errors"] > 0
        print("  ".join(cell(point[c]).ljust(w) for c, w in zip(BENCH_COLUMNS, widths)), flush=True)
        with open(tsv, "a", encoding="utf-8") as f:
            f.write("\t".join(cell(point[c]) for c in BENCH_COLUMNS) + "\n")
        (out / f"c{conc}.json").write_text(
            json.dumps({**point, "per_request": results}, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for r in results:
            if r["error"]:
                print(f"c{conc} {r['audio']}: {r['error']}", file=sys.stderr)
    print(f"wrote {tsv} and c<N>.json per level", file=sys.stderr)
    return 1 if failed else 0


# ---------------------------------------------------------------------------------------------------------------
# Command line


def add_request_options(p, *, chat):
    p.add_argument("--language", help="language of the speech (for example en, de-DE)")
    p.add_argument("--temperature", type=float, help="sampling temperature")
    p.add_argument(
        "--timestamps",
        action="append",
        choices=("word", "segment"),
        help="timestamp granularity (repeatable); on transcribe it implies --format verbose_json",
    )
    p.add_argument(
        "--extra",
        action="append",
        metavar="KEY=VALUE",
        help="extra request field (repeatable; a JSON value is sent as JSON)",
    )
    if chat:
        p.add_argument(
            "--instruction",
            help="the instruction sent as a text part (default: the server's prompt)",
        )
        p.add_argument("--seed", type=int, help="sampling seed")
        p.add_argument("--audio-format", help="input_audio format (default: the file extension)")
    else:
        p.add_argument("--prompt", help="the OpenAI `prompt` field")
        p.add_argument(
            "--format",
            dest="response_format",
            choices=("json", "verbose_json", "text", "srt", "vtt"),
            help="response_format (default json)",
        )
        p.add_argument(
            "--diarize", action="store_true", help="speaker labels (server extension; verbose_json)"
        )
        p.add_argument(
            "--max-speakers", type=int, help="speaker count limit with --diarize (server extension)"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("NEMO_SPEECH_BASE_URL", DEFAULT_BASE_URL),
        help=f"API base URL (default $NEMO_SPEECH_BASE_URL or {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key", help="bearer token (default $NEMO_SPEECH_API_KEY or $OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--model", help="model id (default: the served model listed by GET /v1/models)"
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="per-request timeout in seconds"
    )
    parser.add_argument(
        "--max-retries", type=int, default=0, help="SDK retries on 429/5xx and connection errors"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("models", help="list the served models")
    for name, chat, help_text in (
        ("transcribe", False, "POST /v1/audio/transcriptions"),
        ("chat", True, "POST /v1/chat/completions with an input_audio part"),
    ):
        p = commands.add_parser(name, help=help_text)
        p.add_argument("audio", help="audio file (sent as is; the server decodes it)")
        add_request_options(p, chat=chat)
        p.add_argument("--json", action="store_true", help="print the full response")
        p.add_argument(
            "--show-words", action="store_true", help="print the timed words after the transcript"
        )
        p.add_argument("--output", help="write the result to this file instead of stdout")
        p.add_argument("--stats", action="store_true", help=STATS_HELP)

    p = commands.add_parser("translate", help="POST /v1/audio/translations")
    p.add_argument("audio", help="audio file (sent as is; the server decodes it)")
    p.add_argument("--prompt", help="the OpenAI `prompt` field")
    p.add_argument("--format", dest="response_format", choices=("json", "verbose_json", "text"))
    p.add_argument("--temperature", type=float)
    p.add_argument("--language", help="source language (server extension)")
    p.add_argument("--target-language", help="target language (server extension)")
    p.add_argument(
        "--extra", action="append", metavar="KEY=VALUE", help="extra request field (repeatable)"
    )
    p.add_argument("--json", action="store_true", help="print the full response")
    p.add_argument("--output", help="write the result to this file instead of stdout")
    p.add_argument("--stats", action="store_true", help=STATS_HELP)

    p = commands.add_parser(
        "batch", help="run a JSONL manifest (one {\"audio_filepath\": ...} object per line)"
    )
    p.add_argument("manifest", help="JSONL; relative audio paths are relative to the manifest")
    p.add_argument(
        "--output", required=True, help="result JSONL, one line per row in manifest order"
    )
    p.add_argument("--mode", choices=("transcribe", "chat"), default="transcribe")
    p.add_argument("--workers", type=int, default=8, help="concurrent requests")
    p.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    p.add_argument(
        "--include-response", action="store_true", help="store each full response in its result"
    )
    p.add_argument(
        "--warmup", type=int, default=0, help="untimed requests (the first rows) before the batch"
    )
    p.add_argument("--stats-json", metavar="FILE", help="also write the summary to this file")
    add_batch_request_options(p)

    p = commands.add_parser(
        "bench",
        help="latency and throughput at fixed numbers of requests in flight",
        description="Sends the audio round robin with exactly N requests in flight (each finished request "
        "starts the next) for each --conc N, after one untimed warm-up wave. Per level: at least two waves "
        "(max(2 N, --min-requests) requests); latency from send to the end of the response; RTFx = audio seconds / wall time from the first send to the last response. Writes "
        "OUT/points.tsv (appended) and OUT/c<N>.json with every request.",
    )
    p.add_argument(
        "audio", nargs="+", help="audio files or directories (their audio files in name order)"
    )
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument(
        "--conc", type=int, nargs="+", default=[1, 8, 64], help="requests in flight per level"
    )
    p.add_argument("--min-requests", type=int, default=32, help="minimum requests per level")
    p.add_argument("--warmup", type=int, default=16, help="untimed requests before the first level")
    p.add_argument("--mode", choices=("chat", "transcribe"), default="chat")
    add_batch_request_options(p)
    args = parser.parse_args(argv)
    if args.command == "batch" and args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.command == "bench" and (min(args.conc) < 1 or args.min_requests < 1):
        parser.error("--conc and --min-requests must be at least 1")
    if args.command in ("batch", "bench") and args.warmup < 0:
        parser.error("--warmup must not be negative")
    return args


def add_batch_request_options(p):
    """Request options of batch and bench: the chat ones, and the transcribe ones that do not clash."""
    add_request_options(p, chat=True)
    p.add_argument("--prompt", help="transcribe mode: the OpenAI `prompt` field")
    p.add_argument("--format", dest="response_format", choices=("json", "verbose_json", "text"))
    p.add_argument(
        "--diarize", action="store_true", help="transcribe mode: speaker labels (server extension)"
    )
    p.add_argument(
        "--max-speakers", type=int, help="transcribe mode: speaker count limit with --diarize"
    )


def request_options(args):
    keys = (
        "language",
        "prompt",
        "instruction",
        "response_format",
        "timestamps",
        "diarize",
        "max_speakers",
        "temperature",
        "seed",
        "audio_format",
        "target_language",
    )
    # absent options are None and unset flags False; 0 (a seed, a temperature) is a value
    opts = {k: v for k in keys if (v := getattr(args, k, None)) is not None and v is not False}
    opts["extra"] = parse_extras(getattr(args, "extra", None))
    return opts


def main(argv=None):
    args = parse_args(argv)
    try:
        client = make_client(args)
        if args.command == "models":
            body = client.models.with_raw_response.list().http_response.json()
            print(json.dumps(body, indent=2, ensure_ascii=False))
            return 0
        model = resolve_model(client, args)
        opts = request_options(args)
        if args.command == "batch":
            return run_batch(client, model, args, opts)
        if args.command == "bench":
            return run_bench(client, model, args, opts)
        path = audio_file(args.audio)
        if args.command == "transcribe":
            body, latency = transcription_request(client, model, path, opts)
        elif args.command == "translate":
            body, latency = translation_request(client, model, path, opts)
        else:
            body, latency = chat_request(client, model, path, opts)
        args.show_words = getattr(args, "show_words", False)
        mode = "chat" if args.command == "chat" else "audio"
        emit(render(body, mode, args), args.output)
        if args.stats:
            print(format_stats(request_stats(body, mode, latency, path)), file=sys.stderr)
        return 0
    except UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except RequestFailed as e:
        status = (
            f"HTTP {e.status}" + (f" {e.code}" if e.code else "") if e.status else "request failed"
        )
        param = f" (param: {e.param})" if e.param else ""
        print(f"error: {status}: {e}{param}", file=sys.stderr)
        return 1
    except openai.APIStatusError as e:  # GET /v1/models
        print(f"error: HTTP {e.status_code}: {e.response.text}", file=sys.stderr)
        return 1
    except openai.APIConnectionError as e:
        print(f"error: cannot reach the server: {e.__cause__ or e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
