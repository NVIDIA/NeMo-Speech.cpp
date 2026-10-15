#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import http.client
import json
import math
import os
import platform
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


def distribution(values):
    values = sorted(values)
    return {
        "count": len(values),
        "avg": statistics.fmean(values) if values else None,
        **{
            f"p{percent}": (
                values[max(0, math.ceil(len(values) * percent / 100) - 1)] if values else None
            )
            for percent in (50, 95, 99)
        },
    }


def reconstruct(arrivals, frame_samples, startup, steady, sample_rate):
    chunks, inter_chunk, inter_frame = [], [], []
    completed = 0
    previous_arrival = None
    buffered = 0.0
    deficits = []
    gaps = []
    prior_samples = 0
    for arrival in arrivals:
        stamp, cumulative = arrival["elapsed_ms"], arrival["cumulative_samples"]
        if cumulative == prior_samples:
            continue
        if previous_arrival is not None:
            gap = (stamp - previous_arrival) / 1000
            gaps.append(gap * 1000)
            if gap > buffered:
                deficits.append((gap - buffered) * 1000)
            buffered = max(0.0, buffered - gap)
        buffered += (cumulative - prior_samples) / sample_rate
        previous_arrival, prior_samples = stamp, cumulative
        while True:
            frames = startup[len(chunks)] if len(chunks) < len(startup) else steady
            boundary = completed + frames * frame_samples
            if cumulative < boundary:
                break
            if chunks:
                interval = stamp - chunks[-1]["elapsed_ms"]
                inter_chunk.append(interval)
                inter_frame.append(interval / frames)
            chunks.append({"elapsed_ms": stamp, "frames": frames, "cumulative_samples": boundary})
            completed = boundary
    return {
        "logical_chunk_completions": chunks,
        "inter_chunk_ms": inter_chunk,
        "inter_frame_ms": inter_frame,
        "network_inter_arrival_ms": gaps,
        "terminal_partial_samples": prior_samples - completed,
        "playback_deficit_events": len(deficits),
        "max_playback_deficit_ms": max(deficits, default=0.0),
        "total_playback_deficit_ms": sum(deficits),
    }


def request(args, case, index, trial_start, gate=None):
    result = {
        "request_index": index,
        "case_id": case["id"],
        "seed": args.seed,
        "status": "FAILED",
        "http_status": None,
        "error": None,
        "ttfb_ms": None,
        "ttfa_ms": None,
        "headers_ms": None,
        "completion_ms": None,
        "connect_ms": None,
        "bytes": 0,
        "samples": 0,
        "audio_seconds": 0.0,
        "arrivals": [],
        "sample_rate": args.sample_rate,
        "sample_rate_source": "configured_fallback",
    }
    connection = response = transport = None
    digest = hashlib.sha256()
    started = time.perf_counter()
    deadline = started + args.timeout
    measured = started
    result["worker_start_ms"] = (started - trial_start) * 1000
    body = {
        "model": args.model,
        "input": case["text"],
        "voice": args.voice,
        "language_code": args.language,
        "seed": args.seed,
        "sample_rate": args.sample_rate,
        "response_format": "pcm",
        "stream_format": "audio",
    }
    if args.apply_normalize is not None:
        body["apply_normalize"] = args.apply_normalize == "true"

    def remaining():
        seconds = deadline - time.perf_counter()
        if seconds <= 0:
            raise TimeoutError("request exceeded timeout")
        if transport is not None and transport.fileno() >= 0:
            transport.settimeout(seconds)
        return seconds

    def observe(data):
        if not data:
            return
        stamp = (time.perf_counter() - measured) * 1000
        digest.update(data)
        result["bytes"] += len(data)
        result["samples"] = result["bytes"] // 2
        result["arrivals"].append(
            {
                "elapsed_ms": stamp,
                "bytes": len(data),
                "cumulative_bytes": result["bytes"],
                "cumulative_samples": result["samples"],
            }
        )
        if result["ttfb_ms"] is None:
            result["ttfb_ms"] = stamp
        if result["ttfa_ms"] is None and result["samples"]:
            result["ttfa_ms"] = stamp

    try:
        if gate is not None:
            gate.wait(timeout=args.timeout)
        started = measured = time.perf_counter()
        deadline = started + args.timeout
        result["start_ms"] = (started - trial_start) * 1000
        parsed = urlsplit(args.url)
        connection_class = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_class(parsed.hostname, parsed.port, timeout=remaining())
        connection.connect()
        transport = connection.sock
        result["connect_ms"] = (time.perf_counter() - started) * 1000
        if args.exclude_connect:
            measured = time.perf_counter()
        result["stopwatch_start_ms"] = (measured - trial_start) * 1000
        headers = {"Content-Type": "application/json", "Accept": "audio/pcm", "Connection": "close"}
        if args.api_key_env:
            headers["Authorization"] = "Bearer " + os.environ[args.api_key_env]
        remaining()
        connection.request("POST", parsed.path or "/", json.dumps(body), headers)
        remaining()
        response = connection.getresponse()
        result["http_status"] = response.status
        result["headers_ms"] = (time.perf_counter() - measured) * 1000
        if response.status != 200:
            remaining()
            detail = response.read(8192).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {response.status}: {detail}")
        if response.getheader("Content-Type", "").split(";", 1)[0].strip().lower() != "audio/pcm":
            raise RuntimeError("response Content-Type is not audio/pcm")
        rate = response.getheader("x-audio-sample-rate")
        if rate is not None:
            if int(rate) != args.sample_rate:
                raise RuntimeError(
                    f"server sample rate {rate} differs from requested {args.sample_rate}"
                )
            result["sample_rate_source"] = "response_header"
        result["transfer_encoding"] = response.getheader("Transfer-Encoding")
        result["content_length"] = response.getheader("Content-Length")
        while True:
            remaining()
            try:
                data = response.read1(args.read_bytes)
            except http.client.IncompleteRead as error:
                observe(error.partial)
                raise
            if not data:
                break
            observe(data)
        if result["bytes"] == 0 or result["bytes"] % 2:
            raise RuntimeError("response contains empty or sample-misaligned PCM16")
        if (
            result["content_length"] is not None
            and int(result["content_length"]) != result["bytes"]
        ):
            raise RuntimeError("response ended before Content-Length bytes arrived")
        result["status"] = "COMPLETE"
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        result["completion_ms"] = (time.perf_counter() - measured) * 1000
        result["finish_ms"] = (time.perf_counter() - trial_start) * 1000
        if response is not None:
            response.close()
        if connection is not None:
            connection.close()
    result["audio_seconds"] = result["samples"] / args.sample_rate
    result["pcm_sha256"] = digest.hexdigest()
    result.update(
        reconstruct(
            result["arrivals"],
            args.codec_frame_samples,
            args.startup_chunk_frames,
            args.steady_chunk_frames,
            args.sample_rate,
        )
    )
    return result


def trial_summary(records, wall_seconds):
    complete = [record for record in records if record["status"] == "COMPLETE"]
    audio_seconds = sum(record["audio_seconds"] for record in records)
    complete_seconds = sum(record["audio_seconds"] for record in complete)
    return {
        "requests": len(records),
        "completed": len(complete),
        "failures": len(records) - len(complete),
        "completion_fraction": len(complete) / len(records) if records else 0,
        "wall_seconds_including_drain": wall_seconds,
        "received_audio_seconds": audio_seconds,
        "completed_audio_seconds": complete_seconds,
        "rtfx": audio_seconds / wall_seconds if wall_seconds else 0,
        "completed_rtfx": complete_seconds / wall_seconds if wall_seconds else 0,
        "requests_per_second": len(records) / wall_seconds if wall_seconds else 0,
        "completed_requests_per_second": len(complete) / wall_seconds if wall_seconds else 0,
        "ttfa_ms": distribution([record["ttfa_ms"] for record in complete]),
        "observed_ttfa_ms": distribution(
            [record["ttfa_ms"] for record in records if record["ttfa_ms"] is not None]
        ),
        "completion_ms": distribution([record["completion_ms"] for record in complete]),
        "observed_request_elapsed_ms": distribution(
            [record["completion_ms"] for record in records]
        ),
        "inter_frame_ms": distribution(
            [gap for record in complete for gap in record["inter_frame_ms"]]
        ),
        "inter_chunk_ms": distribution(
            [gap for record in complete for gap in record["inter_chunk_ms"]]
        ),
        "playback_deficit_responses": sum(
            record["playback_deficit_events"] > 0 for record in records
        ),
        "playback_deficit_events": sum(record["playback_deficit_events"] for record in records),
        "max_playback_deficit_ms": max(
            (record["max_playback_deficit_ms"] for record in records), default=0
        ),
    }


def run_trial(args, corpus, count, mode, executor, output):
    started = time.perf_counter()
    output.update(started_utc=datetime.now(timezone.utc).isoformat(), mode=mode, requests=[])
    futures = []
    try:
        if mode == "fixed-cohort":
            for begin in range(0, count, args.concurrency):
                indices = range(begin, min(count, begin + args.concurrency))
                gate = threading.Barrier(len(indices))
                wave = []
                for index in indices:
                    future = executor.submit(
                        request, args, corpus[index % len(corpus)], index, started, gate
                    )
                    futures.append(future)
                    wave.append(future)
                for future in wave:
                    future.result()
        else:
            for index in range(count):
                futures.append(
                    executor.submit(request, args, corpus[index % len(corpus)], index, started)
                )
            for future in futures:
                future.result()
    finally:
        for future in futures:
            output["requests"].append(future.result())
        output["summary"] = trial_summary(output["requests"], time.perf_counter() - started)
        output["planned_requests"] = count
        output["not_submitted"] = count - len(futures)


def summarize(trials):
    summaries = [trial["summary"] for trial in trials if "summary" in trial]
    result = {
        "trials": len(summaries),
        "aggregation": "equal mean of trial statistics; quantiles are not pooled",
    }
    if not summaries:
        return result
    for key in (
        "requests",
        "completed",
        "failures",
        "playback_deficit_responses",
        "playback_deficit_events",
    ):
        result[key] = sum(value[key] for value in summaries)
    for key in (
        "rtfx",
        "completed_rtfx",
        "requests_per_second",
        "completed_requests_per_second",
        "completion_fraction",
        "max_playback_deficit_ms",
    ):
        result[key] = statistics.fmean(value[key] for value in summaries)
    for key in (
        "ttfa_ms",
        "observed_ttfa_ms",
        "completion_ms",
        "observed_request_elapsed_ms",
        "inter_frame_ms",
        "inter_chunk_ms",
    ):
        present = [value[key] for value in summaries if value[key]["count"]]
        result[key] = {
            "contributing_trials": len(present),
            "observations": sum(value["count"] for value in present),
            **{
                stat: statistics.fmean(value[stat] for value in present) if present else None
                for stat in ("avg", "p50", "p95", "p99")
            },
        }
    return result


def save(path, report):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Portable HTTP PCM benchmark; timings differ from historical clients"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/audio/speech")
    parser.add_argument("--mode", choices=("closed-loop", "fixed-cohort"), default="closed-loop")
    parser.add_argument(
        "--width",
        type=int,
        help="declared physical backend width, recorded but not inferred from HTTP",
    )
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--num-requests", type=int, default=128, help="measured requests per trial")
    parser.add_argument("--warmup-waves", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    corpus = parser.add_mutually_exclusive_group()
    corpus.add_argument("--corpus-tsv", type=Path)
    corpus.add_argument(
        "--text", default="The quick brown fox jumps over the lazy dog near the riverbank."
    )
    parser.add_argument("--corpus-sha256")
    parser.add_argument("--model", default="nemotron-tts")
    parser.add_argument("--voice", default="english_sample")
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--apply-normalize", choices=("true", "false"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--read-bytes", type=int, default=65536)
    parser.add_argument("--codec-frame-samples", type=int, default=1764)
    parser.add_argument("--startup-chunk-frames", default="2,3,4,5")
    parser.add_argument("--steady-chunk-frames", type=int, default=8)
    parser.add_argument(
        "--exclude-connect",
        action="store_true",
        help="exclude TCP/TLS connect from request stopwatch, not trial wall time",
    )
    parser.add_argument(
        "--api-key-env",
        help="read a bearer token from this environment variable; token is not recorded",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        args.startup_chunk_frames = [int(value) for value in args.startup_chunk_frames.split(",")]
    except ValueError:
        parser.error("startup-chunk-frames must be comma-separated positive integers")
    for key in (
        "concurrency",
        "num_requests",
        "trials",
        "sample_rate",
        "read_bytes",
        "codec_frame_samples",
        "steady_chunk_frames",
    ):
        if getattr(args, key) <= 0:
            parser.error(key.replace("_", "-") + " must be positive")
    if args.width is not None and args.width <= 0:
        parser.error("width must be positive")
    if args.warmup_waves < 0 or any(value <= 0 for value in args.startup_chunk_frames):
        parser.error("warmup-waves must be nonnegative and chunk frame counts positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not 0 <= args.seed <= 2147483647:
        parser.error("timeout must be positive and seed must be in 0..2147483647")
    if args.num_requests < args.concurrency:
        parser.error("num-requests must be at least concurrency")
    parsed = urlsplit(args.url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        parser.error("url must be an HTTP(S) endpoint without credentials, query, or fragment")
    if args.api_key_env and not os.environ.get(args.api_key_env):
        parser.error("api-key-env names a missing or empty environment variable")
    if args.corpus_sha256 and args.corpus_tsv is None:
        parser.error("corpus-sha256 requires corpus-tsv")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.corpus_tsv:
        raw = args.corpus_tsv.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if args.corpus_sha256 and digest != args.corpus_sha256:
            raise ValueError("corpus SHA256 does not match the requested pin")
        corpus = []
        for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            identifier, separator, text = line.partition("\t")
            if not separator or not identifier or not text.strip():
                raise ValueError(f"invalid corpus row {number}: expected id<TAB>text")
            corpus.append({"id": identifier, "text": text})
    else:
        digest = None
        corpus = [{"id": "text", "text": args.text}]
    if not corpus or any(not value["text"].strip() for value in corpus):
        raise ValueError("corpus must contain nonempty text")
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    report = {
        "schema": "nemo-speech.portable-http-benchmark.v1",
        "status": "RUNNING",
        "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "configuration": config,
        "corpus_sha256": digest,
        "corpus": corpus,
        "warmup": [],
        "trials": [],
        "definitions": {
            "comparability": "distinct portable client; not interchangeable with measure_public_api.py or tts_perf_client",
            "ttfa": "request stopwatch to first complete PCM16 sample returned by HTTPResponse.read1",
            "request_stopwatch": (
                "after TCP/TLS connect" if args.exclude_connect else "before TCP/TLS connect"
            ),
            "connection_pooling": False,
            "request_queue": "stopwatch excludes executor queue and fixed-cohort barrier wait",
            "timeout": "overall request budget including connect, enforced at socket-operation/read boundaries",
            "fixed_cohort": "barrier before connect; next wave starts only after every current request terminates",
            "closed_loop": "bounded thread pool immediately refills completed requests; no wave barrier",
            "rtfx": "sum observed sample-derived audio seconds, including failed partial responses, divided by trial wall time through drain",
            "completed_rtfx": "sum audio seconds from successful responses divided by the same trial wall time",
            "latency_population": "ttfa/completion/ITL use completed requests; observed_ttfa/request_elapsed also retain failures",
            "quantiles": "nearest rank within trial; equal mean of trial quantiles in summary",
            "itl": "sample-threshold reconstruction, not server codec events; same read may complete multiple boundaries with zero interval",
            "inter_frame_ms": "interval between complete logical chunks divided by current chunk frame count",
            "partial_chunk": "retained in sample totals; excluded from reconstructed inter-chunk/frame latency",
            "playback": "playback starts at first aligned sample, pauses on underrun; deficits measured before adding new samples; no post-EOF silence",
            "memory": "not measured; collect GPU telemetry separately",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    executor = ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix="tts-bench")
    try:
        if args.warmup_waves:
            warmup = {"phase": "warmup"}
            report["warmup"].append(warmup)
            run_trial(
                args, corpus, args.warmup_waves * args.concurrency, "fixed-cohort", executor, warmup
            )
            save(args.output, report)
        for index in range(args.trials):
            measured = {"phase": "measured", "trial_index": index}
            report["trials"].append(measured)
            run_trial(args, corpus, args.num_requests, args.mode, executor, measured)
            report["summary"] = summarize(report["trials"])
            save(args.output, report)
            print(json.dumps({"trial": index, **measured["summary"]}), flush=True)
        warmup_failures = sum(value["summary"]["failures"] for value in report["warmup"])
        report["warmup_failures"] = warmup_failures
        report["status"] = "FAIL" if warmup_failures or report["summary"]["failures"] else "PASS"
    except BaseException as error:
        report["status"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "ERROR"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        executor.shutdown(wait=True)
        report["summary"] = summarize(report["trials"])
        save(args.output, report)
    print(
        json.dumps(
            {"status": report["status"], "output": str(args.output), "summary": report["summary"]}
        ),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
