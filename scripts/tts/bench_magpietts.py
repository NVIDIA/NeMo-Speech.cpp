#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure MagpieTTS synthesis speed, and check that it is reproducible.

Self-contained: the text is inlined below, so a run is reproducible from this
file plus a binary and the model files.

    scripts/tts/bench_magpietts.py \
        --bin build/cuda-speech/bin/nemo-speech \
        --magpie MAGPIE.gguf --codec CODEC.gguf --tokenizer TOKENIZER_DIR

Three cases of increasing length -- one line, a five-sentence paragraph, a
twenty-sentence script -- because RTF does not scale flatly with input length
and a single short utterance hides both long-form and codec behaviour.

Two things about the methodology are worth knowing before trusting a number.

**Use --greedy for any A/B comparison.** Sampled decoding picks a different code
sequence when anything perturbs the arithmetic, which changes the audio
*duration*, which moves RTF on its own. Greedy holds the sequence fixed, so RTF
is comparable between arms and the output is byte-reproducible.

**The GPU must be idle.** A background job on the same device moves these
numbers by 2-4x. This script refuses to start until nvidia-smi reports <= 3%
for four consecutive samples, and re-checks between cases.

`--check` skips timing and instead runs each case N times, reporting the sha256
of the decoded WAV. Under --greedy every run must produce the same hash; if they
differ, the pipeline is not reproducible and no benchmark taken against it means
anything. That is not hypothetical -- see the half_snake fusion aliasing guard.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

# The sentence pool. Do not reword or reorder: the sentences and their order
# determine audio duration, and comparability across runs depends on them.
SENTENCES = [
    "Magpie is a text to speech model.",
    "It generates audio codes autoregressively and then decodes them with a neural codec.",
    "This paragraph is long enough that the server treats it as long form input.",
    "We want to know how the wall clock time scales with the number of sentences.",
    "Here is more filler text to push us well past the language threshold.",
]
CASES = [("line", 1), ("paragraph", 5), ("script", 20)]

METRICS = (
    "e2e_rtf",
    "decoder_itl_avg_ms",
    "codec_rtfx",
    "codec_ttfa_ms",
    "e2e_ttfa_ms",
    "e2e_audio_s",
)


def text_for(n):
    return " ".join((SENTENCES * 8)[:n])


def gpu_util():
    # The busiest GPU on the box, not the first row. synthesize does not take a CUDA
    # index -- it initializes the first available device -- and CUDA_VISIBLE_DEVICES can
    # map that onto any physical GPU, so reading row 0 can approve an idle card while
    # the one running the synthesis is saturated.
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    vals = [int(re.sub(r"\D", "", line)) for line in out.splitlines() if re.search(r"\d", line)]
    return max(vals) if vals else 100


def wait_idle(threshold=3, consecutive=4, timeout=600):
    ok, deadline = 0, time.time() + timeout
    while time.time() < deadline:
        ok = ok + 1 if gpu_util() <= threshold else 0
        if ok >= consecutive:
            return True
        time.sleep(2)
    return False


def run_once(a, text, out_path):
    cmd = [
        a.bin,
        "synthesize",
        text,
        "--device",
        a.device,
        "--voice",
        a.voice,
        "--seed",
        str(a.seed),
        "--output",
        out_path,
        "--force",
        "--verbose",
        "--tts.magpie-model",
        a.magpie,
        "--tts.codec-model",
        a.codec,
        "--tts.tokenizer-model-dir",
        a.tokenizer,
    ]
    if a.greedy:
        cmd += ["--top-k", "1"]
    # --extra goes first: command_synthesize applies options in argument order, so
    # anything the benchmark owns has to be appended after it. A duplicated --output
    # would otherwise make the run hash a different file than the one it wrote.
    cmd = cmd[:3] + a.extra + cmd[3:]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"synthesize failed:\n{p.stderr[-2000:]}")
    got = {}
    for m in METRICS:
        hit = re.findall(rf"{m}=([0-9.]+)", p.stderr)
        if hit:
            got[m] = float(hit[-1])
    if "e2e_rtf" not in got:
        sys.exit("no e2e_rtf in output; is this build --verbose capable?")
    return got


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--magpie", required=True)
    ap.add_argument("--codec", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--voice", default="John")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--reps", type=int, default=5, help="runs per case; must be at least 1")
    ap.add_argument("--out", default="/tmp/bench_magpietts.wav")
    ap.add_argument("--label", default="run")
    ap.add_argument("--json", help="write raw per-run metrics here")
    ap.add_argument(
        "--greedy",
        action="store_true",
        help="--top-k 1: holds the code sequence fixed and makes output reproducible",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="report the sha256 of each run's WAV instead of timing it",
    )
    ap.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="skip the idle check (results will not be comparable)",
    )
    ap.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="everything after this is passed to synthesize",
    )
    a = ap.parse_args()
    # Zero or fewer leaves every metric list empty, and the median map then has no
    # e2e_rtf for the report to read.
    if a.reps < 1:
        ap.error("--reps must be at least 1")

    if not a.allow_busy_gpu and a.device.startswith("cuda") and not wait_idle():
        sys.exit("GPU never went idle; refusing to benchmark under load")

    if a.check:
        if not a.greedy:
            print(
                "warning: --check without --greedy; sampled output need not repeat", file=sys.stderr
            )
        print(f"{a.label}: {a.reps} runs per case, checking reproducibility")
        failures = 0
        for name, n in CASES:
            digests = []
            for _ in range(a.reps):
                run_once(a, text_for(n), a.out)
                digests.append(sha256(a.out))
            unique = sorted(set(digests))
            ok = len(unique) == 1
            failures += not ok
            print(
                f"  {name:10s} {'OK  ' if ok else 'FAIL'} "
                f"{len(unique)} distinct output(s) over {a.reps} runs"
            )
            for d in unique:
                print(f"      {d[:16]}  x{digests.count(d)}")
        if failures:
            print(f"{failures} case(s) not reproducible", file=sys.stderr)
        return 1 if failures else 0

    raw = {}
    print(f"{a.label}: {a.reps} reps per case, seed {a.seed}" f"{', greedy' if a.greedy else ''}")
    for name, n in CASES:
        if not a.allow_busy_gpu and a.device.startswith("cuda"):
            wait_idle()
        runs = [run_once(a, text_for(n), a.out) for _ in range(a.reps)]
        raw[name] = runs
        med = {
            m: statistics.median([r[m] for r in runs if m in r])
            for m in METRICS
            if any(m in r for r in runs)
        }
        print(
            f"  {name:10s} rtf {med['e2e_rtf']:.4f}"
            f"   itl {med.get('decoder_itl_avg_ms', float('nan')):.2f} ms"
            f"   codec_rtfx {med.get('codec_rtfx', float('nan')):.1f}"
            f"   codec_ttfa {med.get('codec_ttfa_ms', float('nan')):.2f} ms"
            f"   audio {med.get('e2e_audio_s', float('nan')):.1f} s"
        )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(
                {"label": a.label, "reps": a.reps, "seed": a.seed, "greedy": a.greedy, "runs": raw},
                f,
                indent=1,
            )
        print(f"  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
