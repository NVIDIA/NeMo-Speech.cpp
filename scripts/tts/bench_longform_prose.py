#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure long-form MagpieTTS speed on real English prose, by input length.

    scripts/tts/bench_longform_prose.py \
        --bin build/cuda-speech/bin/nemo-speech \
        --magpie MAGPIE.gguf --codec CODEC.gguf --tokenizer TOKENIZER_DIR \
        --baseline /path/to/stock/nemo-speech

Why this exists alongside bench_magpietts.py. That benchmark repeats five
sentences, all eleven words or longer, which is convenient but not what
long-form input looks like. Real prose has a tenth percentile around six words
and a long tail past thirty, and that distribution changes two things the
repeated text hides:

  * per-chunk decode length varies a lot, so a wave -- which steps its whole
    group in lockstep until the slowest member finishes -- pays for its longest
    chunk. Measured max/mean per-chunk steps is about 1.26 on the repeated text
    and 1.9 to 3.4 here.
  * short sentences make short chunks, which used to crash long-form outright
    when the following chunk asked for more history than its predecessor left.

Sentences come from four Project Gutenberg texts, pinned by sha256 and cached
under --cache. Gatsby is in the set deliberately: the three 19th-century
sources skew long on their own, and 1920s prose pulls the mean back toward
contemporary English. The pool is ~12k sentences at mean 17.1 words, median 15,
which is the usual figure quoted for English prose.

Sampling is seeded, so a given --seed and size list always produce the same
text. Report RTF (compute-seconds per audio-second, lower is better), realtime
multiple, and time to first audio. With --baseline, also the speed-up.
"""
import argparse
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request

SOURCES = [
    (
        "pg64317.txt",
        "https://www.gutenberg.org/cache/epub/64317/pg64317.txt",
        "ce760ec377accd352b41bb8f64504a72d7aa18ab3afb42ded2b56cecacf29e35",
    ),
    (
        "pg1661.txt",
        "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
        "922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0",
    ),
    (
        "1342-0.txt",
        "https://www.gutenberg.org/files/1342/1342-0.txt",
        "81300b79e8a8d65ac530a97578417d06137e3bbc90622a10a65e5036183d2500",
    ),
    (
        "pg11.txt",
        "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e",
    ),
]
ABBREV = re.compile(r"(\b(Mr|Mrs|Ms|Dr|St|Prof|Rev|Hon|Jr|Sr|vs|etc|Co|Inc)|\b[A-Z])\.$")


def fetch(cache):
    os.makedirs(cache, exist_ok=True)
    paths = []
    for name, url, want in SOURCES:
        p = os.path.join(cache, name)
        if not os.path.exists(p):
            sys.stderr.write(f"fetching {name}\n")
            urllib.request.urlretrieve(url, p)
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if got != want:
            sys.exit(f"{p}: sha256 {got}, expected {want}. Delete it and re-run.")
        paths.append(p)
    return paths


def sentences(path):
    raw = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw, re.S)
    if m:
        raw = raw[m.end() :]
    m = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", raw)
    if m:
        raw = raw[: m.start()]
    for a, b in (("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), ("—", " "), ("–", " ")):
        raw = raw.replace(a, b)
    out = []
    for para in re.sub(r"\n{2,}", "\n\n", raw).split("\n\n"):
        para = " ".join(para.split())
        if not para or para.isupper():
            continue
        buf = ""
        for part in re.split(r"(?<=[.!?])\s+", para):
            buf = (buf + " " + part).strip() if buf else part
            if ABBREV.search(buf):  # an abbreviation's period is not a sentence end
                continue
            out.append(buf)
            buf = ""
    return out


def usable(s):
    w = s.split()
    return (
        4 <= len(w) <= 45
        and s.endswith((".", "!", "?"))
        and s.count('"') % 2 == 0
        and not re.search(r"[0-9_\[\]{}<>*]|[^\x00-\x7f]", s)
        and sum(c.isupper() for c in s) <= len(s) * 0.3
    )


def pool(cache):
    seen, out = set(), []
    for p in fetch(cache):
        for s in sentences(p):
            s = s.strip().strip('"').strip()
            if usable(s) and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def gpu_busy():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout
        vals = [int(re.sub(r"\D", "", l)) for l in out.splitlines() if re.search(r"\d", l)]
        return max(vals) if vals else 100
    except Exception:
        return 0


def wait_idle(threshold=3, timeout=600):
    ok, deadline = 0, time.time() + timeout
    while time.time() < deadline:
        ok = ok + 1 if gpu_busy() <= threshold else 0
        if ok >= 4:
            return
        time.sleep(1)


def run(a, binary, text, extra):
    if a.device.startswith("cuda") and not a.allow_busy_gpu:
        wait_idle()
    cmd = [
        binary,
        "synthesize",
        text,
        *extra,
        "--device",
        a.device,
        "--voice",
        a.voice,
        "--seed",
        str(a.seed),
        "--output",
        a.out,
        "--force",
        "--verbose",
        "--top-k",
        "1",
        "--tts.magpie-model",
        a.magpie,
        "--tts.codec-model",
        a.codec,
        "--tts.tokenizer-model-dir",
        a.tokenizer,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "no output"
        return {"failed": tail}
    got = {}
    for m in ("e2e_rtf", "e2e_audio_s", "e2e_ttfa_ms"):
        hit = re.findall(rf"{m}=([0-9.]+)", p.stderr)
        if hit:
            got[m] = float(hit[-1])
    tok = re.findall(r"encoding (\d+) text tokens in (\d+) chunk", p.stderr)
    if tok:
        got["tokens"], got["chunks"] = int(tok[-1][0]), int(tok[-1][1])
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--baseline", help="a second binary to compare against, e.g. stock")
    ap.add_argument("--magpie", required=True)
    ap.add_argument("--codec", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--voice", default="0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sizes", default="1,2,4,8,16,32,64,128", help="sentences per case")
    ap.add_argument("--width", type=int, default=32, help="--tts.batch-size for the test arm")
    ap.add_argument("--chunk-frames", type=int, default=32)
    ap.add_argument("--history", type=int, default=20)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/nemo-speech/prose"))
    ap.add_argument("--out", default="/tmp/bench_longform_prose.wav")
    ap.add_argument("--json", help="write raw per-run metrics here")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    a = ap.parse_args()
    if a.reps < 1:
        ap.error("--reps must be at least 1")

    p = pool(a.cache)
    lens = sorted(len(s.split()) for s in p)
    sys.stderr.write(
        f"pool {len(p)} sentences, mean {statistics.mean(lens):.1f} words, "
        f"median {statistics.median(lens):.0f}, p90 {lens[9 * len(lens) // 10]}\n"
    )

    ours = [
        "--tts.batch-size",
        str(a.width),
        "--tts.longform-history-tokens",
        str(a.history),
        "--tts.chunk-frames",
        str(a.chunk_frames),
    ]
    base = ["--tts.chunk-frames", str(a.chunk_frames)]

    print(
        f"{'sent':>5} {'tokens':>7} {'chunks':>7} {'audio':>8} {'RTF':>8} {'xRT':>7} {'TTFA':>7}"
        + (f" {'base RTF':>9} {'speed-up':>9}" if a.baseline else "")
    )
    raw = []
    for n in [int(x) for x in a.sizes.split(",")]:
        text = " ".join(random.Random(a.seed).sample(p, n))
        arms = [("ours", a.bin, ours)] + ([("base", a.baseline, base)] if a.baseline else [])
        med = {}
        for label, binary, extra in arms:
            runs = [run(a, binary, text, extra) for _ in range(a.reps)]
            raw.append({"n": n, "arm": label, "runs": runs})
            ok = [r for r in runs if "e2e_rtf" in r]
            med[label] = ok and {k: statistics.median(r[k] for r in ok) for k in ok[0]} or None
        o = med["ours"]
        if not o:
            print(f"{n:5d} {'--':>7} {'--':>7}   FAILED: {runs[-1].get('failed', '')[:60]}")
            continue
        line = (
            f"{n:5d} {int(o['tokens']):7d} {int(o['chunks']):7d} {o['e2e_audio_s']:7.1f}s "
            f"{o['e2e_rtf']:8.4f} {1 / o['e2e_rtf']:6.0f}x {o['e2e_ttfa_ms']:6.0f}ms"
        )
        if a.baseline:
            b = med.get("base")
            line += (
                f" {b['e2e_rtf']:9.4f} {b['e2e_rtf'] / o['e2e_rtf']:8.2f}x"
                if b
                else f" {'CRASH':>9} {'--':>9}"
            )
        print(line)
    if a.json:
        json.dump(raw, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
