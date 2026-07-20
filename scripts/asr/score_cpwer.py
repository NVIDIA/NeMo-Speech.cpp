#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cpWER for the ggml ASR+Sortformer pipeline (meeteval).

Drives test_diar_recognizer --seglst over a wav directory, assembles the
hypothesis SegLST, and scores concatenated-minimum-permutation WER against a
reference seglst.json.

Usage:
    python score_cpwer.py --cli <build>/bin/test_diar_recognizer \
        --asr nemotron-speech-streaming-en-0.6b.q8_0.gguf \
        --diar sortformer-v2-f32.gguf \
        --wav-dir /path/to/audio \
        --ref /path/to/reference.seglst.json \
        [--gpu N] [--out hyp.seglst.json] [--max-files N]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import meeteval
from meeteval.io.seglst import SegLST


def normalize_text(s: str) -> str:
    """Match the reference's text normalization: lowercase, strip punctuation
    (keep intra-word apostrophes), collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True, help="test_diar_recognizer binary")
    ap.add_argument("--asr", required=True, help="ASR GGUF")
    ap.add_argument("--diar", required=True, help="Sortformer GGUF")
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--ref", required=True, help="reference seglst.json")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", default=None, help="write the hypothesis seglst.json here")
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    ref = SegLST.load(args.ref)
    sessions = sorted({e["session_id"] for e in ref})
    wavs = {p.stem: p for p in Path(args.wav_dir).glob("*.wav")}
    missing = [s for s in sessions if s not in wavs]
    if missing:
        print(f"[cpwer] WARNING: no wav for sessions {missing}", file=sys.stderr)
    sessions = [s for s in sessions if s in wavs]
    if args.max_files > 0:
        sessions = sessions[: args.max_files]
    print(f"[cpwer] {len(sessions)} sessions")

    hyp_entries = []
    for s in sessions:
        cmd = [args.cli, args.asr, args.diar, str(wavs[s]), "--seglst", s]
        if args.gpu is not None:
            cmd += ["--gpu", str(args.gpu)]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        n_seg = 0
        for ln in out.stdout.splitlines():
            if not ln.startswith("SEG\t"):
                continue
            _, session, spk, t0, t1, text = ln.split("\t", 5)
            hyp_entries.append(
                {
                    "session_id": session,
                    "speaker": spk,
                    "start_time": float(t0),
                    "end_time": float(t1),
                    "words": normalize_text(text),
                }
            )
            n_seg += 1
        print(f"[cpwer]   {s}: {n_seg} segments")

    hyp = SegLST(hyp_entries)
    if args.out:
        Path(args.out).write_text(json.dumps(hyp_entries, indent=1))

    # Reference text is already normalized ("txtnormed"), but run it through
    # the same normalizer so stray punctuation can't skew the comparison.
    ref = ref.map(lambda e: {**e, "words": normalize_text(str(e["words"]))})

    per_session = meeteval.wer.cpwer(ref, hyp)
    from meeteval.wer import combine_error_rates

    combined = combine_error_rates(per_session)
    for sid, er in sorted(per_session.items()):
        print(f"[cpwer]   {sid}: {er.error_rate:.4f} ({er.errors}/{er.length})")
    print(
        f"[cpwer] overall cpWER: {combined.error_rate:.4f} "
        f"(errors {combined.errors} / length {combined.length}; "
        f"ins {combined.insertions} del {combined.deletions} sub {combined.substitutions}; "
        f"missed_spk {combined.missed_speaker} falarm_spk {combined.falarm_speaker})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
