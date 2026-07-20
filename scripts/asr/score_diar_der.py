#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DER smoke test for the ggml Sortformer diarizer.

Runs the C++ streaming CLI (test_diar_streaming --rttm) on a set of wavs,
scores DER against reference RTTMs with pyannote.metrics, and (optionally)
runs NeMo's streaming Sortformer with the same geometry + binarization for an
apples-to-apples delta.

Usage:
    python score_diar_der.py --cli <build>/bin/test_diar_streaming \
        --model sortformer-v2-f32.gguf \
        --wav-dir /path/to/audio \
        --ref /path/to/reference.rttm \
        [--nemo-ckpt diar_streaming_sortformer_4spk-v2.nemo --device cuda] \
        [--gpu] [--max-files N] [--collar 0.25]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

# Geometry presets + segmentation postprocessing, mirroring the C++ single
# sources (keep in sync):
#   riva/asr/diar/aosc_state.h    DiarGeometry::riva_streaming()/riva_offline()
#   riva/asr/diar/diar_pipeline.h DiarSegmentationCfg defaults
# Both the C++ CLI and the NeMo baseline are driven from the same dict below,
# so the ggml-vs-NeMo delta is apples-to-apples by construction.
GEOMETRY_PRESETS = {
    "streaming": dict(chunk=20, lc=0, rc=0, fifo=80, spkcache=160, update=80),
    "offline": dict(chunk=100, lc=0, rc=0, fifo=100, spkcache=312, update=100),
}
DEFAULT_GEOMETRY = "streaming"
POSTPROC_CALLHOME = dict(
    onset=0.641,
    offset=0.561,
    pad_onset=0.229,
    pad_offset=0.079,
    min_duration_on=0.511,
    min_duration_off=0.296,
)
SEC_PER_FRAME = 0.08


def geometry_cli_args(geom: dict) -> list[str]:
    """test_diar_streaming flags for a geometry dict."""
    return [
        "--chunk",
        str(geom["chunk"]),
        "--lc",
        str(geom["lc"]),
        "--rc",
        str(geom["rc"]),
        "--fifo",
        str(geom["fifo"]),
        "--spkcache",
        str(geom["spkcache"]),
        "--update",
        str(geom["update"]),
    ]


def apply_geometry_to_nemo(sortformer_modules, geom: dict) -> None:
    """Configure NeMo's SortformerModules for the same streaming geometry."""
    sortformer_modules.chunk_len = geom["chunk"]
    sortformer_modules.chunk_left_context = geom["lc"]
    sortformer_modules.chunk_right_context = geom["rc"]
    sortformer_modules.fifo_len = geom["fifo"]
    sortformer_modules.spkcache_len = geom["spkcache"]
    sortformer_modules.spkcache_update_period = geom["update"]
    sortformer_modules._check_streaming_parameters()


def parse_rttm_lines(lines):
    ann_by_file: dict[str, Annotation] = {}
    for ln in lines:
        parts = ln.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        uri, t0, dur, spk = parts[1], float(parts[3]), float(parts[4]), parts[7]
        ann = ann_by_file.setdefault(uri, Annotation(uri=uri))
        ann[Segment(t0, t0 + dur)] = spk
    return ann_by_file


def probs_to_annotation(probs, uri, pp=POSTPROC_CALLHOME) -> Annotation:
    """NumPy (T, n_spk) frame probs -> Annotation, DiarStream::segments() rules
    (onset/offset hysteresis -> pad -> fill short gaps -> drop short segments)."""
    ann = Annotation(uri=uri)
    n, n_spk = probs.shape
    total = n * SEC_PER_FRAME
    for s in range(n_spk):
        segs = []
        active, start = False, 0
        for f in range(n):
            p = probs[f, s]
            if not active and p > pp["onset"]:
                active, start = True, f
            elif active and p < pp["offset"]:
                active = False
                segs.append(
                    (
                        max(0.0, start * SEC_PER_FRAME - pp["pad_onset"]),
                        min(total, f * SEC_PER_FRAME + pp["pad_offset"]),
                    )
                )
        if active:
            segs.append((max(0.0, start * SEC_PER_FRAME - pp["pad_onset"]), total))
        merged = []
        for t0, t1 in segs:
            if merged and t0 - merged[-1][1] < pp["min_duration_off"]:
                merged[-1][1] = max(merged[-1][1], t1)
            else:
                merged.append([t0, t1])
        for t0, t1 in merged:
            if t1 - t0 >= pp["min_duration_on"]:
                ann[Segment(t0, t1)] = f"spk{s}"
    return ann


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True, help="test_diar_streaming binary")
    ap.add_argument("--model", required=True, help="sortformer GGUF")
    ap.add_argument("--wav-dir", action="append", required=True)
    ap.add_argument("--ref", required=True, help="reference RTTM (multi-file)")
    ap.add_argument("--nemo-ckpt", default=None, help=".nemo for the NeMo baseline")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", action="store_true", help="run the C++ CLI on GPU")
    ap.add_argument(
        "--preset",
        default=DEFAULT_GEOMETRY,
        choices=sorted(GEOMETRY_PRESETS),
        help="streaming geometry applied to BOTH the C++ CLI and the NeMo "
        "baseline (single source: diar_presets.py)",
    )
    ap.add_argument(
        "--cli-arg",
        action="append",
        default=[],
        help="extra args passed to the CLI (repeat; e.g. --cli-arg=--chunk --cli-arg=20)",
    )
    ap.add_argument("--quiet-per-file", action="store_true")
    ap.add_argument("--max-files", type=int, default=0)
    # NIST/md-eval convention: +-collar seconds around each reference boundary.
    # pyannote's `collar` is the TOTAL window removed, so pass 2x below.
    ap.add_argument("--collar", type=float, default=0.25)
    ap.add_argument(
        "--skip-overlap",
        action="store_true",
        help="exclude reference overlap regions from scoring (md-eval -1 style)",
    )
    args = ap.parse_args()

    refs = parse_rttm_lines(Path(args.ref).read_text().splitlines())
    wavs = []
    for d in args.wav_dir:
        wavs.extend(sorted(Path(d).glob("*.wav")))
    wavs = [w for w in wavs if w.stem in refs]
    if args.max_files > 0:
        wavs = wavs[: args.max_files]
    if not wavs:
        print("no wavs matched the reference RTTM", file=sys.stderr)
        return 1
    print(f"[der] scoring {len(wavs)} files (collar {args.collar})")

    geom = GEOMETRY_PRESETS[args.preset]
    metric_ours = DiarizationErrorRate(collar=2 * args.collar, skip_overlap=args.skip_overlap)
    for w in wavs:
        cmd = [args.cli, args.model, str(w), "--rttm", w.stem]
        cmd += geometry_cli_args(geom) + args.cli_arg
        if args.gpu:
            cmd.append("--gpu")
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        hyp = parse_rttm_lines(out.stdout.splitlines()).get(w.stem, Annotation(uri=w.stem))
        d = metric_ours(refs[w.stem], hyp)
        if not args.quiet_per_file:
            print(f"[der]   ggml {w.stem[:16]}...: {d:.4f}")
    der_ours = abs(metric_ours)
    print(f"[der] ggml overall DER: {der_ours:.4f}")

    if args.nemo_ckpt:
        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel

        device = torch.device(args.device)
        model = SortformerEncLabelModel.restore_from(
            restore_path=args.nemo_ckpt, map_location=device
        )
        model.eval().to(device)
        # Same geometry as the C++ side, from the shared preset table.
        apply_geometry_to_nemo(model.sortformer_modules, geom)
        model.preprocessor.featurizer.dither = 0.0
        model.streaming_mode = True

        import soundfile as sf

        metric_nemo = DiarizationErrorRate(collar=2 * args.collar, skip_overlap=args.skip_overlap)
        with torch.inference_mode():
            for w in wavs:
                audio, sr = sf.read(str(w), dtype="float32")
                if audio.ndim > 1:
                    audio = audio[:, 0]
                sig = torch.from_numpy(audio).unsqueeze(0).to(device)
                sig_len = torch.tensor([sig.shape[1]], device=device)
                preds = model.forward(audio_signal=sig, audio_signal_length=sig_len)
                hyp = probs_to_annotation(preds[0].cpu().numpy(), w.stem)
                d = metric_nemo(refs[w.stem], hyp)
                print(f"[der]   nemo {w.stem[:16]}...: {d:.4f}")
        der_nemo = abs(metric_nemo)
        print(f"[der] nemo overall DER: {der_nemo:.4f}")
        print(f"[der] delta (ggml - nemo): {der_ours - der_nemo:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
