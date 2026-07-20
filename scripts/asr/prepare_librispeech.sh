#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Download and stage a LibriSpeech subset for ASR evaluation. The script walks
# LibriSpeech's per-book `*.trans.txt` files and emits a NeMo-style
# `transcripts.json` manifest with one `audio_filepath` and `text` object per
# line.
#
# Output layout (under $OUT_DIR/<subset_id>/):
#   <spk>/<book>/<spk>-<book>-<utt>.wav   # 16 kHz mono s16le, decoded from flac
#   transcripts.json                       # NeMo manifest, paths relative to here
#
# Usage:
#   scripts/asr/prepare_librispeech.sh test-clean [test-other dev-clean dev-other ...]
#
# Env knobs:
#   OUT_DIR     where to stage datasets (default $ASR_DATASETS_DIR or /tmp/asr_datasets)
#   KEEP_FLAC   1 = don't delete .flac after wav conversion (default 0)
set -euo pipefail

OUT_DIR=${OUT_DIR:-${ASR_DATASETS_DIR:-/tmp/asr_datasets}}
KEEP_FLAC=${KEEP_FLAC:-0}

if [[ "$#" -lt 1 ]]; then
    cat >&2 <<EOF
usage: $0 <subset> [<subset> ...]
subsets: test-clean | test-other | dev-clean | dev-other | train-clean-100 | all
EOF
    exit 1
fi

command -v flac     >/dev/null || { echo "ERROR: flac not on PATH (apt install flac)"     >&2; exit 1; }
command -v parallel >/dev/null || { echo "ERROR: parallel not on PATH (apt install parallel)" >&2; exit 1; }

declare -A SUBSETS=(
    [test-clean]=test-clean.tar.gz
    [test-other]=test-other.tar.gz
    [dev-clean]=dev-clean.tar.gz
    [dev-other]=dev-other.tar.gz
    [train-clean-100]=train-clean-100.tar.gz
)

args=("$@")
if [[ "${args[0]}" == "all" ]]; then
    args=(test-clean test-other dev-clean dev-other)
fi

mkdir -p "${OUT_DIR}"

for subset in "${args[@]}"; do
    tarball=${SUBSETS[${subset}]:-}
    if [[ -z "${tarball}" ]]; then
        echo "ERROR: unknown subset '${subset}'" >&2
        exit 2
    fi
    target_dir="${OUT_DIR}/librispeech-${subset}"
    manifest="${target_dir}/transcripts.json"

    if [[ -f "${manifest}" ]]; then
        echo "[${subset}] manifest already at ${manifest} — skipping"
        continue
    fi

    work_dir="${OUT_DIR}/.work-${subset}"
    mkdir -p "${work_dir}"
    if [[ ! -d "${work_dir}/LibriSpeech/${subset}" ]]; then
        echo "[${subset}] downloading https://www.openslr.org/resources/12/${tarball}"
        wget -q --show-progress -O "${work_dir}/${tarball}" \
            "https://www.openslr.org/resources/12/${tarball}"
        echo "[${subset}] extracting"
        tar -xf "${work_dir}/${tarball}" -C "${work_dir}"
        rm -f "${work_dir}/${tarball}"
    else
        echo "[${subset}] already extracted at ${work_dir}/LibriSpeech/${subset}"
    fi

    src="${work_dir}/LibriSpeech/${subset}"
    flac_count=$(find "${src}" -name "*.flac" | wc -l)
    echo "[${subset}] decoding ${flac_count} FLAC → WAV (16kHz mono s16le)"
    # LibriSpeech FLAC is 16 kHz mono s16le natively, so straight flac -d works.
    find "${src}" -name "*.flac" | parallel --no-notice "flac -s -d {}"
    if [[ "${KEEP_FLAC}" != "1" ]]; then
        find "${src}" -name "*.flac" -delete
    fi

    # Stage final layout: <target>/<spk>/<book>/<spk>-<book>-<utt>.wav
    mkdir -p "${target_dir}"
    rsync -a --include='*/' --include='*.wav' --exclude='*' "${src}/" "${target_dir}/"

    # Build NeMo manifest from per-book trans.txt files.
    # Each line of <spk>-<book>.trans.txt is: "<utt_id> TRANSCRIPT TEXT HERE"
    # where <utt_id> == <spk>-<book>-<utt>.
    echo "[${subset}] building transcripts.json"
    python3 - "${src}" "${manifest}" "${target_dir}" <<'PY'
import json, sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
base = Path(sys.argv[3])
n = 0
with out.open('w') as fh:
    for trans in sorted(src.rglob('*.trans.txt')):
        # Path.stem strips only one extension; '*.trans.txt' has two.
        spk_book = trans.name[: -len('.trans.txt')]    # e.g. 1272-135031
        spk, book = spk_book.split('-')
        for line in trans.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, _, text = line.partition(' ')
            abs_path = str(base / spk / book / f"{utt_id}.wav")
            fh.write(json.dumps({'audio_filepath': abs_path, 'text': text.lower()}) + '\n')
            n += 1
print(f'  wrote {n} entries to {out}')
PY

    rm -rf "${work_dir}"
    echo "[${subset}] ready: ${target_dir} (manifest: ${manifest})"
done

echo "done."
