#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Apply the NeMo-Speech.cpp llama.cpp patch series to the pinned submodule.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${ROOT}/llama.cpp"
PATCHES="${ROOT}/llama-patches"
source "${ROOT}/scripts/patch-series-common.sh"

if [ ! -d "${LLAMA}/src" ]; then
    echo "error: llama.cpp submodule not initialized at ${LLAMA}" >&2
    echo "       run: git submodule update --init llama.cpp" >&2
    exit 1
fi

shopt -s nullglob
patch_files=("${PATCHES}"/*.patch)
if [ "${#patch_files[@]}" -eq 0 ]; then
    echo "error: no .patch files found in ${PATCHES}" >&2
    exit 1
fi

if git -c "safe.directory=${LLAMA}" -C "${LLAMA}" rev-parse --git-dir >/dev/null 2>&1; then
    tmp_dir="$(mktemp -d)"
    expected_index="${tmp_dir}/expected.index"
    current_index="${tmp_dir}/current.index"
    cleanup_indexes() {
        rm -rf "${tmp_dir}"
    }
    trap cleanup_indexes EXIT

    GIT_INDEX_FILE="${expected_index}" git -c "safe.directory=${LLAMA}" -C "${LLAMA}" read-tree HEAD
    for patch in "${patch_files[@]}"; do
        if ! GIT_INDEX_FILE="${expected_index}" git -c "safe.directory=${LLAMA}" -C "${LLAMA}" apply --cached "${patch}"; then
            echo "error: $(basename "${patch}") does not apply to the pinned llama.cpp commit" >&2
            exit 1
        fi
    done

    mapfile -d '' patched_paths \
        < <(GIT_INDEX_FILE="${expected_index}" git -c "safe.directory=${LLAMA}" -C "${LLAMA}" \
            diff --cached --name-only -z HEAD)
    GIT_INDEX_FILE="${current_index}" git -c "safe.directory=${LLAMA}" -C "${LLAMA}" read-tree HEAD
    if [ "${#patched_paths[@]}" -ne 0 ]; then
        GIT_INDEX_FILE="${current_index}" git -c "safe.directory=${LLAMA}" -C "${LLAMA}" add -A -- "${patched_paths[@]}"
    fi
    expected_tree="$(GIT_INDEX_FILE="${expected_index}" git -c "safe.directory=${LLAMA}" -C "${LLAMA}" write-tree)"
    current_tree="$(GIT_INDEX_FILE="${current_index}" git -c "safe.directory=${LLAMA}" -C "${LLAMA}" write-tree)"
    if [ "${current_tree}" = "${expected_tree}" ]; then
        echo "[llama-patch] current series already applied"
        echo "[llama-patch] done"
        exit 0
    fi
else
    # Docker build contexts intentionally omit the submodule's .git file. A
    # per-patch reverse check is insufficient when a later patch changes a
    # hunk introduced by an earlier one, so validate the complete applied
    # series against a temporary index, in reverse order. This never changes
    # the copied source tree.
    if patch_series_applied_without_git "${LLAMA}" "${PATCHES}" "[llama-patch]"; then
        echo "[llama-patch] done"
        exit 0
    fi
fi

for patch in "${patch_files[@]}"; do
    name="$(basename "${patch}")"
    if git -c "safe.directory=${LLAMA}" -C "${LLAMA}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
        echo "[llama-patch] ${name}: already applied"
        continue
    fi
    if ! git -c "safe.directory=${LLAMA}" -C "${LLAMA}" apply --check "${patch}" >/dev/null 2>&1; then
        echo "[llama-patch] ${name}: does NOT apply cleanly to current llama.cpp" >&2
        echo "              restore the pinned submodule, then retry" >&2
        exit 1
    fi
    git -c "safe.directory=${LLAMA}" -C "${LLAMA}" apply "${patch}"
    echo "[llama-patch] ${name}: applied"
done
echo "[llama-patch] done"
