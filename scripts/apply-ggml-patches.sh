#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Apply the in-tree ggml patches (ggml-patches/*.patch, in filename order) onto
# the vendored ggml submodule. This keeps the submodule pinned to clean
# upstream; our project-specific ggml changes (fused rel-pos attention, NVFP4
# quantization, norm fusion, dw-conv F16, skinny-q8 GEMM, FastConformer BF16
# fusions, large-batch CUDA fixes, and MagpieTTS/NanoCodec CUDA ops) are applied
# at build setup time.
#
# Patches that still reverse-apply cleanly are skipped. Apply the complete
# series to a clean submodule for deterministic setup; later patches may refine
# lines introduced by earlier patches, making reverse detection ambiguous on a
# fully patched tree.
# Usage: scripts/apply-ggml-patches.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GGML="${ROOT}/ggml"
PATCHES="${ROOT}/ggml-patches"

if [ ! -d "${GGML}/src" ]; then
    echo "error: ggml submodule not initialized at ${GGML}" >&2
    echo "       run: git submodule update --init ggml" >&2
    exit 1
fi

shopt -s nullglob
patch_files=("${PATCHES}"/*.patch)
if [ "${#patch_files[@]}" -eq 0 ]; then
    echo "error: no .patch files found in ${PATCHES}" >&2
    exit 1
fi

# A per-patch reverse check is not sufficient once a later patch changes a
# hunk introduced by an earlier patch. Compare the complete worktree state
# with the result of applying the full series to the pinned submodule commit.
# Temporary indexes keep both this check and the caller's real index untouched.
if git -C "${GGML}" rev-parse --git-dir >/dev/null 2>&1; then
    tmp_dir="$(mktemp -d)"
    expected_index="${tmp_dir}/expected.index"
    current_index="${tmp_dir}/current.index"
    cleanup_indexes() {
        rm -rf "${tmp_dir}"
    }
    trap cleanup_indexes EXIT

    GIT_INDEX_FILE="${expected_index}" git -C "${GGML}" read-tree HEAD
    for p in "${patch_files[@]}"; do
        if ! GIT_INDEX_FILE="${expected_index}" git -C "${GGML}" apply --cached "${p}"; then
            echo "error: $(basename "${p}") does not apply to the pinned ggml commit" >&2
            exit 1
        fi
    done

    mapfile -d '' patched_paths \
        < <(GIT_INDEX_FILE="${expected_index}" git -C "${GGML}" \
            diff --cached --name-only -z HEAD)
    GIT_INDEX_FILE="${current_index}" git -C "${GGML}" read-tree HEAD
    current_paths=()
    for path in "${patched_paths[@]}"; do
        if [ -e "${GGML}/${path}" ] || [ -L "${GGML}/${path}" ] \
            || git -C "${GGML}" cat-file -e "HEAD:${path}" 2>/dev/null; then
            current_paths+=("${path}")
        fi
    done
    if [ "${#current_paths[@]}" -ne 0 ]; then
        GIT_INDEX_FILE="${current_index}" git -C "${GGML}" \
            add -A -- "${current_paths[@]}"
    fi

    expected_tree="$(GIT_INDEX_FILE="${expected_index}" git -C "${GGML}" write-tree)"
    current_tree="$(GIT_INDEX_FILE="${current_index}" git -C "${GGML}" write-tree)"
    if [ "${current_tree}" = "${expected_tree}" ]; then
        echo "[ggml-patch] current series already applied"
        echo "[ggml-patch] done"
        exit 0
    fi
fi

for p in "${patch_files[@]}"; do
    name="$(basename "${p}")"
    # Already applied? (the reverse patch applies cleanly) -> skip.
    if git -C "${GGML}" apply --reverse --check "${p}" >/dev/null 2>&1; then
        echo "[ggml-patch] ${name}: already applied"
        continue
    fi
    if ! git -C "${GGML}" apply --check "${p}" >/dev/null 2>&1; then
        echo "[ggml-patch] ${name}: does NOT apply cleanly to current ggml" >&2
        echo "             (the tree is modified or contains a stale patch series)" >&2
        echo "             restore the pinned ggml submodule, then retry" >&2
        exit 1
    fi
    git -C "${GGML}" apply "${p}"
    echo "[ggml-patch] ${name}: applied"
done

# Intent-add any files the patches created so a later `git diff`-based patch
# regeneration includes them. Without this, regenerating a patch that owns a
# NEW file silently produces an empty diff (untracked files are invisible to
# `git diff`). Only meaningful on a dev host where ggml is a git repo; the
# docker build copies ggml without its .git, so skip there.
if git -C "${GGML}" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "${GGML}" status --porcelain | awk '$1 == "??" { print $2 }' \
        | while read -r f; do git -C "${GGML}" add -N "${f}"; done
fi
echo "[ggml-patch] done"
