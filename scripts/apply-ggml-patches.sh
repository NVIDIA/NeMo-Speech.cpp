#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Apply the in-tree ggml patches (ggml-patches/*.patch, in filename order) onto
# the vendored ggml submodule. This keeps the submodule pinned to clean
# upstream. Project-specific kernels and runtime extensions are applied during
# build setup.
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
if git -c "safe.directory=${GGML}" -C "${GGML}" rev-parse --git-dir >/dev/null 2>&1; then
    tmp_dir="$(mktemp -d)"
    expected_index="${tmp_dir}/expected.index"
    current_index="${tmp_dir}/current.index"
    cleanup_indexes() {
        rm -rf "${tmp_dir}"
    }
    trap cleanup_indexes EXIT

    GIT_INDEX_FILE="${expected_index}" git -c "safe.directory=${GGML}" -C "${GGML}" read-tree HEAD
    for p in "${patch_files[@]}"; do
        if ! GIT_INDEX_FILE="${expected_index}" git -c "safe.directory=${GGML}" -C "${GGML}" apply --cached "${p}"; then
            echo "error: $(basename "${p}") does not apply to the pinned ggml commit" >&2
            exit 1
        fi
    done

    # note: not mapfile -d '' - that is bash 4+, and macOS ships bash 3.2
    patched_paths=()
    while IFS= read -r -d '' path; do
        patched_paths+=("${path}")
    done < <(GIT_INDEX_FILE="${expected_index}" git -c "safe.directory=${GGML}" -C "${GGML}" \
            diff --cached --name-only -z HEAD)
    GIT_INDEX_FILE="${current_index}" git -c "safe.directory=${GGML}" -C "${GGML}" read-tree HEAD
    current_paths=()
    for path in "${patched_paths[@]}"; do
        if [ -e "${GGML}/${path}" ] || [ -L "${GGML}/${path}" ] \
            || git -c "safe.directory=${GGML}" -C "${GGML}" cat-file -e "HEAD:${path}" 2>/dev/null; then
            current_paths+=("${path}")
        fi
    done
    if [ "${#current_paths[@]}" -ne 0 ]; then
        GIT_INDEX_FILE="${current_index}" git -c "safe.directory=${GGML}" -C "${GGML}" \
            add -A -- "${current_paths[@]}"
    fi

    expected_tree="$(GIT_INDEX_FILE="${expected_index}" git -c "safe.directory=${GGML}" -C "${GGML}" write-tree)"
    current_tree="$(GIT_INDEX_FILE="${current_index}" git -c "safe.directory=${GGML}" -C "${GGML}" write-tree)"
    if [ "${current_tree}" = "${expected_tree}" ]; then
        echo "[ggml-patch] current series already applied"
        echo "[ggml-patch] done"
        exit 0
    fi
else
    # Docker build contexts intentionally omit the submodule's .git file. Test
    # the complete applied series in reverse order against a temporary index;
    # this handles later patches that refine hunks introduced by earlier ones
    # without changing the copied source tree.
    tmp_dir="$(mktemp -d)"
    plain_git_dir="${tmp_dir}/plain.git"
    plain_index="${tmp_dir}/plain.index"
    cleanup_plain_index() {
        rm -rf "${tmp_dir}"
    }
    trap cleanup_plain_index EXIT

    git init --bare --quiet "${plain_git_dir}"
    GIT_INDEX_FILE="${plain_index}" \
        git --git-dir="${plain_git_dir}" --work-tree="${GGML}" read-tree --empty
    mapfile -t patched_paths \
        < <(sed -n -e 's|^--- a/||p' -e 's|^+++ b/||p' "${patch_files[@]}" | sort -u)
    current_paths=()
    for path in "${patched_paths[@]}"; do
        if [ -e "${GGML}/${path}" ] || [ -L "${GGML}/${path}" ]; then
            current_paths+=("${path}")
        fi
    done
    if [ "${#current_paths[@]}" -ne 0 ]; then
        GIT_INDEX_FILE="${plain_index}" \
            git --git-dir="${plain_git_dir}" --work-tree="${GGML}" \
            add -- "${current_paths[@]}"
    fi

    series_applied=ON
    for ((i = ${#patch_files[@]} - 1; i >= 0; --i)); do
        if ! GIT_INDEX_FILE="${plain_index}" \
            git --git-dir="${plain_git_dir}" --work-tree="${GGML}" \
            apply --cached --reverse "${patch_files[i]}" >/dev/null 2>&1; then
            series_applied=OFF
            break
        fi
    done
    if [ "${series_applied}" = ON ]; then
        echo "[ggml-patch] current series already applied"
        echo "[ggml-patch] done"
        exit 0
    fi
fi

for p in "${patch_files[@]}"; do
    name="$(basename "${p}")"
    # Already applied? (the reverse patch applies cleanly) -> skip.
    if git -c "safe.directory=${GGML}" -C "${GGML}" apply --reverse --check "${p}" >/dev/null 2>&1; then
        echo "[ggml-patch] ${name}: already applied"
        continue
    fi
    if ! git -c "safe.directory=${GGML}" -C "${GGML}" apply --check "${p}" >/dev/null 2>&1; then
        echo "[ggml-patch] ${name}: does NOT apply cleanly to current ggml" >&2
        echo "             (the tree is modified or contains a stale patch series)" >&2
        echo "             restore the pinned ggml submodule, then retry" >&2
        exit 1
    fi
    git -c "safe.directory=${GGML}" -C "${GGML}" apply "${p}"
    echo "[ggml-patch] ${name}: applied"
done

# Intent-add any files the patches created so a later `git diff`-based patch
# regeneration includes them. Without this, regenerating a patch that owns a
# NEW file silently produces an empty diff (untracked files are invisible to
# `git diff`). Only meaningful on a dev host where ggml is a git repo; the
# docker build copies ggml without its .git, so skip there.
if git -c "safe.directory=${GGML}" -C "${GGML}" rev-parse --git-dir >/dev/null 2>&1; then
    git -c "safe.directory=${GGML}" -C "${GGML}" status --porcelain \
        | awk '$1 == "??" { print $2 }' \
        | while read -r f; do
            git -c "safe.directory=${GGML}" -C "${GGML}" add -N "${f}"
        done
fi
echo "[ggml-patch] done"
