#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Return success when a source tree without .git contains the complete patch
# series. The reverse applications operate only on a temporary index.
patch_series_applied_without_git() (
    local source_tree="$1"
    local patch_dir="$2"
    local log_prefix="$3"
    local plain_git_dir plain_index series_applied path i
    local -a patch_files patched_paths current_paths

    shopt -s nullglob
    patch_files=("${patch_dir}"/*.patch)
    PATCH_SERIES_TEMP_DIR="$(mktemp -d)"
    plain_git_dir="${PATCH_SERIES_TEMP_DIR}/plain.git"
    plain_index="${PATCH_SERIES_TEMP_DIR}/plain.index"
    trap 'rm -rf -- "${PATCH_SERIES_TEMP_DIR}"' EXIT

    git init --bare --quiet "${plain_git_dir}"
    GIT_INDEX_FILE="${plain_index}" \
        git --git-dir="${plain_git_dir}" --work-tree="${source_tree}" read-tree --empty
    mapfile -t patched_paths \
        < <(sed -n -e 's|^--- a/||p' -e 's|^+++ b/||p' "${patch_files[@]}" | sort -u)
    current_paths=()
    for path in "${patched_paths[@]}"; do
        if [ -e "${source_tree}/${path}" ] || [ -L "${source_tree}/${path}" ]; then
            current_paths+=("${path}")
        fi
    done
    if [ "${#current_paths[@]}" -ne 0 ]; then
        GIT_INDEX_FILE="${plain_index}" \
            git --git-dir="${plain_git_dir}" --work-tree="${source_tree}" \
            add -- "${current_paths[@]}"
    fi

    series_applied=ON
    for ((i = ${#patch_files[@]} - 1; i >= 0; --i)); do
        if ! GIT_INDEX_FILE="${plain_index}" \
            git --git-dir="${plain_git_dir}" --work-tree="${source_tree}" \
            apply --cached --reverse "${patch_files[i]}" >/dev/null 2>&1; then
            series_applied=OFF
            break
        fi
    done
    if [ "${series_applied}" = ON ]; then
        echo "${log_prefix} current series already applied"
        return 0
    fi
    return 1
)
