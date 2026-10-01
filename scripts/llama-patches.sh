#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Maintain the llama.cpp/ggml patch series in patches/.
#
# Builds apply the series automatically (cmake/llama_cpp.cmake); this script is
# only for changing it. It turns the series into one commit per patch inside a
# scratch worktree of the llama.cpp submodule, where it can be edited with plain
# git, and writes it back to patches/. Those scratch commits never leave the
# worktree; what you commit in this repository is patches/series and the
# patches/*.patch files it lists, in apply order.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/llama-patches.sh COMMAND

  edit        Create .deps/llama.cpp-patches: the pinned llama.cpp with one commit
              per patch. Edit and build against it with
                -DNEMO_SPEECH_LLAMA_CPP_SOURCE_DIR=$PWD/.deps/llama.cpp-patches
              and fold changes into a patch with
                git commit --fixup=<patch commit> && git rebase -i --autosquash <pin>
  export      Write the worktree's commits back to patches/.
  rebase REF  Move the series to another llama.cpp commit or tag and pin the
              submodule there. Resolve any conflicts with git in the worktree,
              then run export.
  check       Verify that the series applies to the pinned commit and that
              exporting it reproduces patches/ byte for byte.
  diff [REV]  Show a range-diff of patches/ against the series at REV (default
              HEAD), for reviewing a change to the series.
  done        Remove the scratch worktree (refuses if it has unexported changes).
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${ROOT}/llama.cpp"
PATCHES="${ROOT}/patches"
WORK="${ROOT}/.deps/llama.cpp-patches"

# Identity for the scratch commits. It never appears in patches/.
export GIT_COMMITTER_NAME="llama-patches" GIT_COMMITTER_EMAIL="llama-patches@localhost"

die() {
    echo "llama-patches: $*" >&2
    exit 1
}

llama() {
    git -C "${LLAMA}" "$@"
}

pin() {
    llama rev-parse HEAD
}

# series_files DIR: the patches DIR/series lists, in apply order. Every *.patch in
# DIR must be listed, so a new patch cannot be silently left out.
series_files() {
    local dir="$1" name path listed=" "
    [ -f "${dir}/series" ] || die "missing ${dir}/series"
    while IFS= read -r name || [ -n "${name}" ]; do
        name="${name%%#*}"
        name="$(printf '%s' "${name}" | tr -d '[:space:]')"
        [ -n "${name}" ] || continue
        [ -f "${dir}/${name}" ] || die "${dir}/series lists ${name}, which does not exist"
        listed="${listed}${name} "
        printf '%s\n' "${dir}/${name}"
    done < "${dir}/series"
    for path in "${dir}"/*.patch; do
        [ -e "${path}" ] || continue
        case "${listed}" in
            *" $(basename "${path}") "*) ;;
            *) die "${path} is not listed in ${dir}/series" ;;
        esac
    done
}

# add_series DIR BASE [PATCH_DIR]: a detached worktree at BASE with the series as commits.
add_series() {
    local dir="$1" base="$2" patch_dir="${3:-${PATCHES}}" path listing
    local -a series=()
    listing="$(series_files "${patch_dir}")"
    while IFS= read -r path; do
        [ -n "${path}" ] && series+=("${path}")
    done <<< "${listing}"
    llama worktree add --quiet --detach "${dir}" "${base}"
    if [ "${#series[@]}" -ne 0 ] && ! git -C "${dir}" am --quiet --keep "${series[@]}"; then
        die "the series does not apply to ${base}; see ${dir} (git am --show-current-patch)"
    fi
}

# write_series DIR BASE OUT: one <lowercased subject>.patch per commit plus OUT/series, with
# every format-patch option that could vary pinned. Each header keeps only From: and
# Subject: (unfolded onto one line).
write_series() {
    local dir="$1" base="$2" out="$3" tmp path name listed=" "
    git -C "${dir}" merge-base --is-ancestor "${base}" HEAD \
        || die "${dir} is not based on the pinned llama.cpp commit ${base}"
    tmp="$(mktemp -d)"
    git -C "${dir}" \
        -c diff.noprefix=false -c diff.mnemonicPrefix=false -c diff.relative=false \
        -c diff.algorithm=myers -c diff.indentHeuristic=true -c diff.renames=true \
        -c format.signature= -c format.thread=false -c format.notes=false \
        -c format.coverLetter=false -c format.from=false -c format.useAutoBase=false \
        -c format.suffix=.patch -c format.filenameMaxLength=100 \
        format-patch --quiet --zero-commit --no-signature --keep-subject --no-stat \
        --full-index --no-base -o "${tmp}" "${base}..HEAD"
    mkdir -p "${out}"
    printf '%s\n' "# Apply order for the patches in this directory (scripts/llama-patches.sh export)." \
        > "${tmp}/series"
    for path in "${tmp}"/*.patch; do
        [ -e "${path}" ] || continue
        name="$(basename "${path}")"
        name="$(printf '%s' "${name#[0-9][0-9][0-9][0-9]-}" | tr '[:upper:]' '[:lower:]')"
        case "${listed}" in
            *" ${name} "*) rm -rf "${tmp}"; die "two patches would be named ${name}; change one subject" ;;
        esac
        listed="${listed}${name} "
        trim_header < "${path}" > "${tmp}/${name}"
        rm "${path}"
        printf '%s\n' "${name}" >> "${tmp}/series"
    done
    for path in "${out}"/*.patch; do
        [ -e "${path}" ] || continue
        case "${listed}" in
            *" $(basename "${path}") "*) ;;
            *) rm -f "${path}" ;;
        esac
    done
    # Rewrite only what changed, so unchanged patch files keep their history.
    for path in "${tmp}"/*.patch "${tmp}/series"; do
        [ -e "${path}" ] || continue
        cmp -s "${path}" "${out}/$(basename "${path}")" || cp "${path}" "${out}/"
    done
    rm -rf "${tmp}"
}

# trim_header: drop the mbox separator and Date: from a format-patch header and
# unfold its continuation lines.
trim_header() {
    awk '
        NR == 1 && /^From [0-9a-f]+ Mon Sep 17 00:00:00 2001$/ { next }
        body { print; next }
        /^[ \t]/ { held = held $0; next }
        { if (held != "") print held; held = "" }
        /^$/ { body = 1; print; next }
        /^Date: / { next }
        { held = $0 }
        END { if (held != "") print held }
    '
}

subjects() {
    awk 'FNR == 1 { header = 1 } /^$/ { header = 0 } header && sub(/^Subject: /, "")' \
        "$1"/*.patch 2>/dev/null | sort
}

scratch_trees=()
cleanup() {
    local tree
    for tree in "${scratch_trees[@]+"${scratch_trees[@]}"}"; do
        llama worktree remove --force "${tree}" >/dev/null 2>&1 || true
        rm -rf "$(dirname "${tree}")"
    done
}
trap cleanup EXIT

# scratch_series VAR BASE [PATCH_DIR]: a temporary series worktree, removed on exit.
scratch_series() {
    local scratch_tree
    scratch_tree="$(mktemp -d)/tree"
    scratch_trees+=("${scratch_tree}")
    add_series "${scratch_tree}" "$2" "${3:-${PATCHES}}"
    printf -v "$1" '%s' "${scratch_tree}"
}

[ -e "${LLAMA}/.git" ] || die "llama.cpp submodule is not initialized; run: git submodule update --init llama.cpp"

case "${1:-}" in
    edit)
        [ ! -e "${WORK}" ] || die "${WORK} already exists; run export or done first"
        mkdir -p "$(dirname "${WORK}")"
        add_series "${WORK}" "$(pin)"
        echo "Series applied as commits in ${WORK}."
        # Git Bash on Windows: native CMake needs C:/... rather than /c/...
        work_native="${WORK}"
        if command -v cygpath >/dev/null 2>&1; then
            work_native="$(cygpath -m "${WORK}")"
        fi
        echo "Build against it: cmake ... -DNEMO_SPEECH_LLAMA_CPP_SOURCE_DIR=${work_native}"
        echo "When done: scripts/llama-patches.sh export"
        ;;
    export)
        [ -d "${WORK}" ] || die "no worktree at ${WORK}; run edit first"
        before="$(subjects "${PATCHES}")"
        write_series "${WORK}" "$(pin)" "${PATCHES}"
        after="$(subjects "${PATCHES}")"
        echo "Wrote $(ls "${PATCHES}"/*.patch | wc -l | tr -d ' ') patches and patches/series."
        comm -23 <(printf '%s\n' "${before}") <(printf '%s\n' "${after}") | sed 's/^/  removed: /'
        comm -13 <(printf '%s\n' "${before}") <(printf '%s\n' "${after}") | sed 's/^/  added:   /'
        ;;
    rebase)
        [ -n "${2:-}" ] || die "usage: scripts/llama-patches.sh rebase REF"
        old="$(pin)"
        if ! new="$(llama rev-parse --verify --quiet "${2}^{commit}")"; then
            llama fetch --quiet origin "${2}"
            new="$(llama rev-parse FETCH_HEAD)"
        fi
        [ -d "${WORK}" ] || add_series "${WORK}" "${old}"
        # Pin first, so that export uses the new base once conflicts are resolved.
        llama checkout --quiet --detach "${new}"
        echo "llama.cpp pinned at ${new}; rebasing the series in ${WORK}."
        if ! git -C "${WORK}" rebase --quiet --empty=drop --onto "${new}" "${old}"; then
            echo "Resolve the conflicts in ${WORK} with git (rebase --continue), then run export." >&2
            exit 1
        fi
        echo "Rebased cleanly. Build and test, then run export."
        ;;
    check)
        scratch_series tree "$(pin)"
        write_series "${tree}" "$(pin)" "${tree}.out"
        if ! diff -r "${PATCHES}" "${tree}.out" --exclude='*.md'; then
            die "patches/ is not in exported form; run edit, export and commit the result"
        fi
        echo "patches/ applies to llama.cpp $(pin) and round-trips byte for byte."
        ;;
    diff)
        rev="${2:-HEAD}"
        git -C "${ROOT}" cat-file -e "${rev}:patches" 2>/dev/null || die "${rev} has no patches/ directory"
        old_base="$(git -C "${ROOT}" rev-parse "${rev}:llama.cpp")"
        llama cat-file -e "${old_base}^{commit}" 2>/dev/null || llama fetch --quiet origin "${old_base}"
        old_patches="$(mktemp -d)"
        scratch_trees+=("${old_patches}/none")
        git -C "${ROOT}" archive "${rev}" patches | tar -x -C "${old_patches}" --strip-components=1
        scratch_series old "${old_base}" "${old_patches}"
        scratch_series new "$(pin)"
        llama range-diff "${old_base}..$(git -C "${old}" rev-parse HEAD)" "$(pin)..$(git -C "${new}" rev-parse HEAD)"
        ;;
    done)
        [ -d "${WORK}" ] || exit 0
        if [ "${2:-}" != "--force" ]; then
            out="$(mktemp -d)"
            scratch_trees+=("${out}/none")
            write_series "${WORK}" "$(pin)" "${out}"
            diff -rq "${PATCHES}" "${out}" --exclude='*.md' >/dev/null \
                || die "${WORK} has changes that are not exported; run export, or done --force"
        fi
        llama worktree remove --force "${WORK}"
        ;;
    -h|--help|help|"")
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
