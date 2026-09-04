#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Validate a source checkout, apply the pinned ggml CUDA patches when needed,
# and configure one of the supported CMake presets.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == -h || "${1:-}" == --help || "${1:-}" == help ]]; then
    cat <<'EOF'
Usage: scripts/configure.sh [PRESET] [CMAKE_OPTION ...]

Valid presets:
  cpu-asr cpu-diar cpu-tts cpu-nmt cpu-speech cpu-server
  cuda-asr cuda-diar cuda-tts cuda-nmt cuda-speech cuda-server cuda-full
  metal-asr metal-diar metal-tts metal-nmt metal-speech metal-server
  vulkan-asr vulkan-diar vulkan-tts vulkan-nmt vulkan-speech vulkan-server
  developer

Examples:
  scripts/configure.sh cpu-server
  scripts/configure.sh cuda-server -DNEMO_SPEECH_WITH_NORM=ON

The script checks required submodules and optional feature assets, applies the
pinned ggml patch series for CUDA and Metal presets, and then runs cmake --preset PRESET.
EOF
    exit 0
fi

PRESET="${1:-cpu-asr}"
if [[ "$PRESET" == -* ]]; then
    PRESET=cpu-asr
else
    shift || true
fi

case "$PRESET" in
    cpu-asr|cpu-diar|cpu-tts|cpu-nmt|cpu-speech|cpu-server|\
    cuda-asr|cuda-diar|cuda-tts|cuda-nmt|cuda-speech|cuda-server|cuda-full|\
    metal-asr|metal-diar|metal-tts|metal-nmt|metal-speech|metal-server|\
    vulkan-asr|vulkan-diar|vulkan-tts|vulkan-nmt|vulkan-speech|vulkan-server|\
    developer) ;;
    *)
        echo "error: unknown preset '$PRESET'" >&2
        echo "       run: scripts/configure.sh --help" >&2
        exit 2
        ;;
esac

cd "$ROOT"

if [ ! -f ggml/CMakeLists.txt ]; then
    echo "error: ggml submodule is not initialized" >&2
    echo "       run: git submodule update --init ggml" >&2
    exit 1
fi

cmake_bool_override() { # cmake_bool_override VARIABLE DEFAULT ARGS...
    local variable="$1" value="$2" arg definition key setting
    shift 2
    for arg in "$@"; do
        [[ "$arg" == -D*=* ]] || continue
        definition="${arg#-D}"
        key="${definition%%=*}"
        key="${key%%:*}"
        [ "$key" = "$variable" ] || continue
        setting="${definition#*=}"
        # note: not ${setting^^} - that is bash 4+, and macOS ships bash 3.2
        setting="$(printf '%s' "${setting}" | tr '[:lower:]' '[:upper:]')"
        case "$setting" in
            ON|TRUE|YES|1) value=ON ;;
            OFF|FALSE|NO|0) value=OFF ;;
        esac
    done
    printf '%s' "$value"
}

need_nmt=OFF
need_asr=OFF
need_grpc=OFF
need_http=OFF
need_flashlight=OFF
need_ja=OFF
need_zh=OFF
case "$PRESET" in
    *-nmt|*-speech|*-server|cuda-full|developer) need_nmt=ON ;;
esac
case "$PRESET" in
    *-asr|*-speech|*-server|cuda-full|developer) need_asr=ON ;;
esac
case "$PRESET" in
    cuda-full|developer) need_grpc=ON ;;
esac
case "$PRESET" in
    *-server|cuda-full|developer) need_http=ON ;;
esac
if [ "$PRESET" = cuda-full ]; then
    need_flashlight=ON
    need_ja=ON
    need_zh=ON
fi

need_nmt="$(cmake_bool_override NEMO_SPEECH_BUILD_NMT "$need_nmt" "$@")"
need_nmt="$(cmake_bool_override NEMO_SPEECH_WITH_NMT "$need_nmt" "$@")"
need_asr="$(cmake_bool_override NEMO_SPEECH_BUILD_ASR "$need_asr" "$@")"
need_grpc="$(cmake_bool_override NEMO_SPEECH_BUILD_GRPC "$need_grpc" "$@")"
need_grpc="$(cmake_bool_override NEMO_SPEECH_WITH_GRPC "$need_grpc" "$@")"
need_http="$(cmake_bool_override NEMO_SPEECH_BUILD_HTTP "$need_http" "$@")"
need_flashlight="$(cmake_bool_override NEMO_SPEECH_WITH_FLASHLIGHT "$need_flashlight" "$@")"
need_ja="$(cmake_bool_override NEMO_SPEECH_TTS_WITH_JA "$need_ja" "$@")"
need_zh="$(cmake_bool_override NEMO_SPEECH_TTS_WITH_ZH "$need_zh" "$@")"

lfs_pointer="$(git grep -Il '^version https://git-lfs.github.com/spec/v1$' -- . 2>/dev/null \
    | while read -r path; do
        if [[ "$path" != src/tts/tokenizer/mandarin_data/* || "$need_zh" = ON ]]; then
            printf '%s\n' "$path"
        fi
    done | head -n 1 || true)"
if [ -n "$lfs_pointer" ]; then
    echo "error: required Git LFS content has not been materialized (for example: $lfs_pointer)" >&2
    echo "       run: git lfs install && git lfs pull" >&2
    exit 1
fi

missing=()
require_submodule() { # require_submodule PATH SENTINEL
    if [ ! -e "$1/$2" ]; then
        missing+=("$1")
    fi
}

if [ "$need_http" = ON ]; then
    require_submodule third_party/cpp-httplib httplib.h
fi
if [ "$need_nmt" = ON ]; then
    require_submodule llama.cpp CMakeLists.txt
elif [ "$need_asr" = ON ]; then
    require_submodule llama.cpp vendor/miniaudio/miniaudio.h
fi
if [ "$need_grpc" = ON ]; then
    require_submodule proto/riva-common LICENSE
fi
if [ "$need_flashlight" = ON ]; then
    require_submodule third_party/flashlight-text CMakeLists.txt
    require_submodule third_party/kenlm CMakeLists.txt
fi
if [ "$need_ja" = ON ]; then
    require_submodule third_party/open_jtalk src/CMakeLists.txt
fi
if [ "$need_zh" = ON ]; then
    require_submodule third_party/cppjieba CMakeLists.txt
fi

if [ "${#missing[@]}" -ne 0 ]; then
    echo "error: required submodules are not initialized:" >&2
    printf '         %s\n' "${missing[@]}" >&2
    printf '       run: git submodule update --init --recursive' >&2
    printf ' %q' "${missing[@]}" >&2
    printf '\n' >&2
    exit 1
fi

case "$PRESET" in
    cuda-*|metal-*)
        scripts/apply-ggml-patches.sh
        ;;
esac

cmake --preset "$PRESET" "$@"
