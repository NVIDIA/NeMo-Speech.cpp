#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu

release_base=${NEMO_SPEECH_RELEASE_BASE_URL:-https://github.com/NVIDIA/NeMo-Speech.cpp/releases}
source_url=${NEMO_SPEECH_SOURCE_URL:-https://github.com/NVIDIA/NeMo-Speech.cpp.git}
version_url=${NEMO_SPEECH_VERSION_URL:-https://raw.githubusercontent.com/NVIDIA/NeMo-Speech.cpp/main/VERSION}
version=latest
channel=stable
prefix=
backend=auto
modify_path=1
dry_run=0
install_mode=auto

prerequisite_hint() {
    if [ "$os" = macos ]; then
        printf '%s' "Install the Xcode Command Line Tools with 'xcode-select --install'; install other build tools with Homebrew."
    else
        printf '%s' "On Ubuntu/Debian, install build tools with 'sudo apt-get install build-essential cmake ninja-build git curl'."
    fi
}

require_command() {
    command_name=$1
    description=$2
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "error: $description ('$command_name') was not found." >&2
        echo "       $(prerequisite_hint)" >&2
        exit 1
    }
}

require_cmake_326() {
    cmake_version=$(cmake --version | awk 'NR == 1 { print $3 }')
    cmake_major=${cmake_version%%.*}
    cmake_rest=${cmake_version#*.}
    cmake_minor=${cmake_rest%%.*}
    case "$cmake_major:$cmake_minor" in
        *[!0-9:]*|:*) cmake_major=0; cmake_minor=0 ;;
    esac
    if [ "$cmake_major" -lt 3 ] || { [ "$cmake_major" -eq 3 ] && [ "$cmake_minor" -lt 26 ]; }; then
        echo "error: CMake 3.26 or newer is required; found ${cmake_version:-unknown}." >&2
        echo "       $(prerequisite_hint)" >&2
        exit 1
    fi
}

usage() {
    cat <<'EOF'
Install NeMo-Speech.cpp. Published binaries are preferred; when an
artifact is not available, the installer builds the CLI, HTTP API, and
browser playground from source.

Usage: install.sh [options]
  --version VERSION       Install an exact version
  --channel stable|nightly
  --prefix DIRECTORY      Override the user-level installation directory
  --backend cpu|cuda|metal|vulkan
  --source                Build from source without checking for a binary
  --binary-only           Fail instead of falling back to a source build
  --no-modify-path        Do not update a shell startup file
  --dry-run               Print the resolved installation plan without changes
  -h, --help

Set NEMO_SPEECH_RELEASE_BASE_URL to use a release mirror or local test server.
Set NEMO_SPEECH_SOURCE_URL or NEMO_SPEECH_SOURCE_REF to override GitHub.
Set NEMO_SPEECH_VERSION_URL to override the current-version manifest.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --version) [ "$#" -ge 2 ] || { echo "--version requires a value" >&2; exit 2; }; version=$2; shift 2 ;;
        --channel) [ "$#" -ge 2 ] || { echo "--channel requires a value" >&2; exit 2; }; channel=$2; shift 2 ;;
        --prefix) [ "$#" -ge 2 ] || { echo "--prefix requires a value" >&2; exit 2; }; prefix=$2; shift 2 ;;
        --backend) [ "$#" -ge 2 ] || { echo "--backend requires a value" >&2; exit 2; }; backend=$2; shift 2 ;;
        --source) [ "$install_mode" != binary ] || { echo "--source and --binary-only are mutually exclusive" >&2; exit 2; }; install_mode=source; shift ;;
        --binary-only) [ "$install_mode" != source ] || { echo "--source and --binary-only are mutually exclusive" >&2; exit 2; }; install_mode=binary; shift ;;
        --no-modify-path) modify_path=0; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$channel" in stable|nightly) ;; *) echo "--channel must be stable or nightly" >&2; exit 2 ;; esac
case "$backend" in auto|cpu|cuda|metal|vulkan) ;; *) echo "unsupported backend: $backend" >&2; exit 2 ;; esac

kernel=$(uname -s)
machine=$(uname -m)
case "$kernel" in
    Linux) os=linux ;;
    Darwin) os=macos ;;
    *) echo "unsupported operating system: $kernel" >&2; exit 1 ;;
esac
case "$machine" in
    x86_64|amd64) arch=x86_64 ;;
    arm64|aarch64) arch=aarch64 ;;
    *) echo "unsupported architecture: $machine" >&2; exit 1 ;;
esac
device_model=
if [ -r /proc/device-tree/model ]; then
    device_model=$(tr -d '\000' </proc/device-tree/model | tr '[:upper:]' '[:lower:]')
fi

if [ "$backend" = metal ] && { [ "$os" != macos ] || [ "$arch" != aarch64 ]; }; then
    echo "Metal requires macOS on Apple Silicon; detected $os/$arch." >&2
    exit 1
fi
if [ "$backend" = cuda ] && [ "$os" = macos ]; then
    echo "CUDA builds are not supported on macOS; use --backend metal or --backend cpu." >&2
    exit 1
fi

if [ "$backend" = auto ]; then
    if [ "$os" = macos ] && [ "$arch" = aarch64 ]; then
        backend=metal
    elif command -v nvidia-smi >/dev/null 2>&1; then
        backend=cuda
    elif [ "$os" = linux ] && [ "$arch" = aarch64 ]; then
        case "$device_model" in
            *jetson*|*thor*|*dgx*spark*|*gb10*) backend=cuda ;;
            *) backend=cpu ;;
        esac
    else
        backend=cpu
    fi
fi

artifact_backend=$backend
if [ "$os" = linux ] && [ "$arch" = aarch64 ] && [ "$backend" = cuda ]; then
    cuda_series=${NEMO_SPEECH_CUDA_SERIES:-}
    if [ -z "$cuda_series" ]; then
        case "$device_model" in
            *orin*) cuda_series=12 ;;
            *thor*|*dgx*spark*|*gb10*) cuda_series=13 ;;
        esac
    fi
    if [ -z "$cuda_series" ] && command -v nvidia-smi >/dev/null 2>&1; then
        driver_version=$(nvidia-smi --query-gpu=driver_version \
            --format=csv,noheader 2>/dev/null | sed -n '1p' || true)
        driver_major=${driver_version%%.*}
        case "$driver_major" in
            ''|*[!0-9]*) ;;
            *) [ "$driver_major" -ge 580 ] && cuda_series=13 || cuda_series=12 ;;
        esac
    fi
    if [ -z "$cuda_series" ] && command -v nvcc >/dev/null 2>&1; then
        cuda_series=$(nvcc --version 2>/dev/null |
            sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1)
    fi
    cuda_series=${cuda_series:-12}
    case "$cuda_series" in
        12|13) ;;
        *)
            echo "NEMO_SPEECH_CUDA_SERIES must be 12 or 13; found '$cuda_series'." >&2
            exit 2
            ;;
    esac
    artifact_backend=cuda$cuda_series
fi

binary_candidate=1
[ -n "$release_base" ] || binary_candidate=0
if [ "$install_mode" = binary ] && [ "$binary_candidate" -eq 0 ]; then
    echo "--binary-only requires NEMO_SPEECH_RELEASE_BASE_URL" >&2
    exit 1
fi
if [ "$version" = latest ]; then
    if [ "$channel" = nightly ]; then
        version=nightly
    elif [ "$install_mode" = source ]; then
        version=source
    elif [ "$binary_candidate" -eq 0 ]; then
        version=source
        echo "No release endpoint is configured; building from the current source branch."
    else
        require_command curl "curl is required to resolve and download releases"
        version_manifest=$(curl -fsSL "$version_url" 2>/dev/null || true)
        version=$(printf '%s\n' "$version_manifest" |
            sed -n 's/^NEMO_SPEECH_VERSION:[[:space:]]*//p' | head -n 1)
        if [ -z "$version" ]; then
            binary_candidate=0
        fi
        if [ "$binary_candidate" -eq 0 ]; then
            if [ "$install_mode" = binary ]; then
                echo "could not resolve the latest release" >&2
                exit 1
            fi
            version=source
            echo "Latest release is unavailable; falling back to the main source branch."
        fi
    fi
fi
tag=$version
case "$tag" in v*) release_version=${tag#v} ;; *) release_version=$tag; tag=v$tag ;; esac
if [ "$channel" = nightly ] && [ "$release_version" = nightly ]; then tag=nightly; fi

if [ -n "${NEMO_SPEECH_SOURCE_REF:-}" ]; then
    source_ref=$NEMO_SPEECH_SOURCE_REF
elif [ "$release_version" = source ] && command -v git >/dev/null 2>&1 &&
     [ -e "$source_url/.git" ] &&
     local_source_ref=$(git -C "$source_url" symbolic-ref --quiet --short HEAD 2>/dev/null); then
    source_ref=$local_source_ref
elif [ "$release_version" = nightly ] || [ "$release_version" = source ]; then
    source_ref=main
else
    source_ref=$tag
fi
case "$backend" in
    cpu) source_preset=cpu-server ;;
    cuda) source_preset=cuda-server ;;
    metal) source_preset=metal-server ;;
    vulkan) source_preset=vulkan-server ;;
esac

if [ -z "$prefix" ]; then
    if [ "$os" = macos ]; then
        prefix=$HOME/Library/Application\ Support/NeMoSpeech
    else
        prefix=$HOME/.local/share/nemo-speech
    fi
fi
bin_dir=$HOME/.local/bin
archive=nemo-speech-$release_version-$os-$arch-$artifact_backend.tar.gz
url=$release_base/download/$tag/$archive
checksum_url=$url.sha256

echo "NeMo-Speech.cpp $release_version ($os/$arch, $artifact_backend)"
if [ "$install_mode" != source ] && [ "$binary_candidate" -eq 1 ]; then
    echo "Artifact: $url"
fi
if [ "$install_mode" != binary ]; then
    echo "Source:   $source_url#$source_ref ($source_preset fallback)"
fi
echo "Prefix:   $prefix"
[ "$dry_run" -eq 0 ] || exit 0

install_identity="$release_version $os $arch $artifact_backend"
source_identity="$release_version $os $arch $backend source:$source_ref profile:speech-server"
install_metadata=$prefix/.nemo-speech-install
if [ "$install_mode" != source ] && [ "$binary_candidate" -eq 1 ] &&
   [ -x "$prefix/bin/nemo-speech" ] && [ -f "$install_metadata" ] &&
   [ "$(sed -n '1p' "$install_metadata")" = "$install_identity" ]; then
    mkdir -p "$bin_dir"
    ln -sf "$prefix/bin/nemo-speech" "$bin_dir/nemo-speech"
    echo "Already installed."
    "$prefix/bin/nemo-speech" --version
    exit 0
fi

tmp=$(mktemp -d "${TMPDIR:-/tmp}/nemo-speech-install.XXXXXX")
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
binary_ready=0
if [ "$install_mode" != source ] && [ "$binary_candidate" -eq 1 ]; then
    require_command curl "curl is required to download release archives"
    require_command tar "tar is required to extract release archives"
    if curl -fL --retry 3 --output "$tmp/$archive" "$url" &&
       curl -fL --retry 3 --output "$tmp/$archive.sha256" "$checksum_url"; then
        binary_ready=1
    elif [ "$install_mode" = binary ]; then
        echo "release artifact or checksum is unavailable" >&2
        exit 1
    else
        rm -f "$tmp/$archive" "$tmp/$archive.sha256"
        echo "Release artifact is unavailable; building from source."
    fi
fi

if [ "$binary_ready" -eq 1 ]; then
    expected=$(awk 'NR == 1 { print $1 }' "$tmp/$archive.sha256")
    case "$expected" in
        *[!0-9a-fA-F]*|'') echo "release checksum is not a SHA-256 digest" >&2; exit 1 ;;
    esac
    [ "${#expected}" -eq 64 ] || { echo "release checksum is not a SHA-256 digest" >&2; exit 1; }
    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$tmp/$archive" | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
        actual=$(shasum -a 256 "$tmp/$archive" | awk '{print $1}')
    else
        echo "error: SHA-256 verification requires sha256sum (Linux) or shasum (macOS)." >&2
        exit 1
    fi
    [ "$actual" = "$(printf '%s' "$expected" | tr 'A-F' 'a-f')" ] || {
        echo "SHA-256 verification failed" >&2
        exit 1
    }

    mkdir -p "$tmp/extract"
    tar -xzf "$tmp/$archive" -C "$tmp/extract"
    root=$tmp/extract
    set -- "$tmp/extract"/*
    if [ "$#" -eq 1 ] && [ -d "$1" ]; then root=$1; fi
else
    if [ -x "$prefix/bin/nemo-speech" ] && [ -f "$install_metadata" ] &&
       [ "$(sed -n '1p' "$install_metadata")" = "$source_identity" ]; then
        mkdir -p "$bin_dir"
        ln -sf "$prefix/bin/nemo-speech" "$bin_dir/nemo-speech"
        echo "Already installed from source."
        "$prefix/bin/nemo-speech" --version
        exit 0
    fi

    require_command git "Git is required to build from source (use --binary-only to disable fallback)"
    require_command cmake "CMake 3.26 or newer is required to build from source"
    require_command ninja "Ninja is required to build from source"
    require_command bash "Bash is required by the source configuration helper"
    require_command cc "a C compiler is required to build from source"
    require_command c++ "a C++17 compiler is required to build from source"
    require_cmake_326
    case "$backend" in
        cuda)
            if ! command -v nvcc >/dev/null 2>&1; then
                echo "error: the CUDA compiler ('nvcc') was not found." >&2
                echo "       Install a supported NVIDIA CUDA Toolkit, or choose --backend cpu." >&2
                exit 1
            fi
            ;;
        metal)
            if ! command -v xcrun >/dev/null 2>&1 || ! xcrun --find clang++ >/dev/null 2>&1; then
                echo "error: the Apple developer toolchain required for Metal was not found." >&2
                echo "       Install it with 'xcode-select --install'." >&2
                exit 1
            fi
            ;;
        vulkan)
            if ! command -v glslc >/dev/null 2>&1; then
                echo "error: the Vulkan shader compiler ('glslc') was not found." >&2
                echo "       Install the Vulkan SDK (including headers, loader, glslc, and SPIR-V headers), or choose --backend cpu." >&2
                exit 1
            fi
            ;;
    esac
    source_dir=$tmp/source
    echo "Cloning $source_url at $source_ref..."
    git clone --depth 1 --single-branch --branch "$source_ref" "$source_url" "$source_dir"

    initialize_submodule() {
        submodule_path=$1
        if [ -f "$source_dir/.gitmodules" ] &&
           git -C "$source_dir" config -f .gitmodules --get-regexp '^submodule\..*\.path$' 2>/dev/null |
               awk '{print $2}' | grep -Fx "$submodule_path" >/dev/null 2>&1; then
            git -C "$source_dir" submodule update --init "$submodule_path"
        fi
    }
    initialize_submodule ggml
    initialize_submodule llama.cpp
    initialize_submodule third_party/cpp-httplib

    root=$tmp/source-install
    mkdir -p "$root"
    (
        cd "$source_dir"
        scripts/configure.sh "$source_preset"
        cmake --build --preset "$source_preset"
        cmake --install "build/$source_preset" --prefix "$root"
    )
    install_identity=$source_identity
fi
[ -x "$root/bin/nemo-speech" ] || { echo "installation does not contain bin/nemo-speech" >&2; exit 1; }
printf '%s\n' "$install_identity" >"$root/.nemo-speech-install"

mkdir -p "$(dirname "$prefix")" "$bin_dir"
next=$prefix.new
rm -rf "$next"
mv "$root" "$next"
rm -rf "$prefix.old"
if [ -e "$prefix" ]; then mv "$prefix" "$prefix.old"; fi
if ! mv "$next" "$prefix"; then
    if [ -e "$prefix.old" ]; then mv "$prefix.old" "$prefix"; fi
    echo "could not activate the new installation; the previous version was restored" >&2
    exit 1
fi
rm -rf "$prefix.old"
ln -sf "$prefix/bin/nemo-speech" "$bin_dir/nemo-speech"

if [ "$modify_path" -eq 1 ] && ! printf '%s' ":$PATH:" | grep -q ":$bin_dir:"; then
    shell_name=${SHELL:-}
    shell_name=${shell_name##*/}
    case "$shell_name" in
        zsh) rc=$HOME/.zshrc ;;
        bash) rc=$HOME/.bashrc ;;
        *) rc=$HOME/.profile ;;
    esac
    line='export PATH="$HOME/.local/bin:$PATH"'
    if [ ! -f "$rc" ] || ! grep -F "$line" "$rc" >/dev/null 2>&1; then
        printf '\n# NeMo-Speech.cpp\n%s\n' "$line" >>"$rc"
    fi
    echo "Added $bin_dir to PATH in $rc. For this shell, run:"
    echo '  export PATH="$HOME/.local/bin:$PATH"'
fi

"$prefix/bin/nemo-speech" --version
echo "Next: download a model, then run 'nemo-speech transcribe' or 'nemo-speech serve' (see README.md)."
