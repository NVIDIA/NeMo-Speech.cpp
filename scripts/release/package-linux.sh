#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Package a portable Linux installation for the binary installer.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
install_prefix=
output_dir="$ROOT/release-artifacts"
backend=
artifact_backend=
max_glibc=2.31
version=

usage() {
    cat <<'EOF'
Usage: scripts/release/package-linux.sh --install-prefix DIR --backend cpu|vulkan|cuda [OPTION ...]

Options:
  --artifact-backend NAME
                       Backend label used in the archive name
  --output-dir DIR     Destination for the archive and checksum
  --version VERSION    Override the version read from VERSION ("nightly" for
                       the nightly channel)
  --max-glibc VERSION  Reject binaries requiring a newer glibc (default: 2.31)
  -h, --help

The installed project and GCC runtimes are packaged together; the project must
be built with text normalization (-DNEMO_SPEECH_WITH_NORM=ON and the static
dependencies from scripts/build_itn_deps.sh). CUDA archives also include
libcudart. glibc, GPU drivers, and the Vulkan loader remain host
dependencies. Project binaries must find bundled libraries through DT_RPATH,
which LD_LIBRARY_PATH cannot override.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-prefix)
            [[ $# -ge 2 ]] || { echo "error: --install-prefix requires a value" >&2; exit 2; }
            install_prefix=$2
            shift 2
            ;;
        --backend)
            [[ $# -ge 2 ]] || { echo "error: --backend requires a value" >&2; exit 2; }
            backend=$2
            shift 2
            ;;
        --artifact-backend)
            [[ $# -ge 2 ]] || { echo "error: --artifact-backend requires a value" >&2; exit 2; }
            artifact_backend=$2
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "error: --output-dir requires a value" >&2; exit 2; }
            output_dir=$2
            shift 2
            ;;
        --version)
            [[ $# -ge 2 ]] || { echo "error: --version requires a value" >&2; exit 2; }
            version=$2
            shift 2
            ;;
        --max-glibc)
            [[ $# -ge 2 ]] || { echo "error: --max-glibc requires a value" >&2; exit 2; }
            max_glibc=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n "$install_prefix" ]] || { echo "error: --install-prefix is required" >&2; exit 2; }
[[ -d "$install_prefix" ]] || { echo "error: install prefix does not exist: $install_prefix" >&2; exit 1; }
[[ -x "$install_prefix/bin/nemo-speech" ]] || {
    echo "error: install prefix does not contain bin/nemo-speech" >&2
    exit 1
}
case "$backend" in
    cpu|vulkan|cuda) ;;
    *)
        echo "error: --backend must be cpu, vulkan, or cuda" >&2
        exit 2
        ;;
esac
if [[ -z "$artifact_backend" ]]; then
    artifact_backend=$backend
fi
case "$backend:$artifact_backend" in
    cpu:cpu|vulkan:vulkan|cuda:cuda|cuda:cuda12|cuda:cuda13) ;;
    *)
        echo "error: invalid artifact backend '$artifact_backend' for '$backend'" >&2
        exit 2
        ;;
esac
[[ "$max_glibc" =~ ^[0-9]+(\.[0-9]+)+$ ]] || {
    echo "error: --max-glibc must be a dotted version" >&2
    exit 2
}

if [[ -z "$version" ]]; then
    version="$(sed -n 's/^NEMO_SPEECH_VERSION:[[:space:]]*//p' "$ROOT/VERSION")"
fi
[[ "$version" == nightly || "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.]+)?$ ]] || {
    echo "error: invalid release version '$version'" >&2
    exit 1
}

case "$(uname -m)" in
    x86_64|amd64) arch=x86_64 ;;
    aarch64|arm64) arch=aarch64 ;;
    *)
        echo "error: unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

for command_name in cc c++ ldd readelf strip tar gzip sha256sum sort; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "error: required command not found: $command_name" >&2
        exit 1
    }
done

package_name="nemo-speech-${version}-linux-${arch}-${artifact_backend}"
archive="${output_dir}/${package_name}.tar.gz"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/nemo-speech-package.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT HUP INT TERM
package_root="$work_dir/$package_name"

mkdir -p "$package_root" "$output_dir"
cp -a "$install_prefix"/. "$package_root"/

forbidden_payload="$(find "$package_root" \
    \( -name 'riva_server' -o -name 'riva_server.exe' \
       -o -name 'libgrpc*.so*' -o -path '*/riva-common' \) \
    -print -quit)"
[[ -z "$forbidden_payload" ]] || {
    echo "error: release contains Riva gRPC payload: $forbidden_payload" >&2
    exit 1
}

sentencepiece_license_dir="$package_root/share/licenses/nemo-speech/third_party/sentencepiece"
for license_file in LICENSE absl-LICENSE darts-clone-LICENSE protobuf-lite-LICENSE; do
    [[ -f "$sentencepiece_license_dir/$license_file" ]] || {
        echo "error: release is missing SentencePiece notice: $license_file" >&2
        exit 1
    }
done

# Release archives ship ITN/TN with its statically linked dependencies.
[[ -n "$(find "$package_root/lib" -maxdepth 1 -name 'libnemo_speech_text_normalization.so*' -print -quit)" ]] || {
    echo "error: release is missing text normalization; configure with -DNEMO_SPEECH_WITH_NORM=ON" >&2
    exit 1
}
for license_file in openfst/COPYING sparrowhawk/LICENSE protobuf/LICENSE re2/LICENSE; do
    [[ -f "$package_root/share/licenses/nemo-speech/third_party/$license_file" ]] || {
        echo "error: release is missing text normalization notice: $license_file" >&2
        exit 1
    }
done

runtime_license_dir="$package_root/share/licenses/nemo-speech/third_party/gcc-runtime"
mkdir -p "$package_root/lib" "$runtime_license_dir"
for runtime in libstdc++.so.6 libgcc_s.so.1 libgomp.so.1 libatomic.so.1; do
    # Vulkan drivers (Mesa ICDs) need the host's newer C++ runtime; a bundled
    # copy loaded first through DT_RPATH would make them fail to load.
    if [[ "$backend" == vulkan ]] &&
       [[ "$runtime" == libstdc++.so.6 || "$runtime" == libgcc_s.so.1 ]]; then
        continue
    fi
    if [[ "$runtime" == libstdc++* ]]; then
        compiler=c++
    else
        compiler=cc
    fi
    runtime_path="$("$compiler" -print-file-name="$runtime")"
    [[ "$runtime_path" != "$runtime" && -f "$runtime_path" ]] || {
        echo "error: $compiler could not locate $runtime" >&2
        exit 1
    }
    cp -L "$runtime_path" "$package_root/lib/$runtime"
done

compiler_major="$(cc -dumpfullversion -dumpversion | cut -d. -f1)"
runtime_copyright="/usr/share/doc/gcc-${compiler_major}-base/copyright"
if [[ ! -f "$runtime_copyright" ]]; then
    runtime_copyright="$(find /usr/share/doc -maxdepth 2 -type f \
        \( -path '*/gcc-*-base/copyright' -o -path '*/libstdc++*/copyright' \) \
        -print | sort | head -n 1)"
fi
[[ -n "$runtime_copyright" && -f "$runtime_copyright" ]] || {
    echo "error: could not locate the GCC runtime copyright file" >&2
    exit 1
}
cp "$runtime_copyright" "$runtime_license_dir/copyright"

if [[ "$backend" == cuda ]]; then
    cuda_home="${CUDA_HOME:-${CUDA_PATH:-}}"
    if [[ -z "$cuda_home" ]] && command -v nvcc >/dev/null 2>&1; then
        cuda_home="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    fi
    [[ -n "$cuda_home" && -d "$cuda_home" ]] || {
        echo "error: CUDA_HOME is required when packaging a CUDA build" >&2
        exit 1
    }
    cuda_lib_dir=
    case "$arch" in
        x86_64)
            cuda_targets=(x86_64-linux)
            ;;
        aarch64)
            cuda_targets=(aarch64-linux sbsa-linux)
            ;;
    esac
    for cuda_target in "${cuda_targets[@]}"; do
        candidate="$cuda_home/targets/$cuda_target/lib"
        if [[ -d "$candidate" ]]; then
            cuda_lib_dir="$candidate"
            break
        fi
    done
    [[ -n "$cuda_lib_dir" ]] || {
        echo "error: CUDA target libraries were not found for $arch under $cuda_home/targets" >&2
        exit 1
    }
    cudart_path="$(find -L "$cuda_lib_dir" -maxdepth 1 -type f \
        -name 'libcudart.so.*' -print | sort -V | tail -n 1)"
    [[ -n "$cudart_path" ]] || {
        echo "error: libcudart was not found under $cuda_lib_dir" >&2
        exit 1
    }
    cudart_path="$(readlink -f "$cudart_path")"
    cudart_soname="$(readelf -d "$cudart_path" |
        sed -n 's/.*Library soname: \[\([^]]*\)\].*/\1/p')"
    [[ -n "$cudart_soname" ]] || {
        echo "error: libcudart does not declare a SONAME: $cudart_path" >&2
        exit 1
    }
    cp -L "$cudart_path" "$package_root/lib/$cudart_soname"

    cuda_version="$("$cuda_home/bin/nvcc" --version |
        sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p' | head -n 1)"
    actual_cuda_major="${cuda_version%%-*}"
    if [[ "$artifact_backend" == cuda12 || "$artifact_backend" == cuda13 ]]; then
        expected_cuda_major="${artifact_backend#cuda}"
        [[ "$actual_cuda_major" == "$expected_cuda_major" ]] || {
            echo "error: artifact label '$artifact_backend' does not match CUDA $cuda_version" >&2
            exit 1
        }
    fi
    cublas_soname="libcublas.so.${actual_cuda_major}"
    cublas_shim="$package_root/lib/$cublas_soname"
    [[ -f "$cublas_shim" ]] || {
        echo "error: CUDA release does not contain the $cublas_soname shim" >&2
        exit 1
    }
    actual_cublas_soname="$(readelf -d "$cublas_shim" |
        sed -n 's/.*Library soname: \[\([^]]*\)\].*/\1/p')"
    [[ "$actual_cublas_soname" == "$cublas_soname" ]] || {
        echo "error: cuBLAS shim SONAME is '$actual_cublas_soname'; expected '$cublas_soname'" >&2
        exit 1
    }
    readelf --version-info "$cublas_shim" | grep -Fq "$cublas_soname" || {
        echo "error: cuBLAS shim does not export the $cublas_soname symbol version" >&2
        exit 1
    }
    ggml_cuda="$(find -L "$package_root/lib" -maxdepth 1 -type f \
        -name 'libggml-cuda.so.*' -print | sort -V | tail -n 1)"
    [[ -n "$ggml_cuda" ]] || {
        echo "error: CUDA release does not contain libggml-cuda" >&2
        exit 1
    }
    required_cublas="$(readelf -d "$ggml_cuda" |
        sed -n 's/.*Shared library: \[\(libcublas\.so\.[^]]*\)\].*/\1/p')"
    [[ "$required_cublas" == "$cublas_soname" ]] || {
        echo "error: libggml-cuda requires '$required_cublas'; expected '$cublas_soname'" >&2
        exit 1
    }
    # The shim implements only the cuBLAS calls ggml makes; a call added by a
    # llama.cpp update must be added to kernels/ before it can ship.
    imported_cublas="$work_dir/imported-cublas"
    exported_cublas="$work_dir/exported-cublas"
    nm -D --undefined-only "$ggml_cuda" > "$imported_cublas.raw"
    nm -D --defined-only "$cublas_shim" > "$exported_cublas.raw"
    awk '{ sub(/@.*/, "", $2); if ($2 ~ /^cublas/) print $2 }' "$imported_cublas.raw" |
        sort -u > "$imported_cublas"
    awk '{ sub(/@.*/, "", $3); print $3 }' "$exported_cublas.raw" | sort -u > "$exported_cublas"
    [[ -s "$imported_cublas" && -s "$exported_cublas" ]] || {
        echo "error: could not read the cuBLAS symbols of libggml-cuda or the shim" >&2
        exit 1
    }
    missing_cublas="$(comm -23 "$imported_cublas" "$exported_cublas")"
    [[ -z "$missing_cublas" ]] || {
        echo "error: the cuBLAS shim does not export these functions libggml-cuda calls:" >&2
        sed 's/^/  /' <<< "$missing_cublas" >&2
        exit 1
    }
    cuda_license="/usr/share/doc/cuda-cudart-${cuda_version}/copyright"
    [[ -f "$cuda_license" ]] || {
        echo "error: CUDA runtime license was not found: $cuda_license" >&2
        exit 1
    }
    install -Dm0644 "$cuda_license" \
        "$package_root/share/licenses/nemo-speech/nvidia/cuda-runtime/copyright"
fi

elf_candidates="$work_dir/elf-candidates"
find "$package_root/bin" "$package_root/lib" -type f -print0 > "$elf_candidates"
while IFS= read -r -d '' file; do
    if readelf -h "$file" >/dev/null 2>&1; then
        strip --strip-unneeded "$file"
    fi
done < "$elf_candidates"

abi_versions="$work_dir/glibc-versions"
missing_dependencies="$work_dir/missing-dependencies"
: > "$abi_versions"
: > "$missing_dependencies"
while IFS= read -r -d '' file; do
    readelf -h "$file" >/dev/null 2>&1 || continue
    readelf --version-info "$file" 2>/dev/null |
        grep -Eo 'GLIBC_[0-9]+(\.[0-9]+)*' |
        sed 's/^GLIBC_//' >> "$abi_versions" || true
    LD_LIBRARY_PATH="$package_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ldd "$file" 2>/dev/null |
        awk -v file="$file" -v backend="$backend" \
            '$2 == "not" && $3 == "found" {
                if (!(backend == "cuda" && $1 == "libcuda.so.1")) {
                    print file ": " $1
                }
            }' \
        >> "$missing_dependencies" || true
done < "$elf_candidates"

if [[ -s "$missing_dependencies" ]]; then
    echo "error: packaged ELF dependencies are unresolved:" >&2
    sed 's/^/  /' "$missing_dependencies" >&2
    exit 1
fi

highest_glibc="$(sort -Vu "$abi_versions" | tail -n 1)"
[[ -n "$highest_glibc" ]] || {
    echo "error: no glibc requirements were found in the package" >&2
    exit 1
}
if [[ "$highest_glibc" != "$max_glibc" ]] &&
   [[ "$(printf '%s\n%s\n' "$highest_glibc" "$max_glibc" | sort -V | tail -n 1)" == "$highest_glibc" ]]; then
    echo "error: package requires GLIBC_$highest_glibc; maximum is GLIBC_$max_glibc" >&2
    exit 1
fi

runpath_files="$work_dir/runpath-files"
: > "$runpath_files"
while IFS= read -r -d '' file; do
    readelf -h "$file" >/dev/null 2>&1 || continue
    if readelf -d "$file" 2>/dev/null | grep -q '(RUNPATH)'; then
        echo "$file" >> "$runpath_files"
    fi
done < "$elf_candidates"
if [[ -s "$runpath_files" ]]; then
    echo "error: these binaries use DT_RUNPATH; link with -Wl,--disable-new-dtags:" >&2
    sed 's/^/  /' "$runpath_files" >&2
    exit 1
fi

source_date_epoch="${SOURCE_DATE_EPOCH:-0}"
tar --sort=name \
    --mtime="@$source_date_epoch" \
    --owner=0 --group=0 --numeric-owner \
    -C "$work_dir" -cf - "$package_name" |
    gzip -n -9 > "$archive"
(
    cd "$output_dir"
    sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256"
)

echo "Created: $archive"
echo "SHA-256: $archive.sha256"
echo "Required: GLIBC_$highest_glibc or newer"
