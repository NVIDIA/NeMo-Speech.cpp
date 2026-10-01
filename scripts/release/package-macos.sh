#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Package a macOS installation for the binary installer.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
install_prefix=
output_dir="$ROOT/release-artifacts"
backend=
arch=
version=
min_macos=13.3

usage() {
    cat <<'EOF'
Usage: scripts/release/package-macos.sh --install-prefix DIR --backend cpu|metal --arch aarch64|x86_64 [OPTION ...]

Options:
  --output-dir DIR     Destination for the archive and checksum
  --version VERSION    Override the version read from VERSION ("nightly" for
                       the nightly channel)
  --min-macos VERSION  Reject binaries built for a newer macOS (default: 13.3)
  -h, --help

The project must be built with text normalization (-DNEMO_SPEECH_WITH_NORM=ON).
Every Mach-O file must be built for --arch and link only system libraries or
libraries inside the package. Binaries are stripped of local symbols and
ad-hoc signed.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-prefix|--backend|--arch|--output-dir|--version|--min-macos)
            [[ $# -ge 2 ]] || { echo "error: $1 requires a value" >&2; exit 2; }
            case "$1" in
                --install-prefix) install_prefix=$2 ;;
                --backend) backend=$2 ;;
                --arch) arch=$2 ;;
                --output-dir) output_dir=$2 ;;
                --version) version=$2 ;;
                --min-macos) min_macos=$2 ;;
            esac
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$install_prefix" && -x "$install_prefix/bin/nemo-speech" ]] || {
    echo "error: --install-prefix must contain bin/nemo-speech" >&2
    exit 2
}
case "$backend" in cpu|metal) ;; *) echo "error: --backend must be cpu or metal" >&2; exit 2 ;; esac
case "$arch" in
    aarch64) macho_arch=arm64 ;;
    x86_64) macho_arch=x86_64 ;;
    *) echo "error: --arch must be aarch64 or x86_64" >&2; exit 2 ;;
esac
if [[ "$backend" == metal && "$arch" != aarch64 ]]; then
    echo "error: Metal archives are built for aarch64 only" >&2
    exit 2
fi
if [[ -z "$version" ]]; then
    version="$(sed -n 's/^NEMO_SPEECH_VERSION:[[:space:]]*//p' "$ROOT/VERSION")"
fi
[[ "$version" == nightly || "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.]+)?$ ]] || {
    echo "error: invalid release version '$version'" >&2
    exit 1
}
for command_name in otool lipo codesign strip tar gzip shasum; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "error: required command not found: $command_name" >&2
        exit 1
    }
done

package_name="nemo-speech-${version}-macos-${arch}-${backend}"
archive="${output_dir}/${package_name}.tar.gz"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/nemo-speech-package.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT HUP INT TERM
package_root="$work_dir/$package_name"

mkdir -p "$package_root" "$output_dir"
cp -R "$install_prefix"/. "$package_root"/

if find "$package_root" \( -name 'riva_server' -o -name 'libgrpc*' \) -print | grep -q .; then
    echo "error: release contains Riva gRPC payload" >&2
    exit 1
fi
sentencepiece_license_dir="$package_root/share/licenses/nemo-speech/third_party/sentencepiece"
[[ -f "$sentencepiece_license_dir/LICENSE" ]] || {
    echo "error: release is missing the SentencePiece notice" >&2
    exit 1
}

# Release archives ship ITN/TN with its statically linked dependencies.
[[ -f "$package_root/lib/libnemo_speech_text_normalization.dylib" ]] || {
    echo "error: release is missing text normalization; configure with -DNEMO_SPEECH_WITH_NORM=ON" >&2
    exit 1
}
for license_file in openfst/COPYING sparrowhawk/LICENSE protobuf/LICENSE re2/LICENSE; do
    [[ -f "$package_root/share/licenses/nemo-speech/third_party/$license_file" ]] || {
        echo "error: release is missing text normalization notice: $license_file" >&2
        exit 1
    }
done

version_le() {  # version_le A B: A <= B
    [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -t. -k1,1n -k2,2n -k3,3n | head -n 1)" == "$1" ]]
}

failures="$work_dir/failures"
: > "$failures"
macho_files="$work_dir/macho-files"
find "$package_root/bin" "$package_root/lib" -type f -print | LC_ALL=C sort > "$macho_files"
while IFS= read -r file; do
    otool -h "$file" >/dev/null 2>&1 || continue
    archs="$(lipo -archs "$file" 2>/dev/null || true)"
    [[ "$archs" == "$macho_arch" ]] || echo "$file: built for '$archs', expected $macho_arch" >> "$failures"

    minos="$(otool -l "$file" | awk '$1 == "minos" { print $2; exit }')"
    if [[ -n "$minos" ]] && ! version_le "$minos" "$min_macos"; then
        echo "$file: requires macOS $minos, newer than $min_macos" >> "$failures"
    fi

    # Dependencies: system libraries or @rpath/@loader_path entries present in the package.
    otool -L "$file" | tail -n +2 | awk '{ print $1 }' | while IFS= read -r dependency; do
        case "$dependency" in
            /usr/lib/*|/System/Library/*) ;;
            @rpath/*|@loader_path/*|@executable_path/*)
                name="${dependency##*/}"
                [[ -e "$package_root/lib/$name" || -e "$package_root/bin/$name" ]] ||
                    echo "$file: $dependency is not in the package" >> "$failures"
                ;;
            *) echo "$file: links $dependency, which is outside the package" >> "$failures" ;;
        esac
    done
done < "$macho_files"

if [[ -s "$failures" ]]; then
    echo "error: the package is not self-contained:" >&2
    sed 's/^/  /' "$failures" >&2
    exit 1
fi

# Strip local symbols, then ad-hoc sign: install-time rpath edits invalidate the
# linker's signature, and arm64 macOS refuses to run unsigned code.
while IFS= read -r file; do
    otool -h "$file" >/dev/null 2>&1 || continue
    strip -x "$file"
    codesign --force --sign - "$file"
done < "$macho_files"

source_date_epoch="${SOURCE_DATE_EPOCH:-0}"
stamp="$(date -u -r "$source_date_epoch" +%Y%m%d%H%M.%S)"
find "$package_root" -exec touch -h -t "$stamp" {} +
(cd "$work_dir" && find "$package_name" -print | LC_ALL=C sort > "$work_dir/files")
COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs --uid 0 --gid 0 --uname root --gname wheel \
    -C "$work_dir" -n -T "$work_dir/files" -cf - | gzip -n -9 > "$archive"
(
    cd "$output_dir"
    shasum -a 256 "$(basename "$archive")" > "$(basename "$archive").sha256"
)

echo "Created: $archive"
echo "SHA-256: $archive.sha256"
echo "Requires: macOS $min_macos or newer ($macho_arch)"
