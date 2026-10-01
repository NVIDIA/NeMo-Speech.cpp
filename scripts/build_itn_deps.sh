#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build the Sparrowhawk ITN stack (OpenFST 1.8 + Sparrowhawk) from pinned
# sarane22 forks and install to a user-writable project prefix, enabling
# -DNEMO_SPEECH_WITH_NORM=ON. Shared by the x86_64 and aarch64 images.
#
# Expects autotools and, on Linux, gcc-12 as CC/CXX. docker/Dockerfile sets
# CC/CXX=gcc-12 for this step (gcc-13/14 ICE on OpenFST's heavy templates at
# -O2) while the runtime itself builds with gcc-13. Sparrowhawk uses an in-tree,
# OpenFST-only compatibility implementation for the tiny subset of
# thrax::GrmManager that it calls; no Thrax or fstscript library is built/linked.
#
# STATIC=0 (Linux default) builds shared libraries against the system protobuf
# headers, protoc, and RE2. STATIC=1 (macOS default) also builds pinned protobuf
# and RE2, and installs position-independent static archives only, so
# nemo_speech_text_normalization carries the whole stack privately; the release
# archives use this mode. It additionally requires CMake.
#
# Usage: scripts/build_itn_deps.sh [WORKDIR]   (default: ./.deps/itn-build)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$REPO/.deps/itn-build}"
PREFIX="${PREFIX:-$REPO/.deps/itn}"
JOBS="${JOBS:-8}"
# Cap parallelism: OpenFST's template-heavy translation units can OOM cc1plus.
JOBS="$(( JOBS < 4 ? JOBS : 4 ))"
if [ "$(uname -s)" = Darwin ]; then
    STATIC="${STATIC:-1}"
else
    STATIC="${STATIC:-0}"
fi
CXXO="-std=c++17 -O2"
SHIM="$REPO/src/common/text_normalization/compat/sparrowhawk_compat.h"
ITN_COMPAT="$REPO/src/common/text_normalization/compat"
LICENSE_DIR="$PREFIX/share/licenses/nemo-speech/third_party"

clone() {  # clone <url> <dir> <commit>
    if [ ! -d "$2" ]; then
        git init --quiet "$2"
        git -C "$2" fetch --quiet --depth 1 "$1" "$3"
        git -C "$2" checkout --quiet FETCH_HEAD
    elif [ "$(git -C "$2" rev-parse HEAD)" != "$3" ]; then
        echo "$2 exists at the wrong revision; remove it or choose a clean WORKDIR" >&2
        echo "  expected: $3" >&2
        echo "  actual:   $(git -C "$2" rev-parse HEAD)" >&2
        exit 1
    fi
}

# Stamp the shipped autotools output newer than its inputs so make does not try
# to regenerate it with whichever autoconf/automake the host has.
stamp_autotools() {
    touch -t 202001010000 configure.ac acinclude.m4 2>/dev/null || true
    [ -d m4 ] && touch -t 202001010000 m4/*.m4 2>/dev/null || true
    find . -name 'Makefile.am' -exec touch -t 202001010000 {} +
    touch -t 202001020000 aclocal.m4
    touch -t 202001030000 configure
    find . -name '*.in' -exec touch -t 202001030000 {} +
}

install_license() {  # install_license <file> <component>
    install -d "$LICENSE_DIR/$2"
    install -m 0644 "$1" "$LICENSE_DIR/$2/$(basename "$1")"
}

LIBRARY_KIND=()
if [ "$STATIC" = 1 ]; then
    LIBRARY_KIND=(--disable-shared --enable-static --with-pic)
fi

mkdir -p "$WORK"
cd "$WORK"
# Pinned OpenFST and Sparrowhawk compatibility revisions.
clone https://github.com/sarane22/openfst.git     openfst     fc23b4cf529429284b874a26f28b15c6cc94f404
clone https://github.com/sarane22/sparrowhawk.git sparrowhawk 8b082acc507312077a096be8398584a13832c490

# --------------------------------------------------------- protobuf and RE2
# The last releases that do not require Abseil.
if [ "$STATIC" = 1 ]; then
    clone https://github.com/protocolbuffers/protobuf.git protobuf f0dc78d7e6e331b8c6bb2d5283e06aa26883ca7c  # v21.12
    clone https://github.com/google/re2.git               re2      3a8436ac436124a57a4e22d5c8713a2d42b381d7  # 2023-03-01
    cmake -S "$WORK/protobuf" -B "$WORK/protobuf/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -Dprotobuf_BUILD_SHARED_LIBS=OFF \
        -Dprotobuf_BUILD_TESTS=OFF \
        -Dprotobuf_WITH_ZLIB=OFF
    cmake --build "$WORK/protobuf/build" --target install -j "$JOBS"
    cmake -S "$WORK/re2" -B "$WORK/re2/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DRE2_BUILD_TESTING=OFF
    cmake --build "$WORK/re2/build" --target install -j "$JOBS"
    # Sparrowhawk's configure and Makefiles run protoc from PATH.
    export PATH="$PREFIX/bin:$PATH"
    install_license "$WORK/protobuf/LICENSE" protobuf
    install_license "$WORK/re2/LICENSE" re2
fi

# ---------------------------------------------------------------- OpenFST 1.8
cd "$WORK/openfst"
# FST_FLAGS_v rename missed by the fork.
perl -pi -e 's/\bFLAGS_v\b/FST_FLAGS_v/g' src/include/fst/label-reachable.h
# FAR + PDT cover Sparrowhawk's runtime grammar formats. Disable command-line
# tools/script wrappers: the runtime calls the typed C++ OpenFST API directly.
stamp_autotools
./configure --prefix="$PREFIX" --enable-far --enable-pdt --disable-bin \
    ${LIBRARY_KIND[@]+"${LIBRARY_KIND[@]}"} CXXFLAGS="$CXXO"
make -j"$JOBS"
make install
# Stage all core headers plus the FAR/PDT extension templates used by the
# compatibility GrmManager. Some OpenFST install manifests omit these headers.
cp -a src/include/fst/. "$PREFIX/include/fst/"
cp -f src/include/fst/types.h "$PREFIX/include/fst/" 2>/dev/null || true
for ext in far pdt; do
    mkdir -p "$PREFIX/include/fst/extensions/$ext"
    cp -f "src/include/fst/extensions/$ext"/*.h "$PREFIX/include/fst/extensions/$ext/"
done
[ -f "$PREFIX/include/fst/extensions/pdt/pdt.h" ] || { echo "pdt.h not staged" >&2; exit 1; }
if command -v ldconfig >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
    ldconfig
fi

# -------------------------------------------------------------- Sparrowhawk
cd "$WORK/sparrowhawk"
# (1) Autoconf tarball pins -std=c++11; OpenFST 1.8 headers need C++17.
perl -pi -e 's/-std=c\+\+11/-std=c++17/g' configure
[ -f configure.ac ] && perl -pi -e 's/-std=c\+\+11/-std=c++17/g' configure.ac || true
# (2) Stamp after the edits so make does not try to regenerate configure.
stamp_autotools
./configure --prefix="$PREFIX" --disable-bin ${LIBRARY_KIND[@]+"${LIBRARY_KIND[@]}"} \
    CPPFLAGS="-I$ITN_COMPAT -I$PREFIX/include" \
    LDFLAGS="-L$PREFIX/lib" CXXFLAGS="$CXXO"

# (3) Build + install the proto stubs, library, and headers with the OpenFST 1.8
#     compat shim force-included. src/bin (normalizer_main CLI) is skipped: the
#     runtime links libsparrowhawk directly. Put the compatibility include first so
#     Sparrowhawk resolves <thrax/grm-manager.h> without the Thrax project.
CPPF="-I$ITN_COMPAT -I$PREFIX/include -include $SHIM -funsigned-char"
make -C src/proto              CPPFLAGS="$CPPF"
make -C src/proto   install    CPPFLAGS="$CPPF"
make -j"$JOBS" -C src/lib      CPPFLAGS="$CPPF"
make -C src/lib     install    CPPFLAGS="$CPPF"
make -C src/include install    CPPFLAGS="$CPPF"
# The install manifest misses one generated header.
cp -f src/include/sparrowhawk/serialization_spec.pb.h "$PREFIX/include/sparrowhawk/"

# Strip debug symbols from the installed libs.
for f in "$PREFIX"/lib/lib{fst,fstfar,sparrowhawk}.so.*; do
    [ -f "$f" ] && [ ! -L "$f" ] && strip --strip-unneeded "$f"
done

install_license "$WORK/openfst/COPYING" openfst
install_license "$WORK/sparrowhawk/LICENSE" sparrowhawk
if command -v ldconfig >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
    ldconfig
fi

echo
echo "ITN stack installed to $PREFIX:"
ls -1 "$PREFIX"/lib/lib{sparrowhawk,fstfar,fst}.*
