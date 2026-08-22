#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build the Sparrowhawk ITN stack (OpenFST 1.8 + Sparrowhawk) from pinned
# sarane22 forks and install to a user-writable project prefix, enabling
# -DNEMO_SPEECH_WITH_ITN=ON. Shared by the x86_64 and aarch64 images.
#
# Expects: protobuf headers + protoc, re2, autotools, and gcc-12 as CC/CXX. The
# Dockerfiles set CC/CXX=gcc-12 for this step (gcc-13/14 ICE on OpenFST's heavy
# templates at -O2) while the runtime itself builds with gcc-13. Sparrowhawk uses
# an in-tree, OpenFST-only compatibility implementation for the tiny subset of
# thrax::GrmManager that it calls; no Thrax or fstscript library is built/linked.
#
# Usage: scripts/build_itn_deps.sh [WORKDIR]   (default: ./.deps/itn-build)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$REPO/.deps/itn-build}"
PREFIX="${PREFIX:-$REPO/.deps/itn}"
JOBS="${JOBS:-8}"
# Cap parallelism: OpenFST's template-heavy translation units can OOM cc1plus.
JOBS="$(( JOBS < 4 ? JOBS : 4 ))"
CXXO="-std=c++17 -O2"
SHIM="$REPO/src/common/text_normalization/compat/sparrowhawk_compat.h"
ITN_COMPAT="$REPO/src/common/text_normalization/compat"

clone() {  # clone <url> <dir> <commit>
    if [ ! -d "$2" ]; then
        git clone "$1" "$2"
        git -C "$2" checkout --quiet "$3"
    elif [ "$(git -C "$2" rev-parse HEAD)" != "$3" ]; then
        echo "$2 exists at the wrong revision; remove it or choose a clean WORKDIR" >&2
        echo "  expected: $3" >&2
        echo "  actual:   $(git -C "$2" rev-parse HEAD)" >&2
        exit 1
    fi
}

mkdir -p "$WORK"
cd "$WORK"
# Pinned OpenFST and Sparrowhawk compatibility revisions.
clone https://github.com/sarane22/openfst.git     openfst     fc23b4cf529429284b874a26f28b15c6cc94f404
clone https://github.com/sarane22/sparrowhawk.git sparrowhawk 8b082acc507312077a096be8398584a13832c490

# ---------------------------------------------------------------- OpenFST 1.8
cd "$WORK/openfst"
# FST_FLAGS_v rename missed by the fork.
sed -i 's/\bFLAGS_v\b/FST_FLAGS_v/g' src/include/fst/label-reachable.h
# FAR + PDT cover Sparrowhawk's runtime grammar formats. Disable command-line
# tools/script wrappers: the runtime calls the typed C++ OpenFST API directly.
./configure --prefix="$PREFIX" --enable-far --enable-pdt --disable-bin CXXFLAGS="$CXXO"
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
sed -i 's/-std=c++11/-std=c++17/g' configure
[ -f configure.ac ] && sed -i 's/-std=c++11/-std=c++17/g' configure.ac || true
./configure --prefix="$PREFIX" --disable-bin \
    CPPFLAGS="-I$ITN_COMPAT -I$PREFIX/include" \
    LDFLAGS="-L$PREFIX/lib" CXXFLAGS="$CXXO"

# (2) Stamp generated autotools files so make does not try to regenerate them.
touch -d '2020-01-01 00:00:00' configure.ac acinclude.m4 2>/dev/null || true
[ -d m4 ] && touch -d '2020-01-01 00:00:00' m4/*.m4 2>/dev/null || true
find . -name 'Makefile.am' -exec touch -d '2020-01-01 00:00:00' {} +
touch -d '2020-01-02 00:00:00' aclocal.m4
touch -d '2020-01-03 00:00:00' configure
find . -name '*.in' -exec touch -d '2020-01-03 00:00:00' {} +
touch -d '2020-01-04 00:00:00' config.status
find . -name 'Makefile' -exec touch -d '2020-01-05 00:00:00' {} +

# (3) Build + install the proto stubs, library, and headers with the OpenFST 1.8
#     compat shim force-included. src/bin (normalizer_main CLI) is skipped: the
#     the runtime links libsparrowhawk directly. Put the compatibility include first so
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

LICENSE_DIR="$PREFIX/share/licenses/nemo-speech/third_party"
install -Dm0644 "$WORK/openfst/COPYING" "$LICENSE_DIR/openfst/COPYING"
install -Dm0644 "$WORK/sparrowhawk/LICENSE" "$LICENSE_DIR/sparrowhawk/LICENSE"
if command -v ldconfig >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
    ldconfig
fi

echo
echo "ITN stack installed to $PREFIX:"
ls -1 "$PREFIX"/lib/libsparrowhawk.so "$PREFIX"/lib/libfstfar.so "$PREFIX"/lib/libfst.so
