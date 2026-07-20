#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build SentencePiece as a static archive so its bundled protobuf runtime can
# remain local to the ASR library when ITN and word boosting are enabled.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$REPO/.deps/sentencepiece-build}"
PREFIX="${PREFIX:-$REPO/.deps/sentencepiece}"
JOBS="${JOBS:-8}"
JOBS="$(( JOBS < 4 ? JOBS : 4 ))"
SOURCE="$WORK/source"
BUILD="$WORK/build"
COMMIT=17d7580d6407802f85855d2cc9190634e2c95624

if [ ! -d "$SOURCE/.git" ]; then
    git clone --filter=blob:none --no-checkout https://github.com/google/sentencepiece.git "$SOURCE"
fi
git -C "$SOURCE" fetch --depth 1 origin "$COMMIT"
git -C "$SOURCE" checkout --quiet "$COMMIT"

cmake -G Ninja -S "$SOURCE" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPM_BUILD_TEST=OFF \
    -DSPM_ENABLE_SHARED=OFF \
    -DSPM_ENABLE_TCMALLOC=OFF
cmake --build "$BUILD" --target sentencepiece-static -j "$JOBS"

install -d "$PREFIX/lib" "$PREFIX/include"
install -m 0644 "$BUILD/src/libsentencepiece.a" "$PREFIX/lib/libsentencepiece.a"
install -m 0644 "$SOURCE/src/sentencepiece_processor.h" "$PREFIX/include/sentencepiece_processor.h"

LICENSE_DIR="$PREFIX/share/licenses/nemo-speech/third_party/sentencepiece"
install -Dm0644 "$SOURCE/LICENSE" "$LICENSE_DIR/LICENSE"
install -Dm0644 "$SOURCE/third_party/absl/LICENSE" "$LICENSE_DIR/absl-LICENSE"
install -Dm0644 "$SOURCE/third_party/darts_clone/LICENSE" "$LICENSE_DIR/darts-clone-LICENSE"
install -Dm0644 "$SOURCE/third_party/protobuf-lite/LICENSE" "$LICENSE_DIR/protobuf-lite-LICENSE"
