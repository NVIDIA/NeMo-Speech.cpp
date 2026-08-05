# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Multi-stage build for nemo-speech on Linux x86_64 and aarch64.
#
#   builder - full CUDA dev toolchain (nvcc, headers), gRPC/protobuf/absl dev
#             packages, and CMake. Builds nemo-speech, riva_server, the ggml
#             backend .so files, and optional developer targets.
#   rootfs  - assembles a FROM scratch root with the binaries, ggml .so files,
#             and the exact shared libraries reported by ldd. The CUDA driver
#             (libcuda.so.1) is not bundled; it is injected at run time by the
#             NVIDIA Container Toolkit.
#   runtime - FROM scratch; model-free, mount the GGUF at run time.
#
# Build:
#   docker build -f docker/Dockerfile --target builder \
#       -t nemo-speech-builder:$(uname -m) .
#   docker build -f docker/Dockerfile --target runtime \
#       -t nemo-speech-runtime:$(uname -m) .
#
# Cross-platform builds can pass --platform=linux/amd64 or --platform=linux/arm64
# when the Docker builder supports that target platform.

# Pin to the deployment driver's CUDA (Jetson Thor/L4T ships 13.0/R580). Newer
# toolkit device code miscomputes on an older driver (no cuda-compat on Tegra);
# a 13.0 build runs on both 13.0 and 13.2+ hosts. Bump only when all targets are.
ARG CUDA_VERSION=13.0.0
ARG UBUNTU_VERSION=24.04
ARG ENABLE_CUDA=ON
ARG ENABLE_CUDA_GRAPHS=ON
ARG ENABLE_GRPC=ON
ARG ENABLE_HTTP=ON
ARG ENABLE_FLASHLIGHT=ON
ARG ENABLE_NORM=ON
ARG ENABLE_TTS_JA=OFF
ARG ENABLE_TTS_ZH=OFF
ARG ENABLE_TESTS=OFF
# NMT translation (RivaTranslation) via llama.cpp. OFF by default: it pulls in
# the llama.cpp submodule and libllama. Set ON to build it.
ARG ENABLE_NMT=OFF
# Apply the project ggml patches and build the ASR against them. Set OFF to
# build the ASR against STANDARD upstream ggml (stock ops only, no patches):
# skips apply-ggml-patches.sh and sets -DNEMO_SPEECH_GGML_PATCHED=OFF.
ARG ENABLE_GGML_PATCHES=ON
ARG JOBS=
# GPU arch(es) for ggml-cuda. Empty = ggml's portable default (sm_86/89/120/121
# SASS + sm_80 PTX, runs on any sm_80+ via JIT). CI passes the target GPU's
# arch (80 = A100 on x86_64, 110 = Jetson Thor on aarch64) so kernels run from
# native SASS instead of JIT-compiled PTX.
ARG CUDA_ARCH=

############################################################################
# Stage 1: builder
############################################################################
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG ENABLE_CUDA
ARG ENABLE_CUDA_GRAPHS
ARG ENABLE_GRPC
ARG ENABLE_HTTP
ARG ENABLE_FLASHLIGHT
ARG ENABLE_NORM
ARG ENABLE_TTS_JA
ARG ENABLE_TTS_ZH
ARG ENABLE_TESTS
ARG ENABLE_NMT
ARG ENABLE_GGML_PATCHES
ARG JOBS
ARG CUDA_ARCH

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=${JOBS}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        wget \
        curl \
        pkg-config \
        ninja-build \
        cmake \
        make \
        g++-13 \
        gcc-13 \
        g++-12 \
        gcc-12 \
        bison \
        libgrpc++-dev \
        libprotobuf-dev \
        protobuf-compiler-grpc \
        libabsl-dev \
        libre2-dev \
        portaudio19-dev \
        autoconf \
        automake \
        libtool \
        libssl-dev \
        bison \
        python3 \
        python3-pip \
        python3-venv \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY CMakeLists.txt /work/
COPY cmake /work/cmake
COPY VERSION /work/
COPY LICENSE /work/
COPY NOTICE /work/
COPY THIRD_PARTY_NOTICES.md /work/
COPY README.md CONTRIBUTING.md /work/
COPY docs /work/docs
COPY config /work/config
# Core submodules: ggml (every backend) and llama.cpp (NMT decoder; built only
# when ENABLE_NMT=ON, copied unconditionally so the layer cache is stable).
COPY ggml /work/ggml
COPY llama.cpp /work/llama.cpp
COPY ggml-patches /work/ggml-patches
COPY proto /work/proto
COPY include /work/include
COPY src /work/src
COPY app /work/app
COPY server /work/server
COPY tests /work/tests
COPY examples /work/examples
COPY tools /work/tools
COPY scripts /work/scripts
COPY kernels /work/kernels

# Submodules under third_party/: open_jtalk + cppjieba (optional TTS JA/ZH
# tokenizers) and flashlight-text + kenlm (LM-fused CTC + word boosting, built
# only when ENABLE_FLASHLIGHT=ON). Copied unconditionally for a stable layer
# cache.
COPY third_party /work/third_party

# ITN (Sparrowhawk WFST): build OpenFST 1.8 + Sparrowhawk into
# /usr/local. Compiled with gcc-12 (gcc-13 ICEs on OpenFST's templates at -O2);
# the runtime itself uses gcc-13. Skipped when ENABLE_NORM != ON.
RUN if [ "${ENABLE_NORM}" = "ON" ]; then \
        CC=gcc-12 CXX=g++-12 PREFIX=/usr/local JOBS="${JOBS:-$(nproc)}" \
        bash /work/scripts/build_itn_deps.sh \
        && rm -rf /work/.deps/itn-build; \
    fi

# Keep SentencePiece's bundled protobuf implementation local to the ASR shared
# library; exporting it from libsentencepiece.so interposes Sparrowhawk's
# incompatible protobuf runtime when Flashlight and ITN are enabled together.
RUN if [ "${ENABLE_FLASHLIGHT}" = "ON" ]; then \
        PREFIX=/usr/local JOBS="${JOBS:-$(nproc)}" \
        bash /work/scripts/build_sentencepiece_static.sh \
        && rm -rf /work/.deps/sentencepiece-build; \
    fi

# Apply the project ggml patches (fused rel-pos attention, NVFP4 quantization,
# norm fusion, dw-conv F16, skinny-q8, BF16 fusions, and CUDA correctness) onto the clean
# vendored ggml. Idempotent: a working tree that already has them applied is
# detected and skipped. Skipped entirely when ENABLE_GGML_PATCHES=OFF — the ASR
# then builds against standard upstream ggml (see NEMO_SPEECH_GGML_PATCHED).
RUN if [ "${ENABLE_GGML_PATCHES}" = "ON" ]; then bash /work/scripts/apply-ggml-patches.sh; fi

# NCCL disabled: single-GPU inference does not use multi-GPU collectives,
# and the linked libnccl is otherwise dead runtime weight.
# NEMO_SPEECH_CUBLAS_SHIM=ON builds the drop-in libcublas.so.13 (native GEMM,
# no cuBLASLt) as a normal build target.
RUN cmake -G Ninja -S /work -B /work/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=${ENABLE_CUDA} \
        -DGGML_CUDA_NCCL=OFF \
        -DGGML_CUDA_GRAPHS=${ENABLE_CUDA_GRAPHS} \
        -DGGML_NATIVE=OFF \
        -DNEMO_SPEECH_CUBLAS_SHIM=${ENABLE_CUDA} \
        -DNEMO_SPEECH_GGML_PATCHED=${ENABLE_GGML_PATCHES} \
        ${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}} \
        -DNEMO_SPEECH_BUILD_GRPC=${ENABLE_GRPC} \
        -DNEMO_SPEECH_BUILD_HTTP=${ENABLE_HTTP} \
        -DNEMO_SPEECH_WITH_FLASHLIGHT=${ENABLE_FLASHLIGHT} \
        -DNEMO_SPEECH_WITH_NORM=${ENABLE_NORM} \
        -DNEMO_SPEECH_TTS_WITH_JA=${ENABLE_TTS_JA} \
        -DNEMO_SPEECH_TTS_WITH_ZH=${ENABLE_TTS_ZH} \
        -DNEMO_SPEECH_BUILD_NMT=${ENABLE_NMT} \
        -DNEMO_SPEECH_BUILD_TESTS=${ENABLE_TESTS} \
        -DNEMO_SPEECH_BUILD_TOOLS=${ENABLE_TESTS} \
    && cmake --build /work/build ${JOBS:+-- -j${JOBS}}

RUN mkdir -p /out/bin /out/lib /out/share/nemo-speech \
        /out/share/licenses/nemo-speech/third_party \
    && if [ "${ENABLE_FLASHLIGHT}" = "ON" ]; then \
           test -f /work/build/bin/libkenlm.so; \
           readelf -d /work/build/bin/libnemo_speech_asr.so \
               | grep -q 'Shared library: \[libkenlm.so\]'; \
       fi \
    && for binary in \
           nemo-speech riva_server transcribe_file transcribe_live diarize_file synthesize_text \
           translate_text speech_translate_file; \
       do \
           if [ -f "/work/build/bin/${binary}" ]; then \
               install -Dm0755 "/work/build/bin/${binary}" "/out/bin/${binary}"; \
           fi; \
       done \
    && for library in \
           /work/build/bin/libggml*.so* \
           /work/build/bin/libllama*.so* \
           /work/build/bin/libnemo_speech*.so* \
           /work/build/bin/libkenlm*.so* \
           /work/build/bin/libcublas.so*; \
       do \
           if [ -e "${library}" ]; then cp -a "${library}" /out/lib/; fi; \
       done \
    && cp /work/LICENSE /work/NOTICE /work/THIRD_PARTY_NOTICES.md \
        /out/share/licenses/nemo-speech/ \
    && license_dir=/out/share/licenses/nemo-speech/third_party \
    && install -Dm0644 /work/ggml/LICENSE "$license_dir/ggml/LICENSE" \
    && install -Dm0644 /work/llama.cpp/LICENSE "$license_dir/llama.cpp/LICENSE" \
    && install -Dm0644 /work/proto/riva-common/LICENSE \
        "$license_dir/riva-common/LICENSE" \
    && install -Dm0644 /work/third_party/flashlight-text/LICENSE \
        "$license_dir/flashlight-text/LICENSE" \
    && install -Dm0644 /work/third_party/kenlm/LICENSE \
        "$license_dir/kenlm/LICENSE" \
    && install -Dm0644 /work/third_party/kenlm/COPYING \
        "$license_dir/kenlm/COPYING" \
    && install -Dm0644 /work/third_party/kenlm/COPYING.3 \
        "$license_dir/kenlm/COPYING.3" \
    && install -Dm0644 /work/third_party/kenlm/COPYING.LESSER.3 \
        "$license_dir/kenlm/COPYING.LESSER.3" \
    && install -Dm0644 /work/third_party/cpp-httplib/LICENSE \
        "$license_dir/cpp-httplib/LICENSE" \
    && if [ "${ENABLE_TTS_JA}" = "ON" ]; then \
           install -Dm0644 /work/third_party/open_jtalk/src/COPYING \
               "$license_dir/open_jtalk/COPYING"; \
           install -Dm0644 /work/third_party/open_jtalk/src/mecab/COPYING \
               "$license_dir/mecab/COPYING"; \
           install -Dm0644 /work/third_party/open_jtalk/src/mecab-naist-jdic/COPYING \
               "$license_dir/naist-jdic/COPYING"; \
       fi \
    && if [ "${ENABLE_TTS_ZH}" = "ON" ]; then \
           install -Dm0644 /work/third_party/cppjieba/LICENSE \
               "$license_dir/cppjieba/LICENSE"; \
       fi \
    && if [ -d /usr/local/share/licenses/nemo-speech/third_party ]; then \
           cp -a /usr/local/share/licenses/nemo-speech/third_party/. \
               "$license_dir/"; \
       fi \
    && if [ "${ENABLE_TTS_ZH}" = "ON" ]; then \
           mkdir -p /out/share/nemo-speech/mandarin_g2p; \
           cp -a /work/src/tts/tokenizer/mandarin_data/. \
               /out/share/nemo-speech/mandarin_g2p/; \
       fi \
    && if [ "${ENABLE_TTS_JA}" = "ON" ]; then \
           mkdir -p /out/share/nemo-speech/open_jtalk_dic; \
           cp -a /work/build/open_jtalk_dic/. \
               /out/share/nemo-speech/open_jtalk_dic/; \
       fi

############################################################################
# Stage 2: assemble a scratch rootfs from dependencies reported by ldd
############################################################################
FROM builder AS rootfs

RUN set -eux; \
    root=/opt/nemo-speech-scratch-rootfs; \
    mkdir -p \
        "$root/etc" \
        "$root/home/nemo" \
        "$root/models" \
        "$root/opt/nemo-speech/bin" \
        "$root/opt/nemo-speech/lib" \
        "$root/opt/nemo-speech/share"; \
    printf 'nemo:x:1000:1000:NeMo-Speech.cpp:/home/nemo:/sbin/nologin\n' > "$root/etc/passwd"; \
    printf 'nemo:x:1000:\n' > "$root/etc/group"; \
    printf 'hosts: files dns\npasswd: files\ngroup: files\n' > "$root/etc/nsswitch.conf"; \
    cp -a /out/bin/. "$root/opt/nemo-speech/bin/"; \
    cp -a /out/lib/. "$root/opt/nemo-speech/lib/"; \
    cp -a /out/share/. "$root/opt/nemo-speech/share/"; \
    system_license_dir="$root/opt/nemo-speech/share/licenses/nemo-speech/third_party/system"; \
    mkdir -p "$system_license_dir"; \
    if [ -d /usr/share/common-licenses ]; then \
        cp -a /usr/share/common-licenses "$system_license_dir/debian-common-licenses"; \
    fi; \
    # Strip the build-tree rpath (/work/build/bin). In-tree shared libraries,
    # including ggml and KenLM, resolve from /opt/nemo-speech/lib at run time.
    apt-get update -qq && apt-get install -y --no-install-recommends patchelf >/dev/null; \
    rm -rf /var/lib/apt/lists/*; \
    for f in "$root"/opt/nemo-speech/bin/* "$root"/opt/nemo-speech/lib/*.so*; do \
        if [ -f "$f" ] && [ ! -L "$f" ]; then patchelf --remove-rpath "$f" || true; fi; \
    done; \
    copy_package_notice() { \
        dep="$1"; \
        owner="$(dpkg-query -S "$dep" 2>/dev/null | head -n 1 || true)"; \
        [ -n "$owner" ] || return 0; \
        package="${owner%%:*}"; \
        notice="/usr/share/doc/${package}/copyright"; \
        [ -f "$notice" ] || return 0; \
        install -Dm0644 "$notice" "$system_license_dir/${package}/copyright"; \
    }; \
    copy_dep() { \
        dep="$1"; \
        # Skip the CUDA driver (toolkit-injected at run time) and real cuBLAS/
        # cuBLASLt: the in-tree shim in /opt/nemo-speech/lib provides
        # libcublas.so.13, and nothing else needs cuBLASLt.
        case "$dep" in \
            */libcuda.so*|*/libcublas.so*|*/libcublasLt.so*) return 0;; \
        esac; \
        dest="$root${dep}"; \
        mkdir -p "$(dirname "$dest")"; \
        cp -L "$dep" "$dest"; \
        copy_package_notice "$dep"; \
    }; \
    copy_ldd_deps() { \
        LD_LIBRARY_PATH="$root/opt/nemo-speech/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            ldd "$1" 2>/dev/null \
            | awk '$1 ~ /^\// { print $1 } $2 == "=>" && $3 ~ /^\// { print $3 }' \
            | sort -u \
            | while read -r dep; do copy_dep "$dep"; done; \
    }; \
    for b in "$root"/opt/nemo-speech/bin/*; do [ -f "$b" ] && copy_ldd_deps "$b"; done; \
    for so in "$root"/opt/nemo-speech/lib/*.so*; do \
        [ -f "$so" ] && copy_ldd_deps "$so"; \
    done; \
    for cuda_target in x86_64-linux aarch64-linux; do \
        if [ -d "$root/usr/local/cuda/targets/${cuda_target}/lib" ]; then \
            mkdir -p "$root/usr/local/cuda"; \
            ln -sfn "targets/${cuda_target}/lib" "$root/usr/local/cuda/lib64"; \
            break; \
        fi; \
    done; \
    # The cuBLAS shim (CMake target -DNEMO_SPEECH_CUBLAS_SHIM=ON, staged via
    # /out/lib) is the only libcublas.so.13 in the image and is first on
    # LD_LIBRARY_PATH; real cuBLAS/cuBLASLt were skipped by copy_dep above.
    for b in "$root"/opt/nemo-speech/bin/*; do [ -f "$b" ] && strip --strip-unneeded "$b" || true; done; \
    chmod 0755 "$root"/opt/nemo-speech/bin/*; \
    chown -R 1000:1000 "$root/home/nemo" "$root/opt/nemo-speech"

############################################################################
# Stage 3: runtime (FROM scratch)
############################################################################
FROM scratch AS runtime

COPY --from=rootfs /opt/nemo-speech-scratch-rootfs/ /

ENV HOME=/home/nemo
# Driver (libcuda.so.1) and other NVIDIA libs are mounted at run time by the
# NVIDIA Container Toolkit. The aarch64 Tegra/NVIDIA subdirs are needed for
# Jetson/L4T FROM scratch images because there is no ldconfig-generated cache.
ENV LD_LIBRARY_PATH=/opt/nemo-speech/lib:/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH=/opt/nemo-speech/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/bin:/bin
ENV MAGPIE_MANDARIN_G2P_DIR=/opt/nemo-speech/share/nemo-speech/mandarin_g2p
ENV MAGPIE_OPENJTALK_DIC_DIR=/opt/nemo-speech/share/nemo-speech/open_jtalk_dic

USER 1000:1000
EXPOSE 8080 50051
ENTRYPOINT ["/opt/nemo-speech/bin/nemo-speech"]
CMD ["--help"]
