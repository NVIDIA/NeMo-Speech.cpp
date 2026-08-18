# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.26)

# Single source of truth for the project version: the root VERSION file.
# Parsed here and injected as a
# compile definition so the C ABI version strings (nemo_speech_asr_version /
# nemo_speech_tts_version) stay in sync without a hardcoded literal.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" _nemo_speech_version_lines)
foreach(_line ${_nemo_speech_version_lines})
    if(_line MATCHES
       "^NEMO_SPEECH_VERSION:[ \t]*([0-9]+\\.[0-9]+\\.[0-9]+)([-+][0-9A-Za-z.-]+)?[ \t]*$")
        set(NEMO_SPEECH_VERSION "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
        set(NEMO_SPEECH_PROJECT_VERSION "${CMAKE_MATCH_1}")
    endif()
endforeach()
if(NOT NEMO_SPEECH_VERSION OR NOT NEMO_SPEECH_PROJECT_VERSION)
    message(FATAL_ERROR "could not parse NEMO_SPEECH_VERSION from ${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
endif()

project(nemo_speech VERSION ${NEMO_SPEECH_PROJECT_VERSION} LANGUAGES C CXX)
add_compile_definitions(NEMO_SPEECH_VERSION_STR="${NEMO_SPEECH_VERSION}")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
if(WIN32)
    # Install the Visual C++ runtime required by Windows packages.
    include(InstallRequiredSystemLibraries)
endif()
set(NEMO_SPEECH_LICENSE_INSTALL_DIR
    "${CMAKE_INSTALL_DATADIR}/licenses/nemo-speech")
install(FILES LICENSE NOTICE THIRD_PARTY_NOTICES.md
    DESTINATION "${NEMO_SPEECH_LICENSE_INSTALL_DIR}")
install(FILES README.md CONTRIBUTING.md
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/nemo-speech")
install(DIRECTORY docs/
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/nemo-speech/docs")
install(DIRECTORY config/
    DESTINATION "${CMAKE_INSTALL_DATADIR}/nemo-speech/config")

# Windows: stop <windows.h> - pulled in transitively by the CUDA headers when
# GGML_CUDA=ON - from defining the min()/max() macros, which otherwise clobber
# std::min / std::max / std::numeric_limits<>::max() in the TTS/codec sources
# (C2589/C2059). Applied before add_subdirectory so every target inherits it.
if(WIN32)
    add_compile_definitions(NOMINMAX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Emit build/compile_commands.json for clangd, IDEs, and other tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

# Default to an optimized build. A bare `cmake ..` otherwise yields a no-O2,
# no-NDEBUG build; until now optimization only happened because the Dockerfiles
# passed -DCMAKE_BUILD_TYPE=Release. Skip for multi-config generators (Ninja
# Multi-Config, IDEs), which pick the config at build time instead.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

# Optional sanitizer, off by default. Applied here (before add_subdirectory) so
# the whole program - including vendored ggml - is instrumented, which
# ThreadSanitizer needs to see backend threading.
#   -DNEMO_SPEECH_SANITIZE=thread            (concurrent-streaming stress test)
#   -DNEMO_SPEECH_SANITIZE=address,undefined (memory / UB bugs)
set(NEMO_SPEECH_SANITIZE "" CACHE STRING
    "Sanitizer to enable: address | thread | undefined (comma-separated); empty = off")
if(NEMO_SPEECH_SANITIZE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-fsanitize=${NEMO_SPEECH_SANITIZE} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${NEMO_SPEECH_SANITIZE})
endif()

option(NEMO_SPEECH_WITH_FLASHLIGHT "Link flashlight-text + KenLM for LM decoding" OFF)
option(NEMO_SPEECH_WITH_GRPC       "Deprecated alias for NEMO_SPEECH_BUILD_GRPC" OFF)
option(NEMO_SPEECH_WITH_NMT        "Deprecated alias for NEMO_SPEECH_BUILD_NMT" OFF)
option(NEMO_SPEECH_WITH_NORM       "Build shared Sparrowhawk/OpenFST text normalization" OFF)
option(NEMO_SPEECH_TTS_WITH_JA     "Build Japanese MagpieTTS tokenizer support" OFF)
option(NEMO_SPEECH_TTS_WITH_ZH     "Build Mandarin MagpieTTS tokenizer support" OFF)
option(NEMO_SPEECH_BUILD_ASR       "Build automatic speech recognition" ON)
option(NEMO_SPEECH_BUILD_DIAR      "Build standalone and ASR-integrated diarization" ON)
option(NEMO_SPEECH_BUILD_TTS       "Build text-to-speech" ON)
option(NEMO_SPEECH_BUILD_NMT       "Build text translation (links llama.cpp)" ${NEMO_SPEECH_WITH_NMT})
option(NEMO_SPEECH_BUILD_CLI       "Build the unified nemo-speech CLI" ON)
option(NEMO_SPEECH_BUILD_HTTP      "Build the HTTP server and local playground" OFF)
option(NEMO_SPEECH_HTTP_TLS        "Enable TLS support in the HTTP server (requires OpenSSL)" OFF)
option(NEMO_SPEECH_BUILD_GRPC      "Build Riva-compatible gRPC adapters" ${NEMO_SPEECH_WITH_GRPC})
option(NEMO_SPEECH_GRPC_USE_CONFIG "Resolve protobuf and gRPC from CMake config packages" OFF)
option(NEMO_SPEECH_BUILD_TESTS     "Build first-party test executables" OFF)
option(BUILD_TESTING                 "Build first-party test executables" ${NEMO_SPEECH_BUILD_TESTS})
option(NEMO_SPEECH_BUILD_EXAMPLES  "Build public in-process API examples" OFF)
option(NEMO_SPEECH_BUILD_TOOLS     "Build developer diagnostic tools" OFF)
option(GGML_VULKAN                   "Enable ggml Vulkan backend"            OFF)
option(GGML_METAL                    "Enable ggml Metal backend (Apple Silicon only)" OFF)
set(NEMO_SPEECH_DEPENDENCY_PREFIX "${CMAKE_SOURCE_DIR}/.deps" CACHE PATH
    "User-writable prefix containing optional project-built dependencies")
# GGML_CUDA is forwarded directly to ggml's CMake via -DGGML_CUDA=ON.

# Keep the legacy WITH_NMT/WITH_GRPC options as downstream-compatible aliases.
if(NEMO_SPEECH_WITH_NMT)
    set(NEMO_SPEECH_BUILD_NMT ON CACHE BOOL "Build text translation (links llama.cpp)" FORCE)
endif()
if(NEMO_SPEECH_BUILD_NMT)
    set(NEMO_SPEECH_WITH_NMT ON)
endif()
if(NEMO_SPEECH_WITH_GRPC)
    set(NEMO_SPEECH_BUILD_GRPC ON CACHE BOOL "Build Riva-compatible gRPC adapters" FORCE)
endif()
if(NEMO_SPEECH_BUILD_GRPC)
    set(NEMO_SPEECH_WITH_GRPC ON)
endif()

# Reject invalid component combinations early.
if(WIN32 AND NEMO_SPEECH_WITH_NORM)
    message(FATAL_ERROR
        "NEMO_SPEECH_WITH_NORM is not supported on Windows; disable normalization")
endif()
if(NEMO_SPEECH_BUILD_GRPC AND
   (NOT NEMO_SPEECH_BUILD_ASR OR NOT NEMO_SPEECH_BUILD_TTS))
    message(FATAL_ERROR
        "NEMO_SPEECH_BUILD_GRPC currently requires both ASR and TTS")
endif()
if(NEMO_SPEECH_WITH_FLASHLIGHT AND NOT NEMO_SPEECH_BUILD_ASR)
    message(FATAL_ERROR "NEMO_SPEECH_WITH_FLASHLIGHT requires ASR")
endif()
if((NEMO_SPEECH_TTS_WITH_JA OR NEMO_SPEECH_TTS_WITH_ZH) AND
   NOT NEMO_SPEECH_BUILD_TTS)
    message(FATAL_ERROR "Optional TTS tokenizers require NEMO_SPEECH_BUILD_TTS=ON")
endif()
if(NEMO_SPEECH_HTTP_TLS AND NOT NEMO_SPEECH_BUILD_HTTP)
    message(FATAL_ERROR "NEMO_SPEECH_HTTP_TLS requires NEMO_SPEECH_BUILD_HTTP=ON")
endif()
if(NEMO_SPEECH_BUILD_TESTS)
    set(BUILD_TESTING ON CACHE BOOL "Build first-party test executables" FORCE)
elseif(BUILD_TESTING)
    set(NEMO_SPEECH_BUILD_TESTS ON)
endif()

if(NEMO_SPEECH_BUILD_DIAR AND NOT NEMO_SPEECH_BUILD_ASR)
    message(STATUS
        "Diarization currently shares the ASR runtime library; building that library "
        "without the transcribe CLI/API surface")
endif()

# Drop-in cuBLAS shim (native GEMM, no cuBLASLt). The shipping container image
# substitutes it for real cuBLAS to reduce its runtime closure; put it on
# LD_LIBRARY_PATH ahead of the system cuBLAS to reproduce that GEMM path.
# Disabled by default for source builds, which link the CUDA toolkit's cuBLAS.
# Container builds enable the shim explicitly to reduce the runtime image size.
# It is only built when GGML_CUDA is also ON (see the target below), and is a
# no-op for Metal, Vulkan, and CPU builds.
option(NEMO_SPEECH_CUBLAS_SHIM     "Build the in-tree drop-in cuBLAS shim (native GEMM, no cuBLASLt)" OFF)

# Whether the linked ggml has the project ASR patches applied (ggml-patches/:
# the fused rel-pos attention op and the F16 depthwise-conv kernel). The ASR
# directly references these (a new op symbol + F16 CONV_2D_DW behaviour), so a
# build against STANDARD upstream ggml must set -DNEMO_SPEECH_GGML_PATCHED=OFF
# - the encoder then uses only stock ggml ops (unfused rel-pos, ggml_conv_1d_dw)
# at some latency cost. The remaining low-level patches (NVFP4, skinny-q8,
# norm/BF16 fusions, CUDA graph and launch fixes) are transparent ggml-cuda internals.
option(NEMO_SPEECH_GGML_PATCHED "Linked ggml has the project ASR patches applied (fused rel-pos op, F16 dw-conv)" ON)

# Fused rel-pos attention CUDA op (GGML_OP_FUSED_RELPOS_ATTN). Replaces the
# unfused 3-GEMM + rel-shift + softmax attention sequence with one kernel in the
# non-cache-aware encoder path. Defaults ON whenever GGML_CUDA + a patched ggml
# are present (it links a patch-only op symbol); forced OFF otherwise, where the
# encoder runs the unfused stock-ggml path. Kept as an explicit option so it can
# be toggled OFF for debugging / WER bisection (the kernel stores k/v/p F16).
if(GGML_CUDA AND NEMO_SPEECH_GGML_PATCHED)
    option(NEMO_SPEECH_FUSED_RELPOS_ATTN "Use the fused rel-pos attention CUDA op in the encoder" ON)
else()
    set(NEMO_SPEECH_FUSED_RELPOS_ATTN OFF CACHE BOOL
        "Use the fused rel-pos attention CUDA op in the encoder" FORCE)
    message(STATUS
        "NEMO_SPEECH_FUSED_RELPOS_ATTN forced OFF: requires GGML_CUDA=ON and "
        "NEMO_SPEECH_GGML_PATCHED=ON (patched ggml). Using the unfused path.")
endif()

# Direct depthwise-conv kernel (GGML_OP_CONV_2D_DW) for the conformer/subsample
# convs. Faster than ggml_conv_1d_dw's im2col + matmul, but the conv weights are
# F16 and the direct CONV_2D_DW op only reads F16 kernels correctly on the
# patched CUDA backend (ggml patch 0004); stock ggml (CPU or unpatched CUDA)
# reads them as F32. Defaults ON whenever GGML_CUDA + a patched ggml are present;
# forced OFF otherwise, where nn.cpp falls back to the portable ggml_conv_1d_dw
# lowering (im2col + mul_mat, F16-safe on every backend and on stock ggml). Kept
# as an explicit option so it can be toggled OFF for debugging.
if(GGML_CUDA AND NEMO_SPEECH_GGML_PATCHED)
    option(NEMO_SPEECH_DIRECT_DW_CONV "Use the direct CUDA depthwise-conv kernel in the encoder" ON)
else()
    set(NEMO_SPEECH_DIRECT_DW_CONV OFF CACHE BOOL
        "Use the direct CUDA depthwise-conv kernel in the encoder" FORCE)
    message(STATUS
        "NEMO_SPEECH_DIRECT_DW_CONV forced OFF: requires GGML_CUDA=ON and "
        "NEMO_SPEECH_GGML_PATCHED=ON (patched ggml). Using ggml_conv_1d_dw.")
endif()

# FastConformer graph rewrites that target CUDA-only ggml ops/fusions (sigmoid
# GLU plus BF16 projection epilogues). Keep the symbols out of standard-ggml,
# CPU, Metal, and Vulkan builds. The CUDA backend performs the finer runtime
# architecture checks: native BF16 epilogues require NVIDIA SM80+, and the
# 1024-wide LayerNorm launch specialization requires SM90+.
if(GGML_CUDA AND NEMO_SPEECH_GGML_PATCHED)
    option(NEMO_SPEECH_FASTCONFORMER_CUDA_FUSIONS
        "Use patched CUDA FastConformer graph fusions" ON)
else()
    set(NEMO_SPEECH_FASTCONFORMER_CUDA_FUSIONS OFF CACHE BOOL
        "Use patched CUDA FastConformer graph fusions" FORCE)
endif()

set(GGML_DEPENDENCIES ggml ggml-base ggml-cpu)
if(GGML_CUDA)
    list(APPEND GGML_DEPENDENCIES ggml-cuda)
    # libggml-cuda.so has libcuda.so.1 (the CUDA driver) as a NEEDED entry.
    # The driver is provided at runtime by NVIDIA's container toolkit and is
    # absent from the devel image - only the stub at
    # ${CUDAToolkit_LIBRARY_DIR}/stubs/libcuda.so exists, with no symlink to
    # libcuda.so.1. ld's rpath-link search matches NEEDED entries by exact
    # filename, so pointing it at the stub dir doesn't resolve libcuda.so.1.
    # Resolve it via CUDA::cuda_driver instead (linked PUBLIC on runtime_ggml
    # below) - the imported target points ld at the stub file directly.
    find_package(CUDAToolkit REQUIRED)
    # CUDA Graphs (capture + replay each cached graph's kernel-launch sequence)
    # on by default. ggml-cuda gates it at runtime and disables it for graphs it
    # can't capture (e.g. shapes that change between calls).
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "Enable ggml-cuda CUDA Graphs")
    file(READ "${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda/conv-transpose-1d.cu" GGML_CUDA_CONV_TRANSPOSE_1D)
    if(NOT GGML_CUDA_CONV_TRANSPOSE_1D MATCHES "conv_transpose_1d_use_nanocodec_grouped2")
        message(WARNING
            "MagpieTTS/NanoCodec CUDA ggml patch is not applied. "
            "Run scripts/apply-ggml-patches.sh before compiling CUDA TTS targets.")
    endif()
endif()
if(GGML_VULKAN)
    list(APPEND GGML_DEPENDENCIES ggml-vulkan)
endif()
if(GGML_METAL)
    list(APPEND GGML_DEPENDENCIES ggml-metal)
endif()

add_subdirectory(ggml EXCLUDE_FROM_ALL)

# ggml is an implementation dependency, so install only the runtime libraries
# required by our shared libraries. Its C++ headers, CMake package, pkg-config
# metadata, and tools are not part of the public NeMo-Speech.cpp SDK.
set_property(TARGET ggml PROPERTY PUBLIC_HEADER "")
install(TARGETS ggml ggml-base ggml-cpu
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(TARGET ggml-blas)
    install(TARGETS ggml-blas
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(GGML_CUDA)
    install(TARGETS ggml-cuda
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(GGML_VULKAN)
    install(TARGETS ggml-vulkan
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(GGML_METAL)
    install(TARGETS ggml-metal
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# In-tree cuBLAS shim backed by native GEMM kernels (no cuBLASLt). ggml-cuda
# links real cuBLAS at build time, but the shim shares its major-version SONAME,
# so binaries resolve it instead when it is first on LD_LIBRARY_PATH.
# The cuBLAS shim is a Linux-only container size optimization. It relies on ELF
# SONAME and symbol versioning plus a GNU-ld --version-script, neither of which
# exists with MSVC/link.exe. On Windows, ggml-cuda links the real cuBLAS DLL
# from the CUDA toolkit, so the shim is neither needed nor buildable - skip it.
if(GGML_CUDA AND NEMO_SPEECH_CUBLAS_SHIM AND NOT WIN32)
    enable_language(CUDA)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
        set(NEMO_SPEECH_CUBLAS_SOVERSION 12)
    else()
        set(NEMO_SPEECH_CUBLAS_SOVERSION 13)
    endif()
    add_library(nemo_speech_cublas_shim SHARED kernels/cublas_shim.cu)
    if(CMAKE_CUDA_ARCHITECTURES)
        set(NEMO_SPEECH_CUBLAS_SHIM_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        # Fallback for ad-hoc builds that do not set an architecture. compute_80 PTX is
        # JIT-portable across sm_80..Blackwell, including Thor.
        set(NEMO_SPEECH_CUBLAS_SHIM_ARCHITECTURES "80-virtual")
    endif()
    set_target_properties(nemo_speech_cublas_shim PROPERTIES
        OUTPUT_NAME cublas
        SOVERSION "${NEMO_SPEECH_CUBLAS_SOVERSION}"
        CUDA_ARCHITECTURES "${NEMO_SPEECH_CUBLAS_SHIM_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON)
    configure_file(
        kernels/ver_cublas.map
        "${CMAKE_CURRENT_BINARY_DIR}/ver_cublas.map"
        @ONLY)
    target_link_options(nemo_speech_cublas_shim PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/ver_cublas.map")
elseif(GGML_CUDA AND NEMO_SPEECH_CUBLAS_SHIM AND WIN32)
    message(STATUS
        "NEMO_SPEECH_CUBLAS_SHIM: skipped on Windows; linking the CUDA "
        "toolkit's real cuBLAS instead.")
endif()

if(NEMO_SPEECH_WITH_FLASHLIGHT)
    # Define only KenLM's runtime scoring libraries. flashlight-text checks
    # `TARGET kenlm::kenlm` before its own find_package / FetchContent fallback.
    # Keep LGPL KenLM replaceable while absorbing permissively licensed
    # flashlight-text into the ASR shared library.
    set(_nemo_speech_saved_build_shared "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS ON)
    include(cmake/kenlm_runtime.cmake)

    # flashlight-text has no LGPL replacement requirement and is intentionally
    # kept private inside libnemo_speech_asr.
    set(BUILD_SHARED_LIBS OFF)
    set(FL_TEXT_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
    set(FL_TEXT_BUILD_STANDALONE OFF CACHE BOOL "" FORCE)
    set(FL_TEXT_BUILD_PYTHON     OFF CACHE BOOL "" FORCE)
    set(FL_TEXT_USE_KENLM        ON  CACHE BOOL "" FORCE)
    set(FL_TEXT_KENLM_MAX_ORDER  6   CACHE STRING "" FORCE)
    add_subdirectory(third_party/flashlight-text EXCLUDE_FROM_ALL)
    set(BUILD_SHARED_LIBS "${_nemo_speech_saved_build_shared}")
endif()

# First-party warnings. Placed after the vendored subdirectories (ggml, kenlm,
# flashlight-text) so they aren't compiled under our warning set; applies to
# src/* and tests below. Not -Werror - surface issues without breaking builds.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Scope to C/C++ only: on ARM64-Windows the C++ compiler is clang-cl (so
    # this branch is taken) while nvcc's host compiler is cl.exe, which rejects
    # -Wextra (D8021). A bare add_compile_options would reach the CUDA compile.
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra>")
endif()

# llama.cpp for the NMT decoder. It reuses the ggml target added above (its
# CMake builds its own ggml only when the target is absent), so one ggml backend
# is shared across asr/tts/nmt. Build only libllama.
if(NEMO_SPEECH_WITH_NMT)
    set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
    add_subdirectory(llama.cpp EXCLUDE_FROM_ALL)
    set_property(TARGET llama PROPERTY PUBLIC_HEADER "")
    install(TARGETS llama
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

add_subdirectory(src/runtime/ggml)
add_subdirectory(src/common)
if(NEMO_SPEECH_BUILD_HTTP)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpp-httplib/httplib.h")
        message(FATAL_ERROR
            "cpp-httplib is required for HTTP serving; "
            "run: git submodule update --init third_party/cpp-httplib")
    endif()
    set(HTTPLIB_REQUIRE_OPENSSL ${NEMO_SPEECH_HTTP_TLS} CACHE BOOL "" FORCE)
    set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE ${NEMO_SPEECH_HTTP_TLS} CACHE BOOL "" FORCE)
    set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
    set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
    set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
    set(HTTPLIB_INSTALL OFF CACHE BOOL "" FORCE)
    set(HTTPLIB_TEST OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/cpp-httplib EXCLUDE_FROM_ALL)
endif()
if(NEMO_SPEECH_BUILD_ASR OR NEMO_SPEECH_BUILD_DIAR)
    add_subdirectory(src/asr)
endif()
if(NEMO_SPEECH_BUILD_TTS)
    add_subdirectory(src/tts)
endif()
if(NEMO_SPEECH_BUILD_NMT)
    add_subdirectory(src/nmt)
    if(NEMO_SPEECH_BUILD_ASR)
        add_subdirectory(src/speech)
    endif()
endif()
if(NEMO_SPEECH_BUILD_ASR OR NEMO_SPEECH_BUILD_DIAR OR
   NEMO_SPEECH_BUILD_TTS OR NEMO_SPEECH_BUILD_NMT OR
   NEMO_SPEECH_BUILD_CLI)
    add_subdirectory(src/core)
endif()
if(NEMO_SPEECH_BUILD_HTTP)
    add_subdirectory(server/http)
endif()
if(NEMO_SPEECH_BUILD_GRPC)
    add_subdirectory(proto)
    add_subdirectory(src/services)
    add_subdirectory(src/server)
endif()
if(NEMO_SPEECH_BUILD_CLI)
    add_subdirectory(app)
endif()
if(NEMO_SPEECH_BUILD_EXAMPLES OR BUILD_TESTING)
    add_subdirectory(examples)
endif()
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
if(NEMO_SPEECH_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(NEMO_SPEECH_BUILD_ASR OR NEMO_SPEECH_BUILD_DIAR)
    install(FILES include/nemo_speech/asr.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nemo_speech)
endif()
if(NEMO_SPEECH_BUILD_DIAR)
    install(FILES include/nemo_speech/diar.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nemo_speech)
endif()
if(NEMO_SPEECH_BUILD_NMT)
    install(FILES include/nemo_speech/nmt.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nemo_speech)
endif()
if(NEMO_SPEECH_BUILD_TTS)
    install(FILES include/nemo_speech/tts.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nemo_speech)
endif()

set(NEMO_SPEECH_THIRD_PARTY_LICENSE_DIR
    "${NEMO_SPEECH_LICENSE_INSTALL_DIR}/third_party")
if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    file(GLOB _NEMO_SPEECH_VCPKG_COPYRIGHTS LIST_DIRECTORIES false
        "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/*/copyright")
    foreach(_copyright IN LISTS _NEMO_SPEECH_VCPKG_COPYRIGHTS)
        get_filename_component(_share_dir "${_copyright}" DIRECTORY)
        get_filename_component(_package "${_share_dir}" NAME)
        install(FILES "${_copyright}"
            DESTINATION "${NEMO_SPEECH_THIRD_PARTY_LICENSE_DIR}/vcpkg/${_package}"
            RENAME LICENSE)
    endforeach()
endif()
install(FILES ggml/LICENSE
    DESTINATION "${NEMO_SPEECH_THIRD_PARTY_LICENSE_DIR}/ggml")
if(TARGET nemo_speech_cublas_shim)
    install(TARGETS nemo_speech_cublas_shim
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(NEMO_SPEECH_BUILD_HTTP)
    install(FILES third_party/cpp-httplib/LICENSE
        DESTINATION "${NEMO_SPEECH_THIRD_PARTY_LICENSE_DIR}/cpp-httplib")
endif()
if(NEMO_SPEECH_BUILD_NMT)
    install(FILES llama.cpp/LICENSE
        DESTINATION "${NEMO_SPEECH_THIRD_PARTY_LICENSE_DIR}/llama.cpp")
endif()
if(NEMO_SPEECH_BUILD_GRPC)
    install(FILES proto/riva-common/LICENSE
        DESTINATION "${NEMO_SPEECH_THIRD_PARTY_LICENSE_DIR}/riva-common")
endif()
if(NEMO_SPEECH_WITH_FLASHLIGHT)
    install(FILES third_party/flashlight-text/LICENSE
        DESTINATION "${NEMO_SPEECH_THIRD_PARTY_LICENSE_DIR}/flashlight-text")
    install(FILES
        third_party/kenlm/LICENSE
        third_party/kenlm/COPYING
        third_party/kenlm/COPYING.3
        third_party/kenlm/COPYING.LESSER.3
        DESTINATION "${NEMO_SPEECH_THIRD_PARTY_LICENSE_DIR}/kenlm")
endif()
if(NEMO_SPEECH_TTS_WITH_JA)
    install(FILES
        third_party/open_jtalk/src/COPYING
        DESTINATION "${NEMO_SPEECH_THIRD_PARTY_LICENSE_DIR}/open_jtalk")
    install(FILES
        third_party/open_jtalk/src/mecab/COPYING
        DESTINATION "${NEMO_SPEECH_THIRD_PARTY_LICENSE_DIR}/mecab")
    install(FILES
        third_party/open_jtalk/src/mecab-naist-jdic/COPYING
        DESTINATION "${NEMO_SPEECH_THIRD_PARTY_LICENSE_DIR}/naist-jdic")
endif()
if(NEMO_SPEECH_TTS_WITH_ZH)
    install(FILES third_party/cppjieba/LICENSE
        DESTINATION "${NEMO_SPEECH_THIRD_PARTY_LICENSE_DIR}/cppjieba")
    install(FILES third_party/cppjieba/deps/limonp/LICENSE
        DESTINATION "${NEMO_SPEECH_THIRD_PARTY_LICENSE_DIR}/limonp")
endif()

configure_package_config_file(
    cmake/NeMoSpeechConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NeMoSpeechConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeMoSpeech)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/NeMoSpeechConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NeMoSpeechConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NeMoSpeechConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeMoSpeech)
