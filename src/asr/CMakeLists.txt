# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ASR pipeline. Builds as SHARED so direct C++ consumers can link it as the
# single user-facing .so for this pipeline. Absorbs runtime_ggml internally
# (static lib, hidden visibility) so its symbols don't leak into the public
# ABI.
file(GLOB ASR_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/features/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/encoder/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/decoders/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/vad/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/diar/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/postproc/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pnc/*.cpp"
)
# c_api.cpp is the C ABI boundary - built into its own hidden-visibility lib
# (nemo_speech_asr_c) below, not the default-visibility C++ lib.
list(REMOVE_ITEM ASR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/c_api.cpp")

add_library(nemo_speech_asr SHARED ${ASR_SOURCES})

if(NEMO_SPEECH_GGML_PATCHED)
    target_compile_definitions(nemo_speech_asr PRIVATE NEMO_SPEECH_GGML_PATCHED=1)
endif()

# Windows portability: this lib relies on Linux default symbol visibility to
# export its public API (AsrModel, runners, read_wav_mono_16k). MSVC/clang-cl
# export nothing from a DLL unless annotated with __declspec(dllexport), so the
# generated import lib is empty and direct C++ consumers (tests/, riva_server)
# fail to link with "undefined symbol". Mirror the Linux behaviour by
# auto-exporting all symbols. No-op on non-Windows platforms.
if(WIN32)
    set_target_properties(nemo_speech_asr PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_include_directories(nemo_speech_asr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/features
    ${CMAKE_CURRENT_SOURCE_DIR}/encoder
    ${CMAKE_CURRENT_SOURCE_DIR}/decoders
    ${CMAKE_CURRENT_SOURCE_DIR}/vad
    ${CMAKE_CURRENT_SOURCE_DIR}/diar
    ${CMAKE_CURRENT_SOURCE_DIR}/postproc
    ${CMAKE_CURRENT_SOURCE_DIR}/pnc
)

# Generic config parser (ParameterParser), shared with the other pipelines.
# PUBLIC so config headers' `#include "parameter_parser.h"` and the inline
# Register() methods reach every consumer (server, tests, C ABI).
target_link_libraries(nemo_speech_asr PUBLIC nemo_speech_common)

# Inverse text normalization through the shared WFST runtime. Heavy deps remain
# OFF by default; see scripts/build_itn_deps.sh.
if(NEMO_SPEECH_WITH_NORM)
    target_compile_definitions(nemo_speech_asr PUBLIC NEMO_SPEECH_WITH_NORM=1)
    target_link_libraries(nemo_speech_asr PRIVATE nemo_speech_text_normalization)
endif()

# Link the runtime PUBLIC so the runtime/ggml headers (runtime.h,
# BackendManager) reach direct C++ consumers - the AsrModel ctor takes a
# BackendManager&, so callers need that header. The static lib's symbols are
# hidden via -fvisibility=hidden on the runtime target itself, so they don't
# leak through this .so's ABI.
target_link_libraries(nemo_speech_asr PUBLIC nemo_speech_runtime_ggml)

# SentencePiece is a core ASR dependency: the RNNT head tokenizes word-boosting
# phrases with the model's embedded tokenizer (context biasing in
# rnnt_greedy_decoder.cpp / RnntModel::encode_phrase); the flashlight decoder
# also encodes OOV boost phrases with it. Prefer a static archive on ELF
# platforms so its bundled protobuf symbols stay private.
if(UNIX AND NOT APPLE)
    set(_NEMO_SPEECH_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    find_library(SENTENCEPIECE_STATIC_LIB sentencepiece
        HINTS "${NEMO_SPEECH_DEPENDENCY_PREFIX}/sentencepiece/lib")
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_NEMO_SPEECH_LIBRARY_SUFFIXES}")
    unset(_NEMO_SPEECH_LIBRARY_SUFFIXES)
endif()
if(SENTENCEPIECE_STATIC_LIB)
    find_path(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h
        HINTS "${NEMO_SPEECH_DEPENDENCY_PREFIX}/sentencepiece/include")
    if(NOT SENTENCEPIECE_INCLUDE_DIR)
        message(FATAL_ERROR "SentencePiece archive found without sentencepiece_processor.h")
    endif()
    target_link_libraries(nemo_speech_asr PRIVATE ${SENTENCEPIECE_STATIC_LIB})
    target_include_directories(nemo_speech_asr PRIVATE ${SENTENCEPIECE_INCLUDE_DIR})
    target_link_options(
        nemo_speech_asr PRIVATE "LINKER:--exclude-libs,libsentencepiece.a")
elseif(NEMO_SPEECH_WITH_NORM AND UNIX AND NOT APPLE)
    message(FATAL_ERROR
        "normalization requires private static SentencePiece; "
        "run scripts/build_sentencepiece_static.sh")
else()
    # vcpkg's imported target carries its platform-specific usage requirements.
    find_package(unofficial-sentencepiece CONFIG QUIET)
    if(TARGET unofficial::sentencepiece::sentencepiece)
        target_link_libraries(
            nemo_speech_asr PRIVATE unofficial::sentencepiece::sentencepiece)
    else()
        find_library(SENTENCEPIECE_LIB sentencepiece REQUIRED)
        find_path(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h REQUIRED)
        target_link_libraries(nemo_speech_asr PRIVATE ${SENTENCEPIECE_LIB})
        target_include_directories(nemo_speech_asr PRIVATE ${SENTENCEPIECE_INCLUDE_DIR})
        # When sentencepiece_processor.h includes abseil (upstream since 0.2.1),
        # our objects reference absl::Status directly. System packages (Homebrew)
        # do not propagate that link dependency; add it only when the header
        # needs it.
        file(STRINGS "${SENTENCEPIECE_INCLUDE_DIR}/sentencepiece_processor.h"
            _NEMO_SPEECH_SENTENCEPIECE_ABSL REGEX "#include [<\"](third_party/)?absl/")
        if(_NEMO_SPEECH_SENTENCEPIECE_ABSL)
            find_package(absl CONFIG QUIET)
            if(absl_FOUND)
                target_link_libraries(nemo_speech_asr PRIVATE absl::status absl::strings)
            else()
                message(WARNING
                    "sentencepiece_processor.h includes abseil but no abseil CMake package "
                    "was found; linking nemo_speech_asr may fail")
            endif()
        endif()
        unset(_NEMO_SPEECH_SENTENCEPIECE_ABSL)
    endif()
endif()
set(_NEMO_SPEECH_SENTENCEPIECE_LICENSE_DIR
    "${NEMO_SPEECH_DEPENDENCY_PREFIX}/sentencepiece/share/licenses/nemo-speech/third_party/sentencepiece")
if(EXISTS "${_NEMO_SPEECH_SENTENCEPIECE_LICENSE_DIR}/LICENSE")
    install(DIRECTORY "${_NEMO_SPEECH_SENTENCEPIECE_LICENSE_DIR}/"
        DESTINATION "${NEMO_SPEECH_LICENSE_INSTALL_DIR}/third_party/sentencepiece")
endif()
# SentencePiece (vcpkg) uses absl::flat_hash_map + protobuf internally, but its
# config target doesn't propagate those link deps. On Windows/vcpkg, add them
# explicitly; elsewhere SentencePiece links its own deps (apt's -dev package).
if(WIN32)
    find_package(absl CONFIG QUIET)
    if(absl_FOUND)
        target_link_libraries(nemo_speech_asr PRIVATE
            absl::flat_hash_map absl::flat_hash_set absl::strings absl::str_format absl::hash)
    endif()
    find_package(protobuf CONFIG QUIET)
    if(TARGET protobuf::libprotobuf)
        target_link_libraries(nemo_speech_asr PRIVATE protobuf::libprotobuf)
    endif()
endif()

if(NEMO_SPEECH_WITH_FLASHLIGHT)
    target_compile_definitions(nemo_speech_asr PUBLIC NEMO_SPEECH_WITH_FLASHLIGHT=1)
    target_link_libraries(nemo_speech_asr PRIVATE flashlight-text flashlight-text-kenlm)
endif()

if(NEMO_SPEECH_FUSED_RELPOS_ATTN)
    target_compile_definitions(nemo_speech_asr PRIVATE NEMO_SPEECH_FUSED_RELPOS_ATTN=1)
endif()

if(NEMO_SPEECH_DIRECT_DW_CONV)
    # The encoder's channels-inner conv-module path uses the cwhn direct
    # depthwise kernel (F16-safe only on the patched CUDA backend, like the
    # nn.cpp Conv1D fast path this flag already gates).
    target_compile_definitions(nemo_speech_asr PRIVATE NEMO_SPEECH_DIRECT_DW_CONV=1)
endif()

if(NEMO_SPEECH_FASTCONFORMER_CUDA_FUSIONS)
    target_compile_definitions(
        nemo_speech_asr PRIVATE NEMO_SPEECH_FASTCONFORMER_CUDA_FUSIONS=1)
endif()

# ---------------------------------------------------------------------------
# Stable C ABI shared library (nemo_speech_asr_*). Hidden visibility + a version script
# so only the nemo_speech_asr_* symbols are exported; the C++ implementation is linked
# in privately. This is the installed binary contract.
# ---------------------------------------------------------------------------
add_library(nemo_speech_asr_c SHARED c_api.cpp)
target_link_libraries(nemo_speech_asr_c PRIVATE nemo_speech_asr)
target_include_directories(nemo_speech_asr_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_definitions(nemo_speech_asr_c PRIVATE NEMO_SPEECH_ASR_BUILD)
set_target_properties(nemo_speech_asr_c PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    SOVERSION 1)
if(UNIX AND NOT APPLE)
    target_link_options(nemo_speech_asr_c PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/c_api.map")
endif()

install(TARGETS nemo_speech_asr
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS nemo_speech_asr_c
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
