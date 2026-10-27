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

include(${CMAKE_SOURCE_DIR}/cmake/sentencepiece.cmake)
nemo_speech_link_sentencepiece(nemo_speech_asr)
set(_NEMO_SPEECH_SENTENCEPIECE_LICENSE_DIR
    "${NEMO_SPEECH_DEPENDENCY_PREFIX}/sentencepiece/share/licenses/nemo-speech/third_party/sentencepiece")
if(EXISTS "${_NEMO_SPEECH_SENTENCEPIECE_LICENSE_DIR}/LICENSE")
    install(DIRECTORY "${_NEMO_SPEECH_SENTENCEPIECE_LICENSE_DIR}/"
        DESTINATION "${NEMO_SPEECH_LICENSE_INSTALL_DIR}/third_party/sentencepiece")
endif()

if(NEMO_SPEECH_WITH_FLASHLIGHT)
    target_compile_definitions(nemo_speech_asr PUBLIC NEMO_SPEECH_WITH_FLASHLIGHT=1)
    target_link_libraries(nemo_speech_asr PRIVATE flashlight-text flashlight-text-kenlm)
endif()

# ---------------------------------------------------------------------------
# Stable C ABI shared library (nemo_speech_asr_*). Hidden visibility + a version script
# so only the nemo_speech_asr_* symbols are exported; the C++ implementation is linked
# in privately. This is the installed binary contract.
# ---------------------------------------------------------------------------
add_library(nemo_speech_asr_c SHARED c_api.cpp)
target_link_libraries(nemo_speech_asr_c PRIVATE nemo_speech_asr)
# SALM selection (nemo_speech_asr_salm_config). nemo_speech_speechlm is defined later
# (src/speechlm, which needs this ASR library); CMake resolves the target at generation.
if(NEMO_SPEECH_BUILD_SPEECHLM)
    target_link_libraries(nemo_speech_asr_c PRIVATE nemo_speech_speechlm)
    target_compile_definitions(nemo_speech_asr_c PRIVATE NEMO_SPEECH_C_API_SALM=1)
endif()
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
