# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Generic, domain-agnostic config parser (ParameterParser). Shared by every
# pipeline (asr, tts, s2s, nmt). STATIC + PIC so it links cleanly into each
# pipeline's shared lib; default visibility so the inline Register() methods in
# config headers (which call into here) resolve in whatever binary instantiates
# them. The config-file format is a small nested key/value subset parsed in-house
# (parameter_parser.cpp), so there is no external YAML dependency.
add_library(nemo_speech_common STATIC
    audio_decoder.cpp
    audio_file.cpp
    audio_resampler.cpp
    json.cpp
    parameter_parser.cpp
)

# Public so config headers' `#include "parameter_parser.h"` reaches every
# consumer (server, tests, C ABI). nvtx_utils.h also lives here.
target_include_directories(nemo_speech_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(nemo_speech_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Shared WFST grammar runtime for ASR ITN and TTS TN. Keep this separate from
# nemo_speech_common so builds that do not request text normalization retain
# no OpenFST/Sparrowhawk dependency.
if(NEMO_SPEECH_WITH_NORM)
    add_library(nemo_speech_text_normalization STATIC
        text_normalization/fst_normalizer.cpp)
    target_compile_definitions(
        nemo_speech_text_normalization PUBLIC NEMO_SPEECH_WITH_NORM=1)

    set(_NEMO_SPEECH_ITN_PREFIX "${NEMO_SPEECH_DEPENDENCY_PREFIX}/itn")
    find_library(SPARROWHAWK_LIB sparrowhawk HINTS "${_NEMO_SPEECH_ITN_PREFIX}/lib" REQUIRED)
    find_library(FSTFAR_LIB fstfar HINTS "${_NEMO_SPEECH_ITN_PREFIX}/lib" REQUIRED)
    find_library(FST_LIB fst HINTS "${_NEMO_SPEECH_ITN_PREFIX}/lib" REQUIRED)
    find_library(RE2_LIB re2 REQUIRED)
    find_path(SPARROWHAWK_INCLUDE_DIR sparrowhawk/normalizer.h
        HINTS "${_NEMO_SPEECH_ITN_PREFIX}/include" REQUIRED)
    find_path(OPENFST_INCLUDE_DIR fst/fst.h HINTS "${_NEMO_SPEECH_ITN_PREFIX}/include" REQUIRED)

    find_package(Protobuf REQUIRED)
    find_library(ITN_PROTOBUF_LIB
        NAMES libprotobuf.so.25.3.0 libprotobuf.so.25
        PATHS /lib /lib/x86_64-linux-gnu /usr/lib /usr/lib/x86_64-linux-gnu
        NO_DEFAULT_PATH)
    if(NOT ITN_PROTOBUF_LIB)
        if(TARGET protobuf::libprotobuf)
            set(ITN_PROTOBUF_LIB protobuf::libprotobuf)
        else()
            set(ITN_PROTOBUF_LIB protobuf)
        endif()
    endif()
    find_library(UTF8_RANGE_LIB NAMES utf8_range_lib PATHS /usr/lib /lib)
    file(GLOB ABSL_LIBS /usr/lib/libabsl_*.so /lib/libabsl_*.so)
    set(ITN_PROTOBUF_LIBS ${ITN_PROTOBUF_LIB})
    if(UTF8_RANGE_LIB)
        list(APPEND ITN_PROTOBUF_LIBS ${UTF8_RANGE_LIB})
    endif()

    target_include_directories(nemo_speech_text_normalization
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/text_normalization
            ${CMAKE_CURRENT_SOURCE_DIR}/text_normalization/compat)
    target_include_directories(nemo_speech_text_normalization SYSTEM
        PUBLIC
            ${OPENFST_INCLUDE_DIR}
        PRIVATE
            ${SPARROWHAWK_INCLUDE_DIR})
    target_link_libraries(nemo_speech_text_normalization PUBLIC
        ${SPARROWHAWK_LIB} ${FSTFAR_LIB} ${FST_LIB}
        ${ITN_PROTOBUF_LIBS}
        -Wl,--as-needed ${ABSL_LIBS} -Wl,--no-as-needed
        ${RE2_LIB})
    set_target_properties(
        nemo_speech_text_normalization PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
