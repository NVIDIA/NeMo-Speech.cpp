# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
include_guard(GLOBAL)
function(nemo_speech_link_sentencepiece target)
    # SentencePiece is a core ASR dependency: the RNNT head tokenizes word-boosting
    # phrases with the model's embedded tokenizer (context biasing in
    # rnnt_greedy_decoder.cpp / RnntModel::encode_phrase); the flashlight decoder
    # also encodes OOV boost phrases with it. Prefer a static archive on ELF and
    # Apple platforms so its bundled protobuf symbols stay private.
    if(UNIX AND NOT APPLE)
        set(_NEMO_SPEECH_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
        find_library(SENTENCEPIECE_STATIC_LIB sentencepiece
            HINTS "${NEMO_SPEECH_DEPENDENCY_PREFIX}/sentencepiece/lib")
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${_NEMO_SPEECH_LIBRARY_SUFFIXES}")
        unset(_NEMO_SPEECH_LIBRARY_SUFFIXES)
    elseif(APPLE)
        # Only the project-built archive: Homebrew's libsentencepiece.a needs its
        # shared abseil, which release archives cannot carry.
        find_library(SENTENCEPIECE_STATIC_LIB libsentencepiece.a
            PATHS "${NEMO_SPEECH_DEPENDENCY_PREFIX}/sentencepiece/lib" NO_DEFAULT_PATH)
    endif()
    if(SENTENCEPIECE_STATIC_LIB)
        find_path(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h
            HINTS "${NEMO_SPEECH_DEPENDENCY_PREFIX}/sentencepiece/include")
        if(NOT SENTENCEPIECE_INCLUDE_DIR)
            message(FATAL_ERROR "SentencePiece archive found without sentencepiece_processor.h")
        endif()
        target_include_directories(${target} PRIVATE ${SENTENCEPIECE_INCLUDE_DIR})
        if(APPLE)
            target_link_options(
                ${target} PRIVATE "LINKER:-load_hidden,${SENTENCEPIECE_STATIC_LIB}")
        else()
            target_link_libraries(${target} PRIVATE ${SENTENCEPIECE_STATIC_LIB})
            target_link_options(
                ${target} PRIVATE "LINKER:--exclude-libs,libsentencepiece.a")
        endif()
    elseif(NEMO_SPEECH_WITH_NORM AND UNIX AND NOT APPLE)
        message(FATAL_ERROR
            "normalization requires private static SentencePiece; "
            "run scripts/build_sentencepiece_static.sh")
    else()
        # vcpkg's imported target carries its platform-specific usage requirements.
        find_package(unofficial-sentencepiece CONFIG QUIET)
        if(TARGET unofficial::sentencepiece::sentencepiece)
            target_link_libraries(
                ${target} PRIVATE unofficial::sentencepiece::sentencepiece)
        else()
            find_library(SENTENCEPIECE_LIB sentencepiece REQUIRED)
            find_path(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h REQUIRED)
            target_link_libraries(${target} PRIVATE ${SENTENCEPIECE_LIB})
            target_include_directories(${target} PRIVATE ${SENTENCEPIECE_INCLUDE_DIR})
            # When sentencepiece_processor.h includes abseil (upstream since 0.2.1),
            # our objects reference absl::Status directly. System packages (Homebrew)
            # do not propagate that link dependency; add it only when the header
            # needs it.
            file(STRINGS "${SENTENCEPIECE_INCLUDE_DIR}/sentencepiece_processor.h"
                _NEMO_SPEECH_SENTENCEPIECE_ABSL REGEX "#include [<\"](third_party/)?absl/")
            if(_NEMO_SPEECH_SENTENCEPIECE_ABSL)
                find_package(absl CONFIG QUIET)
                if(absl_FOUND)
                    target_link_libraries(${target} PRIVATE absl::status absl::strings)
                else()
                    message(WARNING
                        "sentencepiece_processor.h includes abseil but no abseil CMake package "
                        "was found; linking ${target} may fail")
                endif()
            endif()
            unset(_NEMO_SPEECH_SENTENCEPIECE_ABSL)
        endif()
    endif()
    # SentencePiece (vcpkg) uses absl::flat_hash_map + protobuf internally, but its
    # config target doesn't propagate those link deps. On Windows/vcpkg, add them
    # explicitly; elsewhere SentencePiece links its own deps (apt's -dev package).
    if(WIN32)
        find_package(absl CONFIG QUIET)
        if(absl_FOUND)
            target_link_libraries(${target} PRIVATE
                absl::flat_hash_map absl::flat_hash_set absl::strings absl::str_format absl::hash)
        endif()
        find_package(protobuf CONFIG QUIET)
        if(TARGET protobuf::libprotobuf)
            target_link_libraries(${target} PRIVATE protobuf::libprotobuf)
        endif()
    endif()
endfunction()
