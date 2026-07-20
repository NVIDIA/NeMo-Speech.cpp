# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set(MAGPIETTS_MODEL_SOURCES
    audio_pp.cpp
    decoder.cpp
    encoder.cpp
    lt.cpp
    model.cpp
)

add_library(nemo_speech_tts SHARED
    ${MAGPIETTS_MODEL_SOURCES}
    config.cpp
    ../c_api.cpp
    ../synthesizer.cpp
    magpietts.cpp
    runtime.cpp
)

# Windows: this DLL relies on Linux default symbol visibility to export its C++ API.
# MSVC/clang-cl export nothing from a DLL unless annotated, so direct C++
# consumers (the TTS C ABI lib, the gRPC TTS service, and smoke tests) would
# fail to link. Auto-export all symbols to mirror Linux.
# Matches nemo_speech_asr; no-op off Windows.
if(WIN32)
    set_target_properties(nemo_speech_tts PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_include_directories(nemo_speech_tts PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../common>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(nemo_speech_tts PUBLIC
    ggml
    nemo_speech_common
    nemo_speech_tts_nanocodec_obj
    nemo_speech_tts_preproc
    nemo_speech_tts_tokenizer_obj
)

# Stable C ABI (nemo_speech_tts_*) and the in-tree C++ API share one implementation
# library. Aliases preserve existing target names without producing extra DSOs.
target_compile_definitions(nemo_speech_tts PRIVATE NEMO_SPEECH_TTS_BUILD)
set_target_properties(nemo_speech_tts PROPERTIES SOVERSION 1)
if(UNIX AND NOT APPLE)
    target_link_options(nemo_speech_tts PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../c_api.map")
endif()

install(TARGETS nemo_speech_tts
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
add_library(nemo_speech_tts_magpietts ALIAS nemo_speech_tts)
add_library(nemo_speech_tts_nanocodec ALIAS nemo_speech_tts)
add_library(nemo_speech_tts_tokenizer ALIAS nemo_speech_tts)
add_library(nemo_speech_tts_c ALIAS nemo_speech_tts)

set(MAGPIETTS_TARGETS
    nemo_speech_tts
)

foreach(target ${MAGPIETTS_TARGETS})
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    )
endforeach()

if(GGML_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND CUDAToolkit_INCLUDE_DIRS)
        foreach(target ${MAGPIETTS_TARGETS})
            target_include_directories(${target} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        endforeach()
    endif()
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        foreach(target ${MAGPIETTS_TARGETS})
            target_sources(${target} PRIVATE magpietts_cuda_sampling.cu)
            target_compile_definitions(${target} PRIVATE GGML_USE_CUDA MAGPIETTS_CUDA_SAMPLING)
            target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::cuda_driver)
            set_target_properties(${target} PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED ON
            )
        endforeach()
        if(TARGET ggml-cuda)
            get_target_property(MAGPIETTS_CUDA_ARCHITECTURES ggml-cuda CUDA_ARCHITECTURES)
        endif()
        if(NOT MAGPIETTS_CUDA_ARCHITECTURES AND DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
            set(MAGPIETTS_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
        endif()
        if(MAGPIETTS_CUDA_ARCHITECTURES)
            foreach(target ${MAGPIETTS_TARGETS})
                set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "${MAGPIETTS_CUDA_ARCHITECTURES}")
            endforeach()
        endif()
    endif()
endif()

if(GGML_METAL)
    foreach(target ${MAGPIETTS_TARGETS})
        target_compile_definitions(${target} PRIVATE GGML_USE_METAL)
    endforeach()
endif()
