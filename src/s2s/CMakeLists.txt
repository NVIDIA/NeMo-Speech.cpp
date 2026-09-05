# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

file(GLOB_RECURSE S2S_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(nemo_speech_s2s SHARED ${S2S_SOURCES})
if(WIN32)
    set_target_properties(nemo_speech_s2s PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_include_directories(nemo_speech_s2s
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/llama.cpp/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/codec
        ${CMAKE_CURRENT_SOURCE_DIR}/eartts
        ${CMAKE_CURRENT_SOURCE_DIR}/llm
        ${CMAKE_CURRENT_SOURCE_DIR}/perception
        ${CMAKE_SOURCE_DIR}/llama.cpp/vendor
)

# The perception/RNN-T side reuses the ASR model implementation and its
# dynamic batching infrastructure. llama.cpp supplies the Nemotron-H and
# Gemma-3 recurrent backbones while sharing this project's patched ggml target.
target_link_libraries(nemo_speech_s2s
    PUBLIC
        nemo_speech_asr
    PRIVATE
        llama
)

if(GGML_CUDA)
    target_link_libraries(nemo_speech_s2s PUBLIC CUDA::cuda_driver)
endif()

install(TARGETS nemo_speech_s2s
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
