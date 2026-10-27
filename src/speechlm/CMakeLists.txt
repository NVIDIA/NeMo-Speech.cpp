# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Speech-LLM transcription: PEE encoder owned by speechlm feeding a llama.cpp
# decoder through input embeddings. llama is linked PRIVATE so neither llama.h nor
# ggml leak through this lib's public ABI (same discipline as nmt).
find_package(Threads REQUIRED)
add_library(nemo_speech_speechlm SHARED salm_engine.cpp salm_recognizer.cpp salm_policy.cpp salm_guards.cpp
    pee/pee_model.cpp pee/pee_rope_transformer.cpp pee/ctc_timestamp.cpp pee/timestamp_alignment.cpp)

if(WIN32)
    set_target_properties(nemo_speech_speechlm PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Word-timestamp alignment reproduces NumPy's and PyTorch's float rounding: no contraction into FMA
# (the fused operations it needs are explicit std::fma calls).
set_source_files_properties(pee/timestamp_alignment.cpp PROPERTIES COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>")

target_include_directories(nemo_speech_speechlm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nemo_speech_speechlm
    PUBLIC nemo_speech_asr nemo_speech_common
    PRIVATE llama llama-common Threads::Threads
)

include(${CMAKE_SOURCE_DIR}/cmake/sentencepiece.cmake)
nemo_speech_link_sentencepiece(nemo_speech_speechlm)

# See src/nmt/CMakeLists.txt: re-link the CUDA driver PUBLIC so executables can
# resolve libggml-cuda's libcuda.so.1 in the devel image (stub only).
if(GGML_CUDA)
    target_link_libraries(nemo_speech_speechlm PUBLIC CUDA::cuda_driver)
    # NVTX phase ranges (common/nvtx_utils.h) for Nsight Systems profiles.
    target_compile_definitions(nemo_speech_speechlm PRIVATE GGML_USE_CUDA=1)
    if(TARGET CUDA::nvtx3)
        target_link_libraries(nemo_speech_speechlm PRIVATE CUDA::nvtx3)
    else()
        target_link_libraries(nemo_speech_speechlm PRIVATE CUDA::cudart)
    endif()
endif()

install(TARGETS nemo_speech_speechlm
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# MTP uses llama-common. Shared builds need that runtime library beside llama.
get_target_property(_salm_common_type llama-common TYPE)
if(_salm_common_type STREQUAL "SHARED_LIBRARY")
    install(TARGETS llama-common
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
