# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from parakeet.cpp:
# Copyright (c) 2025 Jason Ni
# Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
# runtime/ggml/ — the ggml runtime (Module/Session/TensorContainer/
# BackendManager/GGUFLoader + generic NN building blocks). Built as a STATIC
# position-independent lib; absorbed into each pipeline's shared library at
# link time. NOT installed — internal implementation detail.
file(GLOB RUNTIME_GGML_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(nemo_speech_runtime_ggml STATIC ${RUNTIME_GGML_SOURCES})

set_target_properties(nemo_speech_runtime_ggml PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_include_directories(nemo_speech_runtime_ggml PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
)
target_include_directories(nemo_speech_runtime_ggml PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common
)

target_link_libraries(nemo_speech_runtime_ggml PUBLIC ${GGML_DEPENDENCIES})

if(NEMO_SPEECH_GGML_PATCHED)
    target_compile_definitions(nemo_speech_runtime_ggml PRIVATE NEMO_SPEECH_GGML_PATCHED=1)
endif()

# CUDA-only direct depthwise-conv kernel (see CMakeLists.txt option). When off,
# nn.cpp uses the portable ggml_conv_1d_dw lowering (F16-safe on all backends).
if(NEMO_SPEECH_DIRECT_DW_CONV)
    target_compile_definitions(nemo_speech_runtime_ggml PRIVATE NEMO_SPEECH_DIRECT_DW_CONV=1)
endif()

# ggml-cuda links CUDA::cuda_driver PRIVATE, so the stub libcuda.so on the
# link line doesn't propagate to downstream executables. Re-link it PUBLIC
# here so tests/ and riva_server inherit it and ld can resolve libggml-cuda's
# NEEDED libcuda.so.1 at link time. CUDAToolkit was found in the top-level
# CMakeLists.txt under the same GGML_CUDA gate.
if(GGML_CUDA)
    target_link_libraries(nemo_speech_runtime_ggml PUBLIC CUDA::cuda_driver)
endif()
