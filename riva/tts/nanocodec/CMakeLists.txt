# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(nemo_speech_tts_nanocodec_obj OBJECT
    model.cpp
)
set_target_properties(nemo_speech_tts_nanocodec_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nemo_speech_tts_nanocodec_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
)
target_link_libraries(nemo_speech_tts_nanocodec_obj PUBLIC ggml)

set(NANOCODEC_TARGETS
    nemo_speech_tts_nanocodec_obj
)

if(GGML_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND CUDAToolkit_INCLUDE_DIRS)
        foreach(target ${NANOCODEC_TARGETS})
            target_include_directories(${target} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        endforeach()
    endif()
    if(CUDAToolkit_FOUND)
        foreach(target ${NANOCODEC_TARGETS})
            target_compile_definitions(${target} PRIVATE GGML_USE_CUDA)
        endforeach()
        target_link_libraries(nemo_speech_tts_nanocodec_obj PUBLIC CUDA::cuda_driver)
    endif()
endif()

if(GGML_METAL)
    foreach(target ${NANOCODEC_TARGETS})
        target_compile_definitions(${target} PRIVATE GGML_USE_METAL)
    endforeach()
endif()
