# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NEMO_SPEECH_BUILD_ASR OR NEMO_SPEECH_BUILD_DIAR)
    find_package(Threads REQUIRED)

    add_executable(check_backend_coverage check_backend_coverage.cpp)
    target_link_libraries(check_backend_coverage PRIVATE nemo_speech_asr)

    add_executable(bench_asr_batching bench_asr_batching.cpp)
    target_link_libraries(bench_asr_batching PRIVATE nemo_speech_asr Threads::Threads)
    if(GGML_CUDA)
        target_compile_definitions(bench_asr_batching PRIVATE GGML_USE_CUDA)
        target_link_libraries(bench_asr_batching PRIVATE CUDA::cudart)
    endif()
endif()

if(NEMO_SPEECH_BUILD_TTS)
    add_executable(nanocodec nanocodec.cpp)
    target_link_libraries(nanocodec PRIVATE nemo_speech_tts_nanocodec_obj)
    if(GGML_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(nanocodec PRIVATE CUDA::cuda_driver)
    endif()
endif()
