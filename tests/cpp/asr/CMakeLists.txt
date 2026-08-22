# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_executable(test_ctc_offline test_ctc_offline.cpp)
target_link_libraries(test_ctc_offline PRIVATE nemo_speech_asr)

add_executable(test_ctc_streaming test_ctc_streaming.cpp)
target_link_libraries(test_ctc_streaming PRIVATE nemo_speech_asr)

add_executable(test_rnnt_streaming test_rnnt_streaming.cpp)
target_link_libraries(test_rnnt_streaming PRIVATE nemo_speech_asr)

# Multi-threaded crosstalk / deadlock stress test for the shared cache-aware
# encoder Session driven by many per-stream SessionStates at once.
find_package(Threads REQUIRED)
add_executable(test_transducer_offline test_transducer_offline.cpp)
target_link_libraries(test_transducer_offline PRIVATE nemo_speech_asr Threads::Threads)
add_executable(test_rnnt_concurrency test_rnnt_concurrency.cpp)
target_link_libraries(test_rnnt_concurrency PRIVATE nemo_speech_asr Threads::Threads)

add_executable(test_vad test_vad.cpp)
target_link_libraries(test_vad PRIVATE nemo_speech_asr)

add_executable(test_endpointer test_endpointer.cpp)
target_link_libraries(test_endpointer PRIVATE nemo_speech_asr)

add_executable(test_postproc test_postproc.cpp)
target_link_libraries(test_postproc PRIVATE nemo_speech_asr Threads::Threads)

add_executable(test_audio_resampler test_audio_resampler.cpp)
target_link_libraries(test_audio_resampler PRIVATE nemo_speech_common)

add_executable(test_audio_decoder test_audio_decoder.cpp)
target_link_libraries(test_audio_decoder PRIVATE nemo_speech_common)

add_executable(test_decoders test_decoders.cpp)
target_link_libraries(test_decoders PRIVATE nemo_speech_asr)

# Do not link sentencepiece directly; it is provided by nemo_speech_asr.
add_executable(test_oov_boost test_oov_boost.cpp)
target_link_libraries(test_oov_boost PRIVATE nemo_speech_asr)

add_executable(test_pnc test_pnc.cpp)
target_link_libraries(test_pnc PRIVATE nemo_speech_asr)

add_executable(test_sortformer_parity test_sortformer_parity.cpp)
target_link_libraries(test_sortformer_parity PRIVATE nemo_speech_asr)

add_executable(test_diar_streaming test_diar_streaming.cpp)
target_link_libraries(test_diar_streaming PRIVATE nemo_speech_asr)

add_executable(test_diar_state test_diar_state.cpp)
target_link_libraries(test_diar_state PRIVATE nemo_speech_asr)
add_test(NAME diar_state COMMAND test_diar_state)

add_executable(test_diar_recognizer test_diar_recognizer.cpp)
target_link_libraries(test_diar_recognizer PRIVATE nemo_speech_asr)

add_executable(test_batching test_batching.cpp)
target_link_libraries(test_batching PRIVATE nemo_speech_asr Threads::Threads)

# C ABI tests. asr_c_smoke is compiled as C (proves include/nemo_speech/asr.h
# is valid C) and links only the C ABI lib; asr_c_dlopen loads it at runtime.
add_executable(asr_c_smoke ${CMAKE_SOURCE_DIR}/tests/c/asr_c_smoke.c)
target_link_libraries(asr_c_smoke PRIVATE nemo_speech_asr_c)
# The C ABI .so links nemo_speech_asr PRIVATE (it hides the C++/CUDA ABI), so
# the CUDA::cuda_driver stub that runtime_ggml re-links PUBLIC doesn't reach this
# executable. Without it, ld can't resolve libggml-cuda's NEEDED libcuda.so.1 in
# a driver-less build container. Add the stub directly (same fix as runtime_ggml).
if(GGML_CUDA)
    target_link_libraries(asr_c_smoke PRIVATE CUDA::cuda_driver)
endif()

add_executable(asr_c_dlopen ${CMAKE_SOURCE_DIR}/tests/c/asr_c_dlopen.c)
target_link_libraries(asr_c_dlopen PRIVATE ${CMAKE_DL_LIBS})
