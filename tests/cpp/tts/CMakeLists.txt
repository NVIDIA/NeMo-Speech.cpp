# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
find_package(Threads REQUIRED)

add_executable(test_magpietts_file test_magpietts_file.cpp)
target_link_libraries(test_magpietts_file PRIVATE nemo_speech_tts_magpietts)

add_executable(test_magpietts_attention_prior test_magpietts_attention_prior.cpp)
target_link_libraries(test_magpietts_attention_prior PRIVATE nemo_speech_tts_magpietts)

add_executable(test_magpietts_frame_stacking test_magpietts_frame_stacking.cpp)
target_link_libraries(test_magpietts_frame_stacking PRIVATE nemo_speech_tts_magpietts)

if(GGML_CUDA AND NEMO_SPEECH_GGML_PATCHED)
    add_executable(test_magpietts_cached_attention test_magpietts_cached_attention.cpp)
    target_link_libraries(test_magpietts_cached_attention PRIVATE nemo_speech_tts_magpietts)
endif()

add_executable(test_magpietts_asr test_magpietts_asr.cpp)
target_link_libraries(test_magpietts_asr PRIVATE
    tests_cpp_wer
    nemo_speech_tts_magpietts
    nemo_speech_tts_tokenizer
    nemo_speech_asr
)

# C ABI tests. tts_c_smoke is compiled as C (proves include/nemo_speech/tts.h
# is valid C) and links the unified TTS library; tts_c_dlopen loads it at runtime.
add_executable(tts_c_smoke ${CMAKE_SOURCE_DIR}/tests/c/tts_c_smoke.c)
target_link_libraries(tts_c_smoke PRIVATE nemo_speech_tts_c)
# The C ABI .so links the TTS C++ libraries privately, so the CUDA::cuda_driver
# stub that ggml-cuda needs in driver-less build containers may not reach this
# executable. Add the stub directly, matching the ASR C ABI smoke target.
if(GGML_CUDA)
    target_link_libraries(tts_c_smoke PRIVATE CUDA::cuda_driver)
endif()

add_executable(tts_c_dlopen ${CMAKE_SOURCE_DIR}/tests/c/tts_c_dlopen.c)
target_link_libraries(tts_c_dlopen PRIVATE ${CMAKE_DL_LIBS})

add_executable(test_tokenizer_single_chars test_tokenizer_single_chars.cpp)
target_link_libraries(test_tokenizer_single_chars PRIVATE nemo_speech_tts_tokenizer)
if(NEMO_SPEECH_TTS_WITH_JA)
    target_compile_definitions(test_tokenizer_single_chars PRIVATE NEMO_SPEECH_TTS_WITH_JA=1)
endif()
if(NEMO_SPEECH_TTS_WITH_ZH)
    target_compile_definitions(test_tokenizer_single_chars PRIVATE NEMO_SPEECH_TTS_WITH_ZH=1)
endif()

if(NEMO_SPEECH_TTS_WITH_ZH)
    add_executable(test_tokenizer_mandarin test_tokenizer_mandarin.cpp)
    target_link_libraries(
        test_tokenizer_mandarin PRIVATE nemo_speech_tts_tokenizer Threads::Threads)
    target_compile_definitions(test_tokenizer_mandarin PRIVATE
        MANDARIN_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
endif()

add_executable(test_tts_terminal_punctuation test_tts_terminal_punctuation.cpp)
target_link_libraries(test_tts_terminal_punctuation PRIVATE nemo_speech_tts_tokenizer)

add_executable(test_text_normalizer test_text_normalizer.cpp)
target_link_libraries(
    test_text_normalizer PRIVATE nemo_speech_tts_preproc nemo_speech_tts_tokenizer)

if(TARGET nemo_speech_grpc_tts)
    add_executable(test_grpc_tts_config test_grpc_tts_config.cpp)
    target_link_libraries(test_grpc_tts_config PRIVATE nemo_speech_grpc_tts)
    target_compile_definitions(test_grpc_tts_config PRIVATE
        NEMO_SPEECH_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
endif()
