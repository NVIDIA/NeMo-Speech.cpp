# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
foreach(name test_pee_parity test_ctc_timestamp test_salm_recognizer test_salm_guards)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE nemo_speech_speechlm)
endforeach()
target_compile_definitions(test_ctc_timestamp PRIVATE
    TIMESTAMP_GOLDEN_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/data/timestamp_golden.json"
    TIMESTAMP_UNITS_GOLDEN_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/data/timestamp_units_golden.json"
    TIMESTAMP_ALIGNER_GOLDEN_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/data/timestamp_aligner_golden.json")
add_test(NAME ctc_timestamp_dp COMMAND test_ctc_timestamp)
add_test(NAME salm_recognizer COMMAND test_salm_recognizer)
add_test(NAME salm_guards COMMAND test_salm_guards)
# Request rules (salm_policy) against the recorded goldens in salm_policy_goldens.inc. With a GGUF path argument the
# test also checks prompt token ids and detokenization against the real vocabulary (vocab-only load).
add_executable(test_salm_policy test_salm_policy.cpp)
target_link_libraries(test_salm_policy PRIVATE nemo_speech_speechlm llama)
add_test(NAME salm_policy COMMAND test_salm_policy)
# ggml_argmax_bf16 (the ggml patch: the batched path's greedy pick on the device) against policy::greedy_pick (the host's),
# on the CPU backend and, with one, the GPU backend.
if(NEMO_SPEECH_GGML_PATCHED)
    add_executable(test_argmax_bf16 test_argmax_bf16.cpp)
    target_link_libraries(test_argmax_bf16 PRIVATE nemo_speech_speechlm ggml)
    if(GGML_CUDA)
        # Links ggml directly: the stub resolves libggml-cuda's libcuda.so.1 NEEDED entry.
        target_link_libraries(test_argmax_bf16 PRIVATE CUDA::cuda_driver)
    endif()
    add_test(NAME argmax_bf16 COMMAND test_argmax_bf16)
    set_tests_properties(argmax_bf16 PROPERTIES TIMEOUT 120)
endif()
