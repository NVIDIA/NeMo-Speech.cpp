# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(tests_cpp_wer STATIC
    utils/wer.cpp
)
target_include_directories(tests_cpp_wer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(test_shared_utilities
    ${CMAKE_SOURCE_DIR}/tests/cpp/common/test_shared_utilities.cpp
    ${CMAKE_SOURCE_DIR}/app/cli_util.cpp)
target_include_directories(test_shared_utilities PRIVATE ${CMAKE_SOURCE_DIR}/app)
target_link_libraries(test_shared_utilities PRIVATE
    nemo_speech_common nemo_speech_engine_registry ggml)
add_test(NAME shared_utilities COMMAND test_shared_utilities)

add_executable(test_subtitles
    ${CMAKE_SOURCE_DIR}/tests/cpp/common/test_subtitles.cpp)
target_link_libraries(test_subtitles PRIVATE nemo_speech_common)
add_test(NAME subtitles COMMAND test_subtitles)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND AND
   (NEMO_SPEECH_BUILD_ASR OR NEMO_SPEECH_BUILD_DIAR OR
    NEMO_SPEECH_BUILD_NMT OR NEMO_SPEECH_BUILD_TTS))
    add_test(
        NAME installed_sdk_consumer
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/install/sdk_consumer_test.py
                ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR} ${CMAKE_COMMAND})
    set_tests_properties(installed_sdk_consumer PROPERTIES TIMEOUT 180)
endif()

if(TARGET nemo_speech_cli)
    if(Python3_Interpreter_FOUND)
        set(cli_contract_args
            ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/cli/cli_contract_test.py
            $<TARGET_FILE:nemo_speech_cli>)
        if(NEMO_SPEECH_BUILD_ASR AND NEMO_SPEECH_BUILD_MIC_CAPTURE)
            list(APPEND cli_contract_args --expect-live)
        endif()
        add_test(
            NAME cli_contract
            COMMAND ${cli_contract_args})
        find_program(NEMO_SPEECH_CURL_EXECUTABLE NAMES curl curl.exe)
        if(NEMO_SPEECH_CURL_EXECUTABLE)
            add_test(
                NAME cli_model_store
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/cli/model_store_test.py
                        $<TARGET_FILE:nemo_speech_cli>)
        else()
            message(WARNING "curl not found; skipping cli_model_store")
        endif()
        if(UNIX AND NOT APPLE)
            add_test(
                NAME cli_install_linux
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/install/install_sh_test.py)
        endif()
        if(WIN32)
            find_program(NEMO_SPEECH_POWERSHELL_EXECUTABLE NAMES pwsh powershell REQUIRED)
            add_test(
                NAME cli_install_windows
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/install/install_ps1_test.py
                        ${NEMO_SPEECH_POWERSHELL_EXECUTABLE}
                        $<TARGET_FILE:nemo_speech_cli>)
            set_tests_properties(cli_install_windows PROPERTIES TIMEOUT 300)
        endif()
        if(NEMO_SPEECH_BUILD_HTTP AND NEMO_SPEECH_BUILD_ASR AND
           DEFINED ENV{NEMO_SPEECH_TEST_ASR_MODEL} AND
           DEFINED ENV{NEMO_SPEECH_TEST_AUDIO})
            set(http_conformance_args
                --binary $<TARGET_FILE:nemo_speech_cli>
                --asr-model $ENV{NEMO_SPEECH_TEST_ASR_MODEL}
                --audio $ENV{NEMO_SPEECH_TEST_AUDIO})
            set(nemo_speech_http_tts_conformance OFF)
            if(NEMO_SPEECH_BUILD_TTS AND DEFINED ENV{NEMO_SPEECH_TEST_TTS_MODEL} AND
               DEFINED ENV{NEMO_SPEECH_TEST_CODEC_MODEL} AND
               DEFINED ENV{NEMO_SPEECH_TEST_TOKENIZER_DIR})
                set(nemo_speech_http_tts_conformance ON)
                list(APPEND http_conformance_args
                    --tts-model $ENV{NEMO_SPEECH_TEST_TTS_MODEL}
                    --codec-model $ENV{NEMO_SPEECH_TEST_CODEC_MODEL}
                    --tokenizer-dir $ENV{NEMO_SPEECH_TEST_TOKENIZER_DIR})
            endif()
            find_program(NEMO_SPEECH_NODE_EXECUTABLE NAMES node nodejs)
            if(NEMO_SPEECH_NODE_EXECUTABLE AND
               DEFINED ENV{NEMO_SPEECH_TEST_OPENAI_JS_PACKAGE})
                list(APPEND http_conformance_args
                    --node ${NEMO_SPEECH_NODE_EXECUTABLE}
                    --openai-js-package $ENV{NEMO_SPEECH_TEST_OPENAI_JS_PACKAGE}
                    --openai-js-example
                    ${CMAKE_SOURCE_DIR}/tests/integration/clients/openai_client.mjs)
            endif()
            add_test(
                NAME http_openai_conformance
                COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_SOURCE_DIR}/tests/integration/http_conformance_test.py
                        ${http_conformance_args})
            set_tests_properties(http_openai_conformance PROPERTIES TIMEOUT 600)
            if(nemo_speech_http_tts_conformance)
                add_test(
                    NAME http_tts_preemption
                    COMMAND ${Python3_EXECUTABLE}
                            ${CMAKE_SOURCE_DIR}/tests/integration/http_conformance_test.py
                            ${http_conformance_args} --tts-preempt)
                set_tests_properties(http_tts_preemption PROPERTIES TIMEOUT 600)
            endif()
        endif()
    endif()
endif()

if(NEMO_SPEECH_BUILD_ASR OR NEMO_SPEECH_BUILD_DIAR)
    add_subdirectory(asr)
endif()
if(NEMO_SPEECH_BUILD_TTS)
    add_subdirectory(tts)
endif()
if(NEMO_SPEECH_BUILD_NMT)
    add_subdirectory(nmt)
endif()

if(TARGET nemo_speech_cublas_shim)
    add_executable(test_cublas_shim_numerics test_cublas_shim_numerics.cu)
    target_link_libraries(test_cublas_shim_numerics PRIVATE nemo_speech_cublas_shim CUDA::cudart)
    set_target_properties(test_cublas_shim_numerics PROPERTIES
        CUDA_ARCHITECTURES "${NEMO_SPEECH_CUBLAS_SHIM_ARCHITECTURES}")
endif()
