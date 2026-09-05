# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(nemo_speech_cli
    main.cpp
    cli_util.cpp
    doctor.cpp
    model.cpp
    model_store.cpp
    model_utils.cpp
)
set_target_properties(nemo_speech_cli PROPERTIES OUTPUT_NAME nemo-speech)
target_link_libraries(nemo_speech_cli PRIVATE
    nemo_speech_common
    nemo_speech_engine_registry
    ggml
    ggml-base
)
target_include_directories(nemo_speech_cli PRIVATE ${CMAKE_SOURCE_DIR}/include)
if(WIN32)
    target_link_libraries(nemo_speech_cli PRIVATE shell32)
endif()

if(NEMO_SPEECH_BUILD_ASR)
    find_package(Threads REQUIRED)
    target_sources(nemo_speech_cli PRIVATE
        bench.cpp
        transcribe.cpp)
    target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_CLI_ASR=1)
    target_link_libraries(nemo_speech_cli PRIVATE
        nemo_speech_asr
        Threads::Threads)
    if(NEMO_SPEECH_BUILD_MIC_CAPTURE)
        if(NOT EXISTS "${CMAKE_SOURCE_DIR}/llama.cpp/vendor/miniaudio/miniaudio.h")
            message(FATAL_ERROR
                "ASR CLI microphone capture requires the vendored miniaudio header; "
                "run: git submodule update --init llama.cpp")
        endif()
        target_sources(nemo_speech_cli PRIVATE microphone_capture.cpp)
        target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_CLI_LIVE=1)
        target_include_directories(nemo_speech_cli PRIVATE
            ${CMAKE_SOURCE_DIR}/llama.cpp/vendor/miniaudio)
        target_link_libraries(nemo_speech_cli PRIVATE ${CMAKE_DL_LIBS})
        if(APPLE)
            target_link_libraries(nemo_speech_cli PRIVATE
                "-framework AudioToolbox"
                "-framework AudioUnit"
                "-framework CoreAudio"
                "-framework CoreFoundation")
        endif()
    endif()
endif()

if(NEMO_SPEECH_BUILD_HTTP)
    target_sources(nemo_speech_cli PRIVATE health.cpp serve.cpp)
    target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_CLI_HTTP=1)
    target_link_libraries(nemo_speech_cli PRIVATE
        nemo_speech_http_server
        httplib::httplib)
endif()
if(NEMO_SPEECH_BUILD_S2S)
    target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_CLI_S2S=1)
    target_link_libraries(nemo_speech_cli PRIVATE nemo_speech_s2s)
endif()
if(NEMO_SPEECH_BUILD_GRPC)
    target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_GRPC_SERVER_BUILT=1)
endif()
if(GGML_CUDA)
    target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_CLI_CUDA=1)
endif()
if(GGML_METAL)
    target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_CLI_METAL=1)
endif()
if(GGML_VULKAN)
    target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_CLI_VULKAN=1)
endif()
if(NEMO_SPEECH_WITH_FLASHLIGHT)
    target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_CLI_FLASHLIGHT=1)
endif()
if(NEMO_SPEECH_WITH_NORM)
    target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_CLI_NORM=1)
endif()
if(NEMO_SPEECH_BUILD_DIAR)
    target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_CLI_DIAR=1)
    target_sources(nemo_speech_cli PRIVATE diarize.cpp)
    target_link_libraries(nemo_speech_cli PRIVATE nemo_speech_asr)
endif()
if(NEMO_SPEECH_BUILD_TTS)
    target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_CLI_TTS=1)
    target_sources(nemo_speech_cli PRIVATE synthesize.cpp)
    target_link_libraries(nemo_speech_cli PRIVATE nemo_speech_tts_magpietts)
    if(NEMO_SPEECH_TTS_WITH_JA)
        target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_CLI_TTS_JA=1)
    endif()
    if(NEMO_SPEECH_TTS_WITH_ZH)
        target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_CLI_TTS_ZH=1)
    endif()
endif()
if(NEMO_SPEECH_BUILD_NMT)
    target_compile_definitions(nemo_speech_cli PRIVATE NEMO_SPEECH_CLI_NMT=1)
    target_sources(nemo_speech_cli PRIVATE translate.cpp)
    target_link_libraries(nemo_speech_cli PRIVATE nemo_speech_nmt)
endif()

if(UNIX AND NOT APPLE)
    target_link_options(nemo_speech_cli PRIVATE "LINKER:--allow-shlib-undefined")
endif()

install(TARGETS nemo_speech_cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(APPLE)
    set_target_properties(nemo_speech_cli PROPERTIES INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set_target_properties(nemo_speech_cli PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
