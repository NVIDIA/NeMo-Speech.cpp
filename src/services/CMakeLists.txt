# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(nemo_speech_grpc_asr STATIC grpc_asr.cc)
target_include_directories(nemo_speech_grpc_asr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nemo_speech_grpc_asr PUBLIC
    nemo_speech_asr
    riva_proto
)

if(NEMO_SPEECH_BUILD_TTS)
    add_library(nemo_speech_grpc_tts STATIC grpc_tts.cc)
    target_include_directories(nemo_speech_grpc_tts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(nemo_speech_grpc_tts PUBLIC
        nemo_speech_tts_magpietts
        nemo_speech_tts_preproc
        nemo_speech_tts_tokenizer
        riva_proto
    )
endif()

if(NEMO_SPEECH_BUILD_NMT)
    add_library(nemo_speech_grpc_nmt STATIC grpc_nmt.cc)
    target_include_directories(nemo_speech_grpc_nmt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(nemo_speech_grpc_nmt PUBLIC
        nemo_speech_nmt
        nemo_speech_speech
        riva_proto
    )
endif()
