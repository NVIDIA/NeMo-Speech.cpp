# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(nemo_speech_speech STATIC
    speech_translator.cpp
)
target_include_directories(nemo_speech_speech PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(nemo_speech_speech PUBLIC
    nemo_speech_asr
    nemo_speech_nmt
)
if(NEMO_SPEECH_BUILD_TTS)
    target_compile_definitions(nemo_speech_speech PUBLIC NEMO_SPEECH_SPEECH_TTS=1)
    target_link_libraries(nemo_speech_speech PUBLIC nemo_speech_tts)
endif()
