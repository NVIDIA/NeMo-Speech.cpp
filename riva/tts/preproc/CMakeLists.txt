# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(nemo_speech_tts_preproc STATIC
    text_normalizer.cpp
)

target_include_directories(nemo_speech_tts_preproc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

if(NEMO_SPEECH_WITH_NORM)
    target_link_libraries(nemo_speech_tts_preproc PUBLIC nemo_speech_text_normalization)
endif()
