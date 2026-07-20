# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.26)
project(nemo_speech_installed_consumer LANGUAGES C)

set(required_components)
foreach(component IN ITEMS ASR Diarization NMT TTS)
    if(EXPECT_NEMO_SPEECH_${component})
        list(APPEND required_components ${component})
    endif()
endforeach()
find_package(NeMoSpeech REQUIRED COMPONENTS ${required_components})

foreach(component IN ITEMS ASR Diarization NMT TTS)
    if(DEFINED EXPECT_NEMO_SPEECH_${component})
        if(EXPECT_NEMO_SPEECH_${component} AND
           NOT TARGET NeMoSpeech::${component})
            message(FATAL_ERROR "installed package is missing expected ${component} target")
        elseif(NOT EXPECT_NEMO_SPEECH_${component} AND
               TARGET NeMoSpeech::${component})
            message(FATAL_ERROR "installed package unexpectedly exposes ${component} target")
        endif()
    endif()
endforeach()

add_executable(installed_consumer main.c)
if(TARGET NeMoSpeech::ASR)
    target_link_libraries(installed_consumer PRIVATE NeMoSpeech::ASR)
    target_compile_definitions(installed_consumer PRIVATE HAVE_NEMO_SPEECH_ASR=1)
endif()
if(TARGET NeMoSpeech::Diarization)
    target_link_libraries(installed_consumer PRIVATE NeMoSpeech::Diarization)
    target_compile_definitions(installed_consumer PRIVATE HAVE_NEMO_SPEECH_DIAR=1)
endif()
if(TARGET NeMoSpeech::NMT)
    target_link_libraries(installed_consumer PRIVATE NeMoSpeech::NMT)
    target_compile_definitions(installed_consumer PRIVATE HAVE_NEMO_SPEECH_NMT=1)
endif()
if(TARGET NeMoSpeech::TTS)
    target_link_libraries(installed_consumer PRIVATE NeMoSpeech::TTS)
    target_compile_definitions(installed_consumer PRIVATE HAVE_NEMO_SPEECH_TTS=1)
endif()

if(NOT TARGET NeMoSpeech::ASR AND
   NOT TARGET NeMoSpeech::Diarization AND
   NOT TARGET NeMoSpeech::NMT AND
   NOT TARGET NeMoSpeech::TTS)
    message(FATAL_ERROR "installed package exposes no runtime components")
endif()
