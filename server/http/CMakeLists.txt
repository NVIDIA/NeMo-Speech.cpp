# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(nemo_speech_http_server STATIC http_server.cpp)
if(NEMO_SPEECH_BUILD_S2S)
    target_sources(nemo_speech_http_server PRIVATE s2s_realtime.cpp)
endif()
target_include_directories(nemo_speech_http_server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nemo_speech_http_server PUBLIC
    nemo_speech_common
    nemo_speech_engine_registry
    httplib::httplib
)
