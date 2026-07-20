# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_executable(riva_server riva_server.cc)
target_link_libraries(riva_server PRIVATE
    nemo_speech_grpc_asr
    nemo_speech_grpc_tts
)
if(NEMO_SPEECH_BUILD_NMT)
    target_link_libraries(riva_server PRIVATE nemo_speech_grpc_nmt)
    target_compile_definitions(riva_server PRIVATE NEMO_SPEECH_BUILD_NMT=1)
endif()
