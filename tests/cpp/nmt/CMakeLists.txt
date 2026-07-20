# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_executable(test_nmt test_nmt.cpp)
target_link_libraries(test_nmt PRIVATE nemo_speech_nmt)

add_executable(test_langpairs test_langpairs.cpp)
target_link_libraries(test_langpairs PRIVATE nemo_speech_nmt)
