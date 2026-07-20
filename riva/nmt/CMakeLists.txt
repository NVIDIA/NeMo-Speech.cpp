# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# NMT translation pipeline. Runs the Riva-Translate decoder through llama.cpp.
# SHARED so direct C++ consumers link one .so, matching asr/tts. llama is linked
# PRIVATE so neither llama.h nor ggml leak through this lib's public ABI.
file(GLOB NMT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)
# c_api.cpp is the C ABI boundary - built into its own hidden-visibility lib
# (nemo_speech_nmt_c) below, not the default-visibility C++ lib.
list(REMOVE_ITEM NMT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/c_api.cpp")
add_library(nemo_speech_nmt SHARED ${NMT_SOURCES})

if(WIN32)
    set_target_properties(nemo_speech_nmt PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_include_directories(nemo_speech_nmt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nemo_speech_nmt
    PUBLIC nemo_speech_common
    PRIVATE llama
)

# llama is PRIVATE and ggml-cuda links CUDA::cuda_driver PRIVATE, so the stub
# libcuda.so never reaches consumers of this lib. Re-link it PUBLIC (as
# runtime/ggml does) so test_nmt and other executables can resolve
# libggml-cuda's NEEDED libcuda.so.1 in the devel image, where only the stub is
# present. CUDAToolkit was found in the top-level CMakeLists under this gate.
if(GGML_CUDA)
    target_link_libraries(nemo_speech_nmt PUBLIC CUDA::cuda_driver)
endif()

# ---------------------------------------------------------------------------
# Stable C ABI shared library (nemo_speech_nmt_*). Hidden visibility + a version script
# so only the nemo_speech_nmt_* symbols are exported; the C++ implementation is linked in
# privately. This is the installed binary contract.
# ---------------------------------------------------------------------------
add_library(nemo_speech_nmt_c SHARED c_api.cpp)
target_link_libraries(nemo_speech_nmt_c PRIVATE nemo_speech_nmt)
target_include_directories(nemo_speech_nmt_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_definitions(nemo_speech_nmt_c PRIVATE NEMO_SPEECH_NMT_BUILD)
set_target_properties(nemo_speech_nmt_c PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    SOVERSION 1)
if(UNIX AND NOT APPLE)
    target_link_options(nemo_speech_nmt_c PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/c_api.map")
endif()

install(TARGETS nemo_speech_nmt
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS nemo_speech_nmt_c
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
