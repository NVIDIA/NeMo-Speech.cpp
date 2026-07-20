# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# KenLM runtime-only build.
#
# The upstream project configures language-model builders, command-line tools,
# tests, and portability helpers in addition to the scoring libraries used by
# Flashlight.  Keep an explicit allowlist here so none of those programs or
# their sources enter NeMo-Speech.cpp.  In particular, util/getopt.c and
# util/getopt.hh carry the AT&T Public License and must never be compiled or
# linked, including on Windows where upstream adds getopt.c to kenlm_util.

set(_nemo_speech_kenlm_root "${CMAKE_CURRENT_LIST_DIR}/../third_party/kenlm")
if(NOT EXISTS "${_nemo_speech_kenlm_root}/lm/model.cc")
    message(FATAL_ERROR
        "KenLM sources not found at ${_nemo_speech_kenlm_root}; initialize the "
        "third_party/kenlm submodule")
endif()

find_package(Threads REQUIRED)

set(_nemo_speech_kenlm_util_sources
    ${_nemo_speech_kenlm_root}/util/bit_packing.cc
    ${_nemo_speech_kenlm_root}/util/ersatz_progress.cc
    ${_nemo_speech_kenlm_root}/util/exception.cc
    ${_nemo_speech_kenlm_root}/util/file.cc
    ${_nemo_speech_kenlm_root}/util/file_piece.cc
    ${_nemo_speech_kenlm_root}/util/float_to_string.cc
    ${_nemo_speech_kenlm_root}/util/integer_to_string.cc
    ${_nemo_speech_kenlm_root}/util/mmap.cc
    ${_nemo_speech_kenlm_root}/util/murmur_hash.cc
    ${_nemo_speech_kenlm_root}/util/pool.cc
    ${_nemo_speech_kenlm_root}/util/read_compressed.cc
    ${_nemo_speech_kenlm_root}/util/scoped.cc
    ${_nemo_speech_kenlm_root}/util/spaces.cc
    ${_nemo_speech_kenlm_root}/util/string_piece.cc
    ${_nemo_speech_kenlm_root}/util/usage.cc
    ${_nemo_speech_kenlm_root}/util/double-conversion/bignum-dtoa.cc
    ${_nemo_speech_kenlm_root}/util/double-conversion/bignum.cc
    ${_nemo_speech_kenlm_root}/util/double-conversion/cached-powers.cc
    ${_nemo_speech_kenlm_root}/util/double-conversion/fast-dtoa.cc
    ${_nemo_speech_kenlm_root}/util/double-conversion/fixed-dtoa.cc
    ${_nemo_speech_kenlm_root}/util/double-conversion/strtod.cc
    ${_nemo_speech_kenlm_root}/util/double-conversion/double-to-string.cc
    ${_nemo_speech_kenlm_root}/util/double-conversion/string-to-double.cc)

set(_nemo_speech_kenlm_sources
    ${_nemo_speech_kenlm_root}/lm/bhiksha.cc
    ${_nemo_speech_kenlm_root}/lm/binary_format.cc
    ${_nemo_speech_kenlm_root}/lm/config.cc
    ${_nemo_speech_kenlm_root}/lm/lm_exception.cc
    ${_nemo_speech_kenlm_root}/lm/model.cc
    ${_nemo_speech_kenlm_root}/lm/quantize.cc
    ${_nemo_speech_kenlm_root}/lm/read_arpa.cc
    ${_nemo_speech_kenlm_root}/lm/search_hashed.cc
    ${_nemo_speech_kenlm_root}/lm/search_trie.cc
    ${_nemo_speech_kenlm_root}/lm/sizes.cc
    ${_nemo_speech_kenlm_root}/lm/trie.cc
    ${_nemo_speech_kenlm_root}/lm/trie_sort.cc
    ${_nemo_speech_kenlm_root}/lm/value_build.cc
    ${_nemo_speech_kenlm_root}/lm/virtual_interface.cc
    ${_nemo_speech_kenlm_root}/lm/vocab.cc)

# Defense in depth: a future source-list edit must not reintroduce either
# AT&T-licensed getopt file under a different platform condition.
foreach(_source IN LISTS _nemo_speech_kenlm_util_sources _nemo_speech_kenlm_sources)
    if(_source MATCHES "[/\\\\]getopt\\.(c|cc|cpp|h|hh|hpp)$")
        message(FATAL_ERROR "Prohibited KenLM getopt source in runtime build: ${_source}")
    endif()
endforeach()

add_library(kenlm ${_nemo_speech_kenlm_sources} ${_nemo_speech_kenlm_util_sources})
target_include_directories(kenlm PUBLIC
    $<BUILD_INTERFACE:${_nemo_speech_kenlm_root}>
    $<INSTALL_INTERFACE:include/kenlm>)
target_link_libraries(kenlm PRIVATE Threads::Threads)
target_compile_definitions(kenlm PUBLIC KENLM_MAX_ORDER=6)
set_target_properties(kenlm PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(WIN32)
    target_compile_definitions(kenlm PUBLIC _HAS_AUTO_PTR_ETC=1)
    set_target_properties(kenlm PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

find_package(ZLIB QUIET)
if(ZLIB_FOUND)
    set_property(SOURCE ${_nemo_speech_kenlm_root}/util/read_compressed.cc
        APPEND PROPERTY COMPILE_DEFINITIONS HAVE_ZLIB)
    target_link_libraries(kenlm PRIVATE ZLIB::ZLIB)
endif()

find_package(BZip2 QUIET)
if(BZip2_FOUND)
    set_property(SOURCE ${_nemo_speech_kenlm_root}/util/read_compressed.cc
        APPEND PROPERTY COMPILE_DEFINITIONS HAVE_BZLIB)
    target_link_libraries(kenlm PRIVATE BZip2::BZip2)
endif()

find_package(LibLZMA QUIET)
if(LibLZMA_FOUND)
    set_property(SOURCE ${_nemo_speech_kenlm_root}/util/read_compressed.cc
        APPEND PROPERTY COMPILE_DEFINITIONS HAVE_XZLIB)
    target_link_libraries(kenlm PRIVATE LibLZMA::LibLZMA)
endif()

if(UNIX)
    include(CheckLibraryExists)
    check_library_exists(rt clock_gettime "" _nemo_speech_kenlm_clock_gettime_rt)
    if(_nemo_speech_kenlm_clock_gettime_rt)
        target_compile_definitions(kenlm PRIVATE HAVE_CLOCKGETTIME)
        target_link_libraries(kenlm PRIVATE rt)
    else()
        check_library_exists(c clock_gettime "" _nemo_speech_kenlm_clock_gettime_libc)
        if(_nemo_speech_kenlm_clock_gettime_libc)
            target_compile_definitions(kenlm PRIVATE HAVE_CLOCKGETTIME)
        endif()
    endif()
endif()

add_library(kenlm::kenlm ALIAS kenlm)

install(TARGETS kenlm
    EXPORT flashlight-text-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include)

unset(_nemo_speech_kenlm_root)
unset(_nemo_speech_kenlm_util_sources)
unset(_nemo_speech_kenlm_sources)
