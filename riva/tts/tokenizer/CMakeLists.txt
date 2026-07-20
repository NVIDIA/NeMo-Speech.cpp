# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
include(GNUInstallDirs)

if(NEMO_SPEECH_TTS_WITH_JA)
set(OPENJTALK_SRC_DIR ${CMAKE_SOURCE_DIR}/third_party/open_jtalk/src)
set(OPENJTALK_DIC_SRC_DIR ${OPENJTALK_SRC_DIR}/mecab-naist-jdic)
set(OPENJTALK_DIC_BUILD_DIR ${CMAKE_BINARY_DIR}/open_jtalk_dic)
set(OPENJTALK_DIC_WORK_DIR ${CMAKE_BINARY_DIR}/open_jtalk_dic_src)
set(OPENJTALK_DIC_INSTALL_DIR ${CMAKE_INSTALL_DATADIR}/nemo-speech/open_jtalk_dic)
set(OPENJTALK_DIC_INSTALL_FULL_DIR
    ${CMAKE_INSTALL_FULL_DATADIR}/nemo-speech/open_jtalk_dic)
set(OPENJTALK_INCLUDE_DIRS
    ${OPENJTALK_SRC_DIR}/mecab/src
    ${OPENJTALK_SRC_DIR}/mecab2njd
    ${OPENJTALK_SRC_DIR}/njd
    ${OPENJTALK_SRC_DIR}/njd_set_accent_phrase
    ${OPENJTALK_SRC_DIR}/njd_set_accent_type
    ${OPENJTALK_SRC_DIR}/njd_set_digit
    ${OPENJTALK_SRC_DIR}/njd_set_long_vowel
    ${OPENJTALK_SRC_DIR}/njd_set_pronunciation
    ${OPENJTALK_SRC_DIR}/njd_set_unvoiced_vowel
    ${OPENJTALK_SRC_DIR}/text2mecab
)

set(OPENJTALK_COMPILE_DEFS
    CHARSET_UTF_8
    MECAB_UTF8_USE_ONLY
    DIC_VERSION=102
    MECAB_WITHOUT_SHARE_DIC
    MECAB_DEFAULT_RC="dummy"
    PACKAGE="open_jtalk"
    VERSION="1.11.2"
    HAVE_CTYPE_H
    HAVE_FCNTL_H
    HAVE_GETENV
    HAVE_INTTYPES_H
    HAVE_MEMORY_H
    HAVE_SETJMP
    HAVE_STDINT_H
    HAVE_STDLIB_H
    HAVE_STRING_H
    HAVE_SYS_STAT_H
    HAVE_SYS_TYPES_H
)
if(WIN32)
    # mecab's sources branch on _WIN32 + HAVE_WINDOWS_H (CreateFileMapping mmap,
    # FindFirstFile dirent replacement); the POSIX HAVE_* defs below would pull
    # in headers MSVC doesn't ship (dirent.h, unistd.h, sys/mman.h).
    list(APPEND OPENJTALK_COMPILE_DEFS HAVE_WINDOWS_H)
else()
    list(APPEND OPENJTALK_COMPILE_DEFS
        HAVE_DIRENT_H
        HAVE_MMAP
        HAVE_STRINGS_H
        HAVE_SYS_MMAN_H
        HAVE_UNISTD_H
    )
endif()
if(MSVC)
    # mecab uses std::binary_function (removed in C++17). libstdc++ still ships
    # it as deprecated and pulls it in transitively; MSVC's STL needs the
    # _HAS_AUTO_PTR_ETC opt-in AND only defines it inside <functional>, which
    # mecab never includes directly - so force-include it (/FI below).
    list(APPEND OPENJTALK_COMPILE_DEFS _HAS_AUTO_PTR_ETC=1)
endif()

add_library(nemo_speech_openjtalk_frontend STATIC
    ${OPENJTALK_SRC_DIR}/mecab/src/char_property.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/connector.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/context_id.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/dictionary.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/dictionary_compiler.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/dictionary_generator.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/dictionary_rewriter.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/eval.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/feature_index.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/iconv_utils.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/lbfgs.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/learner.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/learner_tagger.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/libmecab.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/mecab.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/nbest_generator.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/param.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/string_buffer.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/tagger.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/tokenizer.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/utils.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/viterbi.cpp
    ${OPENJTALK_SRC_DIR}/mecab/src/writer.cpp
    ${OPENJTALK_SRC_DIR}/mecab2njd/mecab2njd.c
    ${OPENJTALK_SRC_DIR}/njd/njd.c
    ${OPENJTALK_SRC_DIR}/njd/njd_node.c
    ${OPENJTALK_SRC_DIR}/njd_set_accent_phrase/njd_set_accent_phrase.c
    ${OPENJTALK_SRC_DIR}/njd_set_accent_type/njd_set_accent_type.c
    ${OPENJTALK_SRC_DIR}/njd_set_digit/njd_set_digit.c
    ${OPENJTALK_SRC_DIR}/njd_set_long_vowel/njd_set_long_vowel.c
    ${OPENJTALK_SRC_DIR}/njd_set_pronunciation/njd_set_pronunciation.c
    ${OPENJTALK_SRC_DIR}/njd_set_unvoiced_vowel/njd_set_unvoiced_vowel.c
    ${OPENJTALK_SRC_DIR}/text2mecab/text2mecab.c
)

target_include_directories(nemo_speech_openjtalk_frontend PUBLIC ${OPENJTALK_INCLUDE_DIRS})
target_compile_definitions(nemo_speech_openjtalk_frontend PUBLIC ${OPENJTALK_COMPILE_DEFS})
if(MSVC)
    # See _HAS_AUTO_PTR_ETC above: make <functional> (which hosts the restored
    # std::binary_function) visible in every mecab TU.
    target_compile_options(nemo_speech_openjtalk_frontend PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:/FIfunctional>)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # -Wno-register: mecab uses the pre-C++17 `register` keyword, which clang
    # rejects as an error in C++17 (-w silences warnings but not this error).
    target_compile_options(nemo_speech_openjtalk_frontend PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-w -finput-charset=UTF-8 -fexec-charset=UTF-8>
        $<$<COMPILE_LANGUAGE:CXX>:-w -Wno-register -finput-charset=UTF-8 -fexec-charset=UTF-8>
    )
endif()

add_executable(nemo_speech_mecab_dict_index
    ${OPENJTALK_SRC_DIR}/mecab/src/mecab-dict-index.cpp
)
target_link_libraries(nemo_speech_mecab_dict_index PRIVATE nemo_speech_openjtalk_frontend)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nemo_speech_mecab_dict_index PRIVATE -w -Wno-register)
endif()

set(OPENJTALK_DIC_OUTPUTS
    ${OPENJTALK_DIC_BUILD_DIR}/char.bin
    ${OPENJTALK_DIC_BUILD_DIR}/matrix.bin
    ${OPENJTALK_DIC_BUILD_DIR}/sys.dic
    ${OPENJTALK_DIC_BUILD_DIR}/unk.dic
)
set(OPENJTALK_DIC_INPUTS
    ${OPENJTALK_DIC_SRC_DIR}/_left-id.def
    ${OPENJTALK_DIC_SRC_DIR}/_pos-id.def
    ${OPENJTALK_DIC_SRC_DIR}/_rewrite.def
    ${OPENJTALK_DIC_SRC_DIR}/_right-id.def
    ${OPENJTALK_DIC_SRC_DIR}/char.def
    ${OPENJTALK_DIC_SRC_DIR}/feature.def
    ${OPENJTALK_DIC_SRC_DIR}/matrix.def
    ${OPENJTALK_DIC_SRC_DIR}/naist-jdic.csv
    ${OPENJTALK_DIC_SRC_DIR}/unk.def
)
add_custom_command(
    OUTPUT ${OPENJTALK_DIC_OUTPUTS}
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${OPENJTALK_DIC_WORK_DIR} ${OPENJTALK_DIC_BUILD_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OPENJTALK_DIC_WORK_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OPENJTALK_DIC_BUILD_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${OPENJTALK_DIC_SRC_DIR} ${OPENJTALK_DIC_WORK_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENJTALK_DIC_WORK_DIR}/_left-id.def ${OPENJTALK_DIC_WORK_DIR}/left-id.def
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENJTALK_DIC_WORK_DIR}/_pos-id.def ${OPENJTALK_DIC_WORK_DIR}/pos-id.def
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENJTALK_DIC_WORK_DIR}/_rewrite.def ${OPENJTALK_DIC_WORK_DIR}/rewrite.def
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENJTALK_DIC_WORK_DIR}/_right-id.def ${OPENJTALK_DIC_WORK_DIR}/right-id.def
    COMMAND nemo_speech_mecab_dict_index -d ${OPENJTALK_DIC_WORK_DIR} -o ${OPENJTALK_DIC_BUILD_DIR} -f UTF-8 -t UTF-8
    DEPENDS nemo_speech_mecab_dict_index ${OPENJTALK_DIC_INPUTS}
    VERBATIM
)
add_custom_target(nemo_speech_openjtalk_dic DEPENDS ${OPENJTALK_DIC_OUTPUTS})
endif()

set(TTS_TOKENIZER_SOURCES tokenizer.cpp)
if(NEMO_SPEECH_TTS_WITH_ZH)
    list(APPEND TTS_TOKENIZER_SOURCES mandarin_tokenizer.cpp)
endif()

add_library(nemo_speech_tts_tokenizer_obj OBJECT ${TTS_TOKENIZER_SOURCES})
set_target_properties(nemo_speech_tts_tokenizer_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(nemo_speech_tts_tokenizer_obj PUBLIC
    ${CMAKE_SOURCE_DIR}/riva
)
if(NEMO_SPEECH_TTS_WITH_JA)
    target_compile_definitions(nemo_speech_tts_tokenizer_obj PRIVATE
        NEMO_SPEECH_TTS_WITH_JA=1
        NEMO_SPEECH_OPENJTALK_DIC_DIR="${OPENJTALK_DIC_BUILD_DIR}"
        NEMO_SPEECH_OPENJTALK_INSTALLED_DIC_DIR="${OPENJTALK_DIC_INSTALL_FULL_DIR}"
    )
    target_link_libraries(
        nemo_speech_tts_tokenizer_obj PRIVATE nemo_speech_openjtalk_frontend)
    add_dependencies(nemo_speech_tts_tokenizer_obj nemo_speech_openjtalk_dic)
    install(FILES ${OPENJTALK_DIC_OUTPUTS}
        DESTINATION ${OPENJTALK_DIC_INSTALL_DIR}
        COMPONENT Runtime)
endif()

if(NEMO_SPEECH_TTS_WITH_ZH)
    set(MANDARIN_DATA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mandarin_data)
    set(MANDARIN_DATA_INSTALL_DIR
        ${CMAKE_INSTALL_DATADIR}/nemo-speech/mandarin_g2p)
    set(MANDARIN_DATA_INSTALL_FULL_DIR
        ${CMAKE_INSTALL_FULL_DATADIR}/nemo-speech/mandarin_g2p)
    target_include_directories(nemo_speech_tts_tokenizer_obj SYSTEM PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/cppjieba/include
        ${CMAKE_SOURCE_DIR}/third_party/cppjieba/deps/limonp/include
    )
    target_compile_definitions(nemo_speech_tts_tokenizer_obj PRIVATE
        NEMO_SPEECH_TTS_WITH_ZH=1
        NEMO_SPEECH_MANDARIN_DATA_DIR="${MANDARIN_DATA_SRC_DIR}"
        NEMO_SPEECH_MANDARIN_INSTALLED_DATA_DIR="${MANDARIN_DATA_INSTALL_FULL_DIR}"
    )
    install(DIRECTORY ${MANDARIN_DATA_SRC_DIR}/
        DESTINATION ${MANDARIN_DATA_INSTALL_DIR}
        COMPONENT Runtime)
endif()
