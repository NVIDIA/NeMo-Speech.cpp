# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Example programs built on the stable C ABI (nemo_speech_asr_c /
# include/nemo_speech/{asr,diar}.h) - the same public contract external
# consumers use, so they're immune to internal C++ refactors.
#
#   transcribe_file  offline: WAV file -> transcript (no audio dep);
#                    --diar adds speaker turns
#   diarize_file     standalone diarization: WAV file -> speaker segments,
#                    no ASR (nemo_speech_diar_* ABI; streaming or --offline)
#   transcribe_live  streaming: microphone -> transcript (needs PortAudio);
#                    --diar adds speaker turns on finals
#   synthesize_text  TTS: text or token IDs -> WAV
#   translate_text   NMT: text -> translation (only with -DNEMO_SPEECH_BUILD_NMT=ON)

if(NEMO_SPEECH_BUILD_ASR)
    # Offline file transcription (WAV reader shared via wav_reader.h).
    add_executable(transcribe_file transcribe_file.cpp)
    target_link_libraries(transcribe_file PRIVATE nemo_speech_asr_c)
    install(TARGETS transcribe_file RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# The C ABI lib links the C++ pipeline (and its CUDA backend) PRIVATEly, so the
# CUDA driver stub isn't in its link interface. These examples only call the
# nemo_speech_asr_* C ABI; the transitive cu* symbols in libggml-cuda resolve at runtime
# from the toolkit-injected libcuda.so.1. Allow them to stay undefined at link
# time on a driver-less build host. (Linux/GNU ld only; no-op on macOS/Metal.)
    if(UNIX AND NOT APPLE)
        target_link_options(transcribe_file PRIVATE "LINKER:--allow-shlib-undefined")
    endif()

endif()

# Standalone speaker diarization over the nemo_speech_diar_* C ABI (no ASR model).
if(NEMO_SPEECH_BUILD_DIAR)
    add_executable(diarize_file diarize_file.cpp)
    target_link_libraries(diarize_file PRIVATE nemo_speech_asr_c)
    install(TARGETS diarize_file RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    if(UNIX AND NOT APPLE)
        target_link_options(diarize_file PRIVATE "LINKER:--allow-shlib-undefined")
    endif()
endif()

# Offline text-to-speech through the stable TTS C ABI.
if(NEMO_SPEECH_BUILD_TTS)
    add_executable(synthesize_text synthesize_text.cpp)
    target_link_libraries(synthesize_text PRIVATE nemo_speech_tts_c)
    install(TARGETS synthesize_text RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    if(UNIX AND NOT APPLE)
        target_link_options(synthesize_text PRIVATE "LINKER:--allow-shlib-undefined")
    endif()
endif()

# --- translate_text: NMT text translation over the nemo_speech_nmt_* C ABI -----------
# Only built when the NMT component is enabled (it links nemo_speech_nmt_c).
# Placed before the PortAudio block so it is unaffected by whether PortAudio is
# present.
if(NEMO_SPEECH_BUILD_NMT)
    add_executable(translate_text translate_text.cpp)
    target_link_libraries(translate_text PRIVATE nemo_speech_nmt_c)
    install(TARGETS translate_text RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    # See transcribe_file: the C ABI lib links the CUDA backend PRIVATEly, so let
    # libggml-cuda's cu* driver symbols resolve at runtime on a driver-less host.
    if(UNIX AND NOT APPLE)
        target_link_options(translate_text PRIVATE "LINKER:--allow-shlib-undefined")
    endif()
endif()

# A full ASR -> NMT -> TTS example that links only the three stable C APIs.
if(NEMO_SPEECH_BUILD_ASR AND NEMO_SPEECH_BUILD_NMT AND NEMO_SPEECH_BUILD_TTS)
    add_executable(speech_translate_file speech_translate_file.cpp)
    target_link_libraries(speech_translate_file PRIVATE
        nemo_speech_asr_c nemo_speech_nmt_c nemo_speech_tts_c)
    install(TARGETS speech_translate_file RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    if(UNIX AND NOT APPLE)
        target_link_options(speech_translate_file PRIVATE "LINKER:--allow-shlib-undefined")
    endif()
endif()

# --- transcribe_live: needs PortAudio for cross-platform mic capture ---------
# PortAudio is a system package, not vendored:
#   Linux:  apt install portaudio19-dev
#   macOS:  brew install portaudio
#   Windows: vcpkg install portaudio
# If it isn't found, transcribe_live is skipped (a STATUS message says so)
# rather than failing the build - transcribe_file and the core library don't
# need it.
if(NOT NEMO_SPEECH_BUILD_ASR OR NOT NEMO_SPEECH_BUILD_MIC_CAPTURE)
    return()
endif()

set(_pa_found FALSE)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PORTAUDIO QUIET portaudio-2.0)
    if(PORTAUDIO_FOUND)
        set(_pa_found TRUE)
        set(_pa_includes ${PORTAUDIO_INCLUDE_DIRS})
        set(_pa_libs ${PORTAUDIO_LINK_LIBRARIES})
    endif()
endif()

if(NOT _pa_found)
    # Fallback search for systems without pkg-config; hints cover the common
    # non-default install prefixes.
    find_path(PORTAUDIO_INCLUDE_DIR portaudio.h
        HINTS /opt/homebrew/opt/portaudio/include /usr/local/opt/portaudio/include
              /usr/include /usr/local/include)
    find_library(PORTAUDIO_LIBRARY NAMES portaudio
        HINTS /opt/homebrew/opt/portaudio/lib /usr/local/opt/portaudio/lib
              /usr/lib /usr/local/lib)
    if(PORTAUDIO_INCLUDE_DIR AND PORTAUDIO_LIBRARY)
        set(_pa_found TRUE)
        set(_pa_includes ${PORTAUDIO_INCLUDE_DIR})
        set(_pa_libs ${PORTAUDIO_LIBRARY})
    endif()
endif()

if(NOT _pa_found)
    message(STATUS "examples: PortAudio not found - skipping transcribe_live "
                   "(install portaudio19-dev / brew install portaudio to enable)")
    return()
endif()

add_executable(transcribe_live transcribe_live.cpp)
target_include_directories(transcribe_live PRIVATE ${_pa_includes})
target_link_libraries(transcribe_live PRIVATE nemo_speech_asr_c ${_pa_libs})
install(TARGETS transcribe_live RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# See transcribe_file above: resolve libggml-cuda's cu* driver symbols at runtime.
if(UNIX AND NOT APPLE)
    target_link_options(transcribe_live PRIVATE "LINKER:--allow-shlib-undefined")
endif()

# Apple builds: link the system audio frameworks PortAudio needs (required for
# the static archive; harmless with the dylib).
if(APPLE)
    target_link_libraries(transcribe_live PRIVATE
        "-framework CoreFoundation"
        "-framework CoreAudio"
        "-framework AudioUnit"
        "-framework AudioToolbox")
endif()
