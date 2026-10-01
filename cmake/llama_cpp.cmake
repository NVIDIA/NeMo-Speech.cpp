# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Resolve the llama.cpp source tree, which also provides ggml.
#
# The llama.cpp submodule is kept pristine. When NEMO_SPEECH_GGML_PATCHED is ON
# the patches/ series is applied to a copy of its pinned commit in the build
# directory, and the copy is refreshed only when the series or the pin changes.
# A refresh rewrites only the files whose content changed, so editing one patch
# rebuilds only what that edit affects.
#
# Also runs as a script, for tools that need the patched tree outside a build:
#   cmake -DSOURCE_DIR=llama.cpp -DPATCH_DIR=patches -DDEST_DIR=<dir> -P cmake/llama_cpp.cmake

# The patches listed in PATCH_DIR/series, in apply order. Every *.patch file in
# PATCH_DIR must be listed, so a new patch cannot be silently left out.
function(nemo_speech_llama_cpp_series patch_dir out_var)
    if(NOT EXISTS "${patch_dir}/series")
        message(FATAL_ERROR "missing ${patch_dir}/series")
    endif()
    file(STRINGS "${patch_dir}/series" lines)
    set(patches "")
    foreach(line IN LISTS lines)
        string(REGEX REPLACE "#.*$" "" line "${line}")
        string(STRIP "${line}" line)
        if(line STREQUAL "")
            continue()
        endif()
        if(NOT EXISTS "${patch_dir}/${line}")
            message(FATAL_ERROR "${patch_dir}/series lists ${line}, which does not exist")
        endif()
        list(APPEND patches "${patch_dir}/${line}")
    endforeach()
    file(GLOB present LIST_DIRECTORIES false "${patch_dir}/*.patch")
    foreach(patch IN LISTS present)
        list(FIND patches "${patch}" index)
        if(index EQUAL -1)
            message(FATAL_ERROR "${patch} is not listed in ${patch_dir}/series")
        endif()
    endforeach()
    set(${out_var} "${patches}" PARENT_SCOPE)
endfunction()

function(nemo_speech_materialize_llama_cpp source_dir patch_dir dest_dir)
    find_package(Git QUIET)
    if(NOT GIT_EXECUTABLE)
        message(FATAL_ERROR "git is required to apply ${patch_dir} to llama.cpp")
    endif()

    nemo_speech_llama_cpp_series("${patch_dir}" patches)

    # Key the copy by the pinned commit and the series content. Source archives
    # without git metadata fall back to the public headers as the base identity.
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" rev-parse HEAD
        OUTPUT_VARIABLE base RESULT_VARIABLE rc OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    set(have_git_metadata OFF)
    if(rc EQUAL 0)
        set(have_git_metadata ON)
    else()
        file(SHA256 "${source_dir}/ggml/include/ggml.h" ggml_h)
        file(SHA256 "${source_dir}/include/llama.h" llama_h)
        set(base "${ggml_h}-${llama_h}")
    endif()
    set(stamp_input "${base}")
    foreach(patch IN LISTS patches)
        file(SHA256 "${patch}" digest)
        string(APPEND stamp_input "-${digest}")
    endforeach()
    string(SHA256 stamp "${stamp_input}")

    set(stamp_file "${dest_dir}.stamp")
    if(EXISTS "${stamp_file}" AND EXISTS "${dest_dir}")
        file(READ "${stamp_file}" previous_stamp)
        if(previous_stamp STREQUAL stamp)
            return()
        endif()
    endif()
    file(REMOVE "${stamp_file}")

    list(LENGTH patches n_patches)
    message(STATUS "Applying ${n_patches} patches from ${patch_dir} to llama.cpp in ${dest_dir}")
    # Build the new tree in a staging directory, then copy over only what changed.
    set(staging "${dest_dir}.staging")
    file(REMOVE_RECURSE "${staging}")
    file(MAKE_DIRECTORY "${staging}")
    # Model and documentation assets are not needed to build.
    if(have_git_metadata)
        # Only tracked files at the pinned commit: local edits and build
        # directories inside the submodule stay out of the copy.
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" archive --format=tar -o "${staging}.tar" HEAD
                    -- . ":(exclude)models" ":(exclude)docs" ":(exclude)media"
            RESULT_VARIABLE rc)
        if(rc EQUAL 0)
            execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xf "${staging}.tar"
                WORKING_DIRECTORY "${staging}" RESULT_VARIABLE rc)
        endif()
        file(REMOVE "${staging}.tar")
        if(NOT rc EQUAL 0)
            message(FATAL_ERROR "could not export llama.cpp ${base} from ${source_dir}")
        endif()
    else()
        file(GLOB entries RELATIVE "${source_dir}" "${source_dir}/*")
        foreach(entry IN LISTS entries)
            if(NOT entry MATCHES "^(\\.git|models|docs|media)$")
                file(COPY "${source_dir}/${entry}" DESTINATION "${staging}")
            endif()
        endforeach()
    endif()

    # A build directory usually sits inside another git work tree; stop git from
    # discovering it so paths apply relative to the copy.
    get_filename_component(dest_parent "${dest_dir}" DIRECTORY)
    set(normalized "${staging}.patch")
    foreach(patch IN LISTS patches)
        # Windows checkouts may carry CRLF line endings.
        file(READ "${patch}" content)
        string(REPLACE "\r\n" "\n" content "${content}")
        file(WRITE "${normalized}" "${content}")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${dest_parent}"
                    "${GIT_EXECUTABLE}" apply --whitespace=nowarn "${normalized}"
            WORKING_DIRECTORY "${staging}"
            RESULT_VARIABLE rc ERROR_VARIABLE error)
        if(NOT rc EQUAL 0)
            get_filename_component(name "${patch}" NAME)
            message(FATAL_ERROR
                "${name} does not apply to llama.cpp ${base}:\n${error}\n"
                "Restore the pinned submodule (git submodule update llama.cpp) or rebase the "
                "series with scripts/llama-patches.sh rebase.")
        endif()
    endforeach()
    file(REMOVE "${normalized}")

    # Unchanged files keep their timestamps, so the build recompiles only what changed.
    file(GLOB_RECURSE new_files RELATIVE "${staging}" LIST_DIRECTORIES false "${staging}/*")
    foreach(path IN LISTS new_files)
        get_filename_component(dir "${dest_dir}/${path}" DIRECTORY)
        file(MAKE_DIRECTORY "${dir}")
        file(COPY_FILE "${staging}/${path}" "${dest_dir}/${path}" ONLY_IF_DIFFERENT)
    endforeach()
    file(GLOB_RECURSE old_files RELATIVE "${dest_dir}" LIST_DIRECTORIES false "${dest_dir}/*")
    foreach(path IN LISTS old_files)
        if(NOT EXISTS "${staging}/${path}")
            file(REMOVE "${dest_dir}/${path}")
        endif()
    endforeach()
    file(REMOVE_RECURSE "${staging}")
    file(WRITE "${stamp_file}" "${stamp}")
endfunction()

if(CMAKE_SCRIPT_MODE_FILE)
    foreach(var SOURCE_DIR PATCH_DIR DEST_DIR)
        if(NOT DEFINED ${var})
            message(FATAL_ERROR "usage: cmake -DSOURCE_DIR=... -DPATCH_DIR=... -DDEST_DIR=... -P ${CMAKE_SCRIPT_MODE_FILE}")
        endif()
        get_filename_component(${var} "${${var}}" ABSOLUTE)
    endforeach()
    nemo_speech_materialize_llama_cpp("${SOURCE_DIR}" "${PATCH_DIR}" "${DEST_DIR}")
    return()
endif()

set(NEMO_SPEECH_LLAMA_CPP_SOURCE_DIR "" CACHE PATH
    "Build ggml and llama.cpp from this tree as-is (for example a scripts/llama-patches.sh edit worktree)")
set(_nemo_speech_llama_cpp_submodule "${CMAKE_SOURCE_DIR}/llama.cpp")
if(NEMO_SPEECH_LLAMA_CPP_SOURCE_DIR)
    set(NEMO_SPEECH_LLAMA_CPP_DIR "${NEMO_SPEECH_LLAMA_CPP_SOURCE_DIR}")
elseif(NOT EXISTS "${_nemo_speech_llama_cpp_submodule}/ggml/CMakeLists.txt")
    message(FATAL_ERROR
        "The llama.cpp submodule (which also provides ggml) is not initialized.\n"
        "Run: git submodule update --init llama.cpp")
elseif(NEMO_SPEECH_GGML_PATCHED)
    set(NEMO_SPEECH_LLAMA_CPP_DIR "${CMAKE_BINARY_DIR}/_deps/llama.cpp")
    nemo_speech_materialize_llama_cpp(
        "${_nemo_speech_llama_cpp_submodule}" "${CMAKE_SOURCE_DIR}/patches" "${NEMO_SPEECH_LLAMA_CPP_DIR}")
    # Re-run configuration when patches are edited, added, removed, or reordered,
    # or when the submodule moves to another commit.
    nemo_speech_llama_cpp_series("${CMAKE_SOURCE_DIR}/patches" _nemo_speech_patches)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/patches" "${CMAKE_SOURCE_DIR}/patches/series" ${_nemo_speech_patches})
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${_nemo_speech_llama_cpp_submodule}" rev-parse --absolute-git-dir
        OUTPUT_VARIABLE _nemo_speech_llama_cpp_git_dir RESULT_VARIABLE _nemo_speech_rc
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_nemo_speech_rc EQUAL 0 AND EXISTS "${_nemo_speech_llama_cpp_git_dir}/HEAD")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${_nemo_speech_llama_cpp_git_dir}/HEAD")
    endif()
else()
    set(NEMO_SPEECH_LLAMA_CPP_DIR "${_nemo_speech_llama_cpp_submodule}")
endif()
message(STATUS "ggml and llama.cpp source: ${NEMO_SPEECH_LLAMA_CPP_DIR}")
