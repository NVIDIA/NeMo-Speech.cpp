# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Protobuf + gRPC codegen for the vendored riva proto files, compiled into the
# static riva_proto lib. Two dependency-resolution paths:
#   * Config packages (Windows/vcpkg or an explicit Linux prefix): protobuf +
#     gRPC export imported targets (protobuf::protoc, gRPC::grpc_cpp_plugin,
#     gRPC::grpc++_unsecure).
#     gRPC::grpc++_unsecure transitively pulls protobuf + the matching Abseil.
#   * Linux (NGC-style dev image): system protobuf v25 + gRPC 1.62, resolved via
#     find_library with version pinning + an Abseil link set probed by name.

set(RIVA_PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/riva-common)
set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${GEN_DIR})

set(PROTO_FILES
    riva/proto/riva_common.proto
    riva/proto/riva_audio.proto
    riva/proto/riva_asr.proto
    riva/proto/riva_tts.proto
    riva/proto/riva_nmt.proto
)

if(WIN32 OR NEMO_SPEECH_GRPC_USE_CONFIG)
    # Imported targets carry their own include dirs and transitive dependencies,
    # so no path pinning or hand-maintained Abseil library list is needed here.
    find_package(protobuf CONFIG REQUIRED)
    find_package(gRPC CONFIG REQUIRED)
    set(_PROTOC_CMD protobuf::protoc)
    set(_GRPC_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
    if(NOT TARGET gRPC::grpc++_unsecure)
        message(FATAL_ERROR "The installed gRPC package does not provide gRPC::grpc++_unsecure")
    endif()
    set(RIVA_PROTO_LINK_LIBS gRPC::grpc++_unsecure protobuf::libprotobuf)
    set(_PROTO_INCLUDE_DIRS "")
    message(STATUS "riva_proto: using protobuf + gRPC config packages")
else()
    find_package(Protobuf REQUIRED)
    find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

    # /usr/lib/x86_64-linux-gnu/libprotobuf.so → libprotobuf.so.32 (ABI mismatch with
    # protoc v25.3 + the v25.3 headers in /usr/include). Pin to the v25.3 runtime
    # library in /lib so generated code links cleanly.
    find_library(PROTOBUF_LIB_PINNED
        NAMES libprotobuf.so.25.3.0 libprotobuf.so.25
        PATHS /lib /lib/x86_64-linux-gnu /usr/lib /usr/lib/x86_64-linux-gnu
        NO_DEFAULT_PATH
    )
    if(NOT PROTOBUF_LIB_PINNED)
        # No pinned .so.25: use the full path find_package resolved, which works
        # regardless of the linker's default search dirs.
        if(TARGET protobuf::libprotobuf)
            set(PROTOBUF_LIB_PINNED protobuf::libprotobuf)
        elseif(Protobuf_LIBRARIES)
            set(PROTOBUF_LIB_PINNED ${Protobuf_LIBRARIES})
        else()
            set(PROTOBUF_LIB_PINNED protobuf)
        endif()
    endif()
    message(STATUS "Using protobuf runtime: ${PROTOBUF_LIB_PINNED}")

    # The server exposes plaintext gRPC only. Link gRPC's unsecure libraries so
    # the runtime does not carry unused TLS and cryptography implementations.
    set(_GRPC_LIBS "")
    foreach(_g grpc++_unsecure grpc_unsecure gpr)
        find_library(_RIVA_PROTO_${_g} ${_g})
        if(_RIVA_PROTO_${_g})
            list(APPEND _GRPC_LIBS ${_RIVA_PROTO_${_g}})
        else()
            list(APPEND _GRPC_LIBS ${_g})
        endif()
    endforeach()

    # grpc++ + absl link set. Build environments ship different Abseil snapshots, and
    # newer snapshots merge some libs into their parents (e.g. bad_optional_access,
    # strerror), so a hand-curated list breaks across versions. Probe every name with
    # find_library and link only those that exist; symbols from a merged-away lib live
    # in a sibling that is still present. pkg-config would compute this automatically
    # but pulls in a libcares dependency the noble dev image doesn't provide.
    set(_ABSL_LIBS
        absl_statusor absl_status absl_cord absl_cordz_info absl_cord_internal
        absl_cordz_functions absl_exponential_biased absl_cordz_handle
        absl_bad_optional_access absl_strerror absl_str_format_internal
        absl_synchronization absl_graphcycles_internal
        absl_stacktrace absl_symbolize absl_debugging_internal absl_demangle_internal
        absl_malloc_internal absl_time absl_civil_time absl_strings
        absl_strings_internal absl_base absl_spinlock_wait absl_int128
        absl_throw_delegate absl_time_zone absl_bad_variant_access
        absl_raw_logging_internal absl_log_severity
        absl_raw_hash_set absl_hash absl_city absl_low_level_hash
        absl_hashtablez_sampler
        absl_flags_internal absl_flags_reflection
        absl_flags_private_handle_accessor absl_flags_commandlineflag
        absl_flags_commandlineflag_internal absl_flags_config
        absl_flags_program_name absl_flags_marshalling
        absl_random_distributions absl_random_seed_sequences
        absl_random_internal_pool_urbg absl_random_internal_randen
        absl_random_internal_randen_hwaes absl_random_internal_randen_hwaes_impl
        absl_random_internal_randen_slow absl_random_internal_platform
        absl_random_internal_seed_material absl_random_seed_gen_exception
        absl_examine_stack absl_leak_check
        # Names that exist only in some snapshots (probed, skipped when absent):
        absl_crc_cord_state absl_crc32c absl_crc_internal absl_crc_cpu_detect
        absl_kernel_timeout_internal
        absl_string_view absl_flags absl_die_if_null
        absl_log_internal_message absl_log_internal_check_op
        absl_log_internal_nullguard absl_log_internal_format
        absl_log_internal_proto absl_log_internal_log_sink_set
        absl_log_internal_globals absl_log_internal_conditions
        absl_log_globals absl_log_sink absl_log_entry
    )
    set(_ABSL_PRESENT_LIBS "")
    foreach(_lib ${_ABSL_LIBS})
        find_library(_RIVA_PROTO_${_lib} ${_lib})
        if(_RIVA_PROTO_${_lib})
            list(APPEND _ABSL_PRESENT_LIBS ${_RIVA_PROTO_${_lib}})
        endif()
    endforeach()
    list(LENGTH _ABSL_PRESENT_LIBS _n_absl)
    message(STATUS "riva_proto: linking ${_n_absl} absl libs found on this system")

    set(_PROTOC_CMD ${Protobuf_PROTOC_EXECUTABLE})
    set(_GRPC_PLUGIN ${GRPC_CPP_PLUGIN})
    set(RIVA_PROTO_LINK_LIBS ${PROTOBUF_LIB_PINNED} ${_GRPC_LIBS} ${_ABSL_PRESENT_LIBS})
    set(_PROTO_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS})
endif()

set(GENERATED_SRCS)
foreach(proto ${PROTO_FILES})
    set(proto_root ${RIVA_PROTO_DIR})
    if(proto MATCHES "^nemotron/")
        set(proto_root ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
    get_filename_component(proto_dir ${proto} DIRECTORY)
    get_filename_component(proto_we  ${proto} NAME_WE)
    set(pb_cc   ${GEN_DIR}/${proto_dir}/${proto_we}.pb.cc)
    set(pb_h    ${GEN_DIR}/${proto_dir}/${proto_we}.pb.h)
    set(grpc_cc ${GEN_DIR}/${proto_dir}/${proto_we}.grpc.pb.cc)
    set(grpc_h  ${GEN_DIR}/${proto_dir}/${proto_we}.grpc.pb.h)

    add_custom_command(
        OUTPUT ${pb_cc} ${pb_h} ${grpc_cc} ${grpc_h}
        COMMAND ${_PROTOC_CMD}
            --proto_path=${proto_root}
            --cpp_out=${GEN_DIR}
            --grpc_out=${GEN_DIR}
            --plugin=protoc-gen-grpc=${_GRPC_PLUGIN}
            ${proto_root}/${proto}
        DEPENDS ${proto_root}/${proto}
        COMMENT "protoc ${proto}"
        VERBATIM
    )
    list(APPEND GENERATED_SRCS ${pb_cc} ${grpc_cc})
endforeach()

add_library(riva_proto STATIC ${GENERATED_SRCS})
# ${GEN_DIR} for the generated headers; ${_PROTO_INCLUDE_DIRS} so the generated
# .pb.h's `#include <google/protobuf/...>` resolves even when protobuf isn't in a
# default system include path (Linux). On Windows the imported targets carry
# their include dirs transitively.
target_include_directories(riva_proto PUBLIC ${GEN_DIR} ${_PROTO_INCLUDE_DIRS})
target_link_libraries(riva_proto PUBLIC ${RIVA_PROTO_LINK_LIBS})
