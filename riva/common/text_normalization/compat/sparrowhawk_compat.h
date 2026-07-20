// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// OpenFST 1.8 / Sparrowhawk compatibility shim shared by ASR ITN and TTS TN.
//
// Sparrowhawk's public headers (regexp.h, spec_serializer.h, ...) were written
// against OpenFST <=1.7, which exposed legacy aliases (int32/int64), the
// DISALLOW_COPY_AND_ASSIGN macro, a global `using std::string`, and the
// fst::StringTokenType enum via <fst/compat.h>. OpenFST 1.8 removed/renamed all
// of these. Include this BEFORE any <sparrowhawk/...> header so those headers
// compile unchanged against the installed OpenFST 1.8 stack.
//
// Only pulled in by the shared text-normalization target, and only in a
// NEMO_SPEECH_WITH_NORM build.
#pragma once

#include <fst/log.h>
#include <fst/string.h>

#include <cstdint>
#include <string>

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

// Sparrowhawk signatures use bare `string` (OpenFST <=1.7 hoisted it globally).
using std::string;

// OpenFST 1.8 renamed StringTokenType -> TokenType (fst/string.h).
namespace fst {
using StringTokenType = TokenType;
}  // namespace fst

#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName&) = delete;    \
    TypeName& operator=(const TypeName&) = delete
#endif
