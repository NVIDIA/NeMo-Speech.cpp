// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
//
// llama-mmap.cpp (reused by GGUFLoader) logs via LLAMA_LOG_*; this supplies
// the one symbol it needs without linking llama-impl.cpp (format() clash).
#include <cstdarg>
#include <cstdio>

#include "llama-impl.h"
#include "runtime.h"

void
llama_log_internal(ggml_log_level level, const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    ggml_runtime::log_internal(level, __FILE__, __LINE__, "llama_mmap", "%s", buf);
}
