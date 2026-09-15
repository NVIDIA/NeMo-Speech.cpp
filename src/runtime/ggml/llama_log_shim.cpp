// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// llama.cpp's llama-mmap.cpp (reused as-is by GGUFLoader, see loader.cpp)
// logs via the LLAMA_LOG_* macros, which call llama_log_internal. Its usual
// home, llama-impl.cpp, is deliberately not linked here: it also redefines
// format(const char*, ...), which collides at link time with this runtime's
// own definition (logging.cpp) that llama-mmap.cpp's format() calls resolve
// to instead. This is the one remaining symbol llama-mmap.cpp needs.
#include "llama-impl.h"
#include "runtime.h"

#include <cstdarg>
#include <cstdio>

void
llama_log_internal(ggml_log_level level, const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    ggml_runtime::log_internal(level, __FILE__, __LINE__, "llama_mmap", "%s", buf);
}
