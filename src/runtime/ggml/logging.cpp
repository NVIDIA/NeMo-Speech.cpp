// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "runtime.h"

namespace ggml_runtime {

GGML_ATTRIBUTE_FORMAT(5, 6)
void
log_internal(
    ggml_log_level level, const char* file, int line, const char* func, const char* format, ...) {
    va_list args;
    va_start(args, format);
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[1024];
    int len = vsnprintf(buffer, 1024, format, args);
    if (len < 1024) {
        char formatted_buffer[2048];
        snprintf(
            formatted_buffer, sizeof(formatted_buffer), "%s:%d:<%s> %s", file, line, func, buffer);
        ggml_log_callback callback;
        void* user_data;
        ggml_log_get(&callback, &user_data);
        callback(level, formatted_buffer, user_data);
    } else {
        char* buffer2 = new char[len + 1];
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        char formatted_buffer[4096];
        snprintf(
            formatted_buffer, sizeof(formatted_buffer), "%s:%d:<%s> %s", file, line, func, buffer2);
        ggml_log_callback callback;
        void* user_data;
        ggml_log_get(&callback, &user_data);
        callback(level, formatted_buffer, user_data);
        delete[] buffer2;
    }
    va_end(args_copy);
    va_end(args);
}

}  // namespace ggml_runtime

// Global-namespace; called from llama_file and other global helpers.
std::string
format(const char* fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX);  // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}
