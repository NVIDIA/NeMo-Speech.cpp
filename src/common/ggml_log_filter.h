// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstdio>

#include "ggml.h"

namespace nemo_speech {

class GgmlLogFilter {
   public:
    void set_verbose(bool verbose) {
        verbose_.store(verbose, std::memory_order_relaxed);
        continue_log_.store(false, std::memory_order_relaxed);
    }

    bool should_emit(ggml_log_level level) {
        if (level == GGML_LOG_LEVEL_CONT)
            return verbose_.load(std::memory_order_relaxed) ||
                   continue_log_.load(std::memory_order_relaxed);

        const bool emit = verbose_.load(std::memory_order_relaxed) || level >= GGML_LOG_LEVEL_ERROR;
        continue_log_.store(emit, std::memory_order_relaxed);
        return emit;
    }

    static void callback(ggml_log_level level, const char* text, void* user_data) {
        auto& filter = *static_cast<GgmlLogFilter*>(user_data);
        if (filter.should_emit(level)) {
            std::fputs(text, stderr);
            std::fflush(stderr);
        }
    }

   private:
    std::atomic<bool> verbose_{false};
    std::atomic<bool> continue_log_{false};
};

}  // namespace nemo_speech
