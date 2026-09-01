// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Process-wide filtering for GGML and llama.cpp diagnostics.
#pragma once

#include <atomic>
#include <cstdio>

#include "ggml.h"

namespace nemo_speech::common {
namespace detail {

inline std::atomic<bool> model_logging_configured{false};
inline std::atomic<bool> model_logging_verbose{false};
inline thread_local bool continue_model_log = false;

}  // namespace detail

inline void
model_log_callback(ggml_log_level level, const char* text, void*) {
    bool emit = detail::model_logging_verbose.load(std::memory_order_relaxed);
    if (level == GGML_LOG_LEVEL_CONT) {
        emit = emit || detail::continue_model_log;
    } else {
        emit = emit || level >= GGML_LOG_LEVEL_ERROR;
        detail::continue_model_log = emit;
    }
    if (emit && text) {
        std::fputs(text, stderr);
        std::fflush(stderr);
    }
}

// Explicit process-level configuration used by executable frontends.
inline void
configure_ggml_logging(bool verbose) {
    detail::model_logging_verbose.store(verbose, std::memory_order_relaxed);
    detail::continue_model_log = false;
    detail::model_logging_configured.store(true, std::memory_order_release);
    ggml_log_set(model_log_callback, nullptr);
}

// Library entry points default to quiet logging without overriding a setting
// already selected by the host process. Verbose is process-wide and sticky
// once any loaded model requests it.
inline void
ensure_ggml_logging(bool verbose = false) {
    if (verbose) {
        configure_ggml_logging(true);
        return;
    }
    bool expected = false;
    if (detail::model_logging_configured.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        detail::model_logging_verbose.store(false, std::memory_order_relaxed);
        ggml_log_set(model_log_callback, nullptr);
    }
}

}  // namespace nemo_speech::common
