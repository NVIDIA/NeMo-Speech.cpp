// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Shared llama.cpp process-wide initialization for the S2S backbones.
//
// llama_backend_init() must run exactly once per process before any model
// load; both LLMBackbone and EarTTSBackbone call ensure_llama_backend() from
// their constructors. llama_backend_free() is registered with atexit.
#pragma once

#include <cstdlib>
#include <mutex>

#include "llama.h"
#include "model_logging.h"

namespace nemo_speech::s2s {

inline void
configure_llama_logging(bool verbose) {
    common::ensure_ggml_logging(verbose);
    llama_log_set(common::model_log_callback, nullptr);
}

inline void
ensure_llama_backend() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        llama_backend_init();
        std::atexit(llama_backend_free);
    });
}

}  // namespace nemo_speech::s2s
