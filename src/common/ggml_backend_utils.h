// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ggml-backend.h"

namespace nemo_speech::common {

inline bool
is_cpu_backend(ggml_backend_t backend) {
    return ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

inline void
set_cpu_backend_n_threads(ggml_backend_t backend, int n_threads) {
    if (!is_cpu_backend(backend)) {
        return;
    }
    // CPU plugin functions must be accessed through the backend registry.
    auto* reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto* set_n_threads = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_set_n_threads");
    if (set_n_threads) {
        set_n_threads(backend, n_threads);
    }
}

}  // namespace nemo_speech::common
