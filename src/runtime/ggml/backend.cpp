// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
#include <algorithm>
#include <cstdlib>
#include <mutex>

#include "runtime.h"

namespace ggml_runtime {

BackendManager::BackendManager(Params params) {
    this->params = params;
    init_backends();
}

BackendManager::BackendManager(Params params, ggml_backend_t borrowed_gpu_backend) {
    this->params = params;
    borrowed_gpu_backend_ = borrowed_gpu_backend;
    init_backends();
}


void
BackendManager::init_backends() {
#if defined(GGML_USE_VULKAN)
    // Graph optimization is incompatible with in-place persistent cache tensors.
    // Vulkan reads this setting during device initialization; respect user overrides.
    // putenv retains the supplied storage, so the buffer must have static lifetime.
    if (std::getenv("GGML_VK_DISABLE_GRAPH_OPTIMIZE") == nullptr) {
        static char kv[] = "GGML_VK_DISABLE_GRAPH_OPTIMIZE=1";
        putenv(kv);
    }
#endif  // GGML_USE_VULKAN

    ggml_time_init();

    // Initialize ggml's process-wide f16 conversion tables.
    {
        struct ggml_init_params params = {0, NULL, false};
        struct ggml_context* ctx = ggml_init(params);
        ggml_free(ctx);
    }

    auto dev_count = ggml_backend_dev_count();
    GGMLF_LOG_INFO("Found %zu devices.\n", dev_count);

    ggml_backend_dev_t dev = nullptr;
    if (params.use_gpu) {
        if (borrowed_gpu_backend_ != nullptr) {
            dev = ggml_backend_get_device(borrowed_gpu_backend_);
            const auto dev_type = dev ? ggml_backend_dev_type(dev) : GGML_BACKEND_DEVICE_TYPE_CPU;
            if (dev == nullptr ||
                (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
                 dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU)) {
                throw std::runtime_error("borrowed GPU backend is not a GPU device");
            }
            auto* buft = ggml_backend_dev_buffer_type(dev);
            if (buft) {
                buft_list.emplace_back(dev, buft);
            }
            gpu_backend = borrowed_gpu_backend_;
            GGMLF_LOG_INFO("Using borrowed GPU backend: %s\n", ggml_backend_name(gpu_backend));
        } else {
            int idx = 0;
            for (int i = 0; i < dev_count; i++) {
                ggml_backend_dev_t dev_cur = ggml_backend_dev_get(i);
                GGMLF_LOG_INFO("Device %d: %s\n", i, ggml_backend_dev_name(dev_cur));
                const auto dev_type = ggml_backend_dev_type(dev_cur);
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                    dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    // Register buffer types only for the selected device.
                    if (idx == params.gpu_device_idx) {
                        dev = dev_cur;
                        auto* buft = ggml_backend_dev_buffer_type(dev);
                        if (buft) {
                            buft_list.emplace_back(dev, buft);
                        }
                    }

                    if (++idx > params.gpu_device_idx) {
                        break;
                    }
                }
            }
            if (dev == nullptr) {
                throw std::runtime_error(
                    "use_gpu=true but no matching GPU device found (gpu_device_idx=" +
                    std::to_string(params.gpu_device_idx) + ")");
            }
            GGMLF_LOG_INFO("Using GPU backend: %s\n", ggml_backend_dev_name(dev));
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(
                    std::string("use_gpu=true but ggml_backend_dev_init failed for ") +
                    ggml_backend_dev_name(dev));
            }
            gpu_backend = backend;
            backends.emplace_back(backend);
        }
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            GGMLF_LOG_INFO("Using %s backend\n", ggml_backend_dev_name(dev));
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (!backend) {
                GGMLF_LOG_INFO("failed to initialize %s backend\n", ggml_backend_dev_name(dev));
                continue;
            }
            backends.emplace_back(backend);
        }
    }

    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend_cpu == nullptr) {
        throw std::runtime_error("failed to initialize CPU backend");
    }
    GGMLF_LOG_INFO("Using CPU backend\n");
    backends.emplace_back(backend_cpu);

    auto* cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto* cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(
        cpu_reg, "ggml_backend_dev_get_extra_bufts");
    if (get_extra_bufts_fn) {
        ggml_backend_buffer_type_t* extra_bufts = get_extra_bufts_fn(cpu_dev);
        while (extra_bufts && *extra_bufts) {
            buft_list.emplace_back(cpu_dev, *extra_bufts);
            ++extra_bufts;
        }
    }
    buft_list.emplace_back(cpu_dev, ggml_backend_cpu_buffer_type());

    for (const auto& buft : buft_list) {
        ggml_backend_dev_t dev = buft.first;
        ggml_backend_buffer_type_t buft_type = buft.second;
        GGMLF_LOG_INFO("Buffer type: %s\n", ggml_backend_buft_name(buft_type));
    }
}

std::vector<ggml_backend_t>
BackendManager::get_backends() {
    std::vector<ggml_backend_t> handles;
    handles.reserve(backends.size() + (borrowed_gpu_backend_ ? 1 : 0));
    if (borrowed_gpu_backend_) handles.push_back(borrowed_gpu_backend_);
    for (const auto& b : backends) handles.push_back(b.get());
    return handles;
}

BackendManager::~BackendManager() = default;

buft_list_t
BackendManager::get_buft_list() {
    return buft_list;
}


}  // namespace ggml_runtime
