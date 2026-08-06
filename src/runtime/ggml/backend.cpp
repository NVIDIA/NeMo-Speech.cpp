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

#if defined(GGML_USE_CUDA)
#include <ggml-cuda.h>
#endif

namespace ggml_runtime {

namespace {

void
set_process_environment(const char* name, const char* value) {
#if defined(_WIN32)
    if (_putenv_s(name, value) != 0)
        throw std::runtime_error(std::string("failed to set environment variable ") + name);
#else
    if (setenv(name, value, 1) != 0)
        throw std::runtime_error(std::string("failed to set environment variable ") + name);
#endif
}

std::string
compute_capability_name(int cc) {
    if (cc <= 0)
        return "unknown";
    return std::to_string(cc / 100) + "." + std::to_string((cc % 100) / 10);
}

void
configure_skinny_q8(ggml_backend_dev_t device, int gpu_index) {
#if defined(GGML_USE_CUDA)
    const char* mode = std::getenv("NEMO_SPEECH_SKINNY_Q8_MODE");
    if (mode == nullptr)
        return;  // No CLI control: preserve the original GGML environment behavior.

    const int cc = ggml_backend_cuda_get_device_compute_capability(gpu_index);
    const char* description = ggml_backend_dev_description(device);
    const std::string gpu = description ? description : "unknown GPU";
    const std::string cc_name = compute_capability_name(cc);
    if (std::strcmp(mode, "off") == 0) {
        set_process_environment("GGML_SKINNY_Q8", "0");
        GGMLF_LOG_INFO("[cuda] skinny-q8=off source=cli\n");
    } else if (std::strcmp(mode, "on") == 0) {
        if (cc <= 0 || cc < 800) {
            throw std::runtime_error(
                "Skinny Q8 is not compatible with the selected GPU.\nGPU: " + gpu +
                "\nCompute Capability: " + cc_name +
                "\nCurrent kernel requirement: SM 8.0 or higher\nUse: --skinny-q8 off");
        }
        set_process_environment("GGML_SKINNY_Q8", "1");
        GGMLF_LOG_INFO("[cuda] skinny-q8=on source=cli compute-capability=%s\n", cc_name.c_str());
    } else {  // auto
        if (cc <= 0 || cc < 800) {
            set_process_environment("GGML_SKINNY_Q8", "0");
            GGMLF_LOG_INFO(
                "[cuda] skinny-q8=off source=auto reason=compute-capability-%s\n", cc_name.c_str());
        } else {
            GGMLF_LOG_INFO(
                "[cuda] skinny-q8=on source=auto compute-capability=%s\n", cc_name.c_str());
        }
    }
#else
    (void)device;
    (void)gpu_index;
#endif
}

}  // namespace

BackendManager::BackendManager(Params params) {
    this->params = params;
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
        configure_skinny_q8(dev, params.gpu_device_idx);
        if (std::getenv("NEMO_SPEECH_SUPPRESS_CUDA_GRAPH_LOG") != nullptr)
            GGMLF_LOG_INFO("[cuda] cuda-graph-architecture-log=suppressed\n");
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            throw std::runtime_error(
                std::string("use_gpu=true but ggml_backend_dev_init failed for ") +
                ggml_backend_dev_name(dev));
        }
        gpu_backend = backend;
        backends.emplace_back(backend);
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
    handles.reserve(backends.size());
    for (const auto& b : backends) handles.push_back(b.get());
    return handles;
}

BackendManager::~BackendManager() = default;

buft_list_t
BackendManager::get_buft_list() {
    return buft_list;
}


}  // namespace ggml_runtime
