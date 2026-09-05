// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <exception>
#include <iostream>
#include <stdexcept>

#include "tts_device_policy.h"

namespace {

using Backend = nemo_speech::tts::MagpieBackendPreference;
using Config = nemo_speech::tts::MagpieRuntimeConfig;

void
require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void
test_cpu_forces_all_components_to_cpu() {
    Config config;
    apply_tts_device_policy(config, TtsCliDevice::Cpu);

    require(config.lt_backend == Backend::Cpu, "CPU device must select the CPU LT backend");
    require(
        config.sampling_backend == Backend::Cpu, "CPU device must select the CPU sampling backend");
    require(config.magpie_cpu, "CPU device must force MagpieTTS onto CPU");
    require(config.codec_cpu, "CPU device must force NanoCodec onto CPU");
}

void
test_accelerator_preserves_codec_cpu_override() {
    Config config;
    config.codec_cpu = true;

    apply_tts_device_policy(config, TtsCliDevice::Accelerator);

    require(config.lt_backend == Backend::Cpu, "non-CUDA devices must use the CPU LT backend");
    require(
        config.sampling_backend == Backend::Cpu,
        "non-CUDA devices must use the CPU sampling backend");
    require(!config.magpie_cpu, "accelerator device must keep MagpieTTS on the accelerator");
    require(config.codec_cpu, "accelerator device must preserve the NanoCodec CPU override");
}

void
test_accelerator_keeps_default_codec_backend() {
    Config config;
    apply_tts_device_policy(config, TtsCliDevice::Accelerator);

    require(!config.codec_cpu, "accelerator device must not force NanoCodec onto CPU by default");
}

void
test_cuda_preserves_codec_cpu_override() {
    Config config;
    config.codec_cpu = true;

    apply_tts_device_policy(config, TtsCliDevice::Cuda);

    require(config.lt_backend == Backend::Cuda, "CUDA device must select the CUDA LT backend");
    require(config.codec_cpu, "CUDA device must preserve the NanoCodec CPU override");
}

void
test_cuda_keeps_default_codec_backend() {
    Config config;
    apply_tts_device_policy(config, TtsCliDevice::Cuda);

    require(!config.codec_cpu, "CUDA device must not force NanoCodec onto CPU by default");
}

}  // namespace

int
main() {
    try {
        test_cpu_forces_all_components_to_cpu();
        test_accelerator_preserves_codec_cpu_override();
        test_accelerator_keeps_default_codec_backend();
        test_cuda_preserves_codec_cpu_override();
        test_cuda_keeps_default_codec_backend();
        std::cout << "TTS device policy tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "test_tts_device_policy: " << error.what() << '\n';
        return 1;
    }
}
