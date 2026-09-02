// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tts_device_policy.h"

void
apply_tts_device_policy(nemo_speech::tts::MagpieRuntimeConfig& config, TtsCliDevice device) {
    using Backend = nemo_speech::tts::MagpieBackendPreference;
    switch (device) {
        case TtsCliDevice::Cpu:
            config.lt_backend = Backend::Cpu;
            config.sampling_backend = Backend::Cpu;
            config.magpie_cpu = true;
            config.codec_cpu = true;
            break;
        case TtsCliDevice::Cuda:
            config.lt_backend = Backend::Cuda;
            break;
        case TtsCliDevice::Accelerator:
            config.lt_backend = Backend::Cpu;
            config.sampling_backend = Backend::Cpu;
            break;
    }
}
