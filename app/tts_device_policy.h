// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "runtime.h"

enum class TtsCliDevice {
    Cpu,
    Cuda,
    Accelerator,
};

// Apply the aggregate --device selection without discarding a requested
// NanoCodec CPU override. CPU is the exception: selecting it explicitly forces
// every TTS component onto CPU.
void apply_tts_device_policy(nemo_speech::tts::MagpieRuntimeConfig& config, TtsCliDevice device);
