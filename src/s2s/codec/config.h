// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Codec hyperparameters, read from the eartts_codec GGUF KVs at load time.
#pragma once

#include <string>
#include <vector>

#include "runtime.h"

namespace nemo_speech::s2s {

struct CodecConfig {
    int n_quantizers = 0;
    int codebook_size = 0;
    int latent_dim = 0;
    int spec_channels = 0;
    int n_fft = 0;
    int hop_length = 0;
    int sample_rate = 0;
    int base_hidden_size = 0;
    int conv_next_kernel = 0;         // 7
    int intermediate_ratio = 0;       // 4
    int num_blocks_per_stage = 0;     // 3
    std::vector<int> stage_channels;  // {384, 768, 1536}
    std::vector<int> rates;           // {7, 7, 9}; last = bottleneck stride

    int n_bins() const { return n_fft / 2 + 1; }
    int samples_per_frame() const {
        int s = hop_length;
        for (int r : rates) s *= r;
        return s;  // 1764 for the production codec
    }
    int n_blocks() const { return num_blocks_per_stage * static_cast<int>(stage_channels.size()); }

    static CodecConfig from_gguf(const ggml_runtime::GGUFLoader& loader);
};

}  // namespace nemo_speech::s2s
