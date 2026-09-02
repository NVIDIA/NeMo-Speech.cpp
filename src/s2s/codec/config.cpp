// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "config.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace gr = ggml_runtime;

namespace nemo_speech::s2s {

// ===========================================================================
// CodecConfig
// ===========================================================================
CodecConfig
CodecConfig::from_gguf(const gr::GGUFLoader& loader) {
    CodecConfig c;
    c.spec_channels = static_cast<int>(loader.get_u32("eartts_codec.spec_channels"));
    c.n_fft = static_cast<int>(loader.get_u32("eartts_codec.n_fft"));
    c.hop_length = static_cast<int>(loader.get_u32("eartts_codec.hop_length"));
    c.sample_rate = static_cast<int>(loader.get_u32("eartts_codec.sample_rate"));
    c.base_hidden_size = static_cast<int>(loader.get_u32("eartts_codec.base_hidden_size"));
    c.latent_dim = static_cast<int>(loader.get_u32("eartts_codec.latent_size"));
    c.conv_next_kernel = static_cast<int>(loader.get_u32("eartts_codec.conv_next_kernel"));
    c.intermediate_ratio = static_cast<int>(loader.get_u32("eartts_codec.intermediate_ratio"));
    c.num_blocks_per_stage = static_cast<int>(loader.get_u32("eartts_codec.num_blocks_per_stage"));
    c.n_quantizers = static_cast<int>(loader.get_u32("eartts_codec.num_quantizers"));
    c.codebook_size = static_cast<int>(loader.get_u32("eartts_codec.codebook_size"));
    c.stage_channels = loader.get_i32_array("eartts_codec.stage_channels");
    c.rates = loader.get_i32_array("eartts_codec.rates");
    if (c.n_quantizers == 0 || c.latent_dim == 0 || c.n_fft == 0 || c.hop_length == 0 ||
        c.conv_next_kernel == 0 || c.num_blocks_per_stage == 0 || c.stage_channels.empty() ||
        c.rates.empty())
        throw std::runtime_error("codec gguf missing eartts_codec.* metadata");
    return c;
}

}  // namespace nemo_speech::s2s
