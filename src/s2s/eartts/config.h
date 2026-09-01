// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// EarTTS side-network configuration read from s2s_eartts_side.* GGUF metadata.
#pragma once

#include <vector>

#include "runtime.h"

namespace nemo_speech::s2s {

struct EarTTSCharConfig {
    int hidden = 0;
    int intermediate = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int n_layers = 0;
    float rope_theta = 10000.f;
    float rms_eps = 1e-6f;
    float query_pre_attn_scalar = 256.f;
    float attn_logit_softcap = 50.f;
    int max_char_len = 0;
    int char_vocab = 0;
    int subword_vocab = 0;
};

struct EarTTSConfig {
    int hidden = 0;
    int latent = 0;
    int codebook_size = 0;
    int num_quantizers = 0;
    int intermediate = 0;
    int mog_num_predictions = 0;
    int mog_num_layers = 0;
    int mog_low_rank = 0;
    int num_iter = 0;
    float mog_min_log_std = -4.f;
    float mog_eps = 1e-6f;
    float noise_scale = 0.f;
    float top_p_or_k = 1.f;
    float exponent = 3.f;
    float guidance_default = 0.f;
    bool has_guidance_default = false;

    bool use_subword_flag = false;
    bool use_bos_eos = false;
    bool use_gated_fusion = false;
    bool use_audio_prompt_proj = false;

    EarTTSCharConfig char_cfg;

    // MaskGIT unmask schedule: codebooks filled per sampling step, leading
    // zero-k steps stripped (mirrors mog_sampler.py:258-265).
    std::vector<int> sampling_per_step;

    int num_sampling_iter() const { return static_cast<int>(sampling_per_step.size()); }

    static EarTTSConfig from_gguf(const ggml_runtime::GGUFLoader& loader);
};

}  // namespace nemo_speech::s2s
