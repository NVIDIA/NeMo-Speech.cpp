// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "config.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace gr = ggml_runtime;

namespace nemo_speech::s2s {

// ===========================================================================
// EarTTSConfig
// ===========================================================================
EarTTSConfig
EarTTSConfig::from_gguf(const gr::GGUFLoader& loader) {
    EarTTSConfig c;
    c.hidden = static_cast<int>(loader.get_u32("s2s_eartts_side.hidden_size"));
    c.latent = static_cast<int>(loader.get_u32("s2s_eartts_side.latent_size"));
    c.codebook_size = static_cast<int>(loader.get_u32("s2s_eartts_side.codebook_size"));
    c.num_quantizers = static_cast<int>(loader.get_u32("s2s_eartts_side.num_quantizers"));
    c.intermediate = static_cast<int>(loader.get_u32("s2s_eartts_side.intermediate_size"));
    c.mog_num_predictions = static_cast<int>(loader.get_u32("s2s_eartts_side.mog_num_predictions"));
    c.mog_num_layers = static_cast<int>(loader.get_u32("s2s_eartts_side.mog_num_layers"));
    c.mog_low_rank = static_cast<int>(loader.get_u32("s2s_eartts_side.mog_low_rank"));
    c.mog_min_log_std = loader.get_f32("s2s_eartts_side.mog_min_log_std", -4.f);
    c.mog_eps = loader.get_f32("s2s_eartts_side.mog_eps", 1e-6f);
    c.noise_scale = loader.get_f32("s2s_eartts_side.noise_scale", 0.f);
    c.top_p_or_k = loader.get_f32("s2s_eartts_side.top_p_or_k", 1.f);
    c.num_iter = static_cast<int>(loader.get_u32("s2s_eartts_side.num_iter"));
    c.exponent = loader.get_f32("s2s_eartts_side.exponent", 3.f);
    c.has_guidance_default = loader.has_key("s2s_eartts_side.guidance_scale");
    c.guidance_default = loader.get_f32("s2s_eartts_side.guidance_scale", 0.f);

    c.use_subword_flag = loader.get_bool("s2s_eartts_side.use_subword_flag_emb", false);
    c.use_bos_eos = loader.get_bool("s2s_eartts_side.use_bos_eos_emb", false);
    c.use_gated_fusion = loader.get_bool("s2s_eartts_side.use_gated_fusion", false);
    c.use_audio_prompt_proj = loader.get_bool("s2s_eartts_side.use_audio_prompt_proj", false);

    c.char_cfg.hidden = static_cast<int>(loader.get_u32("s2s_eartts_side.char.hidden_size"));
    c.char_cfg.intermediate =
        static_cast<int>(loader.get_u32("s2s_eartts_side.char.intermediate_size"));
    c.char_cfg.n_heads =
        static_cast<int>(loader.get_u32("s2s_eartts_side.char.num_attention_heads"));
    c.char_cfg.n_kv_heads = static_cast<int>(loader.get_u32("s2s_eartts_side.char.num_kv_heads"));
    c.char_cfg.head_dim = static_cast<int>(loader.get_u32("s2s_eartts_side.char.head_dim"));
    c.char_cfg.n_layers =
        static_cast<int>(loader.get_u32("s2s_eartts_side.char.num_hidden_layers"));
    c.char_cfg.rope_theta = loader.get_f32("s2s_eartts_side.char.rope_theta", 10000.f);
    c.char_cfg.rms_eps = loader.get_f32("s2s_eartts_side.char.rms_norm_eps", 1e-6f);
    c.char_cfg.query_pre_attn_scalar =
        loader.get_f32("s2s_eartts_side.char.query_pre_attn_scalar", 256.f);
    c.char_cfg.attn_logit_softcap = loader.get_f32("s2s_eartts_side.char.attn_logit_softcap", 50.f);
    c.char_cfg.max_char_len = static_cast<int>(loader.get_u32("s2s_eartts_side.char.max_char_len"));
    c.char_cfg.char_vocab =
        static_cast<int>(loader.get_u32("s2s_eartts_side.char.char_vocab_size"));
    c.char_cfg.subword_vocab =
        static_cast<int>(loader.get_u32("s2s_eartts_side.char.subword_vocab_size"));

    if (c.hidden == 0 || c.latent == 0 || c.codebook_size == 0 || c.num_quantizers == 0 ||
        c.num_iter == 0 || c.char_cfg.hidden == 0)
        throw std::runtime_error("eartts gguf missing s2s_eartts_side.* metadata");

    // Precompute MaskGIT schedule (mirrors mog_sampler.py:258-265).
    {
        const int num_iter = c.num_iter;
        const int Q = c.num_quantizers;
        const double e = c.exponent;
        std::vector<int> num_maskings(num_iter);
        for (int i = 0; i < num_iter; i++) {
            const double r = static_cast<double>(i) / num_iter;  // linspace[:-1]
            const double masking_rate = std::pow(1.0 - std::pow(r, e), 1.0 / e);
            num_maskings[i] = static_cast<int>(std::ceil(masking_rate * Q));
        }
        std::vector<int> per_step(num_iter);
        for (int i = 0; i < num_iter; i++) {
            const int next_v = (i + 1 < num_iter) ? num_maskings[i + 1] : 0;
            per_step[i] = num_maskings[i] - next_v;
        }
        int first_nonzero = 0;
        for (int i = 0; i < num_iter; i++) {
            if (per_step[i] != 0) {
                first_nonzero = i;
                break;
            }
        }
        c.sampling_per_step.assign(per_step.begin() + first_nonzero, per_step.end());
    }
    return c;
}

}  // namespace nemo_speech::s2s
