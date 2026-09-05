// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "prompt.h"

#include <cstring>
#include <stdexcept>

#include "runtime.h"

namespace nemo_speech::s2s {

EarTTSPrompt::EarTTSPrompt(const std::string& gguf_path) {
    ggml_runtime::GGUFLoader loader(gguf_path);
    T_ = static_cast<int>(loader.get_u32("s2s_tts_prompt.length", 0));
    Q_ = static_cast<int>(loader.get_u32("s2s_tts_prompt.num_quantizers", 0));
    if (T_ <= 0 || Q_ <= 0)
        throw std::runtime_error("tts_prompt gguf missing s2s_tts_prompt.{length,num_quantizers}");

    auto read_i32 = [&](const std::string& name, size_t n) {
        const char* d = loader.get_tensor_file_data(name, n * sizeof(int32_t));
        std::vector<int32_t> v(n);
        std::memcpy(v.data(), d, n * sizeof(int32_t));
        return v;
    };
    auto read_f32 = [&](const std::string& name, size_t n) {
        const char* d = loader.get_tensor_file_data(name, n * sizeof(float));
        std::vector<float> v(n);
        std::memcpy(v.data(), d, n * sizeof(float));
        return v;
    };

    std::vector<int32_t> code = read_i32("tts_prompt.code", static_cast<size_t>(T_) * Q_);
    subword_ids_ = read_i32("tts_prompt.subword_ids", T_);
    subword_mask_ = read_f32("tts_prompt.subword_mask", T_);
    std::vector<float> non_prompt = read_f32("tts_prompt.non_prompt_mask", T_);
    if (loader.has_tensor("tts_prompt.audio_prompt_latent")) {
        H_ = static_cast<int>(loader.get_u32("s2s_tts_prompt.hidden_size", 0));
        if (H_ <= 0)
            throw std::runtime_error(
                "tts_prompt gguf has audio_prompt_latent but no valid hidden_size");
        audio_prompt_latent_ =
            read_f32("tts_prompt.audio_prompt_latent", static_cast<size_t>(T_) * H_);
    }

    // Teacher-forcing shift: drop last frame, prepend zeros.
    code_shifted_.assign(static_cast<size_t>(T_) * Q_, 0);
    if (T_ > 1)
        std::memcpy(
            code_shifted_.data() + Q_, code.data(),
            static_cast<size_t>(T_ - 1) * Q_ * sizeof(int32_t));

    // BOS edge detection.
    bos_mask_.assign(T_, 0.0f);
    for (int t = 0; t < T_; t++) {
        const bool cur = non_prompt[t] > 0.5f;
        const bool prev = (t > 0) && (non_prompt[t - 1] > 0.5f);
        bos_mask_[t] = (cur && !prev) ? 1.0f : 0.0f;
    }
}

}  // namespace nemo_speech::s2s
