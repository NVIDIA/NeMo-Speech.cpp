// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// TTS voice-prompt data (converted from tts_model_init_inputs.pt by
// scripts/convert_tts_prompt_to_gguf.py). Applies the teacher-forcing shift
// and bos-edge derivation EARTTSvLLM.get_prompt_input performs at stream
// creation, so consumers get ready-to-feed prefill arrays.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nemo_speech::s2s {

class EarTTSPrompt {
   public:
    explicit EarTTSPrompt(const std::string& gguf_path);

    int length() const { return T_; }
    int num_quantizers() const { return Q_; }
    int hidden_size() const { return H_; }
    bool has_audio_prompt_latent() const { return !audio_prompt_latent_.empty(); }

    // Teacher-shifted codes: row t = original row t-1 (row 0 = zeros).
    const std::vector<int32_t>& acoustic_tokens() const { return code_shifted_; }  // [T*Q]
    const std::vector<int32_t>& text_tokens() const { return subword_ids_; }       // [T]
    const std::vector<float>& text_mask() const { return subword_mask_; }          // [T]
    // bos_mask[t] = non_prompt[t] && !non_prompt[t-1]  (t=0: prev=false)
    const std::vector<float>& bos_mask() const { return bos_mask_; }  // [T]
    // Current checkpoints provide the frozen speaker projection directly.
    // Empty for older checkpoints, whose projection is computed at runtime.
    const std::vector<float>& audio_prompt_latent() const { return audio_prompt_latent_; }  // [T*H]

   private:
    int T_ = 0, Q_ = 0, H_ = 0;
    std::vector<int32_t> code_shifted_, subword_ids_;
    std::vector<float> subword_mask_, bos_mask_, audio_prompt_latent_;
};

}  // namespace nemo_speech::s2s
