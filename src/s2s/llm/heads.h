// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// LLM auxiliary weights for the S2S pipeline: the input embedding table and
// the function-call head, served from `llm_aux.gguf` (arch "s2s_llm_aux",
// Q8_0 weights).
//
// Two ops, two Sessions (their input shapes and graphs are unrelated):
//   embed_tokens(ids[n])        -> [n, hidden]    f32   (get_rows)
//   function_head(hidden[b,H])  -> [b, vocab]     f32   (mul_mat, F.linear)
//
// The orchestrator calls embed_tokens for the text/function channel
// embeddings each step and for system-prompt prefill; function_head runs on
// the LLM backbone's final hidden state each step.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "runtime.h"

namespace nemo_speech::s2s {

class LLMHeads {
   public:
    LLMHeads(ggml_runtime::BackendManager& bm, const std::string& gguf_path);
    ~LLMHeads();

    LLMHeads(const LLMHeads&) = delete;
    LLMHeads& operator=(const LLMHeads&) = delete;

    int hidden_size() const { return hidden_; }
    int vocab_size() const { return vocab_; }
    int fn_out_dim() const { return fn_out_; }
    float user_channel_weight() const { return user_channel_weight_; }
    float text_channel_weight() const { return text_channel_weight_; }
    float function_channel_weight() const { return function_channel_weight_; }

    // out_emb must hold n * hidden_size() floats; row-major [n, hidden].
    void embed_tokens(const int32_t* ids, int n, float* out_emb);

    // hidden: [batch, hidden] row-major; out_logits: [batch, fn_out] row-major.
    void function_head(const float* hidden, int batch, float* out_logits);

   private:
    class EmbedTokensModule;
    class FunctionHeadModule;

    int hidden_ = 0;
    int vocab_ = 0;
    int fn_out_ = 0;
    float user_channel_weight_ = 1.0f;
    float text_channel_weight_ = 1.0f;
    float function_channel_weight_ = 2.0f;

    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    std::unique_ptr<EmbedTokensModule> embed_module_;
    std::unique_ptr<FunctionHeadModule> fn_module_;
    std::unique_ptr<ggml_runtime::Session> embed_session_;
    std::unique_ptr<ggml_runtime::Session> fn_session_;
    std::mutex embed_mu_;
    std::mutex fn_mu_;
};

}  // namespace nemo_speech::s2s
