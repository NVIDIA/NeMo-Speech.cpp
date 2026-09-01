// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "heads.h"

#include <cmath>
#include <stdexcept>

#include "ggml.h"

namespace gr = ggml_runtime;

namespace nemo_speech::s2s {

// ---------------------------------------------------------------------------
// EmbedTokensModule — out[n, H] = embed_tokens.weight[ids[n]]
// ---------------------------------------------------------------------------
class LLMHeads::EmbedTokensModule : public gr::Module {
   public:
    explicit EmbedTokensModule(int hidden, int vocab) : hidden_(hidden), vocab_(vocab) {}

    void define_tensors(gr::Session* s) override {
        s->model_tensor_container->create_tensor_2d(
            "embed_tokens.weight", s->gguf_loader->get_tensor_type("embed_tokens.weight"), hidden_,
            vocab_);
    }

    gr::TensorBag build_graph(gr::Session* s, gr::TensorBag in, gr::TensorContainer* tc) override {
        auto ids = in.get_tensor(0);  // i32 (n,)
        auto ctx = tc->get_ctx_of_buffer_type(ids.buft);
        auto w = s->model_tensor_container->get_tensor_by_name("embed_tokens.weight");
        ggml_tensor* rows = ggml_get_rows(ctx.ctx, w.tensor, ids.tensor);  // (H, n)
        if (rows->type != GGML_TYPE_F32)
            rows = ggml_cast(ctx.ctx, rows, GGML_TYPE_F32);
        gr::TensorBag out;
        out.add_tensor(gr::ggml_bf_tensor(rows, ids.buft));
        return out;
    }

    void set_data(gr::Session* s) override { s->load_weight("embed_tokens.weight"); }

   private:
    int hidden_, vocab_;
};

// ---------------------------------------------------------------------------
// FunctionHeadModule — out[b, V] = hidden[b, H] @ W^T   (F.linear semantics)
// ---------------------------------------------------------------------------
class LLMHeads::FunctionHeadModule : public gr::Module {
   public:
    explicit FunctionHeadModule(int hidden, int fn_out) : hidden_(hidden), fn_out_(fn_out) {}

    void define_tensors(gr::Session* s) override {
        s->model_tensor_container->create_tensor_2d(
            "function_head.weight", s->gguf_loader->get_tensor_type("function_head.weight"),
            hidden_, fn_out_);
    }

    gr::TensorBag build_graph(gr::Session* s, gr::TensorBag in, gr::TensorContainer* tc) override {
        auto h = in.get_tensor(0);  // f32 (H, b)
        auto ctx = tc->get_ctx_of_buffer_type(h.buft);
        auto w = s->model_tensor_container->get_tensor_by_name("function_head.weight");
        ggml_tensor* logits = ggml_mul_mat(ctx.ctx, w.tensor, h.tensor);  // (V, b)
        if (logits->type != GGML_TYPE_F32)
            logits = ggml_cast(ctx.ctx, logits, GGML_TYPE_F32);
        gr::TensorBag out;
        out.add_tensor(gr::ggml_bf_tensor(logits, h.buft));
        return out;
    }

    void set_data(gr::Session* s) override { s->load_weight("function_head.weight"); }

   private:
    int hidden_, fn_out_;
};

// ---------------------------------------------------------------------------
// LLMHeads facade
// ---------------------------------------------------------------------------
LLMHeads::LLMHeads(gr::BackendManager& bm, const std::string& gguf_path) {
    loader_ = std::make_unique<gr::GGUFLoader>(gguf_path);
    hidden_ = static_cast<int>(loader_->get_u32("s2s_llm_aux.hidden_size", 0));
    vocab_ = static_cast<int>(loader_->get_u32("s2s_llm_aux.vocab_size", 0));
    fn_out_ = static_cast<int>(loader_->get_u32("s2s_llm_aux.function_head.out_dim", vocab_));
    user_channel_weight_ = loader_->get_f32("s2s_llm_aux.user_channel_weight", 1.0f);
    text_channel_weight_ = loader_->get_f32("s2s_llm_aux.text_channel_weight", 1.0f);
    function_channel_weight_ = loader_->get_f32("s2s_llm_aux.function_channel_weight", 2.0f);
    if (hidden_ == 0 || vocab_ == 0)
        throw std::runtime_error("llm_aux gguf missing s2s_llm_aux.{hidden_size,vocab_size}");
    if (!std::isfinite(user_channel_weight_) || !std::isfinite(text_channel_weight_) ||
        !std::isfinite(function_channel_weight_))
        throw std::runtime_error("llm_aux gguf contains a non-finite channel weight");

    embed_module_ = std::make_unique<EmbedTokensModule>(hidden_, vocab_);
    fn_module_ = std::make_unique<FunctionHeadModule>(hidden_, fn_out_);
    embed_session_ = std::make_unique<gr::Session>(bm, embed_module_.get(), loader_.get());
    embed_session_->setup();
    fn_session_ = std::make_unique<gr::Session>(bm, fn_module_.get(), loader_.get());
    fn_session_->setup();
}

LLMHeads::~LLMHeads() = default;

void
LLMHeads::embed_tokens(const int32_t* ids, int n, float* out_emb) {
    std::lock_guard<std::mutex> lock(embed_mu_);
    std::vector<gr::Session::Output> outputs = {
        {0, "", out_emb, static_cast<size_t>(n) * hidden_ * sizeof(float)}};
    embed_session_->run({{"input.ids", GGML_TYPE_I32, ids, {n}}}, outputs);
}

void
LLMHeads::function_head(const float* hidden, int batch, float* out_logits) {
    std::lock_guard<std::mutex> lock(fn_mu_);
    std::vector<gr::Session::Output> outputs = {
        {0, "", out_logits, static_cast<size_t>(batch) * fn_out_ * sizeof(float)}};
    fn_session_->run({{"input.hidden", GGML_TYPE_F32, hidden, {hidden_, batch}}}, outputs);
}

}  // namespace nemo_speech::s2s
