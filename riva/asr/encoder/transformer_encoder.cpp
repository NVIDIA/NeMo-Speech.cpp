// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "transformer_encoder.h"

#include <cmath>

using namespace nemo_speech::asr;

TransformerBlock::TransformerBlock(const std::string& name, const TransformerConfig& cfg)
    : name_(name), cfg_(cfg) {
    const std::string mha = name_ + ".first_sub_layer";
    query_net_ = new ggml_runtime::Linear(mha + ".query_net", cfg.hidden_size, cfg.hidden_size);
    key_net_ = new ggml_runtime::Linear(mha + ".key_net", cfg.hidden_size, cfg.hidden_size);
    value_net_ = new ggml_runtime::Linear(mha + ".value_net", cfg.hidden_size, cfg.hidden_size);
    out_projection_ =
        new ggml_runtime::Linear(mha + ".out_projection", cfg.hidden_size, cfg.hidden_size);
    const int64_t ln_shape[4] = {cfg.hidden_size, 1, 1, 1};
    layer_norm_1_ = new ggml_runtime::LayerNorm(name_ + ".layer_norm_1", ln_shape);
    const std::string ff = name_ + ".second_sub_layer";
    dense_in_ = new ggml_runtime::Linear(ff + ".dense_in", cfg.hidden_size, cfg.inner_size);
    dense_out_ = new ggml_runtime::Linear(ff + ".dense_out", cfg.inner_size, cfg.hidden_size);
    layer_norm_2_ = new ggml_runtime::LayerNorm(name_ + ".layer_norm_2", ln_shape);
}

TransformerBlock::~TransformerBlock() {
    delete query_net_;
    delete key_net_;
    delete value_net_;
    delete out_projection_;
    delete layer_norm_1_;
    delete dense_in_;
    delete dense_out_;
    delete layer_norm_2_;
}

void
TransformerBlock::define_tensors(ggml_runtime::Session* session) {
    query_net_->define_tensors(session);
    key_net_->define_tensors(session);
    value_net_->define_tensors(session);
    out_projection_->define_tensors(session);
    layer_norm_1_->define_tensors(session);
    dense_in_->define_tensors(session);
    dense_out_->define_tensors(session);
    layer_norm_2_->define_tensors(session);
}

void
TransformerBlock::set_data(ggml_runtime::Session* session) {
    query_net_->set_data(session);
    key_net_->set_data(session);
    value_net_->set_data(session);
    out_projection_->set_data(session);
    layer_norm_1_->set_data(session);
    dense_in_->set_data(session);
    dense_out_->set_data(session);
    layer_norm_2_->set_data(session);
}

ggml_runtime::TensorBag
TransformerBlock::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    auto x = input_tensors.get_tensor(0);
    auto bf_ctx = tc->get_ctx_of_buffer_type(x.buft);
    ggml_context* ctx = bf_ctx.ctx;

    const int n_head = cfg_.n_heads;
    const int d_k = cfg_.hidden_size / n_head;
    const int64_t T = x.tensor->ne[1];

    // Scaled dot-product self-attention without positional terms.
    auto q = query_net_->build_graph(session, input_tensors, tc).get_tensor(0);
    auto k = key_net_->build_graph(session, input_tensors, tc).get_tensor(0);
    auto v = value_net_->build_graph(session, input_tensors, tc).get_tensor(0);

    // (hidden, T) -> (d_k, n_head, T) -> (d_k, T, n_head)
    auto q_mh = ggml_reshape_4d(ctx, q.tensor, d_k, n_head, T, 1);
    auto k_mh = ggml_reshape_4d(ctx, k.tensor, d_k, n_head, T, 1);
    auto v_mh = ggml_reshape_4d(ctx, v.tensor, d_k, n_head, T, 1);
    auto q_p = ggml_cont(ctx, ggml_permute(ctx, q_mh, 0, 2, 1, 3));
    auto k_p = ggml_cont(ctx, ggml_permute(ctx, k_mh, 0, 2, 1, 3));

    // scores[kv, q, head] = q . k / sqrt(d_k). NeMo divides q and k each by
    // d_k^0.25; a single 1/sqrt(d_k) on the scores is the same product.
    auto scores = ggml_mul_mat(ctx, k_p, q_p);
    scores = ggml_scale_inplace(ctx, scores, 1.0f / std::sqrt(static_cast<float>(d_k)));
    auto probs = ggml_soft_max_inplace(ctx, scores);

    // context[d_k, q, head] = V^T . probs, with V as (T_kv, d_k, n_head).
    auto v_t = ggml_cont(ctx, ggml_permute(ctx, ggml_permute(ctx, v_mh, 2, 1, 0, 3), 0, 2, 1, 3));
    auto attn = ggml_permute(ctx, ggml_mul_mat(ctx, v_t, probs), 0, 2, 1, 3);
    // merge heads: (d_k, n_head, T) -> (hidden, T)
    auto merged = ggml_reshape_3d(ctx, ggml_cont(ctx, attn), n_head * d_k, T, 1);

    ggml_runtime::TensorBag proj_in;
    proj_in.add_tensor(ggml_runtime::ggml_bf_tensor(merged, x.buft));
    auto attn_out = out_projection_->build_graph(session, proj_in, tc).get_tensor(0);

    // residual + LN1
    auto h = ggml_add(ctx, attn_out.tensor, x.tensor);
    ggml_runtime::TensorBag ln1_in;
    ln1_in.add_tensor(ggml_runtime::ggml_bf_tensor(h, x.buft));
    auto h_ln = layer_norm_1_->build_graph(session, ln1_in, tc).get_tensor(0);

    // Feed-forward projection with ReLU.
    ggml_runtime::TensorBag ff_in;
    ff_in.add_tensor(h_ln);
    auto ff_mid = dense_in_->build_graph(session, ff_in, tc).get_tensor(0);
    ggml_runtime::TensorBag ff_mid_bag;
    ff_mid_bag.add_tensor(ggml_runtime::ggml_bf_tensor(ggml_relu(ctx, ff_mid.tensor), x.buft));
    auto ff_out = dense_out_->build_graph(session, ff_mid_bag, tc).get_tensor(0);

    // residual + LN2
    auto o = ggml_add(ctx, ff_out.tensor, h_ln.tensor);
    ggml_runtime::TensorBag ln2_in;
    ln2_in.add_tensor(ggml_runtime::ggml_bf_tensor(o, x.buft));
    return layer_norm_2_->build_graph(session, ln2_in, tc);
}

TransformerEncoderModule::TransformerEncoderModule(
    const std::string& name, const TransformerConfig& cfg)
    : name_(name), cfg_(cfg) {
    layers_.reserve(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; i++) {
        layers_.push_back(new TransformerBlock(name_ + ".layers." + std::to_string(i), cfg));
    }
}

TransformerEncoderModule::~TransformerEncoderModule() {
    for (auto* l : layers_) delete l;
}

void
TransformerEncoderModule::define_tensors(ggml_runtime::Session* session) {
    for (auto* l : layers_) l->define_tensors(session);
}

void
TransformerEncoderModule::set_data(ggml_runtime::Session* session) {
    for (auto* l : layers_) l->set_data(session);
}

ggml_runtime::TensorBag
TransformerEncoderModule::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    ggml_runtime::TensorBag bag = input_tensors;
    for (auto* l : layers_) {
        bag = l->build_graph(session, bag, tc);
    }
    return bag;
}
