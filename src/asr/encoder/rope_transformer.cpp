// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "rope_transformer.h"

#include <cmath>
#include <stdexcept>

using namespace nemo_speech::asr;

namespace {

ggml_runtime::TensorBag
one(ggml_tensor* tensor, ggml_backend_buffer_type_t buft) {
    ggml_runtime::TensorBag bag;
    bag.add_tensor(ggml_runtime::ggml_bf_tensor(tensor, buft));
    return bag;
}

}  // namespace

RopeTransformerBlock::RopeTransformerBlock(
    const std::string& name, const RopeTransformerConfig& cfg)
    : cfg_(cfg) {
    const int64_t norm_shape[4] = {cfg.d_model, 1, 1, 1};
    norm1_ = new ggml_runtime::LayerNorm(name + ".norm1", norm_shape);
    qkv_ =
        new ggml_runtime::Linear(name + ".attn.w_qkv", cfg.d_model, 3 * cfg.d_model, cfg.qkv_bias);
    out_proj_ = new ggml_runtime::Linear(name + ".attn.out_proj", cfg.d_model, cfg.d_model);
    norm2_ = new ggml_runtime::LayerNorm(name + ".norm2", norm_shape);
    ff_in_ = new ggml_runtime::Linear(name + ".ffn.net.0", cfg.d_model, cfg.d_ff);
    ff_out_ = new ggml_runtime::Linear(name + ".ffn.net.3", cfg.d_ff, cfg.d_model);
}

RopeTransformerBlock::~RopeTransformerBlock() {
    delete norm1_;
    delete qkv_;
    delete out_proj_;
    delete norm2_;
    delete ff_in_;
    delete ff_out_;
}

void
RopeTransformerBlock::define_tensors(ggml_runtime::Session* session) {
    norm1_->define_tensors(session);
    qkv_->define_tensors(session);
    out_proj_->define_tensors(session);
    norm2_->define_tensors(session);
    ff_in_->define_tensors(session);
    ff_out_->define_tensors(session);
}

void
RopeTransformerBlock::set_data(ggml_runtime::Session* session) {
    norm1_->set_data(session);
    qkv_->set_data(session);
    out_proj_->set_data(session);
    norm2_->set_data(session);
    ff_in_->set_data(session);
    ff_out_->set_data(session);
}

ggml_runtime::TensorBag
RopeTransformerBlock::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    auto x = input_tensors.get_tensor(0);
    auto positions = input_tensors.get_tensor(1);
    auto bf_ctx = tc->get_ctx_of_buffer_type(x.buft);
    ggml_context* ctx = bf_ctx.ctx;
    const int64_t time = x.tensor->ne[1];
    const int64_t batch = x.tensor->ne[2];
    const int head_dim = cfg_.d_model / cfg_.n_heads;

    auto xn = norm1_->build_graph(session, one(x.tensor, x.buft), tc).get_tensor(0);
    auto qkv = qkv_->build_graph(session, one(xn.tensor, xn.buft), tc).get_tensor(0);

    auto component = [&](int index) {
        const size_t offset = static_cast<size_t>(index) * cfg_.d_model * qkv.tensor->nb[0];
        // RoPE consumes explicit source strides, so it can read each Q/K/V
        // section directly from the interleaved projection output.
        return ggml_view_4d(
            ctx, qkv.tensor, head_dim, cfg_.n_heads, time, batch,
            static_cast<size_t>(head_dim) * qkv.tensor->nb[0], qkv.tensor->nb[1], qkv.tensor->nb[2],
            offset);
    };
    auto q = component(0);
    auto k = component(1);
    auto v = component(2);

    const int n_rot = static_cast<int>(head_dim * cfg_.rotary_fraction);
    q = ggml_rope_ext(
        ctx, q, positions.tensor, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, cfg_.pos_emb_max_len,
        cfg_.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(
        ctx, k, positions.tensor, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, cfg_.pos_emb_max_len,
        cfg_.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // RoPE indexes positions along ne2 in (head_dim, heads, time, batch).
    // Flash attention consumes (head_dim, time, heads, batch).
    // Its CUDA kernel honors explicit strides (and converts strided K/V to
    // F16 itself), so these views need not be contiguous.
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // ggml's fused operator maps to backend-native flash attention where
    // available and retains a portable CPU implementation otherwise.
    auto attn = ggml_flash_attn_ext(
        ctx, q, k, v, nullptr, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f, 0.0f);
    auto merged = ggml_reshape_3d(ctx, ggml_cont(ctx, attn), cfg_.d_model, time, batch);
    auto attn_out = out_proj_->build_graph(session, one(merged, x.buft), tc).get_tensor(0);
    auto residual = ggml_add(ctx, x.tensor, attn_out.tensor);

    auto rn = norm2_->build_graph(session, one(residual, x.buft), tc).get_tensor(0);
    auto ff = ff_in_->build_graph(session, one(rn.tensor, rn.buft), tc).get_tensor(0);
    auto gelu = ggml_gelu_erf(ctx, ff.tensor);
    auto ff_out = ff_out_->build_graph(session, one(gelu, x.buft), tc).get_tensor(0);

    ggml_runtime::TensorBag out;
    out.add_tensor(ggml_runtime::ggml_bf_tensor(ggml_add(ctx, residual, ff_out.tensor), x.buft));
    out.add_tensor(positions);
    return out;
}

RopeTransformerEncoder::RopeTransformerEncoder(
    const std::string& name, const RopeTransformerConfig& cfg)
    : cfg_(cfg) {
    if (cfg.d_model <= 0 || cfg.n_heads <= 0 || cfg.n_layers <= 0 || cfg.d_ff <= 0 ||
        cfg.pos_emb_max_len <= 0 || !std::isfinite(cfg.rope_base) || cfg.rope_base <= 0 ||
        !std::isfinite(cfg.rotary_fraction) || cfg.rotary_fraction <= 0 ||
        cfg.rotary_fraction > 1 || cfg.d_model % cfg.n_heads != 0 ||
        static_cast<int>(cfg.d_model / cfg.n_heads * cfg.rotary_fraction) == 0 ||
        static_cast<int>(cfg.d_model / cfg.n_heads * cfg.rotary_fraction) % 2 != 0) {
        throw std::invalid_argument("RoPE transformer has incompatible head/rotary dimensions");
    }
    const int64_t norm_shape[4] = {cfg.d_model, 1, 1, 1};
    embed_norm_ = cfg.pre_block_norm ? new ggml_runtime::LayerNorm(name + ".embed_norm", norm_shape)
                                     : nullptr;
    layers_.reserve(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
        layers_.push_back(new RopeTransformerBlock(name + ".layers." + std::to_string(i), cfg));
    }
    final_norm_ = new ggml_runtime::LayerNorm(name + ".final_norm", norm_shape);
}

RopeTransformerEncoder::~RopeTransformerEncoder() {
    delete embed_norm_;
    for (auto* layer : layers_) delete layer;
    delete final_norm_;
}

void
RopeTransformerEncoder::define_tensors(ggml_runtime::Session* session) {
    if (embed_norm_ != nullptr)
        embed_norm_->define_tensors(session);
    for (auto* layer : layers_) layer->define_tensors(session);
    final_norm_->define_tensors(session);
}

void
RopeTransformerEncoder::set_data(ggml_runtime::Session* session) {
    if (embed_norm_ != nullptr)
        embed_norm_->set_data(session);
    for (auto* layer : layers_) layer->set_data(session);
    final_norm_->set_data(session);
}

ggml_runtime::TensorBag
RopeTransformerEncoder::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    auto x = input_tensors.get_tensor(0);
    if (embed_norm_ != nullptr) {
        x = embed_norm_->build_graph(session, one(x.tensor, x.buft), tc).get_tensor(0);
    }
    ggml_runtime::TensorBag bag;
    bag.add_tensor(x);
    bag.add_tensor(input_tensors.get_tensor(1));
    for (auto* layer : layers_) bag = layer->build_graph(session, bag, tc);
    auto out = final_norm_->build_graph(session, one(bag.get_tensor(0).tensor, x.buft), tc);
    return out;
}
