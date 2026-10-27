// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "pee_rope_transformer.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace nemo_speech::speechlm;

namespace {

// rope node like ggml_rope_ext but with an F16 result (same op params and sources).
ggml_tensor*
rope_to_f16(ggml_context* ctx, ggml_tensor* r) {
    ggml_tensor* h = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, r->ne[0], r->ne[1], r->ne[2], r->ne[3]);
    h->op = r->op;
    std::memcpy(h->op_params, r->op_params, sizeof(r->op_params));
    for (int i = 0; i < GGML_MAX_SRC; ++i) h->src[i] = r->src[i];
    return h;
}

ggml_runtime::TensorBag
bag1(ggml_tensor* t, ggml_backend_buffer_type_t buft) {
    ggml_runtime::TensorBag b;
    b.add_tensor(ggml_runtime::ggml_bf_tensor(t, buft));
    return b;
}

}  // namespace

// ------------------------------------------------------------------ block --

RopeTransformerBlock::RopeTransformerBlock(
    const std::string& name, const RopeTransformerConfig& cfg)
    : name_(name), cfg_(cfg) {
    const int64_t ln_d[4] = {cfg.d_model, 1, 1, 1};
    const int64_t ln_k[4] = {cfg.d_k(), 1, 1, 1};
    norm1_ = new ggml_runtime::LayerNorm(name_ + ".norm1", ln_d);
    w_qkv_ =
        new ggml_runtime::Linear(name_ + ".attn.w_qkv", cfg.d_model, 3 * cfg.d_model, cfg.qkv_bias);
    out_proj_ = new ggml_runtime::Linear(name_ + ".attn.out_proj", cfg.d_model, cfg.d_model);
    if (cfg.qk_norm) {
        q_norm_ = new ggml_runtime::LayerNorm(name_ + ".attn.q_norm", ln_k);
        k_norm_ = new ggml_runtime::LayerNorm(name_ + ".attn.k_norm", ln_k);
    }
    norm2_ = new ggml_runtime::LayerNorm(name_ + ".norm2", ln_d);
    ff_in_ = new ggml_runtime::Linear(name_ + ".ffn.net.0", cfg.d_model, cfg.ff_hidden);
    ff_out_ = new ggml_runtime::Linear(name_ + ".ffn.net.3", cfg.ff_hidden, cfg.d_model);
}

RopeTransformerBlock::~RopeTransformerBlock() {
    delete norm1_;
    delete w_qkv_;
    delete out_proj_;
    delete q_norm_;
    delete k_norm_;
    delete norm2_;
    delete ff_in_;
    delete ff_out_;
}

void
RopeTransformerBlock::define_tensors(ggml_runtime::Session* session) {
    norm1_->define_tensors(session);
    w_qkv_->define_tensors(session);
    out_proj_->define_tensors(session);
    if (q_norm_)
        q_norm_->define_tensors(session);
    if (k_norm_)
        k_norm_->define_tensors(session);
    norm2_->define_tensors(session);
    ff_in_->define_tensors(session);
    ff_out_->define_tensors(session);
}

void
RopeTransformerBlock::set_data(ggml_runtime::Session* session) {
    norm1_->set_data(session);
    w_qkv_->set_data(session);
    out_proj_->set_data(session);
    if (q_norm_)
        q_norm_->set_data(session);
    if (k_norm_)
        k_norm_->set_data(session);
    norm2_->set_data(session);
    ff_in_->set_data(session);
    ff_out_->set_data(session);
}

ggml_runtime::TensorBag
RopeTransformerBlock::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    auto x = input_tensors.get_tensor(0);
    auto pos = input_tensors.get_tensor(1);
    auto bf_ctx = tc->get_ctx_of_buffer_type(x.buft);
    ggml_context* ctx = bf_ctx.ctx;

    const int n_head = cfg_.n_heads;
    const int d_k = cfg_.d_k();
    const int d_model = cfg_.d_model;
    const int64_t T = x.tensor->ne[1];
    const int64_t B = x.tensor->ne[2];
    const size_t esz = ggml_element_size(x.tensor);

    // ---- attention sub-layer (pre-LN) ----
    auto h = norm1_->build_graph(session, bag1(x.tensor, x.buft), tc).get_tensor(0);
    auto qkv = w_qkv_->build_graph(session, bag1(h.tensor, h.buft), tc).get_tensor(0);
    // qkv: (3*d_model, T, B); NeMo views it as (3, H, D) => q rows first.
    // Strided 4D views (d_k, H, T, B) - no copies: rope, norm and cast all
    // read strided sources and write contiguous outputs.
    auto split = [&](int i) {
        return ggml_view_4d(
            ctx, qkv.tensor, d_k, n_head, T, B, static_cast<size_t>(d_k) * esz, qkv.tensor->nb[1],
            qkv.tensor->nb[2], static_cast<size_t>(i) * d_model * esz);
    };
    ggml_tensor* q = split(0);
    ggml_tensor* k = split(1);
    ggml_tensor* v = split(2);

    if (cfg_.qk_norm) {
        // Per-head LayerNorm over d_k (ne0); weights (d_k) broadcast.
        q = q_norm_->build_graph(session, bag1(q, x.buft), tc).get_tensor(0).tensor;
        k = k_norm_->build_graph(session, bag1(k, x.buft), tc).get_tensor(0).tensor;
    }

    // RoPE: NeoX half-split rotation of the first rotary_dims of each head.
    q = ggml_rope_ext(
        ctx, q, pos.tensor, nullptr, cfg_.rotary_dims, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(
        ctx, k, pos.tensor, nullptr, cfg_.rotary_dims, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 0.0f);
    const bool kv_f16 = cfg_.kv_f16;
    if (kv_f16)
        k = rope_to_f16(ctx, k);

    ggml_tensor* mask =
        input_tensors.tensor_count() >= 3 ? input_tensors.get_tensor(2).tensor : nullptr;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    // Fused attention: q (d_k, T, H, B) view, k/v (d_k, T_kv, H, B) f16,
    // mask f16 (T_kv, T_q, 1, B); result comes back as (d_k, H, T, B).
    auto q_p = ggml_permute(ctx, q, 0, 2, 1, 3);
    auto k16 = kv_f16 ? ggml_permute(ctx, k, 0, 2, 1, 3)
                      : ggml_cast(ctx, ggml_permute(ctx, k, 0, 2, 1, 3), GGML_TYPE_F16);
    auto v16 = kv_f16 ? ggml_permute(ctx, ggml_cast(ctx, v, GGML_TYPE_F16), 0, 2, 1, 3)
                      : ggml_cast(ctx, ggml_permute(ctx, v, 0, 2, 1, 3), GGML_TYPE_F16);
    // mask arrives f16 from the encoder (cast once, not per layer)
    auto attn = ggml_flash_attn_ext(ctx, q_p, k16, v16, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    ggml_tensor* merged = ggml_reshape_3d(ctx, attn, d_model, T, B);
    auto attn_out = out_proj_->build_graph(session, bag1(merged, x.buft), tc).get_tensor(0);
    ggml_tensor* x1 = ggml_add(ctx, x.tensor, attn_out.tensor);

    // ---- feed-forward sub-layer (pre-LN, exact GELU) ----
    auto h2 = norm2_->build_graph(session, bag1(x1, x.buft), tc).get_tensor(0);
    auto ff_mid = ff_in_->build_graph(session, bag1(h2.tensor, h2.buft), tc).get_tensor(0);
    auto ff_act = ggml_gelu_erf(ctx, ff_mid.tensor);
    auto ff_out = ff_out_->build_graph(session, bag1(ff_act, x.buft), tc).get_tensor(0);
    ggml_tensor* x2 = ggml_add(ctx, x1, ff_out.tensor);

    ggml_runtime::TensorBag out = bag1(x2, x.buft);
    for (size_t i = 1; i < input_tensors.tensor_count(); ++i)
        out.add_tensor(input_tensors.get_tensor(i));
    return out;
}

// ---------------------------------------------------------------- encoder --

RopeTransformerEncoder::RopeTransformerEncoder(
    const std::string& name, const RopeTransformerConfig& cfg)
    : name_(name), cfg_(cfg) {
    if (cfg.d_model % cfg.n_heads != 0)
        throw std::runtime_error(name + ": d_model not divisible by n_heads");
    if (cfg.rotary_dims < 2 || cfg.rotary_dims % 2 || cfg.rotary_dims > cfg.d_k())
        throw std::runtime_error(name + ": bad rotary_dims");
    pre_encode_ = new ggml_runtime::Linear(
        name_ + ".pre_encode.proj", cfg.feat_in * cfg.subsampling_factor, cfg.d_model,
        /*use_bias=*/false);
    const int64_t ln_d[4] = {cfg.d_model, 1, 1, 1};
    if (cfg.pre_block_norm)
        embed_norm_ = new ggml_runtime::LayerNorm(name_ + ".embed_norm", ln_d);
    layers_.reserve(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; i++)
        layers_.push_back(new RopeTransformerBlock(name_ + ".layers." + std::to_string(i), cfg));
    final_norm_ = new ggml_runtime::LayerNorm(name_ + ".final_norm", ln_d);
}

RopeTransformerEncoder::~RopeTransformerEncoder() {
    delete pre_encode_;
    delete embed_norm_;
    for (auto* l : layers_) delete l;
    delete final_norm_;
}

void
RopeTransformerEncoder::define_tensors(ggml_runtime::Session* session) {
    pre_encode_->define_tensors(session);
    if (embed_norm_)
        embed_norm_->define_tensors(session);
    for (auto* l : layers_) l->define_tensors(session);
    final_norm_->define_tensors(session);
}

void
RopeTransformerEncoder::set_data(ggml_runtime::Session* session) {
    pre_encode_->set_data(session);
    if (embed_norm_)
        embed_norm_->set_data(session);
    for (auto* l : layers_) l->set_data(session);
    final_norm_->set_data(session);
}

ggml_runtime::ggml_bf_tensor
RopeTransformerEncoder::pre_encode(
    ggml_runtime::Session* session, const ggml_runtime::ggml_bf_tensor& features,
    ggml_runtime::TensorContainer* tc) {
    ggml_context* ctx = tc->get_ctx_of_buffer_type(features.buft).ctx;
    const int sub = cfg_.subsampling_factor;
    const int64_t T_mel = features.tensor->ne[1];
    const int64_t B = features.tensor->ne[2];
    const int64_t T_out = cfg_.out_frames(static_cast<int>(T_mel));
    const int pad = static_cast<int>(T_out * sub - T_mel);

    // FeatureStacking: zero-pad time to a multiple of `sub`, then view `sub`
    // consecutive frames as one (feat_in*sub) vector (frame-major memory makes
    // this a pure reshape, matching torch's reshape(b, t_new, c*sub)).
    ggml_tensor* x = features.tensor;
    if (pad > 0)
        x = ggml_pad(ctx, x, 0, pad, 0, 0);
    x = ggml_reshape_3d(ctx, ggml_cont(ctx, x), cfg_.feat_in * sub, T_out, B);
    return pre_encode_->build_graph(session, bag1(x, features.buft), tc).get_tensor(0);
}

ggml_runtime::TensorBag
RopeTransformerEncoder::encode_rows(
    ggml_runtime::Session* session, const ggml_runtime::ggml_bf_tensor& rows,
    const ggml_runtime::ggml_bf_tensor& pos, const ggml_runtime::ggml_bf_tensor* kmask,
    ggml_runtime::TensorContainer* tc) {
    ggml_context* ctx = tc->get_ctx_of_buffer_type(rows.buft).ctx;
    const int64_t T = rows.tensor->ne[1];
    const int64_t B = rows.tensor->ne[2];
    ggml_tensor* h = rows.tensor;
    if (embed_norm_)
        h = embed_norm_->build_graph(session, bag1(h, rows.buft), tc).get_tensor(0).tensor;

    ggml_runtime::TensorBag bag = bag1(h, rows.buft);
    bag.add_tensor(pos);
    if (kmask) {
        // (T, 1, 1, B) -> (T, T, 1, B): same key mask for every query row,
        // broadcast over heads by soft_max_ext. Expanded here, not uploaded.
        ggml_tensor* tmpl = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T, T, 1, B);
        ggml_tensor* full = ggml_repeat(ctx, kmask->tensor, tmpl);
        bag.add_tensor(
            ggml_runtime::ggml_bf_tensor(ggml_cast(ctx, full, GGML_TYPE_F16), rows.buft));
    }
    for (auto* l : layers_) bag = l->build_graph(session, bag, tc);

    return final_norm_->build_graph(session, bag1(bag.get_tensor(0).tensor, rows.buft), tc);
}

ggml_runtime::TensorBag
RopeTransformerEncoder::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    const auto mel = input_tensors.get_tensor(0);
    const auto pos = input_tensors.get_tensor(1);
    const auto rows = pre_encode(session, mel, tc);
    if (input_tensors.tensor_count() >= 3) {
        const auto km = input_tensors.get_tensor(2);
        return encode_rows(session, rows, pos, &km, tc);
    }
    return encode_rows(session, rows, pos, nullptr, tc);
}
