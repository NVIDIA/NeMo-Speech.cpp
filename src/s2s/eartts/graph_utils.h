// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Shared ggml graph-building helpers for the EarTTS embedder and sampler
// modules: GGUF declare/load mirroring, Gemma3-style RMSNorm, the char-aware
// subword text encoder stack, and the MoG-head weight set + MLP stack.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.h"
#include "ggml.h"
#include "runtime.h"

namespace nemo_speech::s2s::eartts_graph {

namespace gr = ggml_runtime;

// Declare a tensor on the Session's model container mirroring the GGUF's own
// shape/type.
inline void
declare_like(gr::Session* s, const std::string& name) {
    auto ne = s->gguf_loader->get_tensor_ne(name);
    if (ne.empty())
        throw std::runtime_error("eartts: missing required tensor " + name);
    const ggml_type t = s->gguf_loader->get_tensor_type(name);
    auto* mc = s->model_tensor_container.get();
    switch (ne.size()) {
        case 1:
            mc->create_tensor_1d(name, t, ne[0]);
            break;
        case 2:
            mc->create_tensor_2d(name, t, ne[0], ne[1]);
            break;
        case 3:
            mc->create_tensor_3d(name, t, ne[0], ne[1], ne[2]);
            break;
        default:
            mc->create_tensor_4d(name, t, ne[0], ne[1], ne[2], ne[3]);
            break;
    }
}

// Fetch a weight tensor by name from the model container.
inline ggml_tensor*
W(gr::Session* s, const std::string& name) {
    return s->model_tensor_container->get_tensor_by_name(name).tensor;
}

inline ggml_tensor*
linear(ggml_context* g, ggml_tensor* w, ggml_tensor* x) {
    return ggml_mul_mat(g, w, x);
}

// Gemma3-style RMSNorm: out = rms_norm(x) * (1 + w), realised as
// x_norm + x_norm * w.
inline ggml_tensor*
gemma3_rms_norm(ggml_context* g, ggml_tensor* x, ggml_tensor* w, float eps) {
    ggml_tensor* xn = ggml_rms_norm(g, x, eps);
    ggml_tensor* xn_w = ggml_mul(g, xn, w);
    return ggml_add(g, xn, xn_w);
}

inline ggml_tensor*
to_f32(ggml_context* g, ggml_tensor* t) {
    if (t->type == GGML_TYPE_F32)
        return t;
    return ggml_cast(g, t, GGML_TYPE_F32);
}

// Char encoder one layer (Gemma3-style pre/post-norm sandwich on attention
// and MLP). x: (hidden, T, B).
inline ggml_tensor*
char_layer(
    ggml_context* g, gr::Session* s, const EarTTSConfig& c, int layer, ggml_tensor* x,
    ggml_tensor* pos, ggml_tensor* attn_mask) {
    const int n_head = c.char_cfg.n_heads;
    const int n_kv = c.char_cfg.n_kv_heads;
    const int d_k = c.char_cfg.head_dim;
    const float eps = c.char_cfg.rms_eps;
    const int T = static_cast<int>(x->ne[1]);
    const int B = static_cast<int>(x->ne[2]);
    const std::string p = "char.encoder.layers." + std::to_string(layer) + ".";

    // ---- attention sandwich ----
    ggml_tensor* xn = gemma3_rms_norm(g, x, W(s, p + "pre_self_attn_layernorm.weight"), eps);

    ggml_tensor* q = linear(g, W(s, p + "self_attn.q_proj.weight"), xn);
    ggml_tensor* k = linear(g, W(s, p + "self_attn.k_proj.weight"), xn);
    ggml_tensor* v = linear(g, W(s, p + "self_attn.v_proj.weight"), xn);

    q = ggml_reshape_4d(g, q, d_k, n_head, T, B);
    k = ggml_reshape_4d(g, k, d_k, n_kv, T, B);
    v = ggml_reshape_4d(g, v, d_k, n_kv, T, B);

    // RoPE, NeoX convention (matches transformers T5Gemma2 rotate_half).
    q = ggml_rope_ext(
        g, q, pos, /*freq_factors=*/nullptr, d_k, GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/T,
        c.char_cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(
        g, k, pos, nullptr, d_k, GGML_ROPE_TYPE_NEOX, T, c.char_cfg.rope_theta, 1.0f, 0.0f, 1.0f,
        0.0f, 0.0f);

    q = ggml_cont(g, ggml_permute(g, q, 0, 2, 1, 3));
    k = ggml_cont(g, ggml_permute(g, k, 0, 2, 1, 3));
    v = ggml_cont(g, ggml_permute(g, v, 0, 2, 1, 3));

    // attn = softmax(softcap((q @ k^T) * scaling) + mask). T5Gemma scaling is
    // query_pre_attn_scalar ** -0.5, NOT 1/sqrt(d_k).
    ggml_tensor* scores = ggml_mul_mat(g, k, q);  // (T_kv, T_q, n_head, B)
    const float scaling = 1.0f / std::sqrt(c.char_cfg.query_pre_attn_scalar);
    scores = ggml_scale(g, scores, scaling);
    const float softcap = c.char_cfg.attn_logit_softcap;
    if (softcap > 0.0f) {
        scores = ggml_scale(g, scores, 1.0f / softcap);
        scores = ggml_tanh(g, scores);
        scores = ggml_scale(g, scores, softcap);
    }
    if (attn_mask)
        scores = ggml_add(g, scores, attn_mask);
    ggml_tensor* attn = ggml_soft_max(g, scores);

    ggml_tensor* v_pt = ggml_cont(g, ggml_permute(g, v, 1, 0, 2, 3));
    ggml_tensor* ctx_attn = ggml_mul_mat(g, v_pt, attn);  // (d_k, T_q, n_head, B)
    ctx_attn = ggml_cont(g, ggml_permute(g, ctx_attn, 0, 2, 1, 3));
    ctx_attn = ggml_reshape_3d(g, ctx_attn, n_head * d_k, T, B);

    ggml_tensor* attn_out = linear(g, W(s, p + "self_attn.o_proj.weight"), ctx_attn);
    attn_out = gemma3_rms_norm(g, attn_out, W(s, p + "post_self_attn_layernorm.weight"), eps);
    x = ggml_add(g, x, attn_out);

    // ---- MLP sandwich ----
    ggml_tensor* xn2 = gemma3_rms_norm(g, x, W(s, p + "pre_feedforward_layernorm.weight"), eps);
    ggml_tensor* gate = linear(g, W(s, p + "mlp.gate_proj.weight"), xn2);
    ggml_tensor* up = linear(g, W(s, p + "mlp.up_proj.weight"), xn2);
    gate = ggml_gelu(g, gate);
    ggml_tensor* mlp_out = linear(g, W(s, p + "mlp.down_proj.weight"), ggml_mul(g, gate, up));
    mlp_out = gemma3_rms_norm(g, mlp_out, W(s, p + "post_feedforward_layernorm.weight"), eps);
    x = ggml_add(g, x, mlp_out);
    return x;
}

// CharAwareSubwordEncoder: subword_ids i32 (BT,) -> text emb (hidden, BT).
// Mirrors char_encoder.py:163-176 / tts_ggml.cpp build_char_encoder.
inline ggml_tensor*
build_char_encoder(
    ggml_context* g, gr::Session* s, const EarTTSConfig& c, ggml_tensor* subword_ids, int BT) {
    const int C = c.char_cfg.max_char_len;
    const int H = c.char_cfg.hidden;

    // char_ids[BT, C] from embed_subwords (float storage of exact integers).
    ggml_tensor* char_ids_f = ggml_get_rows(g, W(s, "char.embed_subwords.weight"), subword_ids);
    ggml_tensor* char_ids_i = ggml_cast(g, char_ids_f, GGML_TYPE_I32);  // (C, BT)
    ggml_tensor* char_mask_f =
        ggml_get_rows(g, W(s, "char.embed_subwords_mask.weight"), subword_ids);  // (C, BT)

    ggml_tensor* char_ids_1d = ggml_reshape_1d(g, char_ids_i, C * BT);
    ggml_tensor* char_emb_1d =
        ggml_get_rows(g, W(s, "char.embed_tokens.weight"), char_ids_1d);  // (H, C*BT)
    ggml_tensor* char_emb = ggml_reshape_3d(g, char_emb_1d, H, C, BT);

    // Gemma embed scaling sqrt(H).
    char_emb = ggml_scale(g, char_emb, std::sqrt(static_cast<float>(H)));

    // Additive attention mask over the key axis: (mask - 1) * 1e30, shape
    // (C, 1, 1, BT), broadcast over T_q and heads.
    ggml_tensor* mask_key = ggml_scale(g, ggml_scale_bias(g, char_mask_f, 1.0f, -1.0f), 1e30f);
    mask_key = ggml_reshape_4d(g, mask_key, C, 1, 1, BT);

    ggml_tensor* pos = ggml_arange(g, 0.0f, static_cast<float>(C), 1.0f);
    pos = ggml_cast(g, pos, GGML_TYPE_I32);

    ggml_tensor* x = char_emb;
    for (int L = 0; L < c.char_cfg.n_layers; L++) {
        x = char_layer(g, s, c, L, x, pos, mask_key);
    }

    x = gemma3_rms_norm(g, x, W(s, "char.encoder.norm.weight"), c.char_cfg.rms_eps);

    // Mask-weighted mean-pool over chars.
    ggml_tensor* mask_bcast = ggml_reshape_3d(g, char_mask_f, 1, C, BT);
    ggml_tensor* x_masked = ggml_mul(g, x, mask_bcast);
    ggml_tensor* x_perm = ggml_cont(g, ggml_permute(g, x_masked, 1, 0, 2, 3));  // (C, H, BT)
    ggml_tensor* x_sum_3d = ggml_sum_rows(g, x_perm);                           // (1, H, BT)
    ggml_tensor* x_sum = ggml_reshape_2d(g, x_sum_3d, H, BT);

    ggml_tensor* mask_cont = ggml_cont(g, char_mask_f);
    ggml_tensor* lens = ggml_sum_rows(g, mask_cont);  // (1, BT)
    lens = ggml_clamp(g, lens, 1.0f, 1e30f);
    ggml_tensor* mean_emb = ggml_div(g, x_sum, lens);

    return linear(g, W(s, "char.proj_embedding.weight"), mean_emb);  // (hidden, BT)
}

// MoG MLP stack.
inline ggml_tensor*
mog_mlp_stack(ggml_context* g, gr::Session* s, const EarTTSConfig& c, ggml_tensor* x) {
    const float eps = c.mog_eps;
    for (int i = 0; i < c.mog_num_layers; i++) {
        const std::string p = "sampler.mog_head.mlp_stack." + std::to_string(i) + ".";
        ggml_tensor* h = gemma3_rms_norm(g, x, W(s, p + "pre_norm.weight"), eps);
        ggml_tensor* g_ = ggml_gelu(g, linear(g, W(s, p + "mlp.gate_proj.weight"), h));
        ggml_tensor* u_ = linear(g, W(s, p + "mlp.up_proj.weight"), h);
        ggml_tensor* m = linear(g, W(s, p + "mlp.down_proj.weight"), ggml_mul(g, g_, u_));
        m = gemma3_rms_norm(g, m, W(s, p + "post_norm.weight"), eps);
        x = ggml_add(g, x, m);
    }
    return gemma3_rms_norm(g, x, W(s, "sampler.mog_head.final_norm.weight"), eps);
}

// CFG combine: x_cond + gs * (x_cond - x_uncond), with gs a per-call (1,)
// input tensor (broadcast multiply) so the graph topology is independent of
// the guidance value. gs == 0 reproduces the cond-only path bit-exactly.
inline ggml_tensor*
cfg_combine(ggml_context* g, ggml_tensor* x_cond, ggml_tensor* x_uncond, ggml_tensor* gs) {
    ggml_tensor* delta = ggml_sub(g, x_cond, x_uncond);
    ggml_tensor* gs_2d = ggml_reshape_2d(g, gs, 1, 1);
    return ggml_add(g, x_cond, ggml_mul(g, delta, gs_2d));
}

// Declare / load the MoG-head weight set shared by the step and the
// single-graph sampler modules (each owns its own Session + container).
inline void
declare_mog_head(gr::Session* s, const EarTTSConfig& c) {
    declare_like(s, "sampler.embed_code.weight");
    declare_like(s, "sampler.mog_head.proj_logits.weight");
    declare_like(s, "sampler.mog_head.proj_mus.weight");
    declare_like(s, "sampler.mog_head.proj_logs.weight");
    declare_like(s, "sampler.mog_head.proj_else.weight");
    declare_like(s, "sampler.mog_head.final_norm.weight");
    for (int i = 0; i < c.mog_num_layers; i++) {
        const std::string p = "sampler.mog_head.mlp_stack." + std::to_string(i) + ".";
        declare_like(s, p + "pre_norm.weight");
        declare_like(s, p + "post_norm.weight");
        declare_like(s, p + "mlp.gate_proj.weight");
        declare_like(s, p + "mlp.up_proj.weight");
        declare_like(s, p + "mlp.down_proj.weight");
    }
}

inline void
load_mog_head(gr::Session* s, const EarTTSConfig& c) {
    s->load_weight("sampler.embed_code.weight");
    s->load_weight("sampler.mog_head.proj_logits.weight");
    s->load_weight("sampler.mog_head.proj_mus.weight");
    s->load_weight("sampler.mog_head.proj_logs.weight");
    s->load_weight("sampler.mog_head.proj_else.weight");
    s->load_weight("sampler.mog_head.final_norm.weight");
    for (int i = 0; i < c.mog_num_layers; i++) {
        const std::string p = "sampler.mog_head.mlp_stack." + std::to_string(i) + ".";
        s->load_weight(p + "pre_norm.weight");
        s->load_weight(p + "post_norm.weight");
        s->load_weight(p + "mlp.gate_proj.weight");
        s->load_weight(p + "mlp.up_proj.weight");
        s->load_weight(p + "mlp.down_proj.weight");
    }
}

}  // namespace nemo_speech::s2s::eartts_graph
