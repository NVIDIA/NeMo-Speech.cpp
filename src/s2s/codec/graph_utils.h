// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Shared ggml graph-building helpers for the codec decoder/encoder modules:
// GGUF declare/load mirroring, ConvNeXt blocks with channel-major pointwise
// matmuls, causal depthwise conv, and the matmul-based ConvTranspose1d for
// stride == kernel.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.h"
#include "ggml.h"
#include "runtime.h"

namespace nemo_speech::s2s::codec_graph {

namespace gr = ggml_runtime;

inline std::string
fmt_name(const char* pattern, int i) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), pattern, i);
    return std::string(buf);
}

// Declare a tensor on the Session's model container mirroring the GGUF's own
// shape/type. `force_type` overrides the stored dtype (used to hold the
// pointwise conv weights as F16 — load_weight converts F32->F16 on upload).
inline void
declare_like(gr::Session* s, const std::string& name, ggml_type force_type = GGML_TYPE_COUNT) {
    auto ne = s->gguf_loader->get_tensor_ne(name);
    if (ne.empty())
        throw std::runtime_error("codec: missing required tensor " + name);
    const ggml_type t =
        (force_type != GGML_TYPE_COUNT) ? force_type : s->gguf_loader->get_tensor_type(name);
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

inline bool
gguf_has(gr::Session* s, const std::string& name) {
    return s->gguf_loader->has_tensor(name);
}

// Fetch a weight tensor by name from the model container.
inline ggml_tensor*
W(gr::Session* s, const std::string& name) {
    return s->model_tensor_container->get_tensor_by_name(name).tensor;
}

inline ggml_tensor*
W_opt(gr::Session* s, const std::string& name) {
    if (!s->gguf_loader->has_tensor(name))
        return nullptr;
    return W(s, name);
}

// ---------------------------------------------------------------------------
// Shared graph helpers (faithful port of codec_ggml's conv_next.cpp +
// decoder.cpp, including the channel-major pointwise-matmul ConvNeXt and the
// matmul-based stride==kernel ConvTranspose1d).
// ---------------------------------------------------------------------------

inline ggml_tensor*
add_channel_bias(ggml_context* g, ggml_tensor* x, ggml_tensor* bias) {
    if (!bias)
        return x;
    ggml_tensor* b = ggml_reshape_3d(g, bias, 1, bias->ne[0], 1);
    return ggml_add(g, x, b);
}

// LayerNormNd over the channel dim of [T, C, B]; returns CHANNEL-MAJOR
// [C, T, B] (the block keeps that layout through the pointwise matmuls).
inline ggml_tensor*
layer_norm_channel_cm(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float eps) {
    ggml_tensor* xp = ggml_cont(g, ggml_permute(g, x, 1, 0, 2, 3));
    ggml_tensor* xn = ggml_norm(g, xp, eps);
    ggml_tensor* w3 = ggml_reshape_3d(g, w, w->ne[0], 1, 1);
    ggml_tensor* b3 = ggml_reshape_3d(g, b, b->ne[0], 1, 1);
    xn = ggml_mul(g, xn, w3);
    xn = ggml_add(g, xn, b3);
    return xn;
}

// 1x1 conv over channel-major [C_in, T, B] == mul_mat with w viewed (C_in, C_out).
inline ggml_tensor*
pointwise_conv_cm(ggml_context* g, ggml_tensor* w, ggml_tensor* x_cm) {
    ggml_tensor* w2 = ggml_reshape_2d(g, w, w->ne[1], w->ne[2]);
    return ggml_mul_mat(g, w2, x_cm);
}

inline ggml_tensor*
add_channel_bias_cm(ggml_context* g, ggml_tensor* x_cm, ggml_tensor* bias) {
    ggml_tensor* b3 = ggml_reshape_3d(g, bias, bias->ne[0], 1, 1);
    return ggml_add(g, x_cm, b3);
}

inline ggml_tensor*
to_time_major(ggml_context* g, ggml_tensor* x_cm) {
    return ggml_cont(g, ggml_permute(g, x_cm, 1, 0, 2, 3));
}

struct BlockWeights {
    ggml_tensor *dw_w, *dw_b, *norm_w, *norm_b, *pw1_w, *pw1_b, *pw2_w, *pw2_b;
    int channels;
};

inline BlockWeights
fetch_block(gr::Session* s, const std::string& prefix) {
    BlockWeights bw{};
    bw.dw_w = W(s, prefix + ".dw.weight");
    bw.dw_b = W(s, prefix + ".dw.bias");
    bw.norm_w = W(s, prefix + ".norm.weight");
    bw.norm_b = W(s, prefix + ".norm.bias");
    bw.pw1_w = W(s, prefix + ".pw1.weight");
    bw.pw1_b = W(s, prefix + ".pw1.bias");
    bw.pw2_w = W(s, prefix + ".pw2.weight");
    bw.pw2_b = W(s, prefix + ".pw2.bias");
    bw.channels = static_cast<int>(bw.dw_w->ne[2]);
    return bw;
}

// One ConvNeXt1d block. x: [T, C, B]. cache_in (k-1, C, B) or null for
// zero-left-pad. *cache_out (if non-null) receives the materialized tail
// tensor (caller names + marks output).
inline ggml_tensor*
build_convnext_block(
    ggml_context* g, const BlockWeights& blk, int k, ggml_tensor* x, ggml_tensor* cache_in,
    ggml_tensor** cache_out) {
    const int C = blk.channels;
    const int64_t T = x->ne[0];
    const int64_t B = x->ne[2];
    ggml_tensor* residual = x;

    ggml_tensor* x_padded;
    if (cache_in) {
        x_padded = ggml_concat(g, cache_in, x, /*dim=*/0);  // [k-1+T, C, B]
    } else {
        ggml_tensor* zero_full = ggml_scale(g, x, 0.0f);
        ggml_tensor* zero_pad =
            ggml_view_3d(g, zero_full, k - 1, C, B, zero_full->nb[1], zero_full->nb[2], 0);
        zero_pad = ggml_cont(g, zero_pad);
        x_padded = ggml_concat(g, zero_pad, x, /*dim=*/0);
    }

    if (cache_out) {
        const size_t row_bytes = ggml_element_size(x_padded);
        ggml_tensor* tail = ggml_view_3d(
            g, x_padded, k - 1, C, B, x_padded->nb[1], x_padded->nb[2],
            static_cast<size_t>(T) * row_bytes);
        *cache_out = ggml_cont(g, tail);
        ggml_set_output(*cache_out);
    }

    // Depthwise conv as 2-D direct (H=1) — no im2col slab.
    ggml_tensor* x4 = ggml_reshape_4d(g, x_padded, x_padded->ne[0], 1, C, B);
    ggml_tensor* w4 = ggml_reshape_4d(g, blk.dw_w, k, 1, 1, C);
    ggml_tensor* y4 = ggml_conv_2d_dw_direct(g, w4, x4, 1, 1, 0, 0, 1, 1);
    ggml_tensor* xd = ggml_reshape_3d(g, y4, y4->ne[0], C, B);
    xd = add_channel_bias(g, xd, blk.dw_b);

    // norm + pw1 + GELU + pw2, channel-major throughout.
    ggml_tensor* xn_cm = layer_norm_channel_cm(g, xd, blk.norm_w, blk.norm_b, 1e-6f);
    ggml_tensor* xu_cm = pointwise_conv_cm(g, blk.pw1_w, xn_cm);
    xu_cm = add_channel_bias_cm(g, xu_cm, blk.pw1_b);
    xu_cm = ggml_gelu_erf(g, xu_cm);
    ggml_tensor* xp_cm = pointwise_conv_cm(g, blk.pw2_w, xu_cm);
    xp_cm = add_channel_bias_cm(g, xp_cm, blk.pw2_b);
    ggml_tensor* xp = to_time_major(g, xp_cm);

    return ggml_add(g, residual, xp);
}

// ConvTranspose1d with stride == kernel as matmul + reshape + permute.
// W: (K, C_out, C_in); x: (T, C_in, B) -> (T*K, C_out, B).
inline ggml_tensor*
conv_transpose_1d_stride_eq_kernel(ggml_context* g, ggml_tensor* Wt, ggml_tensor* x) {
    const int K = static_cast<int>(Wt->ne[0]);
    const int C_out = static_cast<int>(Wt->ne[1]);
    const int C_in = static_cast<int>(Wt->ne[2]);
    const int T = static_cast<int>(x->ne[0]);
    const int B = static_cast<int>(x->ne[2]);
    ggml_tensor* W_perm = ggml_cont(g, ggml_permute(g, Wt, 1, 2, 0, 3));  // (C_in, K, C_out)
    ggml_tensor* W_flat = ggml_reshape_2d(g, W_perm, C_in, K * C_out);
    ggml_tensor* x_perm = ggml_cont(g, ggml_permute(g, x, 1, 0, 2, 3));  // (C_in, T, B)
    ggml_tensor* y = ggml_mul_mat(g, W_flat, x_perm);                    // (K*C_out, T, B)
    ggml_tensor* y_4d = ggml_reshape_4d(g, y, K, C_out, T, B);
    ggml_tensor* y_perm = ggml_cont(g, ggml_permute(g, y_4d, 0, 2, 1, 3));
    return ggml_reshape_3d(g, y_perm, T * K, C_out, B);
}

inline ggml_tensor*
elementwise_min(ggml_context* g, ggml_tensor* a, ggml_tensor* b) {
    ggml_tensor* sum = ggml_add(g, a, b);
    ggml_tensor* adif = ggml_abs(g, ggml_sub(g, a, b));
    return ggml_scale(g, ggml_sub(g, sum, adif), 0.5f);
}
inline ggml_tensor*
elementwise_max(ggml_context* g, ggml_tensor* a, ggml_tensor* b) {
    ggml_tensor* sum = ggml_add(g, a, b);
    ggml_tensor* adif = ggml_abs(g, ggml_sub(g, a, b));
    return ggml_scale(g, ggml_add(g, sum, adif), 0.5f);
}

inline void
declare_block(gr::Session* s, const std::string& prefix) {
    declare_like(s, prefix + ".dw.weight");
    declare_like(s, prefix + ".dw.bias");
    declare_like(s, prefix + ".norm.weight");
    declare_like(s, prefix + ".norm.bias");
    // Pointwise conv weights held F16 (tensor-core matmul; load converts).
    declare_like(s, prefix + ".pw1.weight", GGML_TYPE_F16);
    declare_like(s, prefix + ".pw1.bias");
    declare_like(s, prefix + ".pw2.weight", GGML_TYPE_F16);
    declare_like(s, prefix + ".pw2.bias");
}

inline void
load_block(gr::Session* s, const std::string& prefix) {
    for (const char* part :
         {".dw.weight", ".dw.bias", ".norm.weight", ".norm.bias", ".pw1.weight", ".pw1.bias",
          ".pw2.weight", ".pw2.bias"})
        s->load_weight(prefix + part);
}

}  // namespace nemo_speech::s2s::codec_graph
