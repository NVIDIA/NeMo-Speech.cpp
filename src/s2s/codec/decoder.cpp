// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "decoder.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml.h"
#include "graph_utils.h"

namespace gr = ggml_runtime;

namespace nemo_speech::s2s {

using namespace codec_graph;

// ===========================================================================
// CodecDecodeWavModule
// ===========================================================================

int
CodecDecodeWavModule::tail_len() const {
    const int half_spec_padding =
        ((cfg_.n_fft - cfg_.hop_length) / 2 + cfg_.hop_length - 1) / cfg_.hop_length;
    return 2 * half_spec_padding;
}

void
CodecDecodeWavModule::define_tensors(gr::Session* s) {
    // RVQ codebooks (decode needs mus only).
    for (int i = 0; i < cfg_.n_quantizers; i++) declare_like(s, fmt_name("rvq.cb%d.mus", i));

    // Decoder weights.
    declare_like(s, "dec.bottleneck.weight");
    if (gguf_has(s, "dec.bottleneck.bias"))
        declare_like(s, "dec.bottleneck.bias");
    declare_like(s, "dec.proj_out.weight");
    if (gguf_has(s, "dec.proj_out.bias"))
        declare_like(s, "dec.proj_out.bias");
    const int n_stages = static_cast<int>(cfg_.stage_channels.size());
    for (int st = 0; st < n_stages - 1; st++) {
        declare_like(s, fmt_name("dec.us%d.weight", st));
        if (gguf_has(s, fmt_name("dec.us%d.bias", st)))
            declare_like(s, fmt_name("dec.us%d.bias", st));
    }
    for (int b = 0; b < cfg_.n_blocks(); b++) declare_block(s, fmt_name("dec.blk%d", b));

    // Computed post-processing weights.
    const int n_fft = cfg_.n_fft;
    const int n_bins = cfg_.n_bins();
    auto* mc = s->model_tensor_container.get();
    mc->create_tensor_2d("post.idft_re", GGML_TYPE_F32, n_bins, n_fft);
    mc->create_tensor_2d("post.idft_im", GGML_TYPE_F32, n_bins, n_fft);
    mc->create_tensor_1d("post.window", GGML_TYPE_F32, n_fft);
    mc->create_tensor_1d("post.imag_mask", GGML_TYPE_F32, n_bins);
    mc->create_tensor_3d("post.id_overlap_kernel", GGML_TYPE_F32, n_fft, 1, n_fft);
}

void
CodecDecodeWavModule::set_data(gr::Session* s) {
    for (int i = 0; i < cfg_.n_quantizers; i++) s->load_weight(fmt_name("rvq.cb%d.mus", i));
    s->load_weight("dec.bottleneck.weight");
    if (gguf_has(s, "dec.bottleneck.bias"))
        s->load_weight("dec.bottleneck.bias");
    s->load_weight("dec.proj_out.weight");
    if (gguf_has(s, "dec.proj_out.bias"))
        s->load_weight("dec.proj_out.bias");
    const int n_stages = static_cast<int>(cfg_.stage_channels.size());
    for (int st = 0; st < n_stages - 1; st++) {
        s->load_weight(fmt_name("dec.us%d.weight", st));
        if (gguf_has(s, fmt_name("dec.us%d.bias", st)))
            s->load_weight(fmt_name("dec.us%d.bias", st));
    }
    for (int b = 0; b < cfg_.n_blocks(); b++) load_block(s, fmt_name("dec.blk%d", b));

    // Computed post weights (codec_ggml's init_post_weights, verbatim math).
    const int n_fft = cfg_.n_fft;
    const int n_bins = cfg_.n_bins();
    const double TWO_PI = 6.283185307179586476925286766559;
    idft_re_.assign(static_cast<size_t>(n_bins) * n_fft, 0.0f);
    idft_im_.assign(static_cast<size_t>(n_bins) * n_fft, 0.0f);
    for (int n = 0; n < n_fft; ++n) {
        for (int k = 0; k < n_bins; ++k) {
            const double c_k = (k == 0 || k == n_bins - 1) ? (1.0 / n_fft) : (2.0 / n_fft);
            const double ang = TWO_PI * k * n / n_fft;
            idft_re_[static_cast<size_t>(k) + static_cast<size_t>(n) * n_bins] =
                static_cast<float>(c_k * std::cos(ang));
            idft_im_[static_cast<size_t>(k) + static_cast<size_t>(n) * n_bins] =
                static_cast<float>(c_k * std::sin(ang));
        }
    }
    window_.assign(n_fft, 0.0f);
    for (int i = 0; i < n_fft; ++i)
        window_[i] = static_cast<float>(0.5 * (1.0 - std::cos(TWO_PI * i / n_fft)));
    imag_mask_.assign(n_bins, 1.0f);
    imag_mask_[0] = 0.0f;
    imag_mask_[n_bins - 1] = 0.0f;
    id_kernel_.assign(static_cast<size_t>(n_fft) * n_fft, 0.0f);
    for (int c = 0; c < n_fft; ++c)
        id_kernel_[static_cast<size_t>(c) + static_cast<size_t>(c) * n_fft] = 1.0f;
    // Note id_kernel layout: ne (n_fft, 1, n_fft); W[k,0,c]=1 iff k==c →
    // offset k + c*n_fft. (c + c*n_fft above IS k==c.)

    auto upload = [&](const char* name, const std::vector<float>& host) {
        auto t = s->model_tensor_container->get_tensor_by_name(name);
        ggml_backend_tensor_set(t.tensor, host.data(), 0, host.size() * sizeof(float));
    };
    upload("post.idft_re", idft_re_);
    upload("post.idft_im", idft_im_);
    upload("post.window", window_);
    upload("post.imag_mask", imag_mask_);
    upload("post.id_overlap_kernel", id_kernel_);
}

ggml_runtime::TensorBag
CodecDecodeWavModule::build_graph(gr::Session* s, gr::TensorBag in, gr::TensorContainer* tc) {
    const int Q = cfg_.n_quantizers;
    const int L = cfg_.latent_dim;
    const int n_blocks = cfg_.n_blocks();
    const int n_stages = static_cast<int>(cfg_.stage_channels.size());
    const int k = cfg_.conv_next_kernel;
    const int n_fft = cfg_.n_fft;
    const int n_bins = cfg_.n_bins();
    const int hop = cfg_.hop_length;
    const int tl = tail_len();

    // ---- inputs (order fixed by the facade) ----
    auto codes = in.get_tensor(0);  // i32 (BT, Q) — per-q rows contiguous
    auto ctx0 = tc->get_ctx_of_buffer_type(codes.buft);
    ggml_context* g = ctx0.ctx;
    const int BT = static_cast<int>(codes.tensor->ne[0]);
    const int batch_size = 1;  // S2S decodes one stream per call
    const int n_frames = BT;

    std::vector<ggml_tensor*> cache_in(n_blocks);
    for (int b = 0; b < n_blocks; b++) cache_in[b] = in.get_tensor(1 + b).tensor;
    ggml_tensor* tail_re_in = in.get_tensor(1 + n_blocks).tensor;
    ggml_tensor* tail_im_in = in.get_tensor(2 + n_blocks).tensor;
    ggml_tensor* env_in = in.get_tensor(3 + n_blocks).tensor;

    // ---- RVQ depth-sum decode: codes -> latent (L, BT) ----
    ggml_tensor* z = nullptr;
    for (int i = 0; i < Q; i++) {
        ggml_tensor* idx =
            ggml_view_1d(g, codes.tensor, BT, static_cast<size_t>(i) * BT * sizeof(int32_t));
        ggml_tensor* mus = W(s, fmt_name("rvq.cb%d.mus", i));
        ggml_tensor* e = ggml_get_rows(g, mus, idx);  // (L, BT)
        z = z ? ggml_add(g, z, e) : e;
    }

    // ---- decoder ----
    ggml_tensor* lat3 = ggml_reshape_3d(g, z, L, n_frames, batch_size);
    ggml_tensor* x = ggml_cont(g, ggml_permute(g, lat3, 1, 0, 2, 3));  // [T, L, B]

    const int rate_btm = cfg_.rates.back();
    ggml_tensor* btm_w = W(s, "dec.bottleneck.weight");
    if (btm_w->ne[0] == rate_btm) {
        x = conv_transpose_1d_stride_eq_kernel(g, btm_w, x);
    } else {
        x = ggml_conv_transpose_1d(g, btm_w, x, rate_btm, 0, 1);
    }
    x = add_channel_bias(g, x, W_opt(s, "dec.bottleneck.bias"));

    std::vector<ggml_tensor*> cache_out(n_blocks, nullptr);
    int block_idx = 0;
    for (int st = 0; st < n_stages; st++) {
        for (int j = 0; j < cfg_.num_blocks_per_stage; j++) {
            BlockWeights bw = fetch_block(s, fmt_name("dec.blk%d", block_idx));
            x = build_convnext_block(g, bw, k, x, cache_in[block_idx], &cache_out[block_idx]);
            char nm[32];
            std::snprintf(nm, sizeof(nm), "cache_out_%d", block_idx);
            ggml_set_name(cache_out[block_idx], nm);
            tc->cache_tensor(nm, gr::ggml_bf_tensor(cache_out[block_idx], codes.buft));
            block_idx++;
        }
        if (st < n_stages - 1) {
            const int rate_us = cfg_.rates[n_stages - 2 - st];
            ggml_tensor* us_w = W(s, fmt_name("dec.us%d.weight", st));
            if (us_w->ne[0] == rate_us) {
                x = conv_transpose_1d_stride_eq_kernel(g, us_w, x);
            } else {
                x = ggml_conv_transpose_1d(g, us_w, x, rate_us, 0, 1);
            }
            x = add_channel_bias(g, x, W_opt(s, fmt_name("dec.us%d.bias", st)));
        }
    }

    x = ggml_conv_1d(g, W(s, "dec.proj_out.weight"), x, 1, 0, 1);
    x = add_channel_bias(g, x, W_opt(s, "dec.proj_out.bias"));
    ggml_tensor* out_spec = x;  // (out_t, spec_channels, B)

    int out_t = n_frames;
    for (int r : cfg_.rates) out_t *= r;

    // ---- post chain: mag/phase -> IDFT -> overlap-add -> envelope ----
    const float max_mag = 100.0f;
    const float log_max_mag = std::log(max_mag);
    const int pad = (n_fft - hop) / 2;
    const int eff_t = tl + out_t;
    const int half_wav_padding = (tl / 2) * hop;
    const int out_size_full = (eff_t - 1) * hop + n_fft;
    const int out_size_trim = out_size_full - 2 * pad - 2 * half_wav_padding;

    const size_t nb_ch = out_spec->nb[1];
    const size_t nb_b = out_spec->nb[2];
    ggml_tensor* mag_logit = ggml_view_3d(g, out_spec, out_t, n_bins, batch_size, nb_ch, nb_b, 0);
    ggml_tensor* phase = ggml_view_3d(
        g, out_spec, out_t, n_bins, batch_size, nb_ch, nb_b, static_cast<size_t>(n_bins) * nb_ch);

    ggml_tensor* mag_logit_c = ggml_cont(g, mag_logit);
    ggml_tensor* mag = ggml_sigmoid(g, ggml_scale_bias(g, mag_logit_c, 1.0f, -log_max_mag));
    mag = ggml_scale(g, mag, max_mag);

    ggml_tensor* phase_c = ggml_cont(g, phase);
    ggml_tensor* spec_re = ggml_mul(g, mag, ggml_cos(g, phase_c));
    ggml_tensor* spec_im_full = ggml_mul(g, mag, ggml_sin(g, phase_c));
    ggml_tensor* mask_3d = ggml_reshape_3d(g, W(s, "post.imag_mask"), 1, n_bins, 1);
    ggml_tensor* spec_im = ggml_mul(g, spec_im_full, mask_3d);

    ggml_tensor* spec_re_p = ggml_cont(g, ggml_permute(g, spec_re, 1, 0, 2, 3));
    ggml_tensor* spec_im_p = ggml_cont(g, ggml_permute(g, spec_im, 1, 0, 2, 3));

    // Stash this chunk's last tl complex frames for the next call.
    ggml_tensor* tail_re_new = ggml_cont(
        g, ggml_view_3d(
               g, spec_re_p, n_bins, tl, batch_size, spec_re_p->nb[1], spec_re_p->nb[2],
               static_cast<size_t>(out_t - tl) * spec_re_p->nb[1]));
    ggml_tensor* tail_im_new = ggml_cont(
        g, ggml_view_3d(
               g, spec_im_p, n_bins, tl, batch_size, spec_im_p->nb[1], spec_im_p->nb[2],
               static_cast<size_t>(out_t - tl) * spec_im_p->nb[1]));
    ggml_set_name(tail_re_new, "spec_re_tail_out");
    ggml_set_name(tail_im_new, "spec_im_tail_out");
    ggml_set_output(tail_re_new);
    ggml_set_output(tail_im_new);
    tc->cache_tensor("spec_re_tail_out", gr::ggml_bf_tensor(tail_re_new, codes.buft));
    tc->cache_tensor("spec_im_tail_out", gr::ggml_bf_tensor(tail_im_new, codes.buft));

    // Prepend previous tail along time.
    spec_re_p = ggml_concat(g, tail_re_in, spec_re_p, /*dim=*/1);
    spec_im_p = ggml_concat(g, tail_im_in, spec_im_p, /*dim=*/1);

    ggml_tensor* y_re = ggml_mul_mat(g, W(s, "post.idft_re"), spec_re_p);
    ggml_tensor* y_im = ggml_mul_mat(g, W(s, "post.idft_im"), spec_im_p);
    ggml_tensor* y_time = ggml_sub(g, y_re, y_im);  // (n_fft, eff_t, B)

    ggml_tensor* window_3d = ggml_reshape_3d(g, W(s, "post.window"), n_fft, 1, 1);
    ggml_tensor* neg_window_3d = ggml_neg(g, window_3d);
    ggml_tensor* y_clamped =
        elementwise_max(g, elementwise_min(g, y_time, window_3d), neg_window_3d);
    ggml_tensor* y_win = ggml_mul(g, y_clamped, window_3d);

    ggml_tensor* y_perm = ggml_cont(g, ggml_permute(g, y_win, 1, 0, 2, 3));
    ggml_tensor* y2 = ggml_view_2d(g, y_perm, eff_t, n_fft, y_perm->nb[1], 0);
    ggml_tensor* w1 = ggml_conv_transpose_1d(
        g, W(s, "post.id_overlap_kernel"), y2, /*s0=*/hop, /*p0=*/0, /*d0=*/1);
    ggml_tensor* wav_full = ggml_reshape_3d(g, w1, out_size_full, 1, 1);

    const int trim_each_side = pad + half_wav_padding;
    ggml_tensor* wav_trim = ggml_cont(
        g, ggml_view_3d(
               g, wav_full, out_size_trim, 1, batch_size, wav_full->nb[1], wav_full->nb[2],
               static_cast<size_t>(trim_each_side) * wav_full->nb[0]));

    ggml_tensor* env_3d = ggml_reshape_3d(g, env_in, out_size_trim, 1, 1);
    ggml_tensor* wav = ggml_mul(g, wav_trim, env_3d);
    ggml_set_name(wav, "wav_out");
    ggml_set_output(wav);

    gr::TensorBag out;
    out.add_tensor(gr::ggml_bf_tensor(wav, codes.buft));
    // cache_out_* and spec_*_tail_out are fetched by name by the facade.
    for (int b = 0; b < n_blocks; b++) out.add_tensor(gr::ggml_bf_tensor(cache_out[b], codes.buft));
    out.add_tensor(gr::ggml_bf_tensor(tail_re_new, codes.buft));
    out.add_tensor(gr::ggml_bf_tensor(tail_im_new, codes.buft));
    return out;
}

}  // namespace nemo_speech::s2s
