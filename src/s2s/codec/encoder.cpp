// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "encoder.h"

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
// CodecEncodeModule — spec -> codes (encoder + RVQ argmax). Non-streaming.
// ===========================================================================

void
CodecEncodeModule::define_tensors(gr::Session* s) {
    declare_like(s, "enc.proj_in.weight");
    if (gguf_has(s, "enc.proj_in.bias"))
        declare_like(s, "enc.proj_in.bias");
    declare_like(s, "enc.bottleneck.weight");
    if (gguf_has(s, "enc.bottleneck.bias"))
        declare_like(s, "enc.bottleneck.bias");
    const int n_stages = static_cast<int>(cfg_.stage_channels.size());
    for (int st = 0; st < n_stages - 1; st++) {
        declare_like(s, fmt_name("enc.ds%d.weight", st));
        if (gguf_has(s, fmt_name("enc.ds%d.bias", st)))
            declare_like(s, fmt_name("enc.ds%d.bias", st));
    }
    for (int b = 0; b < cfg_.n_blocks(); b++) declare_block(s, fmt_name("enc.blk%d", b));
    for (int i = 0; i < cfg_.n_quantizers; i++) {
        declare_like(s, fmt_name("rvq.cb%d.mus", i));
        // Derived offsets: -0.5*||mus_v||^2, computed in set_data.
        s->model_tensor_container->create_tensor_1d(
            fmt_name("rvq.cb%d.offset", i), GGML_TYPE_F32, cfg_.codebook_size);
    }
}

void
CodecEncodeModule::set_data(gr::Session* s) {
    s->load_weight("enc.proj_in.weight");
    if (gguf_has(s, "enc.proj_in.bias"))
        s->load_weight("enc.proj_in.bias");
    s->load_weight("enc.bottleneck.weight");
    if (gguf_has(s, "enc.bottleneck.bias"))
        s->load_weight("enc.bottleneck.bias");
    const int n_stages = static_cast<int>(cfg_.stage_channels.size());
    for (int st = 0; st < n_stages - 1; st++) {
        s->load_weight(fmt_name("enc.ds%d.weight", st));
        if (gguf_has(s, fmt_name("enc.ds%d.bias", st)))
            s->load_weight(fmt_name("enc.ds%d.bias", st));
    }
    for (int b = 0; b < cfg_.n_blocks(); b++) load_block(s, fmt_name("enc.blk%d", b));

    const int L = cfg_.latent_dim;
    const int V = cfg_.codebook_size;
    for (int i = 0; i < cfg_.n_quantizers; i++) {
        s->load_weight(fmt_name("rvq.cb%d.mus", i));
        // mus on disk: (L, V) f32 — read host-side, compute offsets.
        const std::string mus_name = fmt_name("rvq.cb%d.mus", i);
        const size_t want = static_cast<size_t>(L) * V * sizeof(float);
        const char* data = s->gguf_loader->get_tensor_file_data(mus_name, want);
        const float* mus = reinterpret_cast<const float*>(data);
        std::vector<float> off(V);
        for (int v = 0; v < V; v++) {
            double acc = 0.0;
            const float* row = mus + static_cast<size_t>(v) * L;
            for (int d = 0; d < L; d++) acc += static_cast<double>(row[d]) * row[d];
            off[v] = static_cast<float>(-0.5 * acc);
        }
        auto t = s->model_tensor_container->get_tensor_by_name(fmt_name("rvq.cb%d.offset", i));
        ggml_backend_tensor_set(t.tensor, off.data(), 0, V * sizeof(float));
    }
}

ggml_runtime::TensorBag
CodecEncodeModule::build_graph(gr::Session* s, gr::TensorBag in, gr::TensorContainer* tc) {
    auto spec = in.get_tensor(0);  // (spec_t, spec_channels, B)
    auto ctx0 = tc->get_ctx_of_buffer_type(spec.buft);
    ggml_context* g = ctx0.ctx;

    ggml_tensor* x = ggml_conv_1d(g, W(s, "enc.proj_in.weight"), spec.tensor, 1, 0, 1);
    x = add_channel_bias(g, x, W_opt(s, "enc.proj_in.bias"));

    int block_idx = 0;
    const int n_stages = static_cast<int>(cfg_.stage_channels.size());
    for (int st = 0; st < n_stages; st++) {
        for (int j = 0; j < cfg_.num_blocks_per_stage; j++) {
            BlockWeights bw = fetch_block(s, fmt_name("enc.blk%d", block_idx));
            x = build_convnext_block(g, bw, cfg_.conv_next_kernel, x, nullptr, nullptr);
            block_idx++;
        }
        const int rate = cfg_.rates[st];
        ggml_tensor* w = (st < n_stages - 1) ? W(s, fmt_name("enc.ds%d.weight", st))
                                             : W(s, "enc.bottleneck.weight");
        x = ggml_conv_1d(g, w, x, rate, 0, 1);
    }
    // x: (n_frames, L, B) -> latent (L, BT)
    ggml_tensor* r = ggml_cont(g, ggml_permute(g, x, 1, 0, 2, 3));
    const int64_t n_frames = x->ne[0];
    r = ggml_reshape_2d(g, r, cfg_.latent_dim, n_frames * x->ne[2]);

    gr::TensorBag out;
    ggml_tensor* current = r;
    for (int i = 0; i < cfg_.n_quantizers; i++) {
        ggml_tensor* score = ggml_mul_mat(g, W(s, fmt_name("rvq.cb%d.mus", i)), current);
        ggml_tensor* off2 =
            ggml_reshape_2d(g, W(s, fmt_name("rvq.cb%d.offset", i)), cfg_.codebook_size, 1);
        score = ggml_add(g, score, off2);
        ggml_tensor* idx = ggml_argmax(g, score);
        char nm[32];
        std::snprintf(nm, sizeof(nm), "code_%d", i);
        ggml_set_name(idx, nm);
        ggml_set_output(idx);
        tc->cache_tensor(nm, gr::ggml_bf_tensor(idx, spec.buft));
        out.add_tensor(gr::ggml_bf_tensor(idx, spec.buft));
        ggml_tensor* emb = ggml_get_rows(g, W(s, fmt_name("rvq.cb%d.mus", i)), idx);
        current = ggml_sub(g, current, emb);
    }
    return out;
}

}  // namespace nemo_speech::s2s
