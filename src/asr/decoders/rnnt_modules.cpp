// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "rnnt_modules.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime.h"

namespace nemo_speech::asr {

namespace {

// Per-layer LSTM-cell graph builder. Returns h_out (and via out parameter
// c_out). Both [hidden]. Inputs are flat 1-D tensors.
//
// gates = ih(input) + hh(h_prev)  (Linear adds its bias)
// split [i, f, g, o], apply sigmoid/sigmoid/tanh/sigmoid
// c = f*c_prev + i*g; h = o*tanh(c). The optional second dimension is batch.
//
// CUDA requires contiguous tensors for the gate views, so wrap views in
// ggml_cont. The pre-gate sum lives on the same backend as `input`'s buft.
ggml_tensor*
build_lstm_cell_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorContainer* tc,
    ggml_runtime::Linear* lstm_ih, ggml_runtime::Linear* lstm_hh,
    ggml_runtime::ggml_bf_tensor input, ggml_runtime::ggml_bf_tensor h_prev,
    ggml_runtime::ggml_bf_tensor c_prev, int hidden_size, ggml_tensor** c_out) {
    ggml_runtime::TensorBag in_bag;
    in_bag.add_tensor(input);
    auto ih_out = lstm_ih->build_graph(session, in_bag, tc);
    auto ih = ih_out.get_tensor(0);

    ggml_runtime::TensorBag hh_bag;
    hh_bag.add_tensor(h_prev);
    auto hh_out = lstm_hh->build_graph(session, hh_bag, tc);
    auto hh = hh_out.get_tensor(0);

    auto bf_ctx = tc->get_ctx_of_buffer_type(ih.buft);
    ggml_tensor* gates = ggml_add(bf_ctx.ctx, ih.tensor, hh.tensor);

    const int64_t B = gates->ne[1];
    // Split gates into [i, f, g, o] each [hidden,B].
    const size_t elt = sizeof(float);
    auto gate = [&](int index) {
        return ggml_cont(
            bf_ctx.ctx, ggml_view_2d(
                            bf_ctx.ctx, gates, hidden_size, B, gates->nb[1],
                            static_cast<size_t>(index) * hidden_size * elt));
    };
    ggml_tensor* i_gate = gate(0);
    ggml_tensor* f_gate = gate(1);
    ggml_tensor* g_gate = gate(2);
    ggml_tensor* o_gate = gate(3);

    i_gate = ggml_sigmoid(bf_ctx.ctx, i_gate);
    f_gate = ggml_sigmoid(bf_ctx.ctx, f_gate);
    g_gate = ggml_tanh(bf_ctx.ctx, g_gate);
    o_gate = ggml_sigmoid(bf_ctx.ctx, o_gate);

    // c = f * c_prev + i * g
    ggml_tensor* fc = ggml_mul(bf_ctx.ctx, f_gate, c_prev.tensor);
    ggml_tensor* ig = ggml_mul(bf_ctx.ctx, i_gate, g_gate);
    ggml_tensor* c_new = ggml_add(bf_ctx.ctx, fc, ig);
    // h = o * tanh(c)
    ggml_tensor* h_new = ggml_mul(bf_ctx.ctx, o_gate, ggml_tanh(bf_ctx.ctx, c_new));

    *c_out = c_new;
    return h_new;
}

}  // namespace

RnntPredictorModule::RnntPredictorModule(const std::string& name, const RnntConfig& cfg)
    : name_(name), cfg_(cfg) {
    embed_name_ = "decoder.prediction.embed.weight";
    layers_.resize(cfg_.pred_num_layers);
    // Layer 0 input is the embedding (pred_embed_dim).
    // Layer 1+ input is h_prev of previous layer (pred_hidden).
    for (int l = 0; l < cfg_.pred_num_layers; l++) {
        const int in_dim = (l == 0) ? cfg_.pred_embed_dim : cfg_.pred_hidden;
        const std::string base = "decoder.prediction.dec_rnn.lstm.ih_l" + std::to_string(l);
        const std::string baseh = "decoder.prediction.dec_rnn.lstm.hh_l" + std::to_string(l);
        layers_[l].lstm_ih =
            new ggml_runtime::Linear(base, in_dim, 4 * cfg_.pred_hidden, /*use_bias=*/true);
        layers_[l].lstm_hh = new ggml_runtime::Linear(
            baseh, cfg_.pred_hidden, 4 * cfg_.pred_hidden, /*use_bias=*/true);
    }
}

RnntPredictorModule::~RnntPredictorModule() {
    for (auto& L : layers_) {
        delete L.lstm_ih;
        delete L.lstm_hh;
    }
}


void
RnntPredictorModule::define_tensors(ggml_runtime::Session* session) {
    // Embedding: stored as [embed_dim, vocab_size] (ne[0]=embed_dim, ne[1]=vocab_size).
    // ggml_get_rows picks rows along ne[1]. Honor the stored quantization type
    // (typically F16 - get_rows is dtype-agnostic).
    ggml_type embed_type = session->gguf_loader->get_tensor_type(embed_name_);
    session->model_tensor_container->create_tensor_2d(
        embed_name_, embed_type, cfg_.pred_embed_dim, cfg_.vocab_size);
    for (auto& L : layers_) {
        L.lstm_ih->define_tensors(session);
        L.lstm_hh->define_tensors(session);
    }
}

ggml_runtime::TensorBag
RnntPredictorModule::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    if (input_tensors.tensor_count() < static_cast<size_t>(1 + 2 * cfg_.pred_num_layers)) {
        throw std::runtime_error("RnntPredictorModule: missing inputs");
    }
    auto prev_token = input_tensors.get_tensor(0);  // i32[1]

    // Embedding lookup. ggml_get_rows(embed[embed_dim, vocab_size], idx[1])
    // -> tensor of shape [embed_dim, 1, 1, 1].
    auto embed = session->model_tensor_container->get_tensor_by_name(embed_name_);
    auto bf_ctx = tc->get_ctx_of_buffer_type(embed.buft);
    ggml_tensor* emb = ggml_get_rows(bf_ctx.ctx, embed.tensor, prev_token.tensor);
    const int64_t B = prev_token.tensor->ne[0];
    emb = ggml_cont(bf_ctx.ctx, ggml_reshape_2d(bf_ctx.ctx, emb, cfg_.pred_embed_dim, B));

    ggml_runtime::ggml_bf_tensor cur(emb, bf_ctx.buft);

    ggml_runtime::TensorBag out_bag;
    // Reserve slot [0] for dec_out, filled after the last layer.
    out_bag.add_tensor(ggml_runtime::ggml_bf_tensor(nullptr, nullptr));

    ggml_tensor* last_h = nullptr;
    for (int l = 0; l < cfg_.pred_num_layers; l++) {
        auto h_in = input_tensors.get_tensor(1 + 2 * l);
        auto c_in = input_tensors.get_tensor(1 + 2 * l + 1);
        ggml_tensor* c_new = nullptr;
        ggml_tensor* h_new = build_lstm_cell_graph(
            session, tc, layers_[l].lstm_ih, layers_[l].lstm_hh, cur, h_in, c_in, cfg_.pred_hidden,
            &c_new);
        last_h = h_new;
        out_bag.add_tensor(ggml_runtime::ggml_bf_tensor(h_new, h_in.buft));
        out_bag.add_tensor(ggml_runtime::ggml_bf_tensor(c_new, c_in.buft));
        // Layer l+1 input is h_new (shape [pred_hidden]).
        cur = ggml_runtime::ggml_bf_tensor(h_new, h_in.buft);
    }

    // Slot [0] = dec_out (= final layer's h_out).
    out_bag.set_first_tensor(ggml_runtime::ggml_bf_tensor(last_h, cur.buft));
    return out_bag;
}

void
RnntPredictorModule::set_data(ggml_runtime::Session* session) {
    session->load_weight(embed_name_);
    for (auto& L : layers_) {
        L.lstm_ih->set_data(session);
        L.lstm_hh->set_data(session);
    }
}

RnntJointModule::RnntJointModule(const std::string& name, const RnntConfig& cfg)
    : name_(name), cfg_(cfg) {
    enc_proj_ = new ggml_runtime::Linear("joint.enc", cfg_.d_model, cfg_.joint_dim, true);
    pred_proj_ = new ggml_runtime::Linear("joint.pred", cfg_.pred_hidden, cfg_.joint_dim, true);
    out_proj_ = new ggml_runtime::Linear(
        "joint.joint_net.2", cfg_.joint_dim, cfg_.joint_output_size(), true);
}

RnntJointModule::~RnntJointModule() {
    delete enc_proj_;
    delete pred_proj_;
    delete out_proj_;
}


void
RnntJointModule::define_tensors(ggml_runtime::Session* session) {
    define_encoder_tensors(session);
    define_decoder_tensors(session);
}

void
RnntJointModule::define_encoder_tensors(ggml_runtime::Session* session) {
    enc_proj_->define_tensors(session);
}

void
RnntJointModule::define_decoder_tensors(ggml_runtime::Session* session) {
    pred_proj_->define_tensors(session);
    out_proj_->define_tensors(session);
}

ggml_runtime::TensorBag
RnntJointModule::build_encoder_projection(
    ggml_runtime::Session* session, ggml_runtime::ggml_bf_tensor enc,
    ggml_runtime::TensorContainer* tc) {
    ggml_runtime::TensorBag enc_bag;
    enc_bag.add_tensor(enc);
    return enc_proj_->build_graph(session, enc_bag, tc);
}

ggml_runtime::TensorBag
RnntJointModule::build_predictor_projection(
    ggml_runtime::Session* session, ggml_runtime::ggml_bf_tensor pred,
    ggml_runtime::TensorContainer* tc) {
    ggml_runtime::TensorBag dec_bag;
    dec_bag.add_tensor(pred);
    return pred_proj_->build_graph(session, dec_bag, tc);
}

ggml_runtime::TensorBag
RnntJointModule::build_joint_tail(
    ggml_runtime::Session* session, ggml_runtime::ggml_bf_tensor enc_proj,
    ggml_runtime::ggml_bf_tensor pred_proj, ggml_runtime::TensorContainer* tc, bool argmax_only,
    const ggml_runtime::ggml_bf_tensor* logit_bias) {
    auto bf_ctx = tc->get_ctx_of_buffer_type(enc_proj.buft);
    // pred_proj is [joint_dim,1,B]; ggml_add broadcasts it over the T columns
    // of enc_proj [joint_dim,T,B]. This makes each stream's blank-frame run one
    // joint GEMM while also batching compatible streams.
    const int64_t B =
        pred_proj.tensor->ne[2] > 1 ? pred_proj.tensor->ne[2] : pred_proj.tensor->ne[1];
    const int64_t T = B > 1 ? enc_proj.tensor->ne[1] / B : enc_proj.tensor->ne[1];
    ggml_tensor* act = nullptr;
    if (B == 1) {
        act = ggml_relu(bf_ctx.ctx, ggml_add(bf_ctx.ctx, enc_proj.tensor, pred_proj.tensor));
    } else {
        auto enc_3d = ggml_reshape_3d(bf_ctx.ctx, enc_proj.tensor, enc_proj.tensor->ne[0], T, B);
        auto pred_3d = ggml_reshape_3d(bf_ctx.ctx, pred_proj.tensor, pred_proj.tensor->ne[0], 1, B);
        auto pred_repeated = ggml_repeat(bf_ctx.ctx, pred_3d, enc_3d);
        auto act_3d = ggml_relu(bf_ctx.ctx, ggml_add(bf_ctx.ctx, enc_3d, pred_repeated));
        act = ggml_reshape_2d(bf_ctx.ctx, act_3d, enc_proj.tensor->ne[0], T * B);
    }

    ggml_runtime::TensorBag act_bag;
    act_bag.add_tensor(ggml_runtime::ggml_bf_tensor(act, bf_ctx.buft));
    auto logits = out_proj_->build_graph(session, act_bag, tc);
    if (B > 1) {
        auto l = logits.get_tensor(0);
        logits.set_first_tensor(
            ggml_runtime::ggml_bf_tensor(
                ggml_reshape_3d(bf_ctx.ctx, l.tensor, l.tensor->ne[0], T, B), l.buft));
    }
    if (argmax_only) {
        auto l = logits.get_tensor(0);
        const int64_t V = l.tensor->ne[0];
        const int64_t T = l.tensor->ne[1];
        const int64_t B = l.tensor->ne[2];
        ggml_runtime::TensorBag out;
        if (logit_bias != nullptr) {
            if (cfg_.is_tdt())
                throw std::runtime_error("TDT joint logit bias is not supported");
            // Bias is [V,1,B] and broadcasts over the T axis of [V,T,B].
            l.tensor = ggml_add(bf_ctx.ctx, l.tensor, logit_bias->tensor);
        }

        if (!cfg_.is_tdt()) {
            // GGML ARGMAX reduces rows of a matrix. Flatten T and B together so
            // one compact [T*B] result covers the entire batch and requires one
            // device-to-host transfer rather than one transfer per stream.
            auto flat_logits = ggml_reshape_2d(bf_ctx.ctx, l.tensor, V, T * B);
            auto flat_ids = ggml_argmax(bf_ctx.ctx, flat_logits);
            out.add_tensor(ggml_runtime::ggml_bf_tensor(flat_ids, bf_ctx.buft));
            return out;
        }

        // TDT has two independently decoded slices in the joint vector. Pack
        // each slice across the batch, reducing 2*B result transfers to two.
        const int64_t token_count = cfg_.vocab_size;
        const int64_t duration_count = static_cast<int64_t>(cfg_.durations.size());
        auto token_logits = ggml_view_3d(
            bf_ctx.ctx, l.tensor, token_count, T, B, l.tensor->nb[1], l.tensor->nb[2], 0);
        auto duration_logits = ggml_view_3d(
            bf_ctx.ctx, l.tensor, duration_count, T, B, l.tensor->nb[1], l.tensor->nb[2],
            static_cast<size_t>(token_count) * l.tensor->nb[0]);
        token_logits =
            ggml_reshape_2d(bf_ctx.ctx, ggml_cont(bf_ctx.ctx, token_logits), token_count, T * B);
        duration_logits = ggml_reshape_2d(
            bf_ctx.ctx, ggml_cont(bf_ctx.ctx, duration_logits), duration_count, T * B);
        out.add_tensor(
            ggml_runtime::ggml_bf_tensor(ggml_argmax(bf_ctx.ctx, token_logits), bf_ctx.buft));
        out.add_tensor(
            ggml_runtime::ggml_bf_tensor(ggml_argmax(bf_ctx.ctx, duration_logits), bf_ctx.buft));
        return out;
    }
    return logits;
}

ggml_runtime::TensorBag
RnntJointModule::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    auto enc_p = build_encoder_projection(session, input_tensors.get_tensor(0), tc).get_tensor(0);
    auto dec_p = build_predictor_projection(session, input_tensors.get_tensor(1), tc).get_tensor(0);
    return build_joint_tail(session, enc_p, dec_p, tc, /*argmax_only=*/false);
}

void
RnntJointModule::set_data(ggml_runtime::Session* session) {
    set_encoder_data(session);
    set_decoder_data(session);
}

void
RnntJointModule::set_encoder_data(ggml_runtime::Session* session) {
    enc_proj_->set_data(session);
}

void
RnntJointModule::set_decoder_data(ggml_runtime::Session* session) {
    pred_proj_->set_data(session);
    out_proj_->set_data(session);
}

PromptFusionModule::PromptFusionModule(int d_model, int num_prompts) {
    const int proj_in = d_model + num_prompts;
    const int proj_hidden = 2 * d_model;
    lin0_ = std::make_unique<ggml_runtime::Linear>(
        "prompt_kernel.0", proj_in, proj_hidden, /*use_bias=*/true);
    lin2_ = std::make_unique<ggml_runtime::Linear>(
        "prompt_kernel.2", proj_hidden, d_model, /*use_bias=*/true);
}

PromptFusionModule::~PromptFusionModule() = default;

void
PromptFusionModule::define_tensors(ggml_runtime::Session* session) {
    lin0_->define_tensors(session);
    lin2_->define_tensors(session);
}

ggml_runtime::TensorBag
PromptFusionModule::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    auto enc = input_tensors.get_tensor(0);
    auto onehot = input_tensors.get_tensor(1);
    auto bf = tc->get_ctx_of_buffer_type(enc.buft);
    // cat([encoded, prompt], dim=feature) -> [d_model + num_prompts, T].
    ggml_tensor* cat = ggml_concat(bf.ctx, enc.tensor, onehot.tensor, 0);

    ggml_runtime::TensorBag b0;
    b0.add_tensor(ggml_runtime::ggml_bf_tensor(cat, bf.buft));
    auto h = lin0_->build_graph(session, b0, tc).get_tensor(0);

    ggml_tensor* act = ggml_relu(bf.ctx, h.tensor);
    ggml_runtime::TensorBag b1;
    b1.add_tensor(ggml_runtime::ggml_bf_tensor(act, h.buft));
    return lin2_->build_graph(session, b1, tc);
}

void
PromptFusionModule::set_data(ggml_runtime::Session* session) {
    lin0_->set_data(session);
    lin2_->set_data(session);
}


}  // namespace nemo_speech::asr
