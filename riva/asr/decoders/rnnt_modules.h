// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// RNNT predictor, joint, and optional language-prompt fusion graphs.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nn.h"
#include "rnnt_greedy_decoder.h"  // RnntConfig (the shared RNNT dims)
#include "runtime.h"

namespace nemo_speech::asr {

// Predictor: token (int) -> embedding -> 2-layer LSTM. Returns final hidden.
// State (h, c per layer) is passed in as input tensors, output as separate
// tensors so AsrModel can read them back.
class RnntPredictorModule : public ggml_runtime::Module {
   public:
    RnntPredictorModule(const std::string& name, const RnntConfig& cfg);
    ~RnntPredictorModule() override;

    void define_tensors(ggml_runtime::Session* session) override;
    // Inputs (in this order):
    //   [0] prev_token: i32[1, 1, 1, 1]            (single token id for embedding lookup)
    //   [1] h0_in:      f32[pred_hidden, 1, 1, 1]
    //   [2] c0_in:      f32[pred_hidden, 1, 1, 1]
    //   [3] h1_in:      f32[pred_hidden, 1, 1, 1]
    //   [4] c1_in:      f32[pred_hidden, 1, 1, 1]
    // Outputs:
    //   [0] dec_out: f32[pred_hidden, 1, 1, 1]
    //   [1] h0_out, [2] c0_out, [3] h1_out, [4] c1_out
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

    const RnntConfig& cfg() const { return cfg_; }
    const std::string& embed_name() const { return embed_name_; }

   private:
    std::string name_;
    RnntConfig cfg_;
    std::string embed_name_;  // decoder.prediction.embed.weight
    // Per-layer LSTM weights: weight_ih, weight_hh (bias is rolled into Linear).
    struct Layer {
        ggml_runtime::Linear* lstm_ih;  // input * weight_ih + bias_ih
        ggml_runtime::Linear* lstm_hh;  // h_prev * weight_hh + bias_hh
    };
    std::vector<Layer> layers_;
};

// Joint: g(enc_proj(enc_frame) + pred_proj(dec_out)). NeMo's joint_net is
// Linear(d_model→joint_dim) on enc side, Linear(pred_hidden→joint_dim) on pred
// side, sum + ReLU, then Linear(joint_dim→vocab_size).
class RnntJointModule : public ggml_runtime::Module {
   public:
    RnntJointModule(const std::string& name, const RnntConfig& cfg);
    ~RnntJointModule() override;

    void define_tensors(ggml_runtime::Session* session) override;
    // The encoder projection is owned by the cache-aware encoder Session so
    // raw d_model activations never make a device->host->device round trip.
    // Predictor projection + joint tail stay in the decoder Session.  These
    // split hooks let the two Sessions own only the weights they execute.
    void define_encoder_tensors(ggml_runtime::Session* session);
    void define_decoder_tensors(ggml_runtime::Session* session);
    // Inputs:
    //   [0] enc_frame: f32[d_model, 1, 1, 1]
    //   [1] dec_out  : f32[pred_hidden, 1, 1, 1]
    // Output:
    //   [0] logits   : f32[vocab_size, 1, 1, 1]
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;

    // Optimized staged entry points. Keeping these in one Module lets one
    // Session own one copy of the joint weights while its run cache stores a
    // distinct graph for each stage/shape.
    ggml_runtime::TensorBag build_encoder_projection(
        ggml_runtime::Session* session, ggml_runtime::ggml_bf_tensor enc,
        ggml_runtime::TensorContainer* tc);
    ggml_runtime::TensorBag build_predictor_projection(
        ggml_runtime::Session* session, ggml_runtime::ggml_bf_tensor pred,
        ggml_runtime::TensorContainer* tc);
    ggml_runtime::TensorBag build_joint_tail(
        ggml_runtime::Session* session, ggml_runtime::ggml_bf_tensor enc_proj,
        ggml_runtime::ggml_bf_tensor pred_proj, ggml_runtime::TensorContainer* tc, bool argmax_only,
        const ggml_runtime::ggml_bf_tensor* logit_bias = nullptr);
    void set_data(ggml_runtime::Session* session) override;
    void set_encoder_data(ggml_runtime::Session* session);
    void set_decoder_data(ggml_runtime::Session* session);

   private:
    std::string name_;
    RnntConfig cfg_;
    ggml_runtime::Linear* enc_proj_;
    ggml_runtime::Linear* pred_proj_;
    ggml_runtime::Linear* out_proj_;
};

// Language-ID prompt fusion for prompt-conditioned multilingual RNNT
// (nemotron-3.5 EncDecRNNTBPEModelWithPrompt). Fuses a one-hot language vector
// into the encoder output via the prompt_kernel MLP before the joint. Mirrors
// NeMo's `_apply_prompt_to_encoded`.
class PromptFusionModule : public ggml_runtime::Module {
   public:
    PromptFusionModule(int d_model, int num_prompts);
    ~PromptFusionModule() override;

    void define_tensors(ggml_runtime::Session* session) override;
    // Inputs:  [0] enc_in f32[d_model, T], [1] onehot f32[num_prompts, T].
    // Output:  [0] enc_cond f32[d_model, T].
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

   private:
    std::unique_ptr<ggml_runtime::Linear> lin0_;
    std::unique_ptr<ggml_runtime::Linear> lin2_;
};


}  // namespace nemo_speech::asr
