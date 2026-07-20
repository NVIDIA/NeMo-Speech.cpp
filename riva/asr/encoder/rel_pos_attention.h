// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
// Relative-position positional encoding + multi-head attention. Lives here
// rather than in runtime/ggml/nn because these are FastConformer-specific
// building blocks - the rel-pos formulation, shift trick, and pos_bias_u/v
// tensors mirror NeMo's `RelPositionMultiHeadAttention`.
#pragma once

#include "nn.h"
#include "runtime.h"

namespace ggml_runtime {

class RelPositionalEncoding : public Module {
   public:
    RelPositionalEncoding(const std::string& name, int d_model, int max_len)
        : name(name), d_model(d_model), max_len(max_len) {
        pe_name = this->name + ".pe";
    };
    ~RelPositionalEncoding() = default;

    void define_tensors(Session* session) override;
    TensorBag build_graph(
        Session* session, TensorBag input_tensors,
        TensorContainer* session_tensor_container) override;
    void set_data(Session* session) override;

   private:
    std::string name;
    std::string pe_name;
    int d_model;
    int max_len;

    ggml_bf_tensor get_pe_tensor(Session* session, TensorContainer* session_tensor_container);
};

class RelPositionMultiHeadAttention : public Module {
   public:
    RelPositionMultiHeadAttention(
        const std::string& name, int n_head, int n_feat, bool use_bias = true)
        : name(name), n_head(n_head), n_feat(n_feat), use_bias(use_bias) {
        d_k = n_feat / n_head;
        pos_bias_u_name = this->name + ".pos_bias_u";
        pos_bias_v_name = this->name + ".pos_bias_v";

        // Q/K/V run as ONE fused projection: the three GGUF weights (and
        // biases) are stacked along the output dim into a single
        // [n_feat, 3*n_feat] tensor at load (see set_data), so each chunk pays
        // one mul_mat + one activation-quantize + one bias add instead of
        // three of each. The per-projection tensors are strided views.
        qkv_weight_name = this->name + ".linear_qkv.weight";
        qkv_bias_name = this->name + ".linear_qkv.bias";
        linear_out = new Linear(this->name + ".linear_out", n_feat, n_feat, use_bias);
        // NeMo RelPositionMultiHeadAttention always constructs linear_pos with bias=False.
        linear_pos = new Linear(this->name + ".linear_pos", n_feat, n_feat, false);
    };
    ~RelPositionMultiHeadAttention() {
        delete linear_out;
        delete linear_pos;
    };

    void define_tensors(Session* session) override;
    TensorBag build_graph(
        Session* session, TensorBag input_tensors,
        TensorContainer* session_tensor_container) override;
    // Same as `build_graph`, but inject an additive attention mask between
    // the scaled-dot-product scores and the softmax. Pass `attn_mask=nullptr`
    // to behave identically to `build_graph` - i.e. fully bidirectional
    // attention. Mask must broadcast onto the scores tensor of shape
    // (kv_len, q_len, n_head, batch).
    TensorBag build_graph_masked(
        Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container,
        ggml_tensor* attn_mask);
    void set_data(Session* session) override;

   private:
    std::string name;
    std::string pos_bias_u_name;
    std::string pos_bias_v_name;
    std::string qkv_weight_name;
    std::string qkv_bias_name;
    int n_head;
    int n_feat;
    int d_k;
    bool use_bias;
    Linear* linear_out;
    Linear* linear_pos;
};

}  // namespace ggml_runtime
