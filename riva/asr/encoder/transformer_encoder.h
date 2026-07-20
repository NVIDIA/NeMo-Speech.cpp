// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Post-LN Transformer encoder stack, a port of NeMo's
// `nemo.collections.asr.modules.transformer.transformer_encoders.TransformerEncoder`
// as used by Sortformer (18 layers, hidden 192, inner 768, 8 heads, ReLU FF,
// pre_ln=False, no positional embeddings - encoder states are fed directly).
//
// GGUF tensor names mirror the NeMo state dict with the `transformer.` prefix:
//   transformer.layers.N.first_sub_layer.{query,key,value}_net.{weight,bias}
//   transformer.layers.N.first_sub_layer.out_projection.{weight,bias}
//   transformer.layers.N.layer_norm_{1,2}.{weight,bias}
//   transformer.layers.N.second_sub_layer.dense_{in,out}.{weight,bias}
#pragma once

#include <string>
#include <vector>

#include "nn.h"
#include "runtime.h"

namespace nemo_speech::asr {

struct TransformerConfig {
    int n_layers = 18;
    int hidden_size = 192;
    int inner_size = 768;
    int n_heads = 8;
};

// One block: x -> MHA -> +residual -> LN1 -> FF(relu) -> +residual -> LN2.
// (NeMo TransformerEncoderBlock.forward_postln; dropouts are inference no-ops.)
class TransformerBlock : public ggml_runtime::Module {
   public:
    TransformerBlock(const std::string& name, const TransformerConfig& cfg);
    ~TransformerBlock();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

   private:
    std::string name_;
    TransformerConfig cfg_;
    ggml_runtime::Linear* query_net_;
    ggml_runtime::Linear* key_net_;
    ggml_runtime::Linear* value_net_;
    ggml_runtime::Linear* out_projection_;
    ggml_runtime::LayerNorm* layer_norm_1_;
    ggml_runtime::Linear* dense_in_;
    ggml_runtime::Linear* dense_out_;
    ggml_runtime::LayerNorm* layer_norm_2_;
};

// The full stack. Input/output: (hidden_size, T, 1, 1) f32.
class TransformerEncoderModule : public ggml_runtime::Module {
   public:
    TransformerEncoderModule(const std::string& name, const TransformerConfig& cfg);
    ~TransformerEncoderModule();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

   private:
    std::string name_;
    TransformerConfig cfg_;
    std::vector<TransformerBlock*> layers_;
};

}  // namespace nemo_speech::asr
