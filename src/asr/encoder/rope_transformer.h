// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Pre-LN RoPE Transformer used by high-resolution Sortformer v3.
#pragma once

#include <string>
#include <vector>

#include "nn.h"
#include "runtime.h"

namespace nemo_speech::asr {

struct RopeTransformerConfig {
    int d_model = 512;
    int n_layers = 31;
    int n_heads = 8;
    int d_ff = 2048;
    bool qkv_bias = false;
    bool pre_block_norm = true;
    float rope_base = 10000.0f;
    float rotary_fraction = 1.0f;
    int pos_emb_max_len = 5000;
};

class RopeTransformerBlock : public ggml_runtime::Module {
   public:
    RopeTransformerBlock(const std::string& name, const RopeTransformerConfig& cfg);
    ~RopeTransformerBlock();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

   private:
    RopeTransformerConfig cfg_;
    ggml_runtime::LayerNorm* norm1_;
    ggml_runtime::Linear* qkv_;
    ggml_runtime::Linear* out_proj_;
    ggml_runtime::LayerNorm* norm2_;
    ggml_runtime::Linear* ff_in_;
    ggml_runtime::Linear* ff_out_;
};

// Input/output layout is (d_model, time, batch). The second input is the
// int32 position vector shared by all batch elements.
class RopeTransformerEncoder : public ggml_runtime::Module {
   public:
    RopeTransformerEncoder(const std::string& name, const RopeTransformerConfig& cfg);
    ~RopeTransformerEncoder();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

   private:
    RopeTransformerConfig cfg_;
    ggml_runtime::LayerNorm* embed_norm_;
    std::vector<RopeTransformerBlock*> layers_;
    ggml_runtime::LayerNorm* final_norm_;
};

}  // namespace nemo_speech::asr
