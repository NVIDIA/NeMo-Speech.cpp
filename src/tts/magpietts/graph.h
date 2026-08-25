// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "model.h"

namespace nemo_speech::tts {

class DecoderKvCache;
class DecoderCrossKvCache;

ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight);
ggml_tensor* linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, ggml_tensor* b = nullptr);
ggml_tensor* causal_conv1d(
    ggml_context* ctx, ggml_tensor* x, const std::vector<ggml_tensor*>& kernels);
ggml_tensor* self_attention(
    ggml_context* ctx, const magpietts_transformer& tr, const magpietts_layer& layer,
    ggml_tensor* x);
ggml_tensor* self_attention_cached(
    ggml_context* ctx, ggml_cgraph* gf, const magpietts_transformer& tr,
    const magpietts_layer& layer, DecoderKvCache& kv, int layer_index, int n_past, ggml_tensor* x);
ggml_tensor* cross_attention(
    ggml_context* ctx, const magpietts_transformer& tr, const magpietts_layer& layer,
    ggml_tensor* x, ggml_tensor* memory, ggml_tensor* attn_prior = nullptr,
    ggml_tensor** last_attn = nullptr);
ggml_tensor* cross_attention_cached(
    ggml_context* ctx, const magpietts_transformer& tr, const magpietts_layer& layer,
    const DecoderCrossKvCache& cross_kv, int layer_index, ggml_tensor* x,
    ggml_tensor* attn_prior = nullptr, ggml_tensor** last_attn = nullptr,
    bool prior_is_log = false);
ggml_tensor* transformer_forward(
    ggml_context* ctx, const magpietts_transformer& tr, ggml_tensor* x, ggml_tensor* pos,
    ggml_tensor* cond, ggml_tensor* attn_prior = nullptr,
    std::vector<ggml_tensor*>* alignment_outputs = nullptr);
ggml_tensor* transformer_forward_cached(
    ggml_context* ctx, ggml_cgraph* gf, const magpietts_transformer& tr, ggml_tensor* x,
    ggml_tensor* pos, ggml_tensor* cond, DecoderKvCache& kv, DecoderCrossKvCache* cross_kv,
    int n_past, ggml_tensor* attn_prior = nullptr,
    std::vector<ggml_tensor*>* alignment_outputs = nullptr);
ggml_context* new_graph_context();
void tag_graph_first_node(ggml_cgraph* gf);
bool compute_graph(
    const magpietts_model& model, ggml_context* ctx, ggml_cgraph* gf,
    const std::vector<std::pair<std::string, std::vector<int32_t>>>& i32_inputs,
    const std::vector<std::pair<std::string, std::vector<float>>>& f32_inputs, int threads,
    ggml_gallocr_t* keep_allocr = nullptr);
std::vector<int32_t> positions(int n);
std::vector<int32_t> positions_range(int start, int n);

}  // namespace nemo_speech::tts
