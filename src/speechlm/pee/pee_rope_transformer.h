// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Pre-LN Transformer encoder with rotary position embedding, a port of NeMo's
// `nemo.collections.asr.modules.transformer_encoder.TransformerEncoder` in its
// `self_attention_model: rope` / `subsampling: feature_stacking` configuration
// (the ASR and speaker experts of the Parallel Expert Encoder).
//
//   mel (feat_in, T_mel) -> FeatureStacking: pad T to a multiple of `sub`,
//   reshape to (feat_in*sub, T/sub), Linear(no bias) -> LayerNorm (embed_norm)
//   -> N x block -> LayerNorm (final_norm)                     => (d_model, T')
//
//   block: x + out_proj(SDPA(rope(qk_norm(q)), rope(qk_norm(k)), v))  [pre-LN]
//          x + W3(GELU(W0(norm2(x))))
//
// RoPE is the half-split ("NeoX") rotation on the first `rotary_dims` of each
// head, positions 0..T'-1, base 10000. qk_norm is a per-head LayerNorm over
// d_k applied before the rotation. Attention is full (no mask) for B == 1.
//
// GGUF tensor names mirror the NeMo state dict under a caller-chosen prefix:
//   <p>.pre_encode.proj.weight            (d_model, feat_in*sub)
//   <p>.embed_norm.{weight,bias}          <p>.final_norm.{weight,bias}
//   <p>.layers.N.norm1.{weight,bias}      <p>.layers.N.norm2.{weight,bias}
//   <p>.layers.N.attn.w_qkv.weight        (3*d_model, d_model) [+ .bias]
//   <p>.layers.N.attn.out_proj.{weight,bias}
//   <p>.layers.N.attn.{q_norm,k_norm}.{weight,bias}   (d_k)   [qk_norm only]
//   <p>.layers.N.ffn.net.0.{weight,bias}  <p>.layers.N.ffn.net.3.{weight,bias}
#pragma once

#include <string>
#include <vector>

#include "nn.h"
#include "runtime.h"

namespace nemo_speech::speechlm {

struct RopeTransformerConfig {
    int feat_in = 128;
    int d_model = 512;
    int n_heads = 8;
    int n_layers = 17;
    int ff_hidden = 2048;
    int subsampling_factor = 8;
    int rotary_dims = 64;  // = int(d_k * rotary_fraction), even
    bool qk_norm = false;
    bool qkv_bias = false;
    bool pre_block_norm = true;
    // The K RoPE writes F16 and K/V reach flash attention as permuted views of head-major F16
    // tensors, so the per-layer K cast and the permuting K/V copies go away (same values: flash
    // attention reads K as F16 either way).
    bool kv_f16 = false;

    int d_k() const { return d_model / n_heads; }
    // FeatureStacking output frames for `t_mel` input frames (ceil-div).
    int out_frames(int t_mel) const {
        return (t_mel + subsampling_factor - 1) / subsampling_factor;
    }
};

// Input bag: [0] x (d_model, T, B) f32, [1] positions (T) i32,
// [2] optional additive key mask (T_kv, T_q, 1, B) f32 (0 valid / -1e9 pad).
// Output bag: [0] x (d_model, T, B), then the pass-through extras.
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
    std::string name_;
    RopeTransformerConfig cfg_;
    ggml_runtime::LayerNorm* norm1_;
    ggml_runtime::Linear* w_qkv_;
    ggml_runtime::Linear* out_proj_;
    ggml_runtime::LayerNorm* q_norm_ = nullptr;
    ggml_runtime::LayerNorm* k_norm_ = nullptr;
    ggml_runtime::LayerNorm* norm2_;
    ggml_runtime::Linear* ff_in_;
    ggml_runtime::Linear* ff_out_;
};

// Input bag: [0] mel (feat_in, T_mel, B) f32, [1] positions (T') i32 where
// T' = cfg.out_frames(T_mel), [2] optional key mask (T', 1, 1, B) f32
// (0 valid / -1e9 padded output frame; expanded on-graph to (T', T', 1, B)).
// Output bag: [0] (d_model, T', B).
class RopeTransformerEncoder : public ggml_runtime::Module {
   public:
    RopeTransformerEncoder(const std::string& name, const RopeTransformerConfig& cfg);
    ~RopeTransformerEncoder();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

    const RopeTransformerConfig& cfg() const { return cfg_; }

    // build_graph = encode_rows(pre_encode(features)), split so a caller can prepend rows (the
    // streaming Sortformer's speaker cache and FIFO are pre-encoded embeddings). FeatureStacking
    // (time zero-padded to a multiple of the subsampling factor) + pre_encode: (feat_in, T_mel, B)
    // -> (d_model, T', B).
    ggml_runtime::ggml_bf_tensor pre_encode(
        ggml_runtime::Session* session, const ggml_runtime::ggml_bf_tensor& features,
        ggml_runtime::TensorContainer* tc);
    // embed_norm -> blocks -> final_norm over pre-encoded rows (d_model, T, B). pos: (T) i32 RoPE
    // positions; kmask: optional (T, 1, 1, B) additive key mask (0 valid, -1e9 padding).
    ggml_runtime::TensorBag encode_rows(
        ggml_runtime::Session* session, const ggml_runtime::ggml_bf_tensor& rows,
        const ggml_runtime::ggml_bf_tensor& pos, const ggml_runtime::ggml_bf_tensor* kmask,
        ggml_runtime::TensorContainer* tc);

   private:
    std::string name_;
    RopeTransformerConfig cfg_;
    ggml_runtime::Linear* pre_encode_;
    ggml_runtime::LayerNorm* embed_norm_ = nullptr;
    std::vector<RopeTransformerBlock*> layers_;
    ggml_runtime::LayerNorm* final_norm_;
};

}  // namespace nemo_speech::speechlm
