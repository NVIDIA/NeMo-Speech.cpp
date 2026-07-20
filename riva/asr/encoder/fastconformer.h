// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
// FastConformer encoder, shared between CTC and RNNT heads.
// Parameterized from GGUF metadata.
//
// Supports both Parakeet-style (BatchNorm + symmetric conv + rel_pos MHA) and
// Nemotron cache-aware-style (LayerNorm + causal conv + rel_pos MHA with KV
// cache). The cache-aware path is selected via EncoderConfig::cache_mode and
// adds per-layer K/V/conv state plumbed in/out of each ConformerLayer.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "nn.h"
#include "rel_pos_attention.h"
#include "runtime.h"

namespace nemo_speech::asr {

// Norm flavor used inside ConformerConv. Parakeet CTC uses BatchNorm1d; the
// Nemotron cache-aware-trained encoder uses LayerNorm. (NeMo's checkpoint
// still names the parameters `*.batch_norm.{weight,bias}` even for LN, so we
// keep the GGUF tensor names unchanged.)
enum class ConvNorm { BatchNorm, LayerNorm };

// Depthwise-conv context inside ConformerConv. Symmetric pads on both sides
// (classic Conformer). Causal pads only on the left, so the conv module is
// streaming-safe. The cache-aware encoder additionally maintains a per-layer
// conv state of `kernel-1` past frames to bridge chunk boundaries.
enum class ConvContext { Symmetric, Causal };

// Whether the encoder threads K/V/conv state across calls.
enum class CacheMode { Disabled, Enabled };

struct EncoderConfig {
    int d_model = 1024;
    int n_layers = 42;
    int n_heads = 8;
    int d_ff = 4096;
    int conv_kernel_size = 9;
    int subsampling_factor = 8;
    int subsampling_conv_channels = 256;
    int feat_in = 80;  // mel bins
    int pos_emb_max_len = 5000;
    bool xscaling = true;
    bool use_bias = true;

    // Cache-aware variants - set by AsrModel from GGUF metadata. See
    // asr.encoder.conv_norm / conv_context / cache_supported / train_left_ctx
    // / train_right_ctx / att_context_style keys produced by
    // convert_model.py.
    ConvNorm conv_norm = ConvNorm::BatchNorm;
    ConvContext conv_context = ConvContext::Symmetric;
    bool chunked_limited_attention = false;
    bool cache_supported = false;
    CacheMode cache_mode = CacheMode::Disabled;
    int cache_left_ctx = 70;   // L_att frames threaded across chunks
    int cache_right_ctx = 13;  // right context (R) the model was trained with
    // Encoder-frame chunk length per streaming step (after subsampling).
    // For the Nemotron 0.6B default-latency preset: 1 + train_right_ctx = 14.
    int cache_chunk_frames = 14;
    int cache_state_slots = 1;
    // Frames to drop from the START of the subsampled encoder input to
    // strip the chunk's overlap with the previous chunk's tail (NeMo's
    // `drop_extra_pre_encoded`). For pre_encode_cache_size=9 / sub=8 this is
    // 2 frames, giving cache_chunk_frames = 1 + R after the drop.
    int cache_drop_extra = 2;

    // Offline attention limits are independent of the streaming cache context.
    // -1 means unlimited; finite values preserve a model's trained attention window.
    int offline_left_ctx = -1;
    int offline_right_ctx = -1;

    // Output length after subsampling at feat_in==80, factor 8.
    int subsample_out_length() const {
        int len = feat_in;
        const int n_stages = 3;                                  // log2(8)
        for (int i = 0; i < n_stages; i++) len = (len + 1) / 2;  // ceil-div
        return len;
    }
    int subsample_time_length(int frames) const {
        int len = frames;
        const int n_stages = static_cast<int>(std::log2(subsampling_factor));
        for (int i = 0; i < n_stages; ++i) {
            len = conv_context == ConvContext::Causal ? len / 2 + 1 : (len + 1) / 2;
        }
        return len;
    }
};

// Derive the shared cache-aware geometry used by both the session and runner.
// A negative right_ctx preserves the value stored in the model.
inline EncoderConfig
make_cache_aware_config(const EncoderConfig& base, int right_ctx) {
    EncoderConfig cfg = base;
    cfg.cache_mode = CacheMode::Enabled;
    cfg.conv_context = ConvContext::Causal;  // cache-aware requires causal conv
    if (right_ctx >= 0) {
        cfg.cache_right_ctx = right_ctx;
    }
    cfg.cache_chunk_frames = 1 + cfg.cache_right_ctx;
    return cfg;
}

// Per-layer cache I/O wired into ConformerLayer when cache_mode == Enabled.
// Follows NeMo's per-layer cur/next cache convention (cache_last_channel_cur
// -> cache_last_channel_next, cache_last_time likewise; conformer_encoder.py)
// except we cache projected K/V rather than the pre-projection layer input.
//
//   k_cache_cur / v_cache_cur : (d_model, cache_len) - past keys/values
//   conv_cache_cur            : (d_model, kernel-1)  - past conv state
//   attn_mask                 : (kv_len, 1)          - 0 valid / -1e9 masked
//
// On return, build_graph populates *_next and copies them back into the
// active stream's device-resident indexed cache arena row.
struct LayerCacheIO {
    ggml_tensor* k_cache_cur = nullptr;
    ggml_tensor* v_cache_cur = nullptr;
    ggml_tensor* conv_cache_cur = nullptr;
    ggml_tensor* attn_mask = nullptr;
    // Precomputed per-layer positional projection in head-split layout
    // [d_k, kv+chunk-1, n_head] (see FastConformerEncoder::pos_proj_name).
    // The geometry (kv = cache_left_ctx + chunk) is fixed per config, so the
    // projection is chunk-invariant; when non-null the layer skips its
    // per-chunk `linear_pos @ pos_emb` GEMM and the head permute of its
    // result.
    ggml_tensor* pos_proj = nullptr;
    // Depthwise-conv weight repacked channel-inner ([d_model, k] memory, see
    // FastConformerEncoder::dw_ct_name), enabling the channels-inner (cwhn)
    // direct depthwise kernel so the conv module runs in (d_model, T) layout
    // end-to-end. Null when unavailable — the layer then takes the
    // transpose-based conv path.
    ggml_tensor* dw_conv_w_ct = nullptr;
    int left_context = 0;
    int right_context = 0;

    ggml_tensor* k_cache_next = nullptr;
    ggml_tensor* v_cache_next = nullptr;
    ggml_tensor* conv_cache_next = nullptr;
};

// Conv-based 8x subsampling stem.
class SubSampling : public ggml_runtime::Module {
   public:
    SubSampling(const std::string& name, const EncoderConfig& cfg);
    ~SubSampling();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

   private:
    std::string name_;
    EncoderConfig cfg_;
    ggml_runtime::SequenceModule* conv_;
    ggml_runtime::Linear* out_;
};

// Half feed-forward (one of the two macaron FFs).
class ConformerFF : public ggml_runtime::Module {
   public:
    ConformerFF(const std::string& name, int d_model, int d_ff, bool use_bias);
    ~ConformerFF();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

   private:
    std::string name_;
    ggml_runtime::Linear* linear1_;
    ggml_runtime::Linear* linear2_;
};

// Conformer convolution module (pointwise + GLU + depthwise + norm + pointwise).
//
// Two axes of variation, selected via ConformerConv ctor:
//   norm    : BatchNorm1d (Parakeet) or LayerNorm (Nemotron cache-aware).
//   context : symmetric (classic Conformer) or causal (left-only padding,
//             streaming-safe; the cache-aware path additionally consumes a
//             per-chunk conv_cache_cur to bridge boundaries).
class ConformerConv : public ggml_runtime::Module {
   public:
    ConformerConv(
        const std::string& name, int d_model, int kernel_size, bool use_bias,
        ConvNorm norm = ConvNorm::BatchNorm, ConvContext context = ConvContext::Symmetric);
    ~ConformerConv();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

    // Cache-aware entry point. cache_cur is (d_model, kernel-1) past frames; on
    // success cache_next is set to the (d_model, kernel-1) tail of the
    // concatenated input. Returns the conv module output (d_model, T, B).
    ggml_runtime::TensorBag build_graph_cached(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc, ggml_tensor* cache_cur, ggml_tensor** cache_next);

    // Channels-inner cache-aware variant: keeps (d_model, T, B) layout end to
    // end (pointwise convs as plain mul_mat, GLU on dim-0 halves, LayerNorm
    // without transposes) and runs the depthwise conv via the cwhn direct
    // kernel using `w_ct`, the channel-inner repacked weight. Only valid for
    // Causal context + LayerNorm; caller guarantees w_ct is filled.
    ggml_runtime::TensorBag build_graph_cached_ct(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc, ggml_tensor* cache_in, ggml_tensor** cache_out,
        ggml_tensor* w_ct);

    bool supports_ct_layout() const {
        return context_ == ConvContext::Causal && norm_kind_ == ConvNorm::LayerNorm;
    }

    int kernel_size() const { return kernel_size_; }

   private:
    std::string name_;
    int d_model_;
    int kernel_size_;
    bool use_bias_;
    ConvNorm norm_kind_;
    ConvContext context_;
    ggml_runtime::Conv1D* pointwise_conv1_;
    ggml_runtime::Conv1D* pointwise_conv2_;
    ggml_runtime::Conv1D* depthwise_conv_;
    ggml_runtime::BatchNorm1d* batch_norm_;  // used when norm_kind_ == BatchNorm
    ggml_runtime::LayerNorm* layer_norm_;    // used when norm_kind_ == LayerNorm

    // Shared graph body: takes the (T, d_model, B) "channel-last" tensor that
    // results from the input transpose + pointwise_conv1 + GLU, runs the rest
    // of the conv module, and returns (d_model, T, B). When cache_cur is non-
    // null the depthwise conv is replaced by a manual left-padded conv driven
    // by [cache | x] and cache_next is filled with the tail of that buffer.
    ggml_runtime::TensorBag build_post_glu(
        ggml_runtime::Session* session, ggml_runtime::TensorContainer* tc,
        ggml_runtime::ggml_bf_tensor x_glu, ggml_tensor* cache_cur, ggml_tensor** cache_next);
};

// Single FastConformer block: FF/2 + MHA(rel-pos) + Conv + FF/2 + LN.
//
// build_graph supports both non-cached and cache-aware streaming. Pass a
// non-null LayerCacheIO to opt into the cache-aware path: K/V are concatenated
// with the per-layer cache before attention, the depthwise conv consumes a
// (kernel-1) state, and on return *_cache_next are populated with the updated
// state for the next chunk.
class ConformerLayer : public ggml_runtime::Module {
   public:
    ConformerLayer(const std::string& name, const EncoderConfig& cfg);
    ~ConformerLayer();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

    // Cache-aware entry point. Mutates *cache to fill in the *_cache_next
    // pointers with the updated KV/conv state.
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc, LayerCacheIO* cache);

    int n_heads() const { return cfg_.n_heads; }
    int d_model() const { return cfg_.d_model; }

   private:
    std::string name_;
    EncoderConfig cfg_;
    ggml_runtime::LayerNorm* norm_feed_forward1_;
    ConformerFF* feed_forward1_;
    ggml_runtime::LayerNorm* norm_self_att_;
    ggml_runtime::RelPositionMultiHeadAttention* self_attn_;
    ggml_runtime::LayerNorm* norm_conv_;
    ConformerConv* conv_;
    ggml_runtime::LayerNorm* norm_feed_forward2_;
    ConformerFF* feed_forward2_;
    ggml_runtime::LayerNorm* norm_out_;

    // Cached MHA implementation. The non-cached path delegates to
    // RelPositionMultiHeadAttention::build_graph as before. The positional
    // term comes from the precomputed per-layer projection in
    // cache->pos_proj (no raw pos_emb input).
    ggml_runtime::TensorBag build_mha_cached(
        ggml_runtime::Session* session, ggml_runtime::TensorContainer* tc,
        ggml_runtime::ggml_bf_tensor x, LayerCacheIO* cache);
};

// Weight-load hook for Sessions that load encoder weights. Implements the
// serialized tensor-planar Q8 contract written by convert_model.py
// (--q8-layout planar): when the GGUF declares
// `asr.encoder.q8_layout == tensor_planar_v1`, every Q8_0 `encoder.*` weight
// is flagged GGML_TENSOR_FLAG_Q8_PLANAR for the planar-aware CUDA kernels,
// and rejected on non-CUDA buffers (the bytes are not valid block_q8_0, so
// any other backend would silently decode garbage). Install via
// Session::set_weight_load_hook before setup(); a no-op for models without
// the metadata key.
ggml_runtime::Session::WeightLoadHook planar_q8_weight_load_hook();

// Full FastConformer encoder.
// Input:  features (feat_in, T_audio, 1, 1) float32.
// Output: encoded   (d_model, T_out, 1, 1)   float32 (T_out = T_audio / subsampling_factor).
//
// When cfg.cache_mode == CacheMode::Enabled the encoder additionally owns
// persistent per-layer K/V/conv cache arenas and exposes stream-slot IDs plus
// an attention mask as graph inputs. Each graph run gathers the selected rows
// and scatters the updates in-graph; only the chunk output returns to the host.
class FastConformerEncoder : public ggml_runtime::Module {
   public:
    FastConformerEncoder(const std::string& name, const EncoderConfig& cfg);
    ~FastConformerEncoder();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

    // Split entry points for consumers that drive the two encoder halves
    // separately (Sortformer/NEST: each streaming step pre-encodes only the
    // new mel chunk, concatenates the result with cached [spkcache | fifo]
    // embeddings, and re-encodes the whole concatenation - NeMo's
    // `encoder.pre_encode(x)` + `bypass_pre_encode=True`). Offline
    // (cache_mode == Disabled) only.
    //
    // build_pre_encode: (feat_in, T_mel) mel -> (d_model, T_mel/sub) raw
    // pre-encode embeddings (NO xscaling - matches NeMo, where cached
    // embeddings are stored pre-scale).
    ggml_runtime::TensorBag build_pre_encode(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc);
    // build_graph_from_embeddings: (d_model, T) embeddings -> encoder output;
    // applies xscaling + rel-pos encoding + the conformer stack.
    ggml_runtime::TensorBag build_graph_from_embeddings(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc);

    const EncoderConfig& cfg() const { return cfg_; }

    // Only meaningful when cache_mode is Enabled.
    // Tensor names follow `{encoder_name}.cache.{kind}.{layer}` for KV/conv
    // and `{encoder_name}.cache.attn_mask` for the mask. The cache arenas are
    // persistent model tensors updated in-graph; the runner supplies slot IDs
    // and the per-chunk attention mask before each Session::run.
    std::string kv_cache_name(int layer) const;
    std::string conv_cache_name(int layer) const;
    std::string attn_mask_name() const;

    // Internal tensor names for layer cache-update graph nodes.
    std::string k_cache_out_name(int layer) const;
    std::string v_cache_out_name(int layer) const;
    std::string conv_cache_out_name(int layer) const;

    // Chunk-invariant positional projection for cache-aware mode.
    // The cached pos_emb slice is a fixed window of the PE table (its length
    // depends only on cache_left_ctx and cache_chunk_frames), so each layer's
    // `linear_pos @ pos_emb` is the same every chunk. define_tensors()
    // allocates one persistent [d_k, kv+chunk-1, n_head] tensor per layer and
    // build_precompute_graph() fills them all in one Session::run,
    // triggered once by CacheAwareEncoder right after setup. Chunk graphs
    // then consume the tensors read-only via LayerCacheIO::pos_proj.
    static constexpr const char* kPrecomputeTrigger = "encoder.precompute.trigger";
    std::string pos_proj_name(int layer) const;
    // Depthwise-conv weight repacked channel-inner for the cwhn direct
    // kernel; filled by the same precompute run as pos_proj.
    std::string dw_ct_name(int layer) const;
    ggml_runtime::TensorBag build_precompute_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorContainer* tc);

   private:
    std::string name_;
    EncoderConfig cfg_;
    SubSampling* pre_encode_;
    ggml_runtime::RelPositionalEncoding* pos_enc_;
    // Non-cached path: SequenceModule of ConformerLayer*. Cache-aware path:
    // we own the ConformerLayer* directly so we can pass LayerCacheIO into
    // each one.
    ggml_runtime::SequenceModule* layers_;     // used when cache disabled
    std::vector<ConformerLayer*> layer_ptrs_;  // used when cache enabled
};

}  // namespace nemo_speech::asr
