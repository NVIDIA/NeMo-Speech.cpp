// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "lt.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include "graph.h"
#include "nvtx_utils.h"

#if defined(MAGPIETTS_CUDA_SAMPLING)
#include "ggml-cuda.h"
#endif

namespace nemo_speech::tts {

class LocalTransformerCudaAttentionCache {
   public:
    LocalTransformerCudaAttentionCache() = default;
    ~LocalTransformerCudaAttentionCache();

    LocalTransformerCudaAttentionCache(const LocalTransformerCudaAttentionCache&) = delete;
    LocalTransformerCudaAttentionCache& operator=(const LocalTransformerCudaAttentionCache&) =
        delete;
    LocalTransformerCudaAttentionCache(LocalTransformerCudaAttentionCache&& other) noexcept;
    LocalTransformerCudaAttentionCache& operator=(
        LocalTransformerCudaAttentionCache&& other) noexcept;

    bool init(const magpietts_model& model, int lane_count);
    void reset();

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor*> layers;
    std::vector<ggml_tensor*> cache_states;
    ggml_tensor* slot_ids = nullptr;
    int n_ctx = 0;
    int n_embd = 0;
    int lanes = 0;
};

class LocalTransformerGraph {
   public:
    LocalTransformerGraph() = default;
    ~LocalTransformerGraph();

    LocalTransformerGraph(const LocalTransformerGraph&) = delete;
    LocalTransformerGraph& operator=(const LocalTransformerGraph&) = delete;
    LocalTransformerGraph(LocalTransformerGraph&& other) noexcept;
    LocalTransformerGraph& operator=(LocalTransformerGraph&& other) noexcept;

    void reset();

    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t allocr = nullptr;

    ggml_tensor* dec_cond = nullptr;
    ggml_tensor* dec_uncond = nullptr;
    ggml_tensor* pos_emb = nullptr;
    ggml_tensor* prev_token = nullptr;
    ggml_tensor* cache_state = nullptr;
    ggml_tensor* logits_cond = nullptr;
    ggml_tensor* logits_uncond = nullptr;

    int codebook_idx = -1;
    int seq_len = 0;
    bool pair = false;

    std::vector<float> logits_cond_data;
    std::vector<float> logits_uncond_data;
};

class LocalTransformerGraphBank {
   public:
    LocalTransformerGraphBank() = default;
    ~LocalTransformerGraphBank();

    LocalTransformerGraphBank(const LocalTransformerGraphBank&) = delete;
    LocalTransformerGraphBank& operator=(const LocalTransformerGraphBank&) = delete;
    LocalTransformerGraphBank(LocalTransformerGraphBank&& other) noexcept;
    LocalTransformerGraphBank& operator=(LocalTransformerGraphBank&& other) noexcept;

    void reset();
    bool beginFrame(const magpietts_model& model, bool pair);

    std::vector<LocalTransformerGraph> single_graphs;
    std::vector<LocalTransformerGraph> pair_graphs;
    DecoderKvCache single_cache;
    DecoderKvCache cond_cache;
    DecoderKvCache uncond_cache;
    LocalTransformerCudaAttentionCache pair_cuda_attention_cache;
};

using local_transformer_graph = LocalTransformerGraph;
using local_transformer_graph_bank = LocalTransformerGraphBank;

static void
dump_local_codebook_logits(
    const LocalCodebookLogitDump* dump, int codebook, int sampled, int greedy,
    const std::vector<float>& logits) {
    if (!dump || !dump->path || !dump->path[0]) {
        return;
    }

    FILE* fp = fopen(dump->path, "ab");
    if (!fp) {
        fprintf(stderr, "failed to open MagpieTTS logit dump: %s\n", dump->path);
        return;
    }

    const std::string label = dump->label ? dump->label : "";
    const char magic[4] = {'M', 'L', 'D', 'G'};
    const uint32_t version = 1;
    const uint32_t label_len = (uint32_t)label.size();
    const int32_t chunk_index = dump->chunk_index;
    const int32_t step = dump->step;
    const int32_t frame_index = dump->frame_index;
    const int32_t codebook_i32 = codebook;
    const int32_t vocab_size = (int32_t)logits.size();
    const int32_t sampled_i32 = sampled;
    const int32_t greedy_i32 = greedy;

    fwrite(magic, sizeof(magic), 1, fp);
    fwrite(&version, sizeof(version), 1, fp);
    fwrite(&label_len, sizeof(label_len), 1, fp);
    fwrite(&chunk_index, sizeof(chunk_index), 1, fp);
    fwrite(&step, sizeof(step), 1, fp);
    fwrite(&frame_index, sizeof(frame_index), 1, fp);
    fwrite(&codebook_i32, sizeof(codebook_i32), 1, fp);
    fwrite(&vocab_size, sizeof(vocab_size), 1, fp);
    fwrite(&sampled_i32, sizeof(sampled_i32), 1, fp);
    fwrite(&greedy_i32, sizeof(greedy_i32), 1, fp);
    if (label_len > 0) {
        fwrite(label.data(), 1, label.size(), fp);
    }
    if (!logits.empty()) {
        fwrite(logits.data(), sizeof(float), logits.size(), fp);
    }
    fclose(fp);
}

static bool
magpietts_local_add_tensor(
    ggml_context* ctx, const ggml_tensor* src, ggml_tensor** dst,
    std::vector<std::pair<const ggml_tensor*, ggml_tensor*>>& copies, bool fp32) {
    if (!src) {
        *dst = nullptr;
        return true;
    }
    *dst = fp32 ? ggml_new_tensor(ctx, GGML_TYPE_F32, ggml_n_dims(src), src->ne)
                : ggml_dup_tensor(ctx, src);
    if (!*dst) {
        fprintf(stderr, "failed to mirror local-transformer tensor %s\n", ggml_get_name(src));
        return false;
    }
    ggml_set_name(*dst, ggml_get_name(src));
    copies.push_back({src, *dst});
    return true;
}

static bool
magpietts_local_add_tensor_vector(
    ggml_context* ctx, const std::vector<ggml_tensor*>& src, std::vector<ggml_tensor*>& dst,
    std::vector<std::pair<const ggml_tensor*, ggml_tensor*>>& copies, bool fp32) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        if (!magpietts_local_add_tensor(ctx, src[i], &dst[i], copies, fp32)) {
            return false;
        }
    }
    return true;
}

static bool
magpietts_local_copy_transformer_layout(
    ggml_context* ctx, const magpietts_transformer& src, magpietts_transformer& dst,
    std::vector<std::pair<const ggml_tensor*, ggml_tensor*>>& copies, bool fp32) {
    dst.n_embd = src.n_embd;
    dst.n_head = src.n_head;
    dst.n_cross_head = src.n_cross_head;
    dst.n_cross_dhead = src.n_cross_dhead;
    dst.kernel = src.kernel;
    dst.causal = src.causal;
    dst.has_cross = src.has_cross;
    if (!magpietts_local_add_tensor(ctx, src.norm_out, &dst.norm_out, copies, fp32) ||
        !magpietts_local_add_tensor(ctx, src.pos_emb, &dst.pos_emb, copies, fp32)) {
        return false;
    }

    dst.layers.resize(src.layers.size());
    for (size_t il = 0; il < src.layers.size(); ++il) {
        const magpietts_layer& src_layer = src.layers[il];
        magpietts_layer& dst_layer = dst.layers[il];
        dst_layer.has_cross = src_layer.has_cross;
        dst_layer.kernel = src_layer.kernel;
        if (!magpietts_local_add_tensor(
                ctx, src_layer.norm_self, &dst_layer.norm_self, copies, fp32) ||
            !magpietts_local_add_tensor(
                ctx, src_layer.self_qkv, &dst_layer.self_qkv, copies, fp32) ||
            !magpietts_local_add_tensor(ctx, src_layer.self_o, &dst_layer.self_o, copies, fp32) ||
            !magpietts_local_add_tensor(
                ctx, src_layer.norm_xattn_query, &dst_layer.norm_xattn_query, copies, fp32) ||
            !magpietts_local_add_tensor(ctx, src_layer.cross_q, &dst_layer.cross_q, copies, fp32) ||
            !magpietts_local_add_tensor(
                ctx, src_layer.cross_kv, &dst_layer.cross_kv, copies, fp32) ||
            !magpietts_local_add_tensor(ctx, src_layer.cross_o, &dst_layer.cross_o, copies, fp32) ||
            !magpietts_local_add_tensor(
                ctx, src_layer.norm_xattn_memory, &dst_layer.norm_xattn_memory, copies, fp32) ||
            !magpietts_local_add_tensor(ctx, src_layer.norm_ff, &dst_layer.norm_ff, copies, fp32) ||
            !magpietts_local_add_tensor_vector(
                ctx, src_layer.ff_proj, dst_layer.ff_proj, copies, fp32) ||
            !magpietts_local_add_tensor_vector(
                ctx, src_layer.ff_out, dst_layer.ff_out, copies, fp32)) {
            return false;
        }
    }
    return true;
}

static bool
magpietts_local_copy_tensor_fp32(const ggml_tensor* src, ggml_tensor* dst) {
    const int64_t elements = ggml_nelements(src);
    std::vector<float> values((size_t)elements);
    if (src->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(src, values.data(), 0, values.size() * sizeof(float));
    } else if (src->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> packed((size_t)elements);
        ggml_backend_tensor_get(src, packed.data(), 0, packed.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(packed.data(), values.data(), elements);
    } else if (src->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> packed((size_t)elements);
        ggml_backend_tensor_get(src, packed.data(), 0, packed.size() * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row(packed.data(), values.data(), elements);
    } else {
        fprintf(
            stderr, "cannot convert local-transformer tensor %s from %s to f32\n",
            ggml_get_name(src), ggml_type_name(src->type));
        return false;
    }
    ggml_backend_tensor_set(dst, values.data(), 0, values.size() * sizeof(float));
    return true;
}

static bool
magpietts_model_init_local_transformer_copy(
    const magpietts_model& src, magpietts_model& dst, bool use_cuda, bool fp32) {
    dst.reset();
    dst.hparams = src.hparams;
    dst.cuda_unified_memory = false;

    if (use_cuda) {
        ggml_backend_dev_t device = ggml_backend_get_device(src.backend);
        if (!device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            fprintf(stderr, "cannot create CUDA local-transformer mirror from a non-GPU model\n");
            return false;
        }
        dst.backend = ggml_backend_dev_init(device, nullptr);
    } else {
        dst.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!dst.backend) {
        fprintf(stderr, "failed to initialize local-transformer mirror backend\n");
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/16ull * 1024ull * 1024ull,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    dst.ctx = ggml_init(params);
    if (!dst.ctx) {
        fprintf(stderr, "failed to initialize local-transformer tensor context\n");
        dst.reset();
        return false;
    }

    std::vector<std::pair<const ggml_tensor*, ggml_tensor*>> copies;
    if (!magpietts_local_add_tensor_vector(
            dst.ctx, src.audio_embeddings, dst.audio_embeddings, copies, fp32) ||
        (src.lt_in_w &&
         !magpietts_local_add_tensor(dst.ctx, src.lt_in_w, &dst.lt_in_w, copies, fp32)) ||
        (src.lt_in_b &&
         !magpietts_local_add_tensor(dst.ctx, src.lt_in_b, &dst.lt_in_b, copies, fp32)) ||
        !magpietts_local_add_tensor_vector(dst.ctx, src.lt_out_w, dst.lt_out_w, copies, fp32) ||
        !magpietts_local_add_tensor_vector(dst.ctx, src.lt_out_b, dst.lt_out_b, copies, fp32) ||
        !magpietts_local_copy_transformer_layout(dst.ctx, src.local, dst.local, copies, fp32)) {
        dst.reset();
        return false;
    }

    dst.buffer = ggml_backend_alloc_ctx_tensors(dst.ctx, dst.backend);
    if (!dst.buffer) {
        fprintf(stderr, "failed to allocate local-transformer mirror tensors\n");
        dst.reset();
        return false;
    }

    if (fp32) {
        for (const auto& copy : copies) {
            if (copy.second->type != GGML_TYPE_F32) {
                fprintf(
                    stderr, "local-transformer FP32 mirror contains non-FP32 tensor %s (%s)\n",
                    ggml_get_name(copy.second), ggml_type_name(copy.second->type));
                dst.reset();
                return false;
            }
        }
    }

    for (const auto& copy : copies) {
        if (fp32) {
            if (!magpietts_local_copy_tensor_fp32(copy.first, copy.second)) {
                dst.reset();
                return false;
            }
        } else {
            ggml_backend_tensor_copy(copy.first, copy.second);
        }
    }
    ggml_backend_synchronize(dst.backend);
    fprintf(
        stderr, "MagpieTTS local transformer mirror: %s, precision=%s (%zu tensors)\n",
        ggml_backend_name(dst.backend), fp32 ? "fp32" : "native", copies.size());
    return true;
}

bool
magpietts_model_init_local_transformer_cpu(const magpietts_model& src, magpietts_model& dst) {
    return magpietts_model_init_local_transformer_copy(src, dst, false, false);
}

bool
magpietts_model_init_local_transformer_fp32(
    const magpietts_model& src, magpietts_model& dst, bool use_cuda) {
    return magpietts_model_init_local_transformer_copy(src, dst, use_cuda, true);
}

LocalTransformerCudaAttentionCache::~LocalTransformerCudaAttentionCache() {
    reset();
}

LocalTransformerCudaAttentionCache::LocalTransformerCudaAttentionCache(
    LocalTransformerCudaAttentionCache&& other) noexcept {
    *this = std::move(other);
}

LocalTransformerCudaAttentionCache&
LocalTransformerCudaAttentionCache::operator=(LocalTransformerCudaAttentionCache&& other) noexcept {
    if (this != &other) {
        reset();
        ctx = other.ctx;
        buffer = other.buffer;
        layers = std::move(other.layers);
        cache_states = std::move(other.cache_states);
        slot_ids = other.slot_ids;
        n_ctx = other.n_ctx;
        n_embd = other.n_embd;
        lanes = other.lanes;
        other.ctx = nullptr;
        other.buffer = nullptr;
        other.slot_ids = nullptr;
        other.n_ctx = 0;
        other.n_embd = 0;
        other.lanes = 0;
    }
    return *this;
}

void
LocalTransformerCudaAttentionCache::reset() {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    layers.clear();
    cache_states.clear();
    slot_ids = nullptr;
    n_ctx = 0;
    n_embd = 0;
    lanes = 0;
}

bool
LocalTransformerCudaAttentionCache::init(const magpietts_model& model, int lane_count) {
    const magpietts_hparams& h = model.hparams;
    if (ctx) {
        if (n_ctx == h.lt_ctx && n_embd == h.lt_hidden && lanes == lane_count &&
            static_cast<int>(layers.size()) == h.lt_layers &&
            static_cast<int>(cache_states.size()) == h.stacked_audio_codebooks()) {
            return true;
        }
        reset();
    }

    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() *
            static_cast<size_t>(h.lt_layers + h.stacked_audio_codebooks() + 1),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "failed to allocate local CUDA attention cache context\n");
        return false;
    }

    layers.reserve(static_cast<size_t>(h.lt_layers));
    for (int layer = 0; layer < h.lt_layers; ++layer) {
        ggml_tensor* arena = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F32, static_cast<int64_t>(h.lt_hidden) * h.lt_ctx, lane_count, 2);
        const std::string name = "magpietts_local_cuda_kv_" + std::to_string(layer);
        ggml_set_name(arena, name.c_str());
        layers.push_back(arena);
    }
    slot_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, lane_count);
    ggml_set_name(slot_ids, "magpietts_local_cuda_slot_ids");
    cache_states.reserve(static_cast<size_t>(h.stacked_audio_codebooks()));
    for (int codebook = 0; codebook < h.stacked_audio_codebooks(); ++codebook) {
        ggml_tensor* state_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, lane_count, 2);
        const std::string name = "magpietts_local_cuda_cache_state_" + std::to_string(codebook);
        ggml_set_name(state_tensor, name.c_str());
        cache_states.push_back(state_tensor);
    }

    buffer = ggml_backend_alloc_ctx_tensors(ctx, model.backend);
    if (!buffer) {
        fprintf(stderr, "failed to allocate local CUDA attention cache buffer\n");
        reset();
        return false;
    }
    std::vector<int32_t> slots(static_cast<size_t>(lane_count));
    for (int lane = 0; lane < lane_count; ++lane) slots[static_cast<size_t>(lane)] = lane;
    ggml_backend_tensor_set(slot_ids, slots.data(), 0, slots.size() * sizeof(int32_t));
    for (ggml_tensor* arena : layers) {
        ggml_backend_tensor_memset(arena, 0, 0, ggml_nbytes(arena));
    }
    std::vector<int32_t> state(static_cast<size_t>(lane_count) * 2);
    for (int codebook = 0; codebook < h.stacked_audio_codebooks(); ++codebook) {
        std::fill(state.begin(), state.end(), codebook);
        ggml_backend_tensor_set(
            cache_states[static_cast<size_t>(codebook)], state.data(), 0,
            state.size() * sizeof(int32_t));
    }
    n_ctx = h.lt_ctx;
    n_embd = h.lt_hidden;
    lanes = lane_count;
    return true;
}

LocalTransformerGraph::~LocalTransformerGraph() {
    reset();
}

LocalTransformerGraph::LocalTransformerGraph(LocalTransformerGraph&& other) noexcept {
    *this = std::move(other);
}

LocalTransformerGraph&
LocalTransformerGraph::operator=(LocalTransformerGraph&& other) noexcept {
    if (this != &other) {
        reset();
        ctx = other.ctx;
        gf = other.gf;
        allocr = other.allocr;
        dec_cond = other.dec_cond;
        dec_uncond = other.dec_uncond;
        pos_emb = other.pos_emb;
        prev_token = other.prev_token;
        cache_state = other.cache_state;
        logits_cond = other.logits_cond;
        logits_uncond = other.logits_uncond;
        codebook_idx = other.codebook_idx;
        seq_len = other.seq_len;
        pair = other.pair;
        logits_cond_data = std::move(other.logits_cond_data);
        logits_uncond_data = std::move(other.logits_uncond_data);

        other.ctx = nullptr;
        other.gf = nullptr;
        other.allocr = nullptr;
        other.dec_cond = nullptr;
        other.dec_uncond = nullptr;
        other.pos_emb = nullptr;
        other.prev_token = nullptr;
        other.cache_state = nullptr;
        other.logits_cond = nullptr;
        other.logits_uncond = nullptr;
        other.codebook_idx = -1;
        other.seq_len = 0;
        other.pair = false;
    }
    return *this;
}

void
LocalTransformerGraph::reset() {
    if (allocr) {
        ggml_gallocr_free(allocr);
        allocr = nullptr;
    }
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    gf = nullptr;
    dec_cond = nullptr;
    dec_uncond = nullptr;
    pos_emb = nullptr;
    prev_token = nullptr;
    cache_state = nullptr;
    logits_cond = nullptr;
    logits_uncond = nullptr;
    codebook_idx = -1;
    seq_len = 0;
    pair = false;
    logits_cond_data.clear();
    logits_uncond_data.clear();
}

LocalTransformerGraphBank::~LocalTransformerGraphBank() {
    reset();
}

LocalTransformerGraphBank::LocalTransformerGraphBank(LocalTransformerGraphBank&& other) noexcept {
    *this = std::move(other);
}

LocalTransformerGraphBank&
LocalTransformerGraphBank::operator=(LocalTransformerGraphBank&& other) noexcept {
    if (this != &other) {
        reset();
        single_graphs = std::move(other.single_graphs);
        pair_graphs = std::move(other.pair_graphs);
        single_cache = std::move(other.single_cache);
        cond_cache = std::move(other.cond_cache);
        uncond_cache = std::move(other.uncond_cache);
        pair_cuda_attention_cache = std::move(other.pair_cuda_attention_cache);
    }
    return *this;
}

void
LocalTransformerGraphBank::reset() {
    single_graphs.clear();
    pair_graphs.clear();
    single_cache.reset();
    cond_cache.reset();
    uncond_cache.reset();
    pair_cuda_attention_cache.reset();
}

bool
LocalTransformerGraphBank::beginFrame(const magpietts_model& model, bool pair) {
    const auto& h = model.hparams;
    if (pair) {
        if (magpietts_backend_is_cuda(model.backend)) {
            return pair_cuda_attention_cache.init(model, 2);
        }
        if (!cond_cache.init(
                model.backend, h.lt_layers, h.lt_ctx, h.lt_hidden, "local conditional") ||
            !uncond_cache.init(
                model.backend, h.lt_layers, h.lt_ctx, h.lt_hidden, "local unconditional")) {
            return false;
        }
        cond_cache.clear();
        uncond_cache.clear();
    } else {
        if (!single_cache.init(model.backend, h.lt_layers, h.lt_ctx, h.lt_hidden, "local")) {
            return false;
        }
        single_cache.clear();
    }
    return true;
}

static ggml_tensor*
local_self_attention_cuda_cached_pair(
    ggml_context* ctx, const magpietts_transformer& tr, const magpietts_layer& layer,
    LocalTransformerCudaAttentionCache& cache, int layer_index, ggml_tensor* cache_state,
    ggml_tensor* x) {
    constexpr int kCfgLanes = 2;
    const int64_t n_embd = tr.n_embd;
    const int64_t d_head = n_embd / tr.n_head;
    ggml_tensor* qkv =
        ggml_reshape_3d(ctx, linear(ctx, layer.self_qkv, x), 3 * n_embd, 1, kCfgLanes);
    const size_t element = ggml_element_size(qkv);
    auto split_heads = [&](size_t offset) {
        return ggml_view_4d(
            ctx, qkv, d_head, 1, tr.n_head, kCfgLanes, qkv->nb[1],
            static_cast<size_t>(d_head) * element, qkv->nb[2], offset);
    };
    ggml_tensor* q = split_heads(0);
    ggml_tensor* k = split_heads(static_cast<size_t>(n_embd) * element);
    ggml_tensor* v = split_heads(static_cast<size_t>(2 * n_embd) * element);
    ggml_tensor* heads = ggml_fused_attn_cached(
        ctx, q, k, v, nullptr, cache.layers[static_cast<size_t>(layer_index)], cache.slot_ids,
        cache_state, cache.n_ctx, 1.0f / std::sqrt(static_cast<float>(d_head)), true);
    ggml_tensor* merged =
        ggml_reshape_2d(ctx, ggml_permute(ctx, heads, 0, 2, 1, 3), n_embd, kCfgLanes);
    return linear(ctx, layer.self_o, merged);
}

static ggml_tensor*
local_transformer_forward_cached_fixed_pos(
    ggml_context* ctx, ggml_cgraph* gf, const magpietts_transformer& tr, ggml_tensor* x,
    ggml_tensor* pos_emb, DecoderKvCache& cache, int n_past) {
    pos_emb = ggml_cont(ctx, ggml_cast(ctx, pos_emb, GGML_TYPE_F32));
    x = ggml_add(ctx, x, pos_emb);

    for (int il = 0; il < (int)tr.layers.size(); ++il) {
        const magpietts_layer& layer = tr.layers[(size_t)il];
        ggml_tensor* residual = x;
        ggml_tensor* cur = layer_norm(ctx, x, layer.norm_self);
        cur = self_attention_cached(ctx, gf, tr, layer, cache, il, n_past, cur);
        x = ggml_add(ctx, residual, cur);

        residual = x;
        cur = layer_norm(ctx, x, layer.norm_ff);
        cur = causal_conv1d(ctx, cur, layer.ff_proj);
        cur = ggml_gelu(ctx, cur);
        cur = causal_conv1d(ctx, cur, layer.ff_out);
        x = ggml_add(ctx, residual, cur);
    }

    return tr.norm_out ? layer_norm(ctx, x, tr.norm_out) : x;
}

static ggml_tensor*
local_self_attention_cached_pair(
    ggml_context* ctx, ggml_cgraph* gf, const magpietts_transformer& tr,
    const magpietts_layer& layer, DecoderKvCache& cond_cache, DecoderKvCache& uncond_cache,
    int layer_index, int n_past, ggml_tensor* x) {
    constexpr int kCfgLanes = 2;
    const int64_t n_embd = tr.n_embd;
    const int64_t n_head = tr.n_head;
    const int64_t d_head = n_embd / n_head;
    const int64_t n_tok = x->ne[1];
    const int64_t n_total = n_past + n_tok;

    ggml_tensor* qkv = linear(ctx, layer.self_qkv, x);
    const size_t element = ggml_element_size(qkv);
    auto qkv_slice = [&](size_t offset) {
        return ggml_view_3d(ctx, qkv, n_embd, n_tok, kCfgLanes, qkv->nb[1], qkv->nb[2], offset);
    };
    ggml_tensor* qcur = qkv_slice(0);
    ggml_tensor* kcur = qkv_slice(static_cast<size_t>(n_embd) * element);
    ggml_tensor* vcur = qkv_slice(static_cast<size_t>(2 * n_embd) * element);

    const size_t cache_element = ggml_element_size(cond_cache.memory_k);
    const size_t layer_offset =
        static_cast<size_t>(layer_index) * cond_cache.n_ctx * n_embd * cache_element;
    const size_t write_offset = layer_offset + static_cast<size_t>(n_past) * n_embd * cache_element;
    DecoderKvCache* caches[kCfgLanes] = {&cond_cache, &uncond_cache};
    for (int lane = 0; lane < kCfgLanes; ++lane) {
        ggml_tensor* k_lane = ggml_view_2d(
            ctx, kcur, n_embd, n_tok, kcur->nb[1], static_cast<size_t>(lane) * kcur->nb[2]);
        ggml_tensor* v_lane = ggml_view_2d(
            ctx, vcur, n_embd, n_tok, vcur->nb[1], static_cast<size_t>(lane) * vcur->nb[2]);
        ggml_tensor* k_dst =
            ggml_view_1d(ctx, caches[lane]->memory_k, n_tok * n_embd, write_offset);
        ggml_tensor* v_dst =
            ggml_view_1d(ctx, caches[lane]->memory_v, n_tok * n_embd, write_offset);
        ggml_tensor* k_copy = ggml_cpy(ctx, k_lane, k_dst);
        ggml_set_name(k_copy, "magpietts_local_pair_kv_copy_k");
        ggml_build_forward_expand(gf, k_copy);
        ggml_tensor* v_copy = ggml_cpy(ctx, v_lane, v_dst);
        ggml_set_name(v_copy, "magpietts_local_pair_kv_copy_v");
        ggml_build_forward_expand(gf, v_copy);
    }

    ggml_tensor* q =
        ggml_permute(ctx, ggml_cont_4d(ctx, qcur, d_head, n_head, n_tok, kCfgLanes), 0, 2, 1, 3);
    ggml_tensor* k_cond = ggml_view_2d(
        ctx, cond_cache.memory_k, n_embd, n_total, n_embd * cache_element, layer_offset);
    ggml_tensor* k_uncond = ggml_view_2d(
        ctx, uncond_cache.memory_k, n_embd, n_total, n_embd * cache_element, layer_offset);
    ggml_tensor* k_pair = ggml_concat(ctx, k_cond, k_uncond, 2);
    ggml_tensor* k = ggml_permute(
        ctx, ggml_reshape_4d(ctx, k_pair, d_head, n_head, n_total, kCfgLanes), 0, 2, 1, 3);
    ggml_tensor* scores =
        ggml_scale(ctx, ggml_mul_mat(ctx, k, q), 1.0f / std::sqrt(static_cast<float>(d_head)));
    // Each local graph evaluates exactly one new position and exposes only the populated cache
    // prefix, so all keys are causal-valid and no diagonal mask is needed.
    ggml_tensor* probs = ggml_soft_max(ctx, scores);

    ggml_tensor* v_cond = ggml_view_2d(
        ctx, cond_cache.memory_v, n_embd, n_total, n_embd * cache_element, layer_offset);
    ggml_tensor* v_uncond = ggml_view_2d(
        ctx, uncond_cache.memory_v, n_embd, n_total, n_embd * cache_element, layer_offset);
    ggml_tensor* v_pair = ggml_concat(ctx, v_cond, v_uncond, 2);
    ggml_tensor* v_trans = ggml_cont_4d(
        ctx,
        ggml_permute(
            ctx, ggml_reshape_4d(ctx, v_pair, d_head, n_head, n_total, kCfgLanes), 1, 2, 0, 3),
        n_total, d_head, n_head, kCfgLanes);
    ggml_tensor* weighted = ggml_mul_mat(ctx, v_trans, probs);
    ggml_tensor* merged = ggml_permute(ctx, weighted, 0, 2, 1, 3);
    ggml_tensor* out = ggml_cont_3d(ctx, merged, n_embd, n_tok, kCfgLanes);
    return linear(ctx, layer.self_o, out);
}

static ggml_tensor*
local_transformer_forward_cached_pair_fixed_pos(
    ggml_context* ctx, ggml_cgraph* gf, const magpietts_transformer& tr, ggml_tensor* x,
    ggml_tensor* pos_emb, DecoderKvCache& cond_cache, DecoderKvCache& uncond_cache,
    LocalTransformerCudaAttentionCache* cuda_attention_cache, ggml_tensor* cache_state,
    int n_past) {
    pos_emb = ggml_cont(ctx, ggml_cast(ctx, pos_emb, GGML_TYPE_F32));
    x = ggml_add(ctx, x, pos_emb);
    for (int il = 0; il < static_cast<int>(tr.layers.size()); ++il) {
        const magpietts_layer& layer = tr.layers[static_cast<size_t>(il)];
        ggml_tensor* residual = x;
        ggml_tensor* cur = layer_norm(ctx, x, layer.norm_self);
        cur = cuda_attention_cache
                  ? local_self_attention_cuda_cached_pair(
                        ctx, tr, layer, *cuda_attention_cache, il, cache_state, cur)
                  : local_self_attention_cached_pair(
                        ctx, gf, tr, layer, cond_cache, uncond_cache, il, n_past, cur);
        x = ggml_add(ctx, residual, cur);

        residual = x;
        cur = layer_norm(ctx, x, layer.norm_ff);
        cur = causal_conv1d(ctx, cur, layer.ff_proj);
        cur = ggml_gelu(ctx, cur);
        cur = causal_conv1d(ctx, cur, layer.ff_out);
        x = ggml_add(ctx, residual, cur);
    }
    return tr.norm_out ? layer_norm(ctx, x, tr.norm_out) : x;
}

static bool
local_transformer_graph_init(
    const magpietts_model& model, bool pair, int codebook_idx, local_transformer_graph_bank& bank,
    local_transformer_graph& graph) {
    const ggml_nvtx::range nvtx_range(
        pair ? "magpietts_local_transformer_pair_graph_init"
             : "magpietts_local_transformer_graph_init");
    graph.reset();

    const auto& h = model.hparams;
    if (codebook_idx < 0 || codebook_idx >= h.stacked_audio_codebooks()) {
        fprintf(stderr, "invalid local-transformer codebook index: %d\n", codebook_idx);
        return false;
    }
    if (codebook_idx >= (int)model.audio_embeddings.size() ||
        codebook_idx >= (int)model.lt_out_w.size() || codebook_idx >= (int)model.lt_out_b.size()) {
        fprintf(stderr, "local-transformer tensors missing for codebook index %d\n", codebook_idx);
        return false;
    }
    if (model.local.kernel != 1) {
        fprintf(stderr, "local-transformer KV cache requires kernel size 1\n");
        return false;
    }

    graph.ctx = new_graph_context();
    if (!graph.ctx) {
        fprintf(stderr, "failed to allocate local-transformer graph context\n");
        return false;
    }
    graph.gf = ggml_new_graph_custom(graph.ctx, MAGPIETTS_MAX_NODES, false);
    graph.codebook_idx = codebook_idx;
    graph.seq_len = 1;
    graph.pair = pair;
    const bool cuda_cached_attention = pair && magpietts_backend_is_cuda(model.backend);

    if (cuda_cached_attention) {
        graph.cache_state =
            bank.pair_cuda_attention_cache.cache_states[static_cast<size_t>(codebook_idx)];
    }

    if (!model.local.pos_emb || model.local.pos_emb->ne[0] != model.local.n_embd ||
        model.local.pos_emb->ne[1] <= codebook_idx) {
        fprintf(
            stderr,
            "local-transformer positional embedding shape is incompatible with seq_len=%d\n",
            codebook_idx + 1);
        graph.reset();
        return false;
    }

    {
        const ggml_nvtx::range nvtx_build(
            pair ? "magpietts_build_local_transformer_pair_graph"
                 : "magpietts_build_local_transformer_graph");

        ggml_tensor* input_cond = nullptr;
        ggml_tensor* input_uncond = nullptr;
        if (codebook_idx == 0) {
            graph.dec_cond = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, h.n_embd, 1);
            ggml_set_name(
                graph.dec_cond, pair ? "magpietts_local_transformer_dec_cond"
                                     : "magpietts_local_transformer_dec_last");
            ggml_set_input(graph.dec_cond);
            input_cond = graph.dec_cond;
            if (pair) {
                graph.dec_uncond = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, h.n_embd, 1);
                ggml_set_name(graph.dec_uncond, "magpietts_local_transformer_dec_uncond");
                ggml_set_input(graph.dec_uncond);
                input_uncond = graph.dec_uncond;
            }
        } else {
            const std::string name = "magpietts_local_transformer_prev_code";
            graph.prev_token = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_I32, 1);
            ggml_set_name(graph.prev_token, name.c_str());
            ggml_set_input(graph.prev_token);
            ggml_tensor* emb = ggml_get_rows(
                graph.ctx, model.audio_embeddings[codebook_idx - 1], graph.prev_token);
            input_cond = emb;
            if (pair)
                input_uncond = emb;
        }

        graph.pos_emb = ggml_view_2d(
            graph.ctx, model.local.pos_emb, model.local.n_embd, 1, model.local.pos_emb->nb[1],
            (size_t)codebook_idx * model.local.pos_emb->nb[1]);
        ggml_set_name(graph.pos_emb, "magpietts_local_transformer_pos_emb");

        if (pair) {
            const int pair_dim = cuda_cached_attention ? 1 : 2;
            ggml_tensor* pair_input = ggml_concat(graph.ctx, input_cond, input_uncond, pair_dim);
            ggml_tensor* cur_pair =
                model.lt_in_w ? linear(graph.ctx, model.lt_in_w, pair_input, model.lt_in_b)
                              : pair_input;
            ggml_tensor* out_pair = local_transformer_forward_cached_pair_fixed_pos(
                graph.ctx, graph.gf, model.local, cur_pair, graph.pos_emb, bank.cond_cache,
                bank.uncond_cache,
                cuda_cached_attention ? &bank.pair_cuda_attention_cache : nullptr,
                graph.cache_state, codebook_idx);
            ggml_tensor* logits_pair = linear(
                graph.ctx, model.lt_out_w[codebook_idx], out_pair, model.lt_out_b[codebook_idx]);
            logits_pair = ggml_cont(graph.ctx, ggml_cast(graph.ctx, logits_pair, GGML_TYPE_F32));
            graph.logits_cond =
                ggml_view_2d(graph.ctx, logits_pair, h.audio_vocab_size, 1, logits_pair->nb[1], 0);
            graph.logits_uncond = ggml_view_2d(
                graph.ctx, logits_pair, h.audio_vocab_size, 1, logits_pair->nb[1],
                cuda_cached_attention ? logits_pair->nb[1] : logits_pair->nb[2]);
            ggml_set_name(graph.logits_cond, "magpietts_local_transformer_logits_cond");
            ggml_set_name(graph.logits_uncond, "magpietts_local_transformer_logits_uncond");
            ggml_set_output(graph.logits_cond);
            ggml_set_output(graph.logits_uncond);
            ggml_build_forward_expand(graph.gf, graph.logits_cond);
            ggml_build_forward_expand(graph.gf, graph.logits_uncond);
        } else {
            ggml_tensor* cur_cond =
                model.lt_in_w ? linear(graph.ctx, model.lt_in_w, input_cond, model.lt_in_b)
                              : input_cond;
            ggml_tensor* out_cond = local_transformer_forward_cached_fixed_pos(
                graph.ctx, graph.gf, model.local, cur_cond, graph.pos_emb, bank.single_cache,
                codebook_idx);
            graph.logits_cond = linear(
                graph.ctx, model.lt_out_w[codebook_idx], out_cond, model.lt_out_b[codebook_idx]);
            graph.logits_cond =
                ggml_cont(graph.ctx, ggml_cast(graph.ctx, graph.logits_cond, GGML_TYPE_F32));
            ggml_set_name(graph.logits_cond, "magpietts_local_transformer_logits");
            ggml_set_output(graph.logits_cond);
            ggml_build_forward_expand(graph.gf, graph.logits_cond);
        }
        tag_graph_first_node(graph.gf);
    }

    graph.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!graph.allocr) {
        fprintf(stderr, "failed to create local-transformer graph allocator\n");
        graph.reset();
        return false;
    }
    {
        const ggml_nvtx::range nvtx_alloc("magpietts_local_transformer_graph_alloc");
        if (!ggml_gallocr_alloc_graph(graph.allocr, graph.gf)) {
            fprintf(stderr, "failed to allocate local-transformer graph tensors\n");
            graph.reset();
            return false;
        }
    }

    graph.logits_cond_data.resize(h.audio_vocab_size);
    if (pair) {
        graph.logits_uncond_data.resize(h.audio_vocab_size);
    }
    return true;
}

static bool
local_transformer_graph_eval(
    const magpietts_model& model, local_transformer_graph& graph,
    const std::vector<float>& cond_hidden, const std::vector<float>& uncond_hidden,
    const std::vector<int32_t>& prev_codes, int threads, std::vector<float>& cond_logits,
    std::vector<float>* uncond_logits, MagpiePinnedHostScratch& transfer_staging) {
    const ggml_nvtx::range nvtx_range(
        graph.pair ? "magpietts_local_transformer_pair_graph_eval"
                   : "magpietts_local_transformer_graph_eval");
    const auto& h = model.hparams;

    if (!graph.ctx || !graph.gf || !graph.allocr || !graph.logits_cond ||
        (graph.codebook_idx == 0 && !graph.dec_cond) ||
        (graph.codebook_idx > 0 && !graph.prev_token)) {
        fprintf(stderr, "local-transformer graph is not initialized\n");
        return false;
    }
    if ((int)prev_codes.size() != graph.codebook_idx) {
        fprintf(
            stderr, "local-transformer graph for codebook %d got %zu previous codes\n",
            graph.codebook_idx, prev_codes.size());
        return false;
    }
    if (cond_hidden.size() != (size_t)h.n_embd) {
        fprintf(
            stderr, "local-transformer conditional hidden size %zu does not match n_embd=%d\n",
            cond_hidden.size(), h.n_embd);
        return false;
    }
    if (graph.pair && uncond_hidden.size() != (size_t)h.n_embd) {
        fprintf(
            stderr, "local-transformer unconditional hidden size %zu does not match n_embd=%d\n",
            uncond_hidden.size(), h.n_embd);
        return false;
    }

    {
        const ggml_nvtx::range nvtx_inputs("magpietts_local_transformer_graph_set_inputs");
        if (graph.codebook_idx == 0) {
            magpietts_backend_tensor_set_staged(
                model, transfer_staging, graph.dec_cond, cond_hidden.data(), 0,
                cond_hidden.size() * sizeof(float));
            if (graph.pair) {
                magpietts_backend_tensor_set_staged(
                    model, transfer_staging, graph.dec_uncond, uncond_hidden.data(), 0,
                    uncond_hidden.size() * sizeof(float));
            }
        } else {
            ggml_backend_tensor_set(
                graph.prev_token, &prev_codes.back(), 0, sizeof(prev_codes.back()));
        }
    }

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, threads);
    }

    ggml_status status = GGML_STATUS_FAILED;
    {
        const ggml_nvtx::range nvtx_compute("magpietts_local_transformer_graph_compute");
        status = ggml_backend_graph_compute(model.backend, graph.gf);
    }
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(
            stderr, "local-transformer graph compute failed: %s\n", ggml_status_to_string(status));
        return false;
    }

    const size_t off = 0;
    {
        const ggml_nvtx::range nvtx_outputs("magpietts_local_transformer_graph_get_outputs");
        magpietts_backend_tensor_get_staged(
            model, transfer_staging, graph.logits_cond, graph.logits_cond_data.data(), off,
            graph.logits_cond_data.size() * sizeof(float));
        cond_logits = graph.logits_cond_data;

        if (graph.pair && uncond_logits) {
            magpietts_backend_tensor_get_staged(
                model, transfer_staging, graph.logits_uncond, graph.logits_uncond_data.data(), off,
                graph.logits_uncond_data.size() * sizeof(float));
            *uncond_logits = graph.logits_uncond_data;
        }
    }
    return true;
}

#if defined(MAGPIETTS_CUDA_SAMPLING)
static bool
local_transformer_graph_eval_cuda(
    const magpietts_model& model, local_transformer_graph& graph, const ggml_tensor* cond_hidden,
    const ggml_tensor* uncond_hidden, int prev_code_count, int threads,
    magpietts_cuda_sample_request& cuda_sample, int codebook_idx) {
    const ggml_nvtx::range nvtx_range(
        graph.pair ? "magpietts_local_transformer_pair_graph_eval_cuda"
                   : "magpietts_local_transformer_graph_eval_cuda");

    if (!graph.ctx || !graph.gf || !graph.allocr || !graph.logits_cond ||
        (graph.codebook_idx == 0 && !graph.dec_cond) ||
        (graph.codebook_idx > 0 && !graph.prev_token)) {
        fprintf(stderr, "local-transformer graph is not initialized\n");
        return false;
    }
    if (prev_code_count != graph.codebook_idx) {
        fprintf(
            stderr, "local-transformer graph for codebook %d got %d previous codes\n",
            graph.codebook_idx, prev_code_count);
        return false;
    }
    if (!cond_hidden) {
        fprintf(stderr, "CUDA local-transformer eval requires conditional hidden tensor\n");
        return false;
    }
    if (graph.pair && !uncond_hidden) {
        fprintf(stderr, "CUDA local-transformer pair eval requires unconditional hidden tensor\n");
        return false;
    }
    if (!cuda_sample.sampler) {
        fprintf(stderr, "CUDA local-transformer eval requires a CUDA sampler\n");
        return false;
    }
    const bool building_sequence =
        magpietts_cuda_sampler_sequence_build_active(cuda_sample.sampler);

    {
        const ggml_nvtx::range nvtx_inputs("magpietts_local_transformer_graph_set_device_inputs");
        if (graph.codebook_idx == 0) {
            if (building_sequence) {
                char error[256] = {};
                if (!magpietts_cuda_sampler_sequence_add_device_copy(
                        cuda_sample.sampler, cond_hidden->data, graph.dec_cond->data,
                        ggml_nbytes(graph.dec_cond), error, sizeof(error))) {
                    fprintf(
                        stderr, "CUDA local-transformer conditional input node failed: %s\n",
                        error[0] ? error : "unknown error");
                    return false;
                }
            } else {
                ggml_backend_tensor_copy_async(
                    model.backend, model.backend, cond_hidden, graph.dec_cond);
            }
            if (graph.pair) {
                if (building_sequence) {
                    char error[256] = {};
                    if (!magpietts_cuda_sampler_sequence_add_device_copy(
                            cuda_sample.sampler, uncond_hidden->data, graph.dec_uncond->data,
                            ggml_nbytes(graph.dec_uncond), error, sizeof(error))) {
                        fprintf(
                            stderr, "CUDA local-transformer unconditional input node failed: %s\n",
                            error[0] ? error : "unknown error");
                        return false;
                    }
                } else {
                    ggml_backend_tensor_copy_async(
                        model.backend, model.backend, uncond_hidden, graph.dec_uncond);
                }
            }
        } else {
            if (!graph.prev_token->data) {
                fprintf(
                    stderr, "local-transformer previous-token tensor is not allocated on device\n");
                return false;
            }
            char error[256] = {};
            if (!magpietts_cuda_copy_sampled_code_to_device(
                    cuda_sample.sampler, prev_code_count - 1, graph.prev_token->data, error,
                    sizeof(error))) {
                fprintf(
                    stderr, "CUDA local-transformer previous-token copy failed: %s\n",
                    error[0] ? error : "unknown error");
                return false;
            }
        }
    }

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, threads);
    }

    ggml_status status = GGML_STATUS_FAILED;
    {
        const ggml_nvtx::range nvtx_compute("magpietts_local_transformer_graph_compute_cuda");
        if (building_sequence) {
            void* graph_template = ggml_backend_cuda_get_graph_template(model.backend, graph.gf);
            char error[256] = {};
            if (!graph_template || !magpietts_cuda_sampler_sequence_add_ggml_graph(
                                       cuda_sample.sampler, graph_template, error, sizeof(error))) {
                fprintf(
                    stderr, "CUDA local-transformer child graph is unavailable: %s\n",
                    error[0] ? error : "GGML graph has not completed warm-up");
                return false;
            }
            status = GGML_STATUS_SUCCESS;
        } else {
            status = ggml_backend_graph_compute_async(model.backend, graph.gf);
        }
    }
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(
            stderr, "local-transformer graph compute failed: %s\n", ggml_status_to_string(status));
        return false;
    }

    if (!graph.logits_cond || !graph.logits_cond->data) {
        fprintf(stderr, "CUDA local-transformer conditional logits are not on device\n");
        return false;
    }
    if (graph.pair && (!graph.logits_uncond || !graph.logits_uncond->data)) {
        fprintf(stderr, "CUDA local-transformer unconditional logits are not on device\n");
        return false;
    }

    const size_t off = 0;
    const float* logits_cond = (const float*)graph.logits_cond->data + off;
    const float* logits_uncond =
        graph.pair ? (const float*)graph.logits_uncond->data + off : nullptr;
    char error[256] = {};
    const bool ok = magpietts_cuda_sample_codebooks_device_configured(
        cuda_sample.sampler, logits_cond, logits_uncond, 1, model.hparams.audio_vocab_size,
        model.hparams.audio_codebook_size, model.hparams.audio_eos_id, codebook_idx, codebook_idx,
        error, sizeof(error));
    if (!ok) {
        fprintf(
            stderr, "CUDA local-transformer sampling failed: %s\n",
            error[0] ? error : "unknown error");
        return false;
    }
    return true;
}
#endif

static bool
local_transformer_graph_bank_eval(
    const magpietts_model& model, local_transformer_graph_bank& bank, bool use_cfg,
    const std::vector<float>& cond_hidden, const std::vector<float>& uncond_hidden,
    const std::vector<int32_t>& prev_codes, int codebook_idx, int threads,
    std::vector<float>& cond_logits, std::vector<float>* uncond_logits,
    MagpiePinnedHostScratch& transfer_staging) {
    std::vector<local_transformer_graph>& graphs = use_cfg ? bank.pair_graphs : bank.single_graphs;
    if ((int)graphs.size() <= codebook_idx) {
        graphs.resize((size_t)codebook_idx + 1);
    }

    local_transformer_graph& graph = graphs[(size_t)codebook_idx];
    if (!graph.ctx || !graph.gf || !graph.allocr || graph.codebook_idx != codebook_idx ||
        graph.pair != use_cfg) {
        if (!local_transformer_graph_init(model, use_cfg, codebook_idx, bank, graph)) {
            return false;
        }
    }

    return local_transformer_graph_eval(
        model, graph, cond_hidden, uncond_hidden, prev_codes, threads, cond_logits,
        use_cfg ? uncond_logits : nullptr, transfer_staging);
}

#if defined(MAGPIETTS_CUDA_SAMPLING)
static bool
local_transformer_graph_bank_eval_cuda(
    const magpietts_model& model, local_transformer_graph_bank& bank, bool use_cfg,
    const magpietts_backend_tensor& cond_hidden, const magpietts_backend_tensor& uncond_hidden,
    int prev_code_count, int codebook_idx, int threads,
    magpietts_cuda_sample_request& cuda_sample) {
    std::vector<local_transformer_graph>& graphs = use_cfg ? bank.pair_graphs : bank.single_graphs;
    if ((int)graphs.size() <= codebook_idx) {
        graphs.resize((size_t)codebook_idx + 1);
    }

    local_transformer_graph& graph = graphs[(size_t)codebook_idx];
    if (!graph.ctx || !graph.gf || !graph.allocr || graph.codebook_idx != codebook_idx ||
        graph.pair != use_cfg) {
        if (!local_transformer_graph_init(model, use_cfg, codebook_idx, bank, graph)) {
            return false;
        }
    }

    return local_transformer_graph_eval_cuda(
        model, graph, cond_hidden.tensor, use_cfg ? uncond_hidden.tensor : nullptr, prev_code_count,
        threads, cuda_sample, codebook_idx);
}
#endif

bool
magpietts_stack_forced_code_frames(
    const std::vector<std::vector<int32_t>>& raw_frames, size_t first_frame,
    const magpietts_hparams& h, std::vector<int32_t>& stacked_codes) {
    stacked_codes.clear();
    if (h.audio_codebooks < 1 || h.frame_stacking_factor < 1) {
        fprintf(
            stderr, "invalid forced-code layout: codebooks=%d frame_stacking_factor=%d\n",
            h.audio_codebooks, h.frame_stacking_factor);
        return false;
    }

    const int64_t stacked_count =
        static_cast<int64_t>(h.audio_codebooks) * h.frame_stacking_factor;
    if (stacked_count > std::numeric_limits<int32_t>::max()) {
        fprintf(
            stderr, "forced-code layout has too many stacked codebooks: %lld\n",
            static_cast<long long>(stacked_count));
        return false;
    }

    const size_t lane_count = static_cast<size_t>(h.frame_stacking_factor);
    if (raw_frames.size() % lane_count != 0) {
        fprintf(
            stderr,
            "incomplete forced-code input: %zu raw frame(s) cannot be grouped into %zu-frame "
            "stacks\n",
            raw_frames.size(), lane_count);
        return false;
    }
    if (first_frame % lane_count != 0) {
        fprintf(
            stderr, "forced-code stack starts at unaligned raw frame %zu (stacking factor %zu)\n",
            first_frame, lane_count);
        return false;
    }
    if (first_frame > raw_frames.size() || raw_frames.size() - first_frame < lane_count) {
        const size_t available =
            first_frame < raw_frames.size() ? raw_frames.size() - first_frame : 0;
        fprintf(
            stderr,
            "incomplete forced-code input at frame %zu: found %zu consecutive frame(s), "
            "expected %zu\n",
            first_frame, available, lane_count);
        return false;
    }

    stacked_codes.reserve(static_cast<size_t>(stacked_count));
    for (size_t lane = 0; lane < lane_count; ++lane) {
        const auto& frame = raw_frames[first_frame + lane];
        if (frame.size() != static_cast<size_t>(h.audio_codebooks)) {
            fprintf(
                stderr, "forced-code frame %zu has %zu codebooks, expected %d\n",
                first_frame + lane, frame.size(), h.audio_codebooks);
            stacked_codes.clear();
            return false;
        }
        stacked_codes.insert(stacked_codes.end(), frame.begin(), frame.end());
    }
    return true;
}

static bool
sample_local_codebooks_impl(
    const magpietts_model& model, const std::vector<float>& cond_hidden,
    const std::vector<float>& uncond_hidden, bool use_cfg, float cfg_scale, float temperature,
    int top_k, bool forbid_audio_eos, int threads, local_transformer_graph_bank& local_graphs,
    MagpiePinnedHostScratch& transfer_staging, std::mt19937& rng, std::vector<int32_t>& codes,
    std::vector<int32_t>& argmax_codes, const LocalCodebookLogitDump* logit_dump,
    const std::vector<int32_t>* forced_codes) {
    const ggml_nvtx::range nvtx_range("magpietts_sample_local_codebooks");
    const auto& h = model.hparams;
    codes.clear();
    argmax_codes.clear();
    const int32_t stacked_codebooks = h.stacked_audio_codebooks();
    if (stacked_codebooks < 1) {
        fprintf(stderr, "invalid stacked codebook count: %d\n", stacked_codebooks);
        return false;
    }
    if (forced_codes && forced_codes->size() != static_cast<size_t>(stacked_codebooks)) {
        fprintf(
            stderr, "forced-code input has %zu stacked codebooks, expected %d\n",
            forced_codes->size(), stacked_codebooks);
        return false;
    }
    if (!local_graphs.beginFrame(model, use_cfg)) {
        return false;
    }
    std::vector<int32_t> prev;
    for (int c = 0; c < stacked_codebooks; ++c) {
        std::vector<float> logits;
        if (use_cfg) {
            std::vector<float> uncond;
            if (!local_transformer_graph_bank_eval(
                    model, local_graphs, true, cond_hidden, uncond_hidden, prev, c, threads, logits,
                    &uncond, transfer_staging)) {
                return false;
            }
            for (int i = 0; i < h.audio_vocab_size; ++i) {
                logits[i] = cfg_scale * logits[i] + (1.0f - cfg_scale) * uncond[i];
            }
        } else if (!local_transformer_graph_bank_eval(
                       model, local_graphs, false, cond_hidden, uncond_hidden, prev, c, threads,
                       logits, nullptr, transfer_staging)) {
            return false;
        }
        const int sampled = MagpieCodebookSampler::sampleFromLogits(
            logits, h, temperature, top_k, rng, forbid_audio_eos);
        const int greedy = MagpieCodebookSampler::argmaxFromLogits(logits, h, forbid_audio_eos);
        dump_local_codebook_logits(logit_dump, c, sampled, greedy, logits);
        const int emitted = forced_codes ? (*forced_codes)[c] : sampled;
        codes.push_back(emitted);
        argmax_codes.push_back(greedy);
        prev.push_back(emitted);
    }
    return true;
}

#if defined(MAGPIETTS_CUDA_SAMPLING)
static bool
sample_local_codebooks_cuda_impl(
    const magpietts_model& model, const magpietts_backend_tensor& cond_hidden,
    const magpietts_backend_tensor& uncond_hidden, bool use_cfg, float cfg_scale, float temperature,
    int top_k, bool forbid_audio_eos, int threads, local_transformer_graph_bank& local_graphs,
    magpietts_cuda_sampler* cuda_sampler, uint64_t seed, int frame_index,
    std::vector<int32_t>& codes, std::vector<int32_t>& argmax_codes) {
    const ggml_nvtx::range nvtx_range("magpietts_sample_local_codebooks_cuda");
    const auto& h = model.hparams;
    if (!local_graphs.beginFrame(model, use_cfg)) {
        return false;
    }
    char stream_error[256] = {};
    if (!magpietts_cuda_sampler_bind_stream(
            cuda_sampler, ggml_backend_cuda_get_stream(model.backend), stream_error,
            sizeof(stream_error))) {
        fprintf(
            stderr, "CUDA local-transformer stream binding failed: %s\n",
            stream_error[0] ? stream_error : "unknown error");
        return false;
    }
    codes.clear();
    argmax_codes.clear();
    char error[256] = {};
    if (!magpietts_cuda_sampler_configure(
            cuda_sampler, use_cfg, cfg_scale, temperature, top_k, forbid_audio_eos, seed,
            frame_index, error, sizeof(error))) {
        fprintf(
            stderr, "CUDA local-transformer sampler config failed: %s\n",
            error[0] ? error : "unknown error");
        return false;
    }

    magpietts_cuda_sample_request cuda_sample;
    cuda_sample.sampler = cuda_sampler;
    cuda_sample.use_cfg = use_cfg;
    cuda_sample.cfg_scale = cfg_scale;
    cuda_sample.temperature = temperature;
    cuda_sample.top_k = top_k;
    cuda_sample.forbid_audio_eos = forbid_audio_eos;
    cuda_sample.seed = seed;
    cuda_sample.frame_index = frame_index;
    auto run_chain = [&]() {
        for (int c = 0; c < h.stacked_audio_codebooks(); ++c) {
            if (!local_transformer_graph_bank_eval_cuda(
                    model, local_graphs, use_cfg, cond_hidden, uncond_hidden, c, c, threads,
                    cuda_sample)) {
                return false;
            }
        }
        return true;
    };

    bool chain_ok = false;
    if (magpietts_cuda_sampler_sequence_is_ready(cuda_sampler)) {
        chain_ok = magpietts_cuda_sampler_sequence_launch(cuda_sampler, error, sizeof(error));
    } else if (
        magpietts_cuda_sampler_sequence_is_warm(cuda_sampler) &&
        !magpietts_cuda_sampler_sequence_is_disabled(cuda_sampler)) {
        // Compose the codebook graphs, sampling, and token feedback.
        bool build_ok =
            magpietts_cuda_sampler_sequence_begin_build(cuda_sampler, error, sizeof(error));
        if (build_ok) {
            build_ok = magpietts_cuda_sampler_upload_config(cuda_sampler, error, sizeof(error)) &&
                       run_chain();
        }
        if (build_ok) {
            build_ok = magpietts_cuda_sampler_sequence_finish_build_and_launch(
                cuda_sampler, error, sizeof(error));
        } else {
            magpietts_cuda_sampler_sequence_abort_build(cuda_sampler);
        }
        if (build_ok) {
            fprintf(
                stderr,
                "MagpieTTS local codebook chain: composed one reusable CUDA graph (%d "
                "codebooks)\n",
                h.stacked_audio_codebooks());
            chain_ok = true;
        } else {
            fprintf(
                stderr,
                "warning: CUDA local graph composition unavailable; using async chain: %s\n",
                error[0] ? error : "unknown error");
            magpietts_cuda_sampler_sequence_disable(cuda_sampler);
            chain_ok = magpietts_cuda_sampler_upload_config(cuda_sampler, error, sizeof(error)) &&
                       run_chain();
        }
    } else {
        chain_ok =
            magpietts_cuda_sampler_upload_config(cuda_sampler, error, sizeof(error)) && run_chain();
        if (chain_ok && !magpietts_cuda_sampler_sequence_is_disabled(cuda_sampler)) {
            // Initialize the per-codebook graphs before composing them.
            magpietts_cuda_sampler_sequence_mark_warm(cuda_sampler);
        }
    }
    if (!chain_ok) {
        fprintf(
            stderr, "CUDA local-transformer chain failed: %s\n",
            error[0] ? error : "unknown error");
        return false;
    }
    codes.assign((size_t)h.stacked_audio_codebooks(), 0);
    argmax_codes.assign((size_t)h.stacked_audio_codebooks(), 0);
    error[0] = '\0';
    if (!magpietts_cuda_copy_sampled_codebooks(
            cuda_sampler, h.stacked_audio_codebooks(), codes.data(), argmax_codes.data(), error,
            sizeof(error))) {
        fprintf(
            stderr, "CUDA local-transformer sampled-code host copy failed: %s\n",
            error[0] ? error : "unknown error");
        return false;
    }
    return true;
}
#endif

LocalCodebookSampler::LocalCodebookSampler(const magpietts_model& model, int threads)
    : model_(model), threads_(threads), graph_bank_(std::make_unique<LocalTransformerGraphBank>()) {
}

LocalCodebookSampler::~LocalCodebookSampler() = default;

void
LocalCodebookSampler::setThreads(int threads) {
    threads_ = std::max(1, threads);
}

bool
LocalCodebookSampler::prewarm(bool use_cfg, int passes) {
    const ggml_nvtx::range nvtx_range("magpietts_local_transformer_prewarm");
    const auto& h = model_.hparams;
    const int warmup_passes = std::max(2, passes);
    std::vector<float> cond_hidden((size_t)h.n_embd, 0.0f);
    std::vector<float> uncond_hidden((size_t)h.n_embd, 0.0f);
    std::vector<int32_t> codes;
    std::vector<int32_t> argmax_codes;

    for (int pass = 0; pass < warmup_passes; ++pass) {
        std::mt19937 rng((uint32_t)pass);
        if (!sample_local_codebooks_impl(
                model_, cond_hidden, uncond_hidden, use_cfg, h.cfg_scale, h.temperature, h.top_k,
                false, threads_, *graph_bank_, transfer_staging_, rng, codes, argmax_codes, nullptr,
                nullptr)) {
            return false;
        }
    }
    return true;
}

bool
LocalCodebookSampler::sample(
    const std::vector<float>& cond_hidden, const std::vector<float>& uncond_hidden, bool use_cfg,
    float cfg_scale, float temperature, int top_k, bool forbid_audio_eos, std::mt19937& rng,
    std::vector<int32_t>& codes, std::vector<int32_t>& argmax_codes,
    const LocalCodebookLogitDump* logit_dump, const std::vector<int32_t>* forced_codes) {
    return sample_local_codebooks_impl(
        model_, cond_hidden, uncond_hidden, use_cfg, cfg_scale, temperature, top_k,
        forbid_audio_eos, threads_, *graph_bank_, transfer_staging_, rng, codes, argmax_codes,
        logit_dump, forced_codes);
}

#if defined(MAGPIETTS_CUDA_SAMPLING)
bool
LocalCodebookSampler::sampleCuda(
    const magpietts_backend_tensor& cond_hidden, const magpietts_backend_tensor& uncond_hidden,
    bool use_cfg, float cfg_scale, float temperature, int top_k, bool forbid_audio_eos,
    magpietts_cuda_sampler* cuda_sampler, uint64_t seed, int frame_index,
    std::vector<int32_t>& codes, std::vector<int32_t>& argmax_codes) {
    return sample_local_codebooks_cuda_impl(
        model_, cond_hidden, uncond_hidden, use_cfg, cfg_scale, temperature, top_k,
        forbid_audio_eos, threads_, *graph_bank_, cuda_sampler, seed, frame_index, codes,
        argmax_codes);
}
#endif

}  // namespace nemo_speech::tts
