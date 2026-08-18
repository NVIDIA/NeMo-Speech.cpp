// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "lt.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include "graph.h"
#include "nvtx_utils.h"

namespace nemo_speech::tts {

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
}

bool
LocalTransformerGraphBank::beginFrame(const magpietts_model& model, bool pair) {
    const auto& h = model.hparams;
    if (pair) {
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

        ggml_tensor* cur_cond = nullptr;
        ggml_tensor* cur_uncond = nullptr;
        if (codebook_idx == 0) {
            graph.dec_cond = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, h.n_embd, 1);
            ggml_set_name(
                graph.dec_cond, pair ? "magpietts_local_transformer_dec_cond"
                                     : "magpietts_local_transformer_dec_last");
            ggml_set_input(graph.dec_cond);
            cur_cond = model.lt_in_w
                           ? linear(graph.ctx, model.lt_in_w, graph.dec_cond, model.lt_in_b)
                           : graph.dec_cond;
            if (pair) {
                graph.dec_uncond = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, h.n_embd, 1);
                ggml_set_name(graph.dec_uncond, "magpietts_local_transformer_dec_uncond");
                ggml_set_input(graph.dec_uncond);
                cur_uncond = model.lt_in_w
                                 ? linear(graph.ctx, model.lt_in_w, graph.dec_uncond, model.lt_in_b)
                                 : graph.dec_uncond;
            }
        } else {
            const std::string name = "magpietts_local_transformer_prev_code";
            graph.prev_token = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_I32, 1);
            ggml_set_name(graph.prev_token, name.c_str());
            ggml_set_input(graph.prev_token);
            ggml_tensor* emb = ggml_get_rows(
                graph.ctx, model.audio_embeddings[codebook_idx - 1], graph.prev_token);
            cur_cond = model.lt_in_w ? linear(graph.ctx, model.lt_in_w, emb, model.lt_in_b) : emb;
            if (pair) {
                cur_uncond = cur_cond;
            }
        }

        graph.pos_emb = ggml_view_2d(
            graph.ctx, model.local.pos_emb, model.local.n_embd, 1, model.local.pos_emb->nb[1],
            (size_t)codebook_idx * model.local.pos_emb->nb[1]);
        ggml_set_name(graph.pos_emb, "magpietts_local_transformer_pos_emb");

        DecoderKvCache& cond_cache = pair ? bank.cond_cache : bank.single_cache;
        ggml_tensor* out_cond = local_transformer_forward_cached_fixed_pos(
            graph.ctx, graph.gf, model.local, cur_cond, graph.pos_emb, cond_cache, codebook_idx);
        graph.logits_cond =
            linear(graph.ctx, model.lt_out_w[codebook_idx], out_cond, model.lt_out_b[codebook_idx]);
        graph.logits_cond =
            ggml_cont(graph.ctx, ggml_cast(graph.ctx, graph.logits_cond, GGML_TYPE_F32));
        ggml_set_name(
            graph.logits_cond, pair ? "magpietts_local_transformer_logits_cond"
                                    : "magpietts_local_transformer_logits");
        ggml_set_output(graph.logits_cond);
        ggml_build_forward_expand(graph.gf, graph.logits_cond);

        if (pair) {
            ggml_tensor* out_uncond = local_transformer_forward_cached_fixed_pos(
                graph.ctx, graph.gf, model.local, cur_uncond, graph.pos_emb, bank.uncond_cache,
                codebook_idx);
            graph.logits_uncond = linear(
                graph.ctx, model.lt_out_w[codebook_idx], out_uncond, model.lt_out_b[codebook_idx]);
            graph.logits_uncond =
                ggml_cont(graph.ctx, ggml_cast(graph.ctx, graph.logits_uncond, GGML_TYPE_F32));
            ggml_set_name(graph.logits_uncond, "magpietts_local_transformer_logits_uncond");
            ggml_set_output(graph.logits_uncond);
            ggml_build_forward_expand(graph.gf, graph.logits_uncond);
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

    {
        const ggml_nvtx::range nvtx_inputs("magpietts_local_transformer_graph_set_device_inputs");
        if (graph.codebook_idx == 0) {
            ggml_backend_tensor_copy(cond_hidden, graph.dec_cond);
            if (graph.pair) {
                ggml_backend_tensor_copy(uncond_hidden, graph.dec_uncond);
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
        status = ggml_backend_graph_compute(model.backend, graph.gf);
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

    ggml_backend_synchronize(model.backend);
    const size_t off = 0;
    const float* logits_cond = (const float*)graph.logits_cond->data + off;
    const float* logits_uncond =
        graph.pair ? (const float*)graph.logits_uncond->data + off : nullptr;
    char error[256] = {};
    const bool ok = magpietts_cuda_sample_codebooks_device(
        cuda_sample.sampler, logits_cond, logits_uncond, 1, model.hparams.audio_vocab_size,
        model.hparams.audio_codebook_size, model.hparams.audio_eos_id, cuda_sample.use_cfg,
        cuda_sample.cfg_scale, cuda_sample.temperature, cuda_sample.top_k,
        cuda_sample.forbid_audio_eos, cuda_sample.seed, cuda_sample.frame_index, codebook_idx,
        codebook_idx, error, sizeof(error));
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
    if (!local_graphs.beginFrame(model, use_cfg)) {
        return false;
    }
    codes.clear();
    argmax_codes.clear();
    std::vector<int32_t> prev;
    for (int c = 0; c < h.stacked_audio_codebooks(); ++c) {
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
        const int emitted = forced_codes && (int)forced_codes->size() == h.stacked_audio_codebooks()
                                ? (*forced_codes)[c]
                                : sampled;
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
    codes.clear();
    argmax_codes.clear();
    for (int c = 0; c < h.stacked_audio_codebooks(); ++c) {
        magpietts_cuda_sample_request cuda_sample;
#if defined(MAGPIETTS_CUDA_SAMPLING)
        cuda_sample.sampler = cuda_sampler;
#else
        (void)cuda_sampler;
#endif
        cuda_sample.use_cfg = use_cfg;
        cuda_sample.cfg_scale = cfg_scale;
        cuda_sample.temperature = temperature;
        cuda_sample.top_k = top_k;
        cuda_sample.forbid_audio_eos = forbid_audio_eos;
        cuda_sample.seed = seed;
        cuda_sample.frame_index = frame_index;
        const bool ok = local_transformer_graph_bank_eval_cuda(
            model, local_graphs, use_cfg, cond_hidden, uncond_hidden, c, c, threads, cuda_sample);
        if (!ok) {
            return false;
        }
    }
    codes.assign((size_t)h.stacked_audio_codebooks(), 0);
    argmax_codes.assign((size_t)h.stacked_audio_codebooks(), 0);
    char error[256] = {};
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
