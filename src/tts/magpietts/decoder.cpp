// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "decoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "../../runtime/ggml/runtime.h"
#include "graph.h"
#include "nvtx_utils.h"

namespace nemo_speech::tts {

static bool decoder_eval_impl(
    const magpietts_model& model, const std::vector<float>& text_cond, int text_len,
    const std::vector<std::vector<int32_t>>& audio_codes, int speaker, bool conditional,
    int threads, decoder_result& result, magpietts_cuda_sample_request* cuda_sample,
    const magpietts_backend_tensor* text_cond_device, magpietts_backend_tensor* hidden_out,
    MagpiePinnedHostScratch& output_staging, const magpietts_decoder_attention* attention);
static bool decoder_eval_pair_impl(
    const magpietts_model& model, const std::vector<float>& text_cond, int text_len,
    const std::vector<std::vector<int32_t>>& audio_codes, int speaker, int threads,
    decoder_result& cond_result, decoder_result& uncond_result,
    magpietts_cuda_sample_request* cuda_sample, const magpietts_backend_tensor* text_cond_device,
    magpietts_backend_tensor* cond_hidden_out, magpietts_backend_tensor* uncond_hidden_out,
    MagpiePinnedHostScratch& output_staging, const magpietts_decoder_attention* attention);
static bool decoder_eval_cached_impl(
    const magpietts_model& model, const std::vector<float>& text_cond, int text_len,
    const std::vector<std::vector<int32_t>>& audio_codes, int speaker, bool conditional,
    int threads, DecoderKvCache& kv_state, decoder_result& result,
    magpietts_cuda_sample_request* cuda_sample, const magpietts_backend_tensor* text_cond_device,
    magpietts_backend_tensor* hidden_out, DecoderCrossKvCache* cross_kv,
    MagpiePinnedHostScratch& output_staging, const magpietts_decoder_attention* attention);
static bool decoder_eval_cached_pair_impl(
    const magpietts_model& model, const std::vector<float>& text_cond, int text_len,
    const std::vector<std::vector<int32_t>>& audio_codes, int speaker, int threads,
    DecoderKvCache& cond_kv, DecoderKvCache& uncond_kv, decoder_result& cond_result,
    decoder_result& uncond_result, magpietts_cuda_sample_request* cuda_sample,
    const magpietts_backend_tensor* text_cond_device, magpietts_backend_tensor* cond_hidden_out,
    magpietts_backend_tensor* uncond_hidden_out, DecoderCrossKvCache* cond_cross_kv,
    MagpiePinnedHostScratch& output_staging, const magpietts_decoder_attention* attention);

DecoderKvCache::~DecoderKvCache() {
    reset();
}

DecoderKvCache::DecoderKvCache(DecoderKvCache&& other) noexcept {
    *this = std::move(other);
}

DecoderKvCache&
DecoderKvCache::operator=(DecoderKvCache&& other) noexcept {
    if (this != &other) {
        reset();
        ctx = other.ctx;
        buffer = other.buffer;
        memory_k = other.memory_k;
        memory_v = other.memory_v;
        n_ctx = other.n_ctx;
        n_layers = other.n_layers;
        n_embd = other.n_embd;
        n_tokens = other.n_tokens;
        other.ctx = nullptr;
        other.buffer = nullptr;
        other.memory_k = nullptr;
        other.memory_v = nullptr;
        other.n_ctx = 0;
        other.n_layers = 0;
        other.n_embd = 0;
        other.n_tokens = 0;
    }
    return *this;
}

void
DecoderKvCache::reset() {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    memory_k = nullptr;
    memory_v = nullptr;
    n_ctx = 0;
    n_layers = 0;
    n_embd = 0;
    n_tokens = 0;
}

void
DecoderKvCache::clear() {
    n_tokens = 0;
}

bool
DecoderKvCache::init(const magpietts_model& model) {
    const magpietts_hparams& h = model.hparams;
    return init(model.backend, h.n_dec_layer, h.n_ctx, h.n_embd, "decoder");
}

bool
DecoderKvCache::init(
    ggml_backend_t backend, int layer_count, int context_length, int embedding_dim,
    const char* label) {
    if (ctx) {
        if (n_ctx != context_length || n_layers != layer_count || n_embd != embedding_dim) {
            reset();
        } else {
            return true;
        }
    }

    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "failed to allocate %s KV context\n", label);
        return false;
    }

    const int64_t n_elements = (int64_t)layer_count * context_length * embedding_dim;
    memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(
            stderr, "failed to allocate %s KV buffer on backend %s\n", label,
            ggml_backend_name(backend));
        reset();
        return false;
    }

    n_ctx = context_length;
    n_layers = layer_count;
    n_embd = embedding_dim;
    n_tokens = 0;
    return true;
}

DecoderCrossKvCache::~DecoderCrossKvCache() {
    reset();
}

DecoderCrossKvCache::DecoderCrossKvCache(DecoderCrossKvCache&& other) noexcept {
    *this = std::move(other);
}

DecoderCrossKvCache&
DecoderCrossKvCache::operator=(DecoderCrossKvCache&& other) noexcept {
    if (this != &other) {
        reset();
        ctx = other.ctx;
        buffer = other.buffer;
        memory_k = other.memory_k;
        memory_v = other.memory_v;
        text_len = other.text_len;
        n_layers = other.n_layers;
        n_cross_dim = other.n_cross_dim;
        valid = other.valid;
        other.ctx = nullptr;
        other.buffer = nullptr;
        other.memory_k = nullptr;
        other.memory_v = nullptr;
        other.text_len = 0;
        other.n_layers = 0;
        other.n_cross_dim = 0;
        other.valid = false;
    }
    return *this;
}

void
DecoderCrossKvCache::reset() {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    memory_k = nullptr;
    memory_v = nullptr;
    text_len = 0;
    n_layers = 0;
    n_cross_dim = 0;
    valid = false;
}

void
DecoderCrossKvCache::clear() {
    valid = false;
}

bool
DecoderCrossKvCache::validFor(const magpietts_model& model, int requested_text_len) const {
    const magpietts_hparams& h = model.hparams;
    const int cross_dim = h.n_cross_head * h.n_cross_dhead;
    return valid && ctx && memory_k && memory_v && text_len == requested_text_len &&
           n_layers == h.n_dec_layer && n_cross_dim == cross_dim;
}

bool
DecoderCrossKvCache::init(const magpietts_model& model, int requested_text_len) {
    if (requested_text_len <= 0) {
        fprintf(stderr, "decoder cross KV cache requires positive text length\n");
        return false;
    }

    const magpietts_hparams& h = model.hparams;
    const int cross_dim = h.n_cross_head * h.n_cross_dhead;
    if (cross_dim <= 0 || h.n_dec_layer <= 0) {
        fprintf(
            stderr, "decoder cross KV cache has invalid dimensions: layers=%d cross_dim=%d\n",
            h.n_dec_layer, cross_dim);
        return false;
    }

    if (ctx) {
        if (text_len != requested_text_len || n_layers != h.n_dec_layer ||
            n_cross_dim != cross_dim) {
            reset();
        } else {
            return true;
        }
    }

    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "failed to allocate decoder cross KV context\n");
        return false;
    }

    const int64_t n_elements = (int64_t)h.n_dec_layer * requested_text_len * cross_dim;
    memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    buffer = ggml_backend_alloc_ctx_tensors(ctx, model.backend);
    if (!buffer) {
        fprintf(
            stderr, "failed to allocate decoder cross KV buffer on backend %s\n",
            ggml_backend_name(model.backend));
        reset();
        return false;
    }

    text_len = requested_text_len;
    n_layers = h.n_dec_layer;
    n_cross_dim = cross_dim;
    valid = false;
    return true;
}

static bool
ensure_decoder_cross_kv_cache(
    const magpietts_model& model, const std::vector<float>& text_cond, int text_len, int threads,
    DecoderCrossKvCache* cross_kv, const magpietts_backend_tensor* text_cond_device) {
    if (!cross_kv || !model.decoder.has_cross) {
        return true;
    }
    if (cross_kv->validFor(model, text_len)) {
        return true;
    }
    if (!cross_kv->init(model, text_len)) {
        return false;
    }

    const ggml_nvtx::range nvtx_range("magpietts_decoder_cross_kv_cache_build");
    const magpietts_transformer& tr = model.decoder;
    const int64_t cross_dim = (int64_t)tr.n_cross_head * tr.n_cross_dhead;

    ggml_context* ctx = new_graph_context();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAGPIETTS_MAX_NODES, false);
    std::vector<std::pair<std::string, std::vector<float>>> f32_inputs;
    const bool use_device_text = text_cond_device && text_cond_device->tensor;

    ggml_tensor* memory = nullptr;
    if (use_device_text) {
        memory = text_cond_device->tensor;
    } else {
        memory = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.n_embd, text_len);
        ggml_set_name(memory, "magpietts_decoder_cross_text_cond");
        ggml_set_input(memory);
        f32_inputs.push_back({"magpietts_decoder_cross_text_cond", text_cond});
    }

    for (int il = 0; il < (int)tr.layers.size(); ++il) {
        const magpietts_layer& layer = tr.layers[il];
        if (!layer.has_cross) {
            continue;
        }
        ggml_tensor* mem_norm = layer_norm(ctx, memory, layer.norm_xattn_memory);
        ggml_tensor* kv = linear(ctx, layer.cross_kv, mem_norm);
        ggml_tensor* kcur = ggml_view_2d(ctx, kv, cross_dim, text_len, kv->nb[1], 0);
        ggml_tensor* vcur = ggml_view_2d(
            ctx, kv, cross_dim, text_len, kv->nb[1], (size_t)ggml_element_size(kv) * cross_dim);

        const size_t layer_offset =
            (size_t)il * text_len * cross_dim * ggml_element_size(cross_kv->memory_k);
        ggml_tensor* k_dst =
            ggml_view_1d(ctx, cross_kv->memory_k, text_len * cross_dim, layer_offset);
        ggml_tensor* v_dst =
            ggml_view_1d(ctx, cross_kv->memory_v, text_len * cross_dim, layer_offset);

        ggml_tensor* k_copy = ggml_cpy(ctx, kcur, k_dst);
        const std::string k_name = "magpietts_decoder_cross_kv_copy_k_" + std::to_string(il);
        ggml_set_name(k_copy, k_name.c_str());
        ggml_build_forward_expand(gf, k_copy);

        ggml_tensor* v_copy = ggml_cpy(ctx, vcur, v_dst);
        const std::string v_name = "magpietts_decoder_cross_kv_copy_v_" + std::to_string(il);
        ggml_set_name(v_copy, v_name.c_str());
        ggml_build_forward_expand(gf, v_copy);
    }

    const bool ok = compute_graph(model, ctx, gf, {}, f32_inputs, threads);
    ggml_free(ctx);
    if (!ok) {
        cross_kv->clear();
        return false;
    }
    cross_kv->valid = true;
    return true;
}

static bool
prepare_attention_prior_input(
    ggml_context* ctx, int text_len, const magpietts_decoder_attention* attention,
    std::vector<std::pair<std::string, std::vector<float>>>& f32_inputs, ggml_tensor*& prior) {
    prior = nullptr;
    if (!attention || !attention->prior) {
        return true;
    }
    if ((int)attention->prior->size() != text_len) {
        fprintf(
            stderr, "attention prior length %zu does not match text length %d\n",
            attention->prior->size(), text_len);
        return false;
    }
    prior = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, text_len);
    ggml_set_name(prior, "magpietts_decoder_attention_prior");
    ggml_set_input(prior);
    f32_inputs.push_back({"magpietts_decoder_attention_prior", *attention->prior});
    return true;
}

static bool
read_alignment_outputs(
    const magpietts_model& model, MagpiePinnedHostScratch& output_staging,
    const std::vector<ggml_tensor*>& outputs, int text_len,
    const magpietts_decoder_attention* attention) {
    if (!attention || !attention->alignment_scores) {
        return true;
    }
    attention->alignment_scores->clear();
    if (outputs.empty()) {
        return true;
    }

    std::vector<float> sum((size_t)text_len, 0.0f);
    int count = 0;
    for (ggml_tensor* t : outputs) {
        if (!t || t->ne[0] != text_len || t->ne[1] <= 0) {
            fprintf(stderr, "unexpected cross-attention output shape while reading alignment\n");
            return false;
        }
        const int heads = (int)t->ne[1];
        std::vector<float> tmp((size_t)text_len * heads);
        magpietts_backend_tensor_get_staged(
            model, output_staging, t, tmp.data(), 0, tmp.size() * sizeof(float));
        for (int h = 0; h < heads; ++h) {
            const size_t off = (size_t)h * text_len;
            for (int i = 0; i < text_len; ++i) {
                sum[(size_t)i] += tmp[off + i];
            }
        }
        count += heads;
    }
    if (count <= 0) {
        return true;
    }
    attention->alignment_scores->resize((size_t)text_len);
    for (int i = 0; i < text_len; ++i) {
        (*attention->alignment_scores)[(size_t)i] = sum[(size_t)i] / (float)count;
    }
    return true;
}

namespace {

constexpr int kMagpieCfgLanes = 2;

bool
runtime_layer_selected(const std::vector<int32_t>& layers, int layer) {
    return layers.empty() ||
           std::find(layers.begin(), layers.end(), static_cast<int32_t>(layer)) != layers.end();
}

std::string
runtime_kv_name(int layer) {
    return "magpietts.decoder.runtime.kv." + std::to_string(layer);
}

// Single-token decoder graph with external cross-K/V and persistent self-K/V storage.
class PersistentDecoderModule final : public ggml_runtime::Module {
   public:
    PersistentDecoderModule(
        const magpietts_model& model, const DecoderCrossKvCache& cross_kv, int text_len,
        int cache_len)
        : model_(model), cross_kv_(cross_kv), text_len_(text_len), cache_len_(cache_len) {
        for (int layer = 0; layer < static_cast<int>(model_.decoder.layers.size()); ++layer) {
            if (model_.decoder.layers[layer].has_cross &&
                runtime_layer_selected(model_.decoder.estimate_alignment_from_layers, layer)) {
                ++alignment_count_;
            }
        }
    }

    void define_tensors(ggml_runtime::Session* session) override {
        std::unordered_set<const ggml_tensor*> seen;
        int imported = 0;
        auto import = [&](ggml_tensor* tensor) {
            if (!tensor || !seen.insert(tensor).second) {
                return;
            }
            session->import_model_tensor(
                "magpietts.decoder.external." + std::to_string(imported++), tensor);
        };

        for (ggml_tensor* embedding : model_.audio_embeddings) import(embedding);
        import(model_.decoder.pos_emb);
        import(model_.decoder.norm_out);
        for (const magpietts_layer& layer : model_.decoder.layers) {
            import(layer.norm_self);
            import(layer.self_qkv);
            import(layer.self_o);
            import(layer.norm_xattn_query);
            import(layer.cross_q);
            import(layer.cross_o);
            import(layer.norm_ff);
            for (ggml_tensor* tensor : layer.ff_proj) import(tensor);
            for (ggml_tensor* tensor : layer.ff_out) import(tensor);
        }
        import(cross_kv_.memory_k);
        import(cross_kv_.memory_v);

        session->model_tensor_container->create_tensor_1d(
            "magpietts.decoder.runtime.slot_ids", GGML_TYPE_I32, kMagpieCfgLanes);
        for (int layer = 0; layer < model_.hparams.n_dec_layer; ++layer) {
            session->model_tensor_container->create_tensor_3d(
                runtime_kv_name(layer), GGML_TYPE_F32,
                static_cast<int64_t>(model_.hparams.n_embd) * cache_len_, kMagpieCfgLanes, 2);
        }
    }

    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag inputs,
        ggml_runtime::TensorContainer* tc) override {
        if (inputs.tensor_count() != 4) {
            throw std::runtime_error("Magpie persistent decoder expects four inputs");
        }
        const auto tokens = inputs.get_tensor(0);
        const auto position = inputs.get_tensor(1);
        const auto cache_meta = inputs.get_tensor(2);
        const auto prior = inputs.get_tensor(3);
        const auto bf_ctx = tc->get_ctx_of_buffer_type(tokens.buft);
        ggml_context* ctx = bf_ctx.ctx;
        const magpietts_hparams& h = model_.hparams;
        const magpietts_transformer& tr = model_.decoder;

        ggml_tensor* audio = nullptr;
        for (int codebook = 0; codebook < h.stacked_audio_codebooks(); ++codebook) {
            ggml_tensor* token = ggml_view_1d(
                ctx, tokens.tensor, 1, static_cast<size_t>(codebook) * sizeof(int32_t));
            ggml_tensor* embedding = ggml_get_rows(ctx, model_.audio_embeddings[codebook], token);
            audio = audio ? ggml_add(ctx, audio, embedding) : embedding;
        }
        audio = ggml_scale(ctx, audio, 1.0f / static_cast<float>(h.stacked_audio_codebooks()));
        audio = ggml_add(ctx, audio, ggml_get_rows(ctx, tr.pos_emb, position.tensor));
        // Store CFG lanes as projection columns for two-column MMVF.
        ggml_tensor* x = ggml_concat(ctx, audio, audio, 1);  // [E,CFG=2]

        const int64_t d_head = tr.n_embd / tr.n_head;
        auto slots = session->model_tensor_container->get_tensor_by_name(
            "magpietts.decoder.runtime.slot_ids");
        std::vector<ggml_tensor*> alignment_outputs;

        for (int layer_index = 0; layer_index < static_cast<int>(tr.layers.size()); ++layer_index) {
            const magpietts_layer& layer = tr.layers[layer_index];
            ggml_tensor* residual = x;
            ggml_tensor* cur = layer_norm(ctx, x, layer.norm_self);
            ggml_tensor* qkv = linear(ctx, layer.self_qkv, cur);
            const size_t element = ggml_element_size(qkv);
            auto split_heads = [&](size_t offset) {
                // Expose Q/K/V as [d_head,1,n_head,B] views without staging copies.
                return ggml_view_4d(
                    ctx, qkv, d_head, 1, tr.n_head, kMagpieCfgLanes, qkv->nb[1],
                    static_cast<size_t>(d_head) * element, qkv->nb[1], offset);
            };
            ggml_tensor* q = split_heads(0);
            ggml_tensor* k = split_heads(static_cast<size_t>(tr.n_embd) * element);
            ggml_tensor* v = split_heads(static_cast<size_t>(2 * tr.n_embd) * element);
            auto kv =
                session->model_tensor_container->get_tensor_by_name(runtime_kv_name(layer_index));
            ggml_tensor* heads = ggml_fused_attn_cached(
                ctx, q, k, v, nullptr, kv.tensor, slots.tensor, cache_meta.tensor, cache_len_,
                1.0f / std::sqrt(static_cast<float>(d_head)), true);
            ggml_tensor* merged = ggml_reshape_2d(
                ctx, ggml_permute(ctx, heads, 0, 2, 1, 3), tr.n_embd, kMagpieCfgLanes);
            x = ggml_add(ctx, residual, linear(ctx, layer.self_o, merged));

            // Text cross-attention applies only to the conditional lane.
            if (tr.has_cross && layer.has_cross) {
                ggml_tensor* cond = ggml_view_2d(ctx, x, tr.n_embd, 1, x->nb[1], 0);
                ggml_tensor* uncond = ggml_view_2d(ctx, x, tr.n_embd, 1, x->nb[1], x->nb[1]);
                ggml_tensor* cross_in = layer_norm(ctx, cond, layer.norm_xattn_query);
                ggml_tensor* last_attn = nullptr;
                const bool apply_prior =
                    tr.apply_attention_prior &&
                    runtime_layer_selected(tr.apply_prior_to_layers, layer_index);
                const bool collect =
                    runtime_layer_selected(tr.estimate_alignment_from_layers, layer_index);
                ggml_tensor* cross = cross_attention_cached(
                    ctx, tr, layer, cross_kv_, layer_index, cross_in,
                    apply_prior ? prior.tensor : nullptr, collect ? &last_attn : nullptr, true);
                cond = ggml_add(ctx, cond, cross);
                x = ggml_concat(ctx, cond, uncond, 1);
                if (last_attn) {
                    alignment_outputs.push_back(last_attn);
                }
            }

            residual = x;
            cur = layer_norm(ctx, x, layer.norm_ff);
            cur = causal_conv1d(ctx, cur, layer.ff_proj);
            cur = ggml_gelu(ctx, cur);
            cur = causal_conv1d(ctx, cur, layer.ff_out);
            x = ggml_add(ctx, residual, cur);
        }
        if (tr.norm_out)
            x = layer_norm(ctx, x, tr.norm_out);

        ggml_tensor* cond =
            ggml_cont_2d(ctx, ggml_view_2d(ctx, x, tr.n_embd, 1, x->nb[1], 0), tr.n_embd, 1);
        ggml_tensor* uncond =
            ggml_cont_2d(ctx, ggml_view_2d(ctx, x, tr.n_embd, 1, x->nb[1], x->nb[1]), tr.n_embd, 1);
        ggml_set_name(cond, "magpietts_decoder_runtime_hidden_cond");
        ggml_set_name(uncond, "magpietts_decoder_runtime_hidden_uncond");
        ggml_runtime::TensorBag outputs;
        outputs.add_tensor({cond, bf_ctx.buft});
        outputs.add_tensor({uncond, bf_ctx.buft});
        if (alignment_outputs.size() != alignment_count_) {
            throw std::runtime_error("Magpie persistent decoder alignment topology changed");
        }
        if (!alignment_outputs.empty()) {
            ggml_tensor* sum = nullptr;
            for (ggml_tensor* alignment : alignment_outputs) {
                ggml_tensor* per_layer = alignment;
                if (tr.n_cross_head > 1) {
                    per_layer = ggml_reshape_1d(
                        ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, alignment))),
                        text_len_);
                } else {
                    per_layer = ggml_reshape_1d(ctx, alignment, text_len_);
                }
                sum = sum ? ggml_add(ctx, sum, per_layer) : per_layer;
            }
            ggml_tensor* mean = ggml_scale(
                ctx, sum, 1.0f / static_cast<float>(alignment_outputs.size() * tr.n_cross_head));
            ggml_set_name(mean, "magpietts_decoder_runtime_alignment_mean");
            outputs.add_tensor({mean, bf_ctx.buft});
        }
        return outputs;
    }

    void set_data(ggml_runtime::Session* session) override {
        const int32_t slots[kMagpieCfgLanes] = {0, 1};
        auto slot_ids = session->model_tensor_container->get_tensor_by_name(
            "magpietts.decoder.runtime.slot_ids");
        ggml_backend_tensor_set(slot_ids.tensor, slots, 0, sizeof(slots));
        for (int layer = 0; layer < model_.hparams.n_dec_layer; ++layer) {
            auto kv = session->model_tensor_container->get_tensor_by_name(runtime_kv_name(layer));
            ggml_backend_tensor_memset(kv.tensor, 0, 0, ggml_nbytes(kv.tensor));
        }
    }

    size_t alignment_count() const { return alignment_count_; }

   private:
    const magpietts_model& model_;
    const DecoderCrossKvCache& cross_kv_;
    int text_len_ = 0;
    int cache_len_ = 0;
    size_t alignment_count_ = 0;
};

}  // namespace

class MagpieDecoder::PersistentDecoderRuntime {
   public:
    PersistentDecoderRuntime(
        const magpietts_model& model, const DecoderCrossKvCache& cross_kv, int text_len,
        int stacked_position_budget)
        : model_(model), cross_kv_(&cross_kv), text_len_(text_len),
          stacked_position_budget_(stacked_position_budget),
          cache_len_(model.hparams.baked_context_length + stacked_position_budget - 1),
          backend_manager_(ggml_runtime::Params{true, 0, nullptr}, model.backend),
          module_(model, cross_kv, text_len, cache_len_),
          session_(backend_manager_, &module_, nullptr) {
        if (cache_len_ <= 0 || cache_len_ >= model.hparams.n_ctx) {
            throw std::runtime_error("invalid persistent Magpie decoder cache length");
        }
        session_.set_run_cache_capacity(1);
        session_.setup();
    }

    bool matches(
        const DecoderCrossKvCache* cross_kv, int text_len, int stacked_position_budget) const {
        return cross_kv == cross_kv_ && text_len == text_len_ &&
               stacked_position_budget == stacked_position_budget_;
    }

    bool sequence_matches(int n_tokens) const { return n_tokens == n_tokens_; }

    void seed(const DecoderKvCache& cond, const DecoderKvCache& uncond) {
        if (cond.n_tokens <= 0 || cond.n_tokens != uncond.n_tokens || cond.n_tokens > cache_len_) {
            throw std::runtime_error("cannot seed persistent decoder from incompatible KV caches");
        }
        ggml_context* ctx = new_graph_context();
        const size_t element = sizeof(float);
        const size_t source_layer_bytes = static_cast<size_t>(cond.n_ctx) * cond.n_embd * element;
        const size_t copy_elements = static_cast<size_t>(cond.n_tokens) * cond.n_embd;
        const size_t destination_token = static_cast<size_t>(cache_len_ - cond.n_tokens);
        for (int layer = 0; layer < cond.n_layers; ++layer) {
            auto arena =
                session_.model_tensor_container->get_tensor_by_name(runtime_kv_name(layer));
            for (int plane = 0; plane < 2; ++plane) {
                ggml_tensor* dst_base = arena.tensor;
                ggml_tensor* cond_src_base = plane == 0 ? cond.memory_k : cond.memory_v;
                ggml_tensor* uncond_src_base = plane == 0 ? uncond.memory_k : uncond.memory_v;
                const ggml_tensor* sources[kMagpieCfgLanes] = {cond_src_base, uncond_src_base};
                for (int lane = 0; lane < kMagpieCfgLanes; ++lane) {
                    ggml_tensor* src = ggml_view_1d(
                        ctx, const_cast<ggml_tensor*>(sources[lane]), copy_elements,
                        static_cast<size_t>(layer) * source_layer_bytes);
                    const size_t dst_offset = static_cast<size_t>(plane) * dst_base->nb[2] +
                                              static_cast<size_t>(lane) * dst_base->nb[1] +
                                              destination_token * cond.n_embd * element;
                    ggml_tensor* dst = ggml_view_1d(ctx, dst_base, copy_elements, dst_offset);
                    ggml_backend_tensor_copy_async(model_.backend, model_.backend, src, dst);
                }
            }
        }
        ggml_backend_synchronize(model_.backend);
        ggml_free(ctx);
        n_tokens_ = cond.n_tokens;
        valid_tokens_ = cond.n_tokens;
        ring_head_ = 0;
    }

    bool eval(
        const std::vector<std::vector<int32_t>>& audio_codes, DecoderKvCache& cond_kv,
        DecoderKvCache& uncond_kv, decoder_result& cond_result, decoder_result& uncond_result,
        magpietts_backend_tensor* cond_hidden_out, magpietts_backend_tensor* uncond_hidden_out,
        const magpietts_decoder_attention* attention) {
        const ggml_nvtx::range nvtx_range("magpietts_persistent_decoder_eval");
        const magpietts_hparams& h = model_.hparams;
        if (!cond_hidden_out || !uncond_hidden_out || !cond_hidden_out->tensor ||
            !uncond_hidden_out->tensor ||
            static_cast<int>(audio_codes.size()) != h.audio_codebooks || audio_codes.empty()) {
            return false;
        }
        const size_t raw_len = audio_codes[0].size();
        if (raw_len == 0 || raw_len % h.frame_stacking_factor != 0)
            return false;
        for (const auto& codes : audio_codes) {
            if (codes.size() != raw_len)
                return false;
        }
        const int total_len =
            h.baked_context_length + static_cast<int>(raw_len / h.frame_stacking_factor);
        if (total_len != n_tokens_ + 1 || n_tokens_ >= cache_len_)
            return false;

        std::vector<int32_t> tokens(static_cast<size_t>(h.stacked_audio_codebooks()));
        const size_t frame_start = raw_len - static_cast<size_t>(h.frame_stacking_factor);
        for (int lane = 0; lane < h.frame_stacking_factor; ++lane) {
            for (int codebook = 0; codebook < h.audio_codebooks; ++codebook) {
                tokens[static_cast<size_t>(codebook + lane * h.audio_codebooks)] =
                    audio_codes[static_cast<size_t>(codebook)][frame_start + lane];
            }
        }
        const int32_t position = n_tokens_;
        // Column-major [CFG lane, {ring head, valid length}]. The fused attention kernel reads
        // only the active suffix while the graph and arena shapes remain constant.
        const int32_t cache_meta[kMagpieCfgLanes * 2] = {
            ring_head_, ring_head_, valid_tokens_, valid_tokens_};
        std::vector<float> log_prior(static_cast<size_t>(text_len_), 0.0f);
        if (attention && attention->prior) {
            if (static_cast<int>(attention->prior->size()) != text_len_)
                return false;
            for (int i = 0; i < text_len_; ++i) {
                log_prior[static_cast<size_t>(i)] =
                    std::log(std::max((*attention->prior)[static_cast<size_t>(i)], 1.0e-20f));
            }
        }

        std::vector<ggml_runtime::Session::Input> inputs = {
            {"magpietts.decoder.runtime.tokens",
             GGML_TYPE_I32,
             tokens.data(),
             {h.stacked_audio_codebooks()}},
            {"magpietts.decoder.runtime.position", GGML_TYPE_I32, &position, {1}},
            {"magpietts.decoder.runtime.cache_meta",
             GGML_TYPE_I32,
             cache_meta,
             {kMagpieCfgLanes, 2}},
            {"magpietts.decoder.runtime.prior", GGML_TYPE_F32, log_prior.data(), {text_len_}}};

        ggml_runtime::DeviceTensor cond_device;
        ggml_runtime::DeviceTensor uncond_device;
        std::vector<float> alignment(static_cast<size_t>(text_len_));
        const bool has_alignment = module_.alignment_count() > 0;
        std::vector<ggml_runtime::Session::Output> outputs(has_alignment ? 3 : 2);
        outputs[0].index = 0;
        outputs[0].device_tensor = &cond_device;
        outputs[1].index = 1;
        outputs[1].device_tensor = &uncond_device;
        if (has_alignment) {
            outputs[2].index = 2;
            outputs[2].host_buffer = alignment.data();
            outputs[2].nbytes = alignment.size() * sizeof(float);
        }
        session_.run(inputs, outputs);
        ggml_backend_tensor_copy_async(
            model_.backend, model_.backend, cond_device.tensor, cond_hidden_out->tensor);
        ggml_backend_tensor_copy_async(
            model_.backend, model_.backend, uncond_device.tensor, uncond_hidden_out->tensor);

        if (attention && attention->alignment_scores) {
            *attention->alignment_scores = alignment;
        }

        ++n_tokens_;
        valid_tokens_ = std::min(cache_len_, valid_tokens_ + 1);
        ring_head_ = (ring_head_ + 1) % cache_len_;
        cond_kv.n_tokens = n_tokens_;
        uncond_kv.n_tokens = n_tokens_;
        cond_result.hidden_last.clear();
        uncond_result.hidden_last.clear();
        return true;
    }

   private:
    const magpietts_model& model_;
    const DecoderCrossKvCache* cross_kv_ = nullptr;
    int text_len_ = 0;
    int stacked_position_budget_ = 0;
    int cache_len_ = 0;
    int n_tokens_ = 0;
    int valid_tokens_ = 0;
    int ring_head_ = 0;
    ggml_runtime::BackendManager backend_manager_;
    PersistentDecoderModule module_;
    ggml_runtime::Session session_;
};

MagpieDecoder::MagpieDecoder(const magpietts_model& model) : model_(model) {}

MagpieDecoder::~MagpieDecoder() = default;

bool
MagpieDecoder::eval(
    const std::vector<float>& text_cond, int text_len,
    const std::vector<std::vector<int32_t>>& audio_codes, int speaker, bool conditional,
    int threads, decoder_result& result, magpietts_cuda_sample_request* cuda_sample,
    const magpietts_backend_tensor* text_cond_device, magpietts_backend_tensor* hidden_out,
    const magpietts_decoder_attention* attention) const {
    return decoder_eval_impl(
        model_, text_cond, text_len, audio_codes, speaker, conditional, threads, result,
        cuda_sample, text_cond_device, hidden_out, output_staging_, attention);
}

bool
MagpieDecoder::evalPair(
    const std::vector<float>& text_cond, int text_len,
    const std::vector<std::vector<int32_t>>& audio_codes, int speaker, int threads,
    decoder_result& cond_result, decoder_result& uncond_result,
    magpietts_cuda_sample_request* cuda_sample, const magpietts_backend_tensor* text_cond_device,
    magpietts_backend_tensor* cond_hidden_out, magpietts_backend_tensor* uncond_hidden_out,
    const magpietts_decoder_attention* attention) const {
    return decoder_eval_pair_impl(
        model_, text_cond, text_len, audio_codes, speaker, threads, cond_result, uncond_result,
        cuda_sample, text_cond_device, cond_hidden_out, uncond_hidden_out, output_staging_,
        attention);
}

bool
MagpieDecoder::evalCached(
    const std::vector<float>& text_cond, int text_len,
    const std::vector<std::vector<int32_t>>& audio_codes, int speaker, bool conditional,
    int threads, DecoderKvCache& kv_state, decoder_result& result,
    magpietts_cuda_sample_request* cuda_sample, const magpietts_backend_tensor* text_cond_device,
    magpietts_backend_tensor* hidden_out, DecoderCrossKvCache* cross_kv,
    const magpietts_decoder_attention* attention) const {
    return decoder_eval_cached_impl(
        model_, text_cond, text_len, audio_codes, speaker, conditional, threads, kv_state, result,
        cuda_sample, text_cond_device, hidden_out, cross_kv, output_staging_, attention);
}

bool
MagpieDecoder::evalCachedPair(
    const std::vector<float>& text_cond, int text_len,
    const std::vector<std::vector<int32_t>>& audio_codes, int speaker, int threads,
    DecoderKvCache& cond_kv, DecoderKvCache& uncond_kv, decoder_result& cond_result,
    decoder_result& uncond_result, int stacked_position_budget,
    magpietts_cuda_sample_request* cuda_sample, const magpietts_backend_tensor* text_cond_device,
    magpietts_backend_tensor* cond_hidden_out, magpietts_backend_tensor* uncond_hidden_out,
    DecoderCrossKvCache* cond_cross_kv, const magpietts_decoder_attention* attention) const {
    const bool persistent_candidate =
        cuda_sample == nullptr && cond_hidden_out != nullptr && uncond_hidden_out != nullptr &&
        cond_cross_kv != nullptr && cond_cross_kv->validFor(model_, text_len) &&
        model_.hparams.dec_kernel == 1 && magpietts_backend_is_cuda(model_.backend) &&
        cond_kv.n_tokens > 0 && cond_kv.n_tokens == uncond_kv.n_tokens;
    if (persistent_candidate) {
        try {
            if (persistent_runtime_ &&
                !persistent_runtime_->matches(cond_cross_kv, text_len, stacked_position_budget)) {
                // Cross-cache address, shape, or request budget changes require a new graph.
                persistent_runtime_.reset();
                cond_kv.clear();
                uncond_kv.clear();
            }
            if (cond_kv.n_tokens > 0 && !persistent_runtime_) {
                persistent_runtime_ = std::make_unique<PersistentDecoderRuntime>(
                    model_, *cond_cross_kv, text_len, stacked_position_budget);
                persistent_runtime_->seed(cond_kv, uncond_kv);
                fprintf(
                    stderr,
                    "MagpieTTS decoder runtime: fixed-shape CUDA graph, CFG batch=2, "
                    "device K/V arena enabled\n");
            }
            if (persistent_runtime_ && !persistent_runtime_->sequence_matches(cond_kv.n_tokens)) {
                // Reuse the graph and reseed the cache suffix.
                persistent_runtime_->seed(cond_kv, uncond_kv);
            }
            if (persistent_runtime_ &&
                persistent_runtime_->eval(
                    audio_codes, cond_kv, uncond_kv, cond_result, uncond_result, cond_hidden_out,
                    uncond_hidden_out, attention)) {
                return true;
            }
            persistent_runtime_.reset();
            cond_kv.clear();
            uncond_kv.clear();
        }
        catch (const std::exception& e) {
            fprintf(stderr, "MagpieTTS persistent decoder failed: %s\n", e.what());
            persistent_runtime_.reset();
            cond_kv.clear();
            uncond_kv.clear();
        }
    }
    return decoder_eval_cached_pair_impl(
        model_, text_cond, text_len, audio_codes, speaker, threads, cond_kv, uncond_kv, cond_result,
        uncond_result, cuda_sample, text_cond_device, cond_hidden_out, uncond_hidden_out,
        cond_cross_kv, output_staging_, attention);
}

bool
MagpieCodebookSampler::runCuda(
    ggml_backend_t backend, const magpietts_hparams& h, magpietts_cuda_sample_request* request,
    const ggml_tensor* logits_cond, const ggml_tensor* logits_uncond, size_t logits_off_floats,
    int codebooks, int codebook_offset) {
    if (!request) {
        return false;
    }
#if defined(MAGPIETTS_CUDA_SAMPLING)
    if (!request->sampler || !logits_cond || !logits_cond->data) {
        fprintf(stderr, "CUDA sampling requested with invalid sampler or logits\n");
        return false;
    }
    request->codes.assign(codebooks, 0);
    request->argmax_codes.assign(codebooks, 0);
    ggml_backend_synchronize(backend);
    const float* cond = (const float*)logits_cond->data + logits_off_floats;
    const float* uncond = logits_uncond && logits_uncond->data
                              ? (const float*)logits_uncond->data + logits_off_floats
                              : nullptr;
    char error[256] = {};
    const bool ok = magpietts_cuda_sample_codebooks(
        request->sampler, cond, uncond, codebooks, h.audio_vocab_size, h.audio_codebook_size,
        h.audio_eos_id, request->use_cfg, request->cfg_scale, request->temperature, request->top_k,
        request->forbid_audio_eos, request->seed, request->frame_index, codebook_offset,
        request->codes.data(), request->argmax_codes.data(), error, sizeof(error));
    if (!ok) {
        fprintf(stderr, "CUDA sampling failed: %s\n", error[0] ? error : "unknown error");
        return false;
    }
    return true;
#else
    (void)backend;
    (void)h;
    (void)logits_cond;
    (void)logits_uncond;
    (void)logits_off_floats;
    (void)codebooks;
    (void)codebook_offset;
    fprintf(stderr, "CUDA sampling was not compiled into this MagpieTTS build\n");
    return false;
#endif
}

static ggml_tensor*
build_audio_embedding(
    ggml_context* ctx, const magpietts_model& model,
    const std::vector<ggml_tensor*>& audio_tok_inputs) {
    ggml_tensor* sum = nullptr;
    for (int c = 0; c < model.hparams.stacked_audio_codebooks(); ++c) {
        ggml_tensor* emb = ggml_get_rows(ctx, model.audio_embeddings[c], audio_tok_inputs[c]);
        sum = sum ? ggml_add(ctx, sum, emb) : emb;
    }
    return ggml_scale(ctx, sum, 1.0f / (float)model.hparams.stacked_audio_codebooks());
}

static bool
stack_audio_codes(
    const std::vector<std::vector<int32_t>>& audio_codes, const magpietts_hparams& h,
    std::vector<std::vector<int32_t>>& stacked) {
    if ((int)audio_codes.size() != h.audio_codebooks || audio_codes.empty() ||
        audio_codes[0].empty() || (int)audio_codes[0].size() % h.frame_stacking_factor != 0) {
        return false;
    }
    const int raw_len = (int)audio_codes[0].size();
    for (const auto& codes : audio_codes) {
        if ((int)codes.size() != raw_len) {
            return false;
        }
    }
    const int stacked_len = raw_len / h.frame_stacking_factor;
    stacked.assign((size_t)h.stacked_audio_codebooks(), std::vector<int32_t>(stacked_len));
    for (int lane = 0; lane < h.frame_stacking_factor; ++lane) {
        for (int codebook = 0; codebook < h.audio_codebooks; ++codebook) {
            auto& dst = stacked[(size_t)(codebook + lane * h.audio_codebooks)];
            const auto& src = audio_codes[(size_t)codebook];
            for (int pos = 0; pos < stacked_len; ++pos) {
                dst[(size_t)pos] = src[(size_t)(pos * h.frame_stacking_factor + lane)];
            }
        }
    }
    return true;
}

static bool
decoder_eval_impl(
    const magpietts_model& model, const std::vector<float>& text_cond, int text_len,
    const std::vector<std::vector<int32_t>>& audio_codes, int speaker, bool conditional,
    int threads, decoder_result& result, magpietts_cuda_sample_request* cuda_sample,
    const magpietts_backend_tensor* text_cond_device, magpietts_backend_tensor* hidden_out,
    MagpiePinnedHostScratch& output_staging, const magpietts_decoder_attention* attention) {
    const ggml_nvtx::range nvtx_range(
        conditional ? "magpietts_decoder_eval_cond" : "magpietts_decoder_eval_uncond");
    const auto& h = model.hparams;
    std::vector<std::vector<int32_t>> stacked_audio;
    if (!stack_audio_codes(audio_codes, h, stacked_audio)) {
        fprintf(stderr, "decoder_eval requires at least one audio token\n");
        return false;
    }
    const int audio_len = (int)stacked_audio[0].size();
    const int total_len = h.baked_context_length + audio_len;

    ggml_context* ctx = new_graph_context();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAGPIETTS_MAX_NODES, false);

    std::vector<ggml_tensor*> audio_tok_inputs(h.stacked_audio_codebooks());
    std::vector<std::pair<std::string, std::vector<int32_t>>> i32_inputs;
    std::vector<std::pair<std::string, std::vector<float>>> f32_inputs;

    ggml_tensor* speaker_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speaker_in, "magpietts_decoder_speaker");
    ggml_set_input(speaker_in);
    i32_inputs.push_back({"magpietts_decoder_speaker", {speaker}});

    for (int c = 0; c < h.stacked_audio_codebooks(); ++c) {
        const std::string name = "magpietts_decoder_audio_tokens_" + std::to_string(c);
        audio_tok_inputs[c] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, audio_len);
        ggml_set_name(audio_tok_inputs[c], name.c_str());
        ggml_set_input(audio_tok_inputs[c]);
        i32_inputs.push_back({name, stacked_audio[c]});
    }

    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total_len);
    ggml_set_name(pos, "magpietts_decoder_positions");
    ggml_set_input(pos);
    i32_inputs.push_back({"magpietts_decoder_positions", positions(total_len)});

    ggml_tensor* ctx_flat = ggml_get_rows(ctx, model.baked_context, speaker_in);
    ggml_tensor* ctx_emb = ggml_reshape_2d(ctx, ctx_flat, h.n_embd, h.baked_context_length);
    if (!conditional) {
        ctx_emb = ggml_scale(ctx, ctx_emb, 0.0f);
    }

    ggml_tensor* audio_emb = build_audio_embedding(ctx, model, audio_tok_inputs);
    ggml_tensor* dec_in = ggml_concat(ctx, ctx_emb, audio_emb, 1);

    ggml_tensor* cond = nullptr;
    if (conditional) {
        if (text_cond_device && text_cond_device->tensor) {
            cond = text_cond_device->tensor;
        } else {
            cond = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, text_len);
            ggml_set_name(cond, "magpietts_decoder_text_cond");
            ggml_set_input(cond);
            f32_inputs.push_back({"magpietts_decoder_text_cond", text_cond});
        }
    }

    ggml_tensor* attn_prior = nullptr;
    std::vector<ggml_tensor*> alignment_outputs;
    if (conditional &&
        !prepare_attention_prior_input(ctx, text_len, attention, f32_inputs, attn_prior)) {
        ggml_free(ctx);
        return false;
    }

    const bool collect_alignment = conditional && attention && attention->alignment_scores;
    ggml_tensor* dec_out = transformer_forward(
        ctx, model.decoder, dec_in, pos, cond, attn_prior,
        collect_alignment ? &alignment_outputs : nullptr);
    dec_out = ggml_cont(ctx, ggml_cast(ctx, dec_out, GGML_TYPE_F32));
    ggml_set_name(dec_out, "magpietts_decoder_out");
    ggml_set_output(dec_out);

    const bool compute_logits = cuda_sample || result.logits_required;
    ggml_tensor* logits = nullptr;
    if (compute_logits) {
        logits = linear(ctx, model.final_proj_w, dec_out, model.final_proj_b);
        logits = ggml_cont(ctx, ggml_cast(ctx, logits, GGML_TYPE_F32));
        ggml_set_name(logits, "magpietts_decoder_logits");
        ggml_set_output(logits);
    }

    ggml_gallocr_t allocr = nullptr;
    const size_t hidden_off = (size_t)h.n_embd * (total_len - 1) * sizeof(float);
    ggml_tensor* hidden_last = nullptr;
    if (hidden_out) {
        hidden_last = ggml_view_2d(ctx, dec_out, h.n_embd, 1, dec_out->nb[1], hidden_off);
        ggml_set_name(hidden_last, "magpietts_decoder_hidden_last");
        ggml_set_output(hidden_last);
    }
    if (logits) {
        ggml_build_forward_expand(gf, logits);
    }
    ggml_build_forward_expand(gf, dec_out);
    if (hidden_last) {
        ggml_build_forward_expand(gf, hidden_last);
    }
    for (ggml_tensor* t : alignment_outputs) {
        ggml_build_forward_expand(gf, t);
    }

    const bool ok = compute_graph(model, ctx, gf, i32_inputs, f32_inputs, threads, &allocr);
    if (!ok) {
        ggml_free(ctx);
        return false;
    }
    if (!read_alignment_outputs(
            model, output_staging, alignment_outputs, text_len,
            conditional ? attention : nullptr)) {
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return false;
    }

    const size_t logits_last_size = (size_t)h.stacked_audio_codebooks() * h.audio_vocab_size;
    const size_t logits_off_floats = logits_last_size * (total_len - 1);
    if (hidden_out && hidden_last) {
        ggml_backend_tensor_copy(hidden_last, hidden_out->tensor);
    }
    if (cuda_sample) {
        const bool sampled = MagpieCodebookSampler::runCuda(
            model.backend, h, cuda_sample, logits, nullptr, logits_off_floats,
            h.stacked_audio_codebooks(), 0);
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return sampled;
    }
    if (hidden_out) {
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return true;
    }

    if (result.logits_required) {
        result.logits_last.resize(logits_last_size);
    }
    result.hidden_last.resize(h.n_embd);
    const size_t logits_off = logits_off_floats * sizeof(float);
    if (result.logits_required) {
        magpietts_backend_tensor_get_staged(
            model, output_staging, logits, result.logits_last.data(), logits_off,
            result.logits_last.size() * sizeof(float));
    }
    magpietts_backend_tensor_get_staged(
        model, output_staging, dec_out, result.hidden_last.data(), hidden_off,
        result.hidden_last.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return true;
}

static bool
decoder_eval_pair_impl(
    const magpietts_model& model, const std::vector<float>& text_cond, int text_len,
    const std::vector<std::vector<int32_t>>& audio_codes, int speaker, int threads,
    decoder_result& cond_result, decoder_result& uncond_result,
    magpietts_cuda_sample_request* cuda_sample, const magpietts_backend_tensor* text_cond_device,
    magpietts_backend_tensor* cond_hidden_out, magpietts_backend_tensor* uncond_hidden_out,
    MagpiePinnedHostScratch& output_staging, const magpietts_decoder_attention* attention) {
    const ggml_nvtx::range nvtx_range("magpietts_decoder_eval_pair");
    const auto& h = model.hparams;
    std::vector<std::vector<int32_t>> stacked_audio;
    if (!stack_audio_codes(audio_codes, h, stacked_audio)) {
        fprintf(stderr, "decoder_eval_pair requires at least one audio token\n");
        return false;
    }
    const int audio_len = (int)stacked_audio[0].size();
    const int total_len = h.baked_context_length + audio_len;

    ggml_context* ctx = new_graph_context();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAGPIETTS_MAX_NODES, false);

    std::vector<ggml_tensor*> audio_tok_inputs(h.stacked_audio_codebooks());
    std::vector<std::pair<std::string, std::vector<int32_t>>> i32_inputs;
    std::vector<std::pair<std::string, std::vector<float>>> f32_inputs;

    ggml_tensor* speaker_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speaker_in, "magpietts_decoder_speaker");
    ggml_set_input(speaker_in);
    i32_inputs.push_back({"magpietts_decoder_speaker", {speaker}});

    for (int c = 0; c < h.stacked_audio_codebooks(); ++c) {
        const std::string name = "magpietts_decoder_audio_tokens_" + std::to_string(c);
        audio_tok_inputs[c] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, audio_len);
        ggml_set_name(audio_tok_inputs[c], name.c_str());
        ggml_set_input(audio_tok_inputs[c]);
        i32_inputs.push_back({name, stacked_audio[c]});
    }

    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total_len);
    ggml_set_name(pos, "magpietts_decoder_positions");
    ggml_set_input(pos);
    i32_inputs.push_back({"magpietts_decoder_positions", positions(total_len)});

    ggml_tensor* text = nullptr;
    if (text_cond_device && text_cond_device->tensor) {
        text = text_cond_device->tensor;
    } else {
        text = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, text_len);
        ggml_set_name(text, "magpietts_decoder_text_cond");
        ggml_set_input(text);
        f32_inputs.push_back({"magpietts_decoder_text_cond", text_cond});
    }

    ggml_tensor* attn_prior = nullptr;
    std::vector<ggml_tensor*> alignment_outputs;
    if (!prepare_attention_prior_input(ctx, text_len, attention, f32_inputs, attn_prior)) {
        ggml_free(ctx);
        return false;
    }

    ggml_tensor* ctx_flat = ggml_get_rows(ctx, model.baked_context, speaker_in);
    ggml_tensor* ctx_emb_cond = ggml_reshape_2d(ctx, ctx_flat, h.n_embd, h.baked_context_length);
    ggml_tensor* ctx_emb_uncond = ggml_scale(ctx, ctx_emb_cond, 0.0f);
    ggml_tensor* audio_emb = build_audio_embedding(ctx, model, audio_tok_inputs);

    ggml_tensor* dec_in_cond = ggml_concat(ctx, ctx_emb_cond, audio_emb, 1);
    ggml_tensor* dec_in_uncond = ggml_concat(ctx, ctx_emb_uncond, audio_emb, 1);

    const bool collect_alignment = attention && attention->alignment_scores;
    ggml_tensor* dec_out_cond = transformer_forward(
        ctx, model.decoder, dec_in_cond, pos, text, attn_prior,
        collect_alignment ? &alignment_outputs : nullptr);
    dec_out_cond = ggml_cont(ctx, ggml_cast(ctx, dec_out_cond, GGML_TYPE_F32));
    ggml_set_name(dec_out_cond, "magpietts_decoder_out_cond");
    ggml_set_output(dec_out_cond);

    const bool compute_logits =
        cuda_sample || cond_result.logits_required || uncond_result.logits_required;
    ggml_tensor* logits_cond = nullptr;
    if (compute_logits) {
        logits_cond = linear(ctx, model.final_proj_w, dec_out_cond, model.final_proj_b);
        logits_cond = ggml_cont(ctx, ggml_cast(ctx, logits_cond, GGML_TYPE_F32));
        ggml_set_name(logits_cond, "magpietts_decoder_logits_cond");
        ggml_set_output(logits_cond);
    }

    ggml_tensor* dec_out_uncond =
        transformer_forward(ctx, model.decoder, dec_in_uncond, pos, nullptr);
    dec_out_uncond = ggml_cont(ctx, ggml_cast(ctx, dec_out_uncond, GGML_TYPE_F32));
    ggml_set_name(dec_out_uncond, "magpietts_decoder_out_uncond");
    ggml_set_output(dec_out_uncond);

    ggml_tensor* logits_uncond = nullptr;
    if (compute_logits) {
        logits_uncond = linear(ctx, model.final_proj_w, dec_out_uncond, model.final_proj_b);
        logits_uncond = ggml_cont(ctx, ggml_cast(ctx, logits_uncond, GGML_TYPE_F32));
        ggml_set_name(logits_uncond, "magpietts_decoder_logits_uncond");
        ggml_set_output(logits_uncond);
    }

    if (logits_cond) {
        ggml_build_forward_expand(gf, logits_cond);
    }
    ggml_build_forward_expand(gf, dec_out_cond);
    if (logits_uncond) {
        ggml_build_forward_expand(gf, logits_uncond);
    }
    ggml_build_forward_expand(gf, dec_out_uncond);
    for (ggml_tensor* t : alignment_outputs) {
        ggml_build_forward_expand(gf, t);
    }

    ggml_gallocr_t allocr = nullptr;
    const size_t hidden_off = (size_t)h.n_embd * (total_len - 1) * sizeof(float);
    ggml_tensor* cond_hidden_last = nullptr;
    ggml_tensor* uncond_hidden_last = nullptr;
    if (cond_hidden_out) {
        cond_hidden_last =
            ggml_view_2d(ctx, dec_out_cond, h.n_embd, 1, dec_out_cond->nb[1], hidden_off);
        ggml_set_name(cond_hidden_last, "magpietts_decoder_hidden_last_cond");
        ggml_set_output(cond_hidden_last);
        ggml_build_forward_expand(gf, cond_hidden_last);
    }
    if (uncond_hidden_out) {
        uncond_hidden_last =
            ggml_view_2d(ctx, dec_out_uncond, h.n_embd, 1, dec_out_uncond->nb[1], hidden_off);
        ggml_set_name(uncond_hidden_last, "magpietts_decoder_hidden_last_uncond");
        ggml_set_output(uncond_hidden_last);
        ggml_build_forward_expand(gf, uncond_hidden_last);
    }

    const bool ok = compute_graph(model, ctx, gf, i32_inputs, f32_inputs, threads, &allocr);
    if (!ok) {
        ggml_free(ctx);
        return false;
    }
    if (!read_alignment_outputs(model, output_staging, alignment_outputs, text_len, attention)) {
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return false;
    }

    const size_t logits_last_size = (size_t)h.stacked_audio_codebooks() * h.audio_vocab_size;
    const size_t logits_off_floats = logits_last_size * (total_len - 1);
    if (cond_hidden_out && cond_hidden_last) {
        ggml_backend_tensor_copy(cond_hidden_last, cond_hidden_out->tensor);
    }
    if (uncond_hidden_out && uncond_hidden_last) {
        ggml_backend_tensor_copy(uncond_hidden_last, uncond_hidden_out->tensor);
    }
    if (cuda_sample) {
        const bool sampled = MagpieCodebookSampler::runCuda(
            model.backend, h, cuda_sample, logits_cond, logits_uncond, logits_off_floats,
            h.stacked_audio_codebooks(), 0);
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return sampled;
    }
    if (cond_hidden_out || uncond_hidden_out) {
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return true;
    }

    if (cond_result.logits_required) {
        cond_result.logits_last.resize(logits_last_size);
    }
    cond_result.hidden_last.resize(h.n_embd);
    if (uncond_result.logits_required) {
        uncond_result.logits_last.resize(logits_last_size);
    }
    uncond_result.hidden_last.resize(h.n_embd);
    const size_t logits_off = logits_off_floats * sizeof(float);
    if (cond_result.logits_required) {
        magpietts_backend_tensor_get_staged(
            model, output_staging, logits_cond, cond_result.logits_last.data(), logits_off,
            cond_result.logits_last.size() * sizeof(float));
    }
    magpietts_backend_tensor_get_staged(
        model, output_staging, dec_out_cond, cond_result.hidden_last.data(), hidden_off,
        cond_result.hidden_last.size() * sizeof(float));
    if (uncond_result.logits_required) {
        magpietts_backend_tensor_get_staged(
            model, output_staging, logits_uncond, uncond_result.logits_last.data(), logits_off,
            uncond_result.logits_last.size() * sizeof(float));
    }
    magpietts_backend_tensor_get_staged(
        model, output_staging, dec_out_uncond, uncond_result.hidden_last.data(), hidden_off,
        uncond_result.hidden_last.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return true;
}

static bool
decoder_eval_cached_impl(
    const magpietts_model& model, const std::vector<float>& text_cond, int text_len,
    const std::vector<std::vector<int32_t>>& audio_codes, int speaker, bool conditional,
    int threads, DecoderKvCache& kv_state, decoder_result& result,
    magpietts_cuda_sample_request* cuda_sample, const magpietts_backend_tensor* text_cond_device,
    magpietts_backend_tensor* hidden_out, DecoderCrossKvCache* cross_kv,
    MagpiePinnedHostScratch& output_staging, const magpietts_decoder_attention* attention) {
    const ggml_nvtx::range nvtx_range(
        conditional ? "magpietts_decoder_eval_cached_cond"
                    : "magpietts_decoder_eval_cached_uncond");
    const auto& h = model.hparams;
    std::vector<std::vector<int32_t>> stacked_audio;
    if (!stack_audio_codes(audio_codes, h, stacked_audio)) {
        fprintf(stderr, "decoder_eval_cached requires at least one audio token\n");
        return false;
    }
    const int audio_len = (int)stacked_audio[0].size();
    if (h.dec_kernel != 1) {
        return decoder_eval_impl(
            model, text_cond, text_len, audio_codes, speaker, conditional, threads, result,
            cuda_sample, text_cond_device, hidden_out, output_staging, attention);
    }
    if (!kv_state.init(model)) {
        return false;
    }
    if (conditional && !ensure_decoder_cross_kv_cache(
                           model, text_cond, text_len, threads, cross_kv, text_cond_device)) {
        return false;
    }

    const int total_len = h.baked_context_length + audio_len;
    if (total_len > h.n_ctx) {
        fprintf(
            stderr, "decoder KV cache length %d exceeds context length %d\n", total_len, h.n_ctx);
        return false;
    }
    if (kv_state.n_tokens != 0 && total_len != kv_state.n_tokens + 1) {
        kv_state.n_tokens = 0;
    }

    const int n_past = kv_state.n_tokens;
    const bool refill = n_past == 0;
    const int n_audio_in = refill ? audio_len : 1;
    const int n_graph_tokens = refill ? total_len : 1;

    ggml_context* ctx = new_graph_context();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAGPIETTS_MAX_NODES, false);

    std::vector<ggml_tensor*> audio_tok_inputs(h.stacked_audio_codebooks());
    std::vector<std::pair<std::string, std::vector<int32_t>>> i32_inputs;
    std::vector<std::pair<std::string, std::vector<float>>> f32_inputs;

    ggml_tensor* dec_in = nullptr;
    if (refill) {
        ggml_tensor* speaker_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_name(speaker_in, "magpietts_decoder_speaker");
        ggml_set_input(speaker_in);
        i32_inputs.push_back({"magpietts_decoder_speaker", {speaker}});

        ggml_tensor* ctx_flat = ggml_get_rows(ctx, model.baked_context, speaker_in);
        ggml_tensor* ctx_emb = ggml_reshape_2d(ctx, ctx_flat, h.n_embd, h.baked_context_length);
        if (!conditional) {
            ctx_emb = ggml_scale(ctx, ctx_emb, 0.0f);
        }

        for (int c = 0; c < h.stacked_audio_codebooks(); ++c) {
            const std::string name = "magpietts_decoder_audio_tokens_" + std::to_string(c);
            audio_tok_inputs[c] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_audio_in);
            ggml_set_name(audio_tok_inputs[c], name.c_str());
            ggml_set_input(audio_tok_inputs[c]);
            i32_inputs.push_back({name, stacked_audio[c]});
        }
        ggml_tensor* audio_emb = build_audio_embedding(ctx, model, audio_tok_inputs);
        dec_in = ggml_concat(ctx, ctx_emb, audio_emb, 1);
    } else {
        for (int c = 0; c < h.stacked_audio_codebooks(); ++c) {
            const std::string name = "magpietts_decoder_audio_tokens_" + std::to_string(c);
            audio_tok_inputs[c] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_audio_in);
            ggml_set_name(audio_tok_inputs[c], name.c_str());
            ggml_set_input(audio_tok_inputs[c]);
            i32_inputs.push_back({name, {stacked_audio[c].back()}});
        }
        dec_in = build_audio_embedding(ctx, model, audio_tok_inputs);
    }

    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_graph_tokens);
    ggml_set_name(pos, "magpietts_decoder_positions");
    ggml_set_input(pos);
    i32_inputs.push_back({"magpietts_decoder_positions", positions_range(n_past, n_graph_tokens)});

    const bool use_cached_cross =
        conditional && cross_kv && cross_kv->valid && cross_kv->text_len == text_len;
    ggml_tensor* cond = nullptr;
    if (conditional && !use_cached_cross) {
        if (text_cond_device && text_cond_device->tensor) {
            cond = text_cond_device->tensor;
        } else {
            cond = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, text_len);
            ggml_set_name(cond, "magpietts_decoder_text_cond");
            ggml_set_input(cond);
            f32_inputs.push_back({"magpietts_decoder_text_cond", text_cond});
        }
    }

    ggml_tensor* attn_prior = nullptr;
    std::vector<ggml_tensor*> alignment_outputs;
    if (conditional &&
        !prepare_attention_prior_input(ctx, text_len, attention, f32_inputs, attn_prior)) {
        ggml_free(ctx);
        return false;
    }

    const bool collect_alignment = conditional && attention && attention->alignment_scores;
    ggml_tensor* dec_out = transformer_forward_cached(
        ctx, gf, model.decoder, dec_in, pos, cond, kv_state, conditional ? cross_kv : nullptr,
        n_past, attn_prior, collect_alignment ? &alignment_outputs : nullptr);
    dec_out = ggml_cont(ctx, ggml_cast(ctx, dec_out, GGML_TYPE_F32));
    ggml_set_name(dec_out, "magpietts_decoder_out_cached");
    ggml_set_output(dec_out);

    const bool compute_logits = cuda_sample || result.logits_required;
    ggml_tensor* logits = nullptr;
    if (compute_logits) {
        logits = linear(ctx, model.final_proj_w, dec_out, model.final_proj_b);
        logits = ggml_cont(ctx, ggml_cast(ctx, logits, GGML_TYPE_F32));
        ggml_set_name(logits, "magpietts_decoder_logits_cached");
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
    }
    ggml_build_forward_expand(gf, dec_out);
    for (ggml_tensor* t : alignment_outputs) {
        ggml_build_forward_expand(gf, t);
    }

    ggml_gallocr_t allocr = nullptr;
    const size_t hidden_off = (size_t)h.n_embd * (n_graph_tokens - 1) * sizeof(float);
    ggml_tensor* hidden_last = nullptr;
    if (hidden_out) {
        hidden_last = ggml_view_2d(ctx, dec_out, h.n_embd, 1, dec_out->nb[1], hidden_off);
        ggml_set_name(hidden_last, "magpietts_decoder_hidden_last_cached");
        ggml_set_output(hidden_last);
        ggml_build_forward_expand(gf, hidden_last);
    }

    const bool ok = compute_graph(model, ctx, gf, i32_inputs, f32_inputs, threads, &allocr);
    if (!ok) {
        ggml_free(ctx);
        return false;
    }
    if (!read_alignment_outputs(
            model, output_staging, alignment_outputs, text_len,
            conditional ? attention : nullptr)) {
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return false;
    }

    const size_t logits_last_size = (size_t)h.stacked_audio_codebooks() * h.audio_vocab_size;
    const size_t logits_off_floats = logits_last_size * (n_graph_tokens - 1);
    if (hidden_out && hidden_last) {
        ggml_backend_tensor_copy(hidden_last, hidden_out->tensor);
    }
    if (cuda_sample) {
        const bool sampled = MagpieCodebookSampler::runCuda(
            model.backend, h, cuda_sample, logits, nullptr, logits_off_floats,
            h.stacked_audio_codebooks(), 0);
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        if (sampled) {
            kv_state.n_tokens = total_len;
        }
        return sampled;
    }
    if (hidden_out) {
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        kv_state.n_tokens = total_len;
        return true;
    }

    if (result.logits_required) {
        result.logits_last.resize(logits_last_size);
    }
    result.hidden_last.resize(h.n_embd);
    const size_t logits_off = logits_off_floats * sizeof(float);
    if (result.logits_required) {
        magpietts_backend_tensor_get_staged(
            model, output_staging, logits, result.logits_last.data(), logits_off,
            result.logits_last.size() * sizeof(float));
    }
    magpietts_backend_tensor_get_staged(
        model, output_staging, dec_out, result.hidden_last.data(), hidden_off,
        result.hidden_last.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);

    kv_state.n_tokens = total_len;
    return true;
}

static bool
decoder_eval_cached_pair_impl(
    const magpietts_model& model, const std::vector<float>& text_cond, int text_len,
    const std::vector<std::vector<int32_t>>& audio_codes, int speaker, int threads,
    DecoderKvCache& cond_kv, DecoderKvCache& uncond_kv, decoder_result& cond_result,
    decoder_result& uncond_result, magpietts_cuda_sample_request* cuda_sample,
    const magpietts_backend_tensor* text_cond_device, magpietts_backend_tensor* cond_hidden_out,
    magpietts_backend_tensor* uncond_hidden_out, DecoderCrossKvCache* cond_cross_kv,
    MagpiePinnedHostScratch& output_staging, const magpietts_decoder_attention* attention) {
    const ggml_nvtx::range nvtx_range("magpietts_decoder_eval_cached_pair");
    const auto& h = model.hparams;
    std::vector<std::vector<int32_t>> stacked_audio;
    if (!stack_audio_codes(audio_codes, h, stacked_audio)) {
        fprintf(stderr, "decoder_eval_cached_pair requires at least one audio token\n");
        return false;
    }
    const int audio_len = (int)stacked_audio[0].size();
    if (h.dec_kernel != 1) {
        return decoder_eval_pair_impl(
            model, text_cond, text_len, audio_codes, speaker, threads, cond_result, uncond_result,
            cuda_sample, text_cond_device, cond_hidden_out, uncond_hidden_out, output_staging,
            attention);
    }
    if (!cond_kv.init(model) || !uncond_kv.init(model)) {
        return false;
    }
    if (!ensure_decoder_cross_kv_cache(
            model, text_cond, text_len, threads, cond_cross_kv, text_cond_device)) {
        return false;
    }

    const int total_len = h.baked_context_length + audio_len;
    if (total_len > h.n_ctx) {
        fprintf(
            stderr, "decoder KV cache length %d exceeds context length %d\n", total_len, h.n_ctx);
        return false;
    }
    if (cond_kv.n_tokens != 0 && total_len != cond_kv.n_tokens + 1) {
        cond_kv.n_tokens = 0;
    }
    if (uncond_kv.n_tokens != 0 && total_len != uncond_kv.n_tokens + 1) {
        uncond_kv.n_tokens = 0;
    }
    if (cond_kv.n_tokens != uncond_kv.n_tokens) {
        if (cuda_sample) {
            cond_kv.n_tokens = 0;
            uncond_kv.n_tokens = 0;
        } else {
            const bool cond_ok = decoder_eval_cached_impl(
                model, text_cond, text_len, audio_codes, speaker, true, threads, cond_kv,
                cond_result, nullptr, text_cond_device, cond_hidden_out, cond_cross_kv,
                output_staging, attention);
            const bool uncond_ok =
                cond_ok && decoder_eval_cached_impl(
                               model, text_cond, text_len, audio_codes, speaker, false, threads,
                               uncond_kv, uncond_result, nullptr, text_cond_device,
                               uncond_hidden_out, nullptr, output_staging, nullptr);
            return cond_ok && uncond_ok;
        }
    }

    const int n_past = cond_kv.n_tokens;
    const bool refill = n_past == 0;
    const int n_audio_in = refill ? audio_len : 1;
    const int n_graph_tokens = refill ? total_len : 1;

    ggml_context* ctx = new_graph_context();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAGPIETTS_MAX_NODES, false);

    std::vector<ggml_tensor*> audio_tok_inputs(h.stacked_audio_codebooks());
    std::vector<std::pair<std::string, std::vector<int32_t>>> i32_inputs;
    std::vector<std::pair<std::string, std::vector<float>>> f32_inputs;

    ggml_tensor* dec_in_cond = nullptr;
    ggml_tensor* dec_in_uncond = nullptr;
    if (refill) {
        ggml_tensor* speaker_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_name(speaker_in, "magpietts_decoder_speaker");
        ggml_set_input(speaker_in);
        i32_inputs.push_back({"magpietts_decoder_speaker", {speaker}});

        for (int c = 0; c < h.stacked_audio_codebooks(); ++c) {
            const std::string name = "magpietts_decoder_audio_tokens_" + std::to_string(c);
            audio_tok_inputs[c] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_audio_in);
            ggml_set_name(audio_tok_inputs[c], name.c_str());
            ggml_set_input(audio_tok_inputs[c]);
            i32_inputs.push_back({name, stacked_audio[c]});
        }

        ggml_tensor* ctx_flat = ggml_get_rows(ctx, model.baked_context, speaker_in);
        ggml_tensor* ctx_emb_cond =
            ggml_reshape_2d(ctx, ctx_flat, h.n_embd, h.baked_context_length);
        ggml_tensor* ctx_emb_uncond = ggml_scale(ctx, ctx_emb_cond, 0.0f);
        ggml_tensor* audio_emb = build_audio_embedding(ctx, model, audio_tok_inputs);
        dec_in_cond = ggml_concat(ctx, ctx_emb_cond, audio_emb, 1);
        dec_in_uncond = ggml_concat(ctx, ctx_emb_uncond, audio_emb, 1);
    } else {
        for (int c = 0; c < h.stacked_audio_codebooks(); ++c) {
            const std::string name = "magpietts_decoder_audio_tokens_" + std::to_string(c);
            audio_tok_inputs[c] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_audio_in);
            ggml_set_name(audio_tok_inputs[c], name.c_str());
            ggml_set_input(audio_tok_inputs[c]);
            i32_inputs.push_back({name, {stacked_audio[c].back()}});
        }
        ggml_tensor* audio_emb = build_audio_embedding(ctx, model, audio_tok_inputs);
        dec_in_cond = audio_emb;
        dec_in_uncond = audio_emb;
    }

    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_graph_tokens);
    ggml_set_name(pos, "magpietts_decoder_positions");
    ggml_set_input(pos);
    i32_inputs.push_back({"magpietts_decoder_positions", positions_range(n_past, n_graph_tokens)});

    const bool use_cached_cross =
        cond_cross_kv && cond_cross_kv->valid && cond_cross_kv->text_len == text_len;
    ggml_tensor* text = nullptr;
    if (!use_cached_cross) {
        if (text_cond_device && text_cond_device->tensor) {
            text = text_cond_device->tensor;
        } else {
            text = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, text_len);
            ggml_set_name(text, "magpietts_decoder_text_cond");
            ggml_set_input(text);
            f32_inputs.push_back({"magpietts_decoder_text_cond", text_cond});
        }
    }

    ggml_tensor* attn_prior = nullptr;
    std::vector<ggml_tensor*> alignment_outputs;
    if (!prepare_attention_prior_input(ctx, text_len, attention, f32_inputs, attn_prior)) {
        ggml_free(ctx);
        return false;
    }

    const bool collect_alignment = attention && attention->alignment_scores;
    ggml_tensor* dec_out_cond = transformer_forward_cached(
        ctx, gf, model.decoder, dec_in_cond, pos, text, cond_kv, cond_cross_kv, n_past, attn_prior,
        collect_alignment ? &alignment_outputs : nullptr);
    dec_out_cond = ggml_cont(ctx, ggml_cast(ctx, dec_out_cond, GGML_TYPE_F32));
    ggml_set_name(dec_out_cond, "magpietts_decoder_out_cond_cached");
    ggml_set_output(dec_out_cond);

    const bool compute_logits =
        cuda_sample || cond_result.logits_required || uncond_result.logits_required;
    ggml_tensor* logits_cond = nullptr;
    if (compute_logits) {
        logits_cond = linear(ctx, model.final_proj_w, dec_out_cond, model.final_proj_b);
        logits_cond = ggml_cont(ctx, ggml_cast(ctx, logits_cond, GGML_TYPE_F32));
        ggml_set_name(logits_cond, "magpietts_decoder_logits_cond_cached");
        ggml_set_output(logits_cond);
    }

    ggml_tensor* dec_out_uncond = transformer_forward_cached(
        ctx, gf, model.decoder, dec_in_uncond, pos, nullptr, uncond_kv, nullptr, n_past);
    dec_out_uncond = ggml_cont(ctx, ggml_cast(ctx, dec_out_uncond, GGML_TYPE_F32));
    ggml_set_name(dec_out_uncond, "magpietts_decoder_out_uncond_cached");
    ggml_set_output(dec_out_uncond);

    ggml_tensor* logits_uncond = nullptr;
    if (compute_logits) {
        logits_uncond = linear(ctx, model.final_proj_w, dec_out_uncond, model.final_proj_b);
        logits_uncond = ggml_cont(ctx, ggml_cast(ctx, logits_uncond, GGML_TYPE_F32));
        ggml_set_name(logits_uncond, "magpietts_decoder_logits_uncond_cached");
        ggml_set_output(logits_uncond);
    }

    if (logits_cond) {
        ggml_build_forward_expand(gf, logits_cond);
    }
    ggml_build_forward_expand(gf, dec_out_cond);
    if (logits_uncond) {
        ggml_build_forward_expand(gf, logits_uncond);
    }
    ggml_build_forward_expand(gf, dec_out_uncond);
    for (ggml_tensor* t : alignment_outputs) {
        ggml_build_forward_expand(gf, t);
    }

    ggml_gallocr_t allocr = nullptr;
    const size_t hidden_off = (size_t)h.n_embd * (n_graph_tokens - 1) * sizeof(float);
    ggml_tensor* cond_hidden_last = nullptr;
    ggml_tensor* uncond_hidden_last = nullptr;
    if (cond_hidden_out) {
        cond_hidden_last =
            ggml_view_2d(ctx, dec_out_cond, h.n_embd, 1, dec_out_cond->nb[1], hidden_off);
        ggml_set_name(cond_hidden_last, "magpietts_decoder_hidden_last_cond_cached");
        ggml_set_output(cond_hidden_last);
        ggml_build_forward_expand(gf, cond_hidden_last);
    }
    if (uncond_hidden_out) {
        uncond_hidden_last =
            ggml_view_2d(ctx, dec_out_uncond, h.n_embd, 1, dec_out_uncond->nb[1], hidden_off);
        ggml_set_name(uncond_hidden_last, "magpietts_decoder_hidden_last_uncond_cached");
        ggml_set_output(uncond_hidden_last);
        ggml_build_forward_expand(gf, uncond_hidden_last);
    }

    const bool ok = compute_graph(model, ctx, gf, i32_inputs, f32_inputs, threads, &allocr);
    if (!ok) {
        ggml_free(ctx);
        return false;
    }
    if (!read_alignment_outputs(model, output_staging, alignment_outputs, text_len, attention)) {
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return false;
    }

    const size_t logits_last_size = (size_t)h.stacked_audio_codebooks() * h.audio_vocab_size;
    const size_t logits_off_floats = logits_last_size * (n_graph_tokens - 1);
    if (cond_hidden_out && cond_hidden_last) {
        ggml_backend_tensor_copy(cond_hidden_last, cond_hidden_out->tensor);
    }
    if (uncond_hidden_out && uncond_hidden_last) {
        ggml_backend_tensor_copy(uncond_hidden_last, uncond_hidden_out->tensor);
    }
    if (cuda_sample) {
        const bool sampled = MagpieCodebookSampler::runCuda(
            model.backend, h, cuda_sample, logits_cond, logits_uncond, logits_off_floats,
            h.stacked_audio_codebooks(), 0);
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        if (sampled) {
            cond_kv.n_tokens = total_len;
            uncond_kv.n_tokens = total_len;
        }
        return sampled;
    }
    if (cond_hidden_out || uncond_hidden_out) {
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        cond_kv.n_tokens = total_len;
        uncond_kv.n_tokens = total_len;
        return true;
    }

    if (cond_result.logits_required) {
        cond_result.logits_last.resize(logits_last_size);
    }
    cond_result.hidden_last.resize(h.n_embd);
    if (uncond_result.logits_required) {
        uncond_result.logits_last.resize(logits_last_size);
    }
    uncond_result.hidden_last.resize(h.n_embd);
    const size_t logits_off = logits_off_floats * sizeof(float);
    if (cond_result.logits_required) {
        magpietts_backend_tensor_get_staged(
            model, output_staging, logits_cond, cond_result.logits_last.data(), logits_off,
            cond_result.logits_last.size() * sizeof(float));
    }
    magpietts_backend_tensor_get_staged(
        model, output_staging, dec_out_cond, cond_result.hidden_last.data(), hidden_off,
        cond_result.hidden_last.size() * sizeof(float));
    if (uncond_result.logits_required) {
        magpietts_backend_tensor_get_staged(
            model, output_staging, logits_uncond, uncond_result.logits_last.data(), logits_off,
            uncond_result.logits_last.size() * sizeof(float));
    }
    magpietts_backend_tensor_get_staged(
        model, output_staging, dec_out_uncond, uncond_result.hidden_last.data(), hidden_off,
        uncond_result.hidden_last.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);

    cond_kv.n_tokens = total_len;
    uncond_kv.n_tokens = total_len;
    return true;
}

static void
clear_forbidden_logits(
    std::vector<float>& logits, const magpietts_hparams& h, bool forbid_audio_eos) {
    const int base = h.audio_codebook_size;
    const int forbidden[] = {
        base + 0,  // AUDIO_BOS
        base + 2,  // AUDIO_CONTEXT_BOS
        base + 3,  // AUDIO_CONTEXT_EOS
        base + 4,  // MASK_TOKEN
        base + 5, base + 6, base + 7,
    };
    for (int id : forbidden) {
        if (0 <= id && id < (int)logits.size()) {
            logits[id] = -INFINITY;
        }
    }
    if (forbid_audio_eos && 0 <= h.audio_eos_id && h.audio_eos_id < (int)logits.size()) {
        logits[h.audio_eos_id] = -INFINITY;
    }
}

struct prepared_codebook_sample {
    std::vector<int> top_ids;
    std::vector<double> weights;
    int greedy = 0;
};

static prepared_codebook_sample
prepare_codebook_sample(
    std::vector<float> logits, const magpietts_hparams& h, float temperature, int top_k,
    bool forbid_audio_eos) {
    clear_forbidden_logits(logits, h, forbid_audio_eos);

    prepared_codebook_sample prepared;
    prepared.greedy = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());

    top_k = std::max(1, std::min(top_k, (int)logits.size()));
    std::vector<int> idx(logits.size());
    for (int i = 0; i < (int)idx.size(); ++i) {
        idx[i] = i;
    }
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), [&](int a, int b) {
        return logits[a] > logits[b];
    });
    idx.resize(top_k);

    prepared.top_ids = std::move(idx);
    if (temperature > 0.0f) {
        float max_logit = -INFINITY;
        for (int id : prepared.top_ids) {
            max_logit = std::max(max_logit, logits[id]);
        }
        prepared.weights.reserve(prepared.top_ids.size());
        for (int id : prepared.top_ids) {
            const double w =
                std::isfinite(logits[id]) ? std::exp((logits[id] - max_logit) / temperature) : 0.0;
            prepared.weights.push_back(w);
        }
    }
    return prepared;
}

static int
sample_from_prepared_codebook(
    const prepared_codebook_sample& prepared, float temperature, std::mt19937& rng) {
    if (temperature <= 0.0f) {
        return prepared.top_ids[0];
    }
    std::discrete_distribution<int> dist(prepared.weights.begin(), prepared.weights.end());
    return prepared.top_ids[dist(rng)];
}

int
MagpieCodebookSampler::sampleFromLogits(
    std::vector<float> logits, const magpietts_hparams& h, float temperature, int top_k,
    std::mt19937& rng, bool forbid_audio_eos) {
    const prepared_codebook_sample prepared =
        prepare_codebook_sample(std::move(logits), h, temperature, top_k, forbid_audio_eos);
    return sample_from_prepared_codebook(prepared, temperature, rng);
}

int
MagpieCodebookSampler::argmaxFromLogits(
    std::vector<float> logits, const magpietts_hparams& h, bool forbid_audio_eos) {
    clear_forbidden_logits(logits, h, forbid_audio_eos);
    return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

static std::vector<float>
slice_codebook_logits(const std::vector<float>& logits, const magpietts_hparams& h, int codebook) {
    std::vector<float> out(h.audio_vocab_size);
    const size_t off = (size_t)codebook * h.audio_vocab_size;
    std::copy(logits.begin() + off, logits.begin() + off + h.audio_vocab_size, out.begin());
    return out;
}

std::vector<int32_t>
MagpieCodebookSampler::sampleParallel(
    const std::vector<float>& cond_logits, const std::vector<float>& uncond_logits,
    const magpietts_hparams& h, bool use_cfg, float cfg_scale, float temperature, int top_k,
    bool forbid_audio_eos, std::mt19937& rng, std::vector<int32_t>* argmax_codes) {
    const ggml_nvtx::range nvtx_range("magpietts_sample_parallel_codebooks");
    std::vector<int32_t> codes(h.stacked_audio_codebooks());
    if (argmax_codes) {
        argmax_codes->assign(h.stacked_audio_codebooks(), 0);
    }

    std::vector<std::future<prepared_codebook_sample>> futures;
    futures.reserve(h.stacked_audio_codebooks());
    for (int c = 0; c < h.stacked_audio_codebooks(); ++c) {
        futures.push_back(std::async(std::launch::async, [&, c]() {
            std::vector<float> logits = slice_codebook_logits(cond_logits, h, c);
            if (use_cfg) {
                const size_t off = (size_t)c * h.audio_vocab_size;
                for (int i = 0; i < h.audio_vocab_size; ++i) {
                    logits[i] = cfg_scale * logits[i] + (1.0f - cfg_scale) * uncond_logits[off + i];
                }
            }
            return prepare_codebook_sample(
                std::move(logits), h, temperature, top_k, forbid_audio_eos);
        }));
    }

    std::vector<prepared_codebook_sample> prepared((size_t)h.stacked_audio_codebooks());
    for (int c = 0; c < h.stacked_audio_codebooks(); ++c) {
        prepared[(size_t)c] = futures[(size_t)c].get();
    }
    for (int c = 0; c < h.stacked_audio_codebooks(); ++c) {
        codes[c] = sample_from_prepared_codebook(prepared[(size_t)c], temperature, rng);
        if (argmax_codes) {
            (*argmax_codes)[c] = prepared[(size_t)c].greedy;
        }
    }
    return codes;
}

bool
MagpieCodebookSampler::hasEos(
    const std::vector<int32_t>& a, const std::vector<int32_t>& b, int eos_id) {
    return std::find(a.begin(), a.end(), eos_id) != a.end() ||
           std::find(b.begin(), b.end(), eos_id) != b.end();
}

}  // namespace nemo_speech::tts
