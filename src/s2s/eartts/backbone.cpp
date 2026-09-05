// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "backbone.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "llm/llama_common.h"

namespace nemo_speech::s2s {

EarTTSBackbone::EarTTSBackbone(const std::string& gguf_path, int max_streams) {
    ensure_llama_backend();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    model_ = llama_model_load_from_file(gguf_path.c_str(), mp);
    if (!model_) {
        throw std::runtime_error("EarTTSBackbone: failed to load model from " + gguf_path);
    }
    char attention_scale[32];
    if (llama_model_meta_val_str(
            model_, "gemma3.attention.scale", attention_scale, sizeof(attention_scale)) < 0) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error(
            "EarTTSBackbone: model is missing gemma3.attention.scale; reconvert the model");
    }

    n_embd_ = llama_model_n_embd(model_);
    if (max_streams < 1)
        throw std::invalid_argument("EarTTSBackbone: max_streams must be positive");
    // Two reserved template rows plus cond+uncond for every live stream.
    n_seq_max_ = 2 * (max_streams + 1);

    llama_context_params cp = llama_context_default_params();
    // n_ctx is the TOTAL KV pool shared by all sequences: 4500 positions each.
    cp.n_ctx = kPositionsPerSequence * (uint32_t)n_seq_max_;
    cp.n_seq_max = (uint32_t)n_seq_max_;
    cp.n_batch = kMaxBatchTokens;
    cp.n_ubatch = kMaxBatchTokens;
    cp.embeddings = true;

    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("EarTTSBackbone: failed to create llama context");
    }
    llama_set_embeddings(ctx_, true);

    batch_ = llama_batch_init(kMaxBatchTokens, n_embd_, 1);

    slot_pos_.assign(n_seq_max_, 0);
    slot_pool_.reserve(n_seq_max_ - 2);
    for (int i = 2; i < n_seq_max_; i++) slot_pool_.push_back(i);
}

EarTTSBackbone::~EarTTSBackbone() {
    llama_batch_free(batch_);
    if (ctx_)
        llama_free(ctx_);
    if (model_)
        llama_model_free(model_);
}

int
EarTTSBackbone::step_capacity() const {
    if (!prompt_template_ready_)
        throw std::runtime_error("EarTTSBackbone: prompt template is not initialized");
    return kPositionsPerSequence - slot_pos_[template_cond_slot_];
}

std::pair<int, int>
EarTTSBackbone::allocate_pair() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!prompt_template_ready_)
        throw std::runtime_error("EarTTSBackbone: prompt template is not initialized");
    if (slot_pool_.size() < 2) {
        throw std::runtime_error(
            "EarTTSBackbone: no free seq slots for cond+uncond pair (need 2, have " +
            std::to_string(slot_pool_.size()) + ")");
    }
    int s1 = slot_pool_.back();
    slot_pool_.pop_back();
    int s2 = slot_pool_.back();
    slot_pool_.pop_back();
    // Consecutive ascending: cond < uncond, uncond == cond + 1. The pool is
    // initialised 0..n-1 and popped/pushed in pairs, so the two popped slots
    // are always a consecutive pair; order them.
    const int cond = std::min(s1, s2);
    const int uncond = std::max(s1, s2);
    llama_memory_t mem = llama_get_memory(ctx_);
    if (!mem)
        throw std::runtime_error("EarTTSBackbone: llama_get_memory returned null");
    llama_memory_seq_cp(mem, template_cond_slot_, cond, -1, -1);
    llama_memory_seq_cp(mem, template_uncond_slot_, uncond, -1, -1);
    slot_pos_[cond] = slot_pos_[template_cond_slot_];
    slot_pos_[uncond] = slot_pos_[template_uncond_slot_];
    return {cond, uncond};
}

void
EarTTSBackbone::free_pair(int cond_slot, int uncond_slot) {
    std::lock_guard<std::mutex> lock(mu_);
    llama_memory_t mem = llama_get_memory(ctx_);
    if (!mem)
        throw std::runtime_error("EarTTSBackbone: llama_get_memory returned null");
    for (int s : {cond_slot, uncond_slot}) {
        llama_memory_seq_rm(mem, s, -1, -1);
        slot_pos_[s] = 0;
    }
    slot_pool_.push_back(cond_slot);
    slot_pool_.push_back(uncond_slot);
}

void
EarTTSBackbone::reset_pair(int cond_slot, int uncond_slot) {
    std::lock_guard<std::mutex> lock(mu_);
    llama_memory_t mem = llama_get_memory(ctx_);
    if (!mem)
        throw std::runtime_error("EarTTSBackbone: llama_get_memory returned null");
    for (int s : {cond_slot, uncond_slot}) llama_memory_seq_rm(mem, s, -1, -1);
    llama_memory_seq_cp(mem, template_cond_slot_, cond_slot, -1, -1);
    llama_memory_seq_cp(mem, template_uncond_slot_, uncond_slot, -1, -1);
    slot_pos_[cond_slot] = slot_pos_[template_cond_slot_];
    slot_pos_[uncond_slot] = slot_pos_[template_uncond_slot_];
}

void
EarTTSBackbone::initialize_prompt_template(
    const float* cond_emb, const float* uncond_emb, int n_tokens, float* out_h_cond,
    float* out_h_uncond) {
    std::lock_guard<std::mutex> lock(mu_);
    if (prompt_template_ready_)
        throw std::runtime_error("EarTTSBackbone: prompt template initialized twice");
    decode_chunked(template_cond_slot_, cond_emb, n_tokens);
    fetch_last_hidden(out_h_cond);
    decode_chunked(template_uncond_slot_, uncond_emb, n_tokens);
    fetch_last_hidden(out_h_uncond);
    prompt_template_ready_ = true;
}

void
EarTTSBackbone::decode_chunked(int slot, const float* embs, int n_tokens) {
    if (n_tokens <= 0)
        throw std::invalid_argument("EarTTSBackbone: n_tokens must be positive");
    llama_pos pos = slot_pos_[slot];
    for (int start = 0; start < n_tokens; start += kMaxBatchTokens) {
        const int n = std::min(kMaxBatchTokens, n_tokens - start);

        batch_.n_tokens = n;
        std::memcpy(
            batch_.embd, embs + (size_t)start * n_embd_, (size_t)n * n_embd_ * sizeof(float));
        for (int i = 0; i < n; i++) {
            batch_.pos[i] = pos + i;
            batch_.n_seq_id[i] = 1;
            batch_.seq_id[i][0] = slot;
            // Logits only on the very last row of the very last chunk.
            batch_.logits[i] = (start + n == n_tokens && i == n - 1) ? 1 : 0;
        }
        pos += n;

        const int32_t ret = llama_decode(ctx_, batch_);
        if (ret != 0) {
            throw std::runtime_error(
                "EarTTSBackbone: llama_decode failed (seq=" + std::to_string(slot) +
                ", chunk=" + std::to_string(start) + "): " + std::to_string(ret));
        }
    }
    slot_pos_[slot] = pos;
}

void
EarTTSBackbone::fetch_last_hidden(float* out) const {
    const float* h = llama_get_embeddings_ith(ctx_, -1);
    if (!h)
        throw std::runtime_error("EarTTSBackbone: llama_get_embeddings_ith(-1) returned null");
    std::memcpy(out, h, (size_t)n_embd_ * sizeof(float));
}

void
EarTTSBackbone::prefill_pair(
    int cond_slot, int uncond_slot, const float* cond_emb, const float* uncond_emb, int n_tokens,
    float* out_h_cond, float* out_h_uncond) {
    std::lock_guard<std::mutex> lock(mu_);
    decode_chunked(cond_slot, cond_emb, n_tokens);
    fetch_last_hidden(out_h_cond);
    decode_chunked(uncond_slot, uncond_emb, n_tokens);
    fetch_last_hidden(out_h_uncond);
}

void
EarTTSBackbone::step_pair(
    int cond_slot, int uncond_slot, const float* cond_emb, const float* uncond_emb,
    float* out_h_cond, float* out_h_uncond) {
    step_pairs(&cond_slot, &uncond_slot, cond_emb, uncond_emb, 1, out_h_cond, out_h_uncond);
}

void
EarTTSBackbone::step_pairs(
    const int* cond_slots, const int* uncond_slots, const float* cond_emb, const float* uncond_emb,
    int batch, float* out_h_cond, float* out_h_uncond) {
    if (batch <= 0 || 2 * batch > kMaxBatchTokens)
        throw std::invalid_argument("EarTTSBackbone: invalid CFG step batch size");
    std::lock_guard<std::mutex> lock(mu_);
    batch_.n_tokens = 2 * batch;
    struct Row {
        int slot;
        int stream;
        bool conditional;
    };
    std::vector<Row> rows;
    rows.reserve(static_cast<size_t>(2 * batch));
    for (int b = 0; b < batch; ++b) {
        rows.push_back({cond_slots[b], b, true});
        rows.push_back({uncond_slots[b], b, false});
    }
    std::sort(rows.begin(), rows.end(), [](const Row& lhs, const Row& rhs) {
        return lhs.slot < rhs.slot;
    });
    for (int row = 0; row < 2 * batch; ++row) {
        const Row& item = rows[static_cast<size_t>(row)];
        const float* source = item.conditional
                                  ? cond_emb + static_cast<size_t>(item.stream) * n_embd_
                                  : uncond_emb + static_cast<size_t>(item.stream) * n_embd_;
        std::memcpy(
            batch_.embd + static_cast<size_t>(row) * n_embd_, source,
            static_cast<size_t>(n_embd_) * sizeof(float));
        batch_.pos[row] = slot_pos_[item.slot];
        batch_.n_seq_id[row] = 1;
        batch_.seq_id[row][0] = item.slot;
        batch_.logits[row] = 1;
    }

    const int32_t ret = llama_decode(ctx_, batch_);
    if (ret != 0) {
        throw std::runtime_error(
            "EarTTSBackbone: llama_decode (batched CFG step) failed: " + std::to_string(ret));
    }
    for (int row = 0; row < 2 * batch; ++row) {
        const Row& item = rows[static_cast<size_t>(row)];
        ++slot_pos_[item.slot];
        const float* hidden = llama_get_embeddings_ith(ctx_, row);
        if (!hidden)
            throw std::runtime_error("EarTTSBackbone: batched hidden state is null");
        float* output = item.conditional
                            ? out_h_cond + static_cast<size_t>(item.stream) * n_embd_
                            : out_h_uncond + static_cast<size_t>(item.stream) * n_embd_;
        std::memcpy(output, hidden, static_cast<size_t>(n_embd_) * sizeof(float));
    }
}

}  // namespace nemo_speech::s2s
