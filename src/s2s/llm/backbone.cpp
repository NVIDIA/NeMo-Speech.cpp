// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "backbone.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "llama_common.h"

namespace nemo_speech::s2s {

LLMBackbone::LLMBackbone(const std::string& gguf_path, int max_streams) {
    ensure_llama_backend();

    if (max_streams < 1)
        throw std::invalid_argument("LLMBackbone: max_streams must be positive");
    n_seq_max_ = max_streams;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    model_ = llama_model_load_from_file(gguf_path.c_str(), mp);
    if (!model_) {
        throw std::runtime_error("LLMBackbone: failed to load model from " + gguf_path);
    }

    n_embd_ = llama_model_n_embd(model_);
    vocab_ = llama_model_get_vocab(model_);
    n_vocab_ = llama_vocab_n_tokens(vocab_);

    llama_context_params cp = llama_context_default_params();
    // llama.cpp divides n_ctx across n_seq_max. Keep the original 5120-token
    // budget for every conversation as stream capacity grows.
    cp.n_ctx = 5120 * static_cast<uint32_t>(n_seq_max_);
    cp.n_seq_max = static_cast<uint32_t>(n_seq_max_);
    cp.n_batch = kMaxBatchTokens;
    cp.n_ubatch = kMaxBatchTokens;
    cp.embeddings = true;

    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("LLMBackbone: failed to create llama context");
    }
    llama_set_embeddings(ctx_, true);

    batch_ = llama_batch_init(kMaxBatchTokens, n_embd_, 1);

    slot_pool_.reserve(n_seq_max_);
    for (int slot = n_seq_max_ - 1; slot >= 0; --slot) slot_pool_.push_back(slot);
}

LLMBackbone::~LLMBackbone() {
    llama_batch_free(batch_);
    if (ctx_)
        llama_free(ctx_);
    if (model_)
        llama_model_free(model_);
}

int
LLMBackbone::map_slot(int64_t ext_seq_id) {
    auto it = ext_to_slot_.find(ext_seq_id);
    if (it != ext_to_slot_.end())
        return it->second;
    if (slot_pool_.empty()) {
        throw std::runtime_error(
            "LLMBackbone: no free sequence slots (max " + std::to_string(n_seq_max_) + ")");
    }
    int slot = slot_pool_.back();
    slot_pool_.pop_back();
    ext_to_slot_[ext_seq_id] = slot;
    return slot;
}

void
LLMBackbone::prefill(int64_t ext_seq_id, const float* embeds, int n_tokens) {
    std::lock_guard<std::mutex> lock(mu_);
    const int slot = map_slot(ext_seq_id);
    llama_pos pos = slot_pos_.count(slot) ? slot_pos_[slot] : 0;

    for (int start = 0; start < n_tokens; start += kMaxBatchTokens) {
        const int n = std::min(kMaxBatchTokens, n_tokens - start);

        batch_.n_tokens = n;
        std::memcpy(
            batch_.embd, embeds + (size_t)start * n_embd_, (size_t)n * n_embd_ * sizeof(float));
        for (int i = 0; i < n; i++) {
            batch_.pos[i] = pos + i;
            batch_.n_seq_id[i] = 1;
            batch_.seq_id[i][0] = slot;
            batch_.logits[i] = (i == n - 1) ? 1 : 0;
        }
        pos += n;

        const int32_t ret = llama_decode(ctx_, batch_);
        if (ret != 0) {
            throw std::runtime_error(
                "LLMBackbone: llama_decode (prefill) failed with code " + std::to_string(ret));
        }
    }
    slot_pos_[slot] = pos;
}

void
LLMBackbone::step(
    int64_t ext_seq_id, const float* embed, float* out_text_logits, float* out_hidden) {
    step_batch(&ext_seq_id, embed, 1, out_text_logits, out_hidden);
}

void
LLMBackbone::step_batch(
    const int64_t* ext_seq_ids, const float* embeds, int batch, float* out_text_logits,
    float* out_hidden) {
    if (batch <= 0 || batch > kMaxBatchTokens)
        throw std::invalid_argument("LLMBackbone: invalid step batch size");
    std::lock_guard<std::mutex> lock(mu_);
    batch_.n_tokens = batch;
    std::vector<int> slots(static_cast<size_t>(batch));
    for (int i = 0; i < batch; ++i) {
        slots[static_cast<size_t>(i)] = map_slot(ext_seq_ids[i]);
    }
    std::vector<int> order(static_cast<size_t>(batch));
    for (int i = 0; i < batch; ++i) order[static_cast<size_t>(i)] = i;
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return slots[static_cast<size_t>(lhs)] < slots[static_cast<size_t>(rhs)];
    });
    for (int row = 0; row < batch; ++row) {
        const int input = order[static_cast<size_t>(row)];
        const int slot = slots[static_cast<size_t>(input)];
        std::memcpy(
            batch_.embd + static_cast<size_t>(row) * n_embd_,
            embeds + static_cast<size_t>(input) * n_embd_,
            static_cast<size_t>(n_embd_) * sizeof(float));
        batch_.pos[row] = slot_pos_.count(slot) ? slot_pos_[slot] : 0;
        batch_.n_seq_id[row] = 1;
        batch_.seq_id[row][0] = slot;
        batch_.logits[row] = 1;
    }

    const int32_t ret = llama_decode(ctx_, batch_);
    if (ret != 0) {
        throw std::runtime_error(
            "LLMBackbone: llama_decode (batched step) failed with code " + std::to_string(ret));
    }
    for (int row = 0; row < batch; ++row) {
        const int output = order[static_cast<size_t>(row)];
        ++slot_pos_[slots[static_cast<size_t>(output)]];
        const float* logits = llama_get_logits_ith(ctx_, row);
        const float* hidden = llama_get_embeddings_ith(ctx_, row);
        if (!logits || !hidden)
            throw std::runtime_error("LLMBackbone: batched output buffer is null");
        std::memcpy(
            out_text_logits + static_cast<size_t>(output) * n_vocab_, logits,
            static_cast<size_t>(n_vocab_) * sizeof(float));
        std::memcpy(
            out_hidden + static_cast<size_t>(output) * n_embd_, hidden,
            static_cast<size_t>(n_embd_) * sizeof(float));
    }
}

void
LLMBackbone::cleanup(int64_t ext_seq_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = ext_to_slot_.find(ext_seq_id);
    if (it == ext_to_slot_.end())
        return;
    const int slot = it->second;
    ext_to_slot_.erase(it);

    llama_memory_t mem = llama_get_memory(ctx_);
    if (!mem)
        throw std::runtime_error("LLMBackbone: llama_get_memory returned null");
    llama_memory_seq_rm(mem, slot, -1, -1);

    slot_pos_.erase(slot);
    slot_pool_.push_back(slot);
}

std::vector<int32_t>
LLMBackbone::text_to_ids(const std::string& text) const {
    std::vector<int32_t> ids(text.size() + 8);
    int32_t n = llama_tokenize(
        vocab_, text.c_str(), (int32_t)text.size(), ids.data(), (int32_t)ids.size(),
        /*add_special=*/false,
        /*parse_special=*/true);
    if (n < 0) {
        ids.resize(-n);
        n = llama_tokenize(
            vocab_, text.c_str(), (int32_t)text.size(), ids.data(), (int32_t)ids.size(), false,
            true);
        if (n < 0)
            throw std::runtime_error("LLMBackbone: llama_tokenize failed");
    }
    ids.resize(n);
    return ids;
}

std::string
LLMBackbone::ids_to_text(const int32_t* ids, int n) const {
    std::string out;
    char buf[256];
    for (int i = 0; i < n; i++) {
        // special=true: special tokens are rendered as their literal text.
        const int32_t len = llama_token_to_piece(
            vocab_, ids[i], buf, (int32_t)sizeof(buf),
            /*lstrip=*/0, /*special=*/true);
        if (len < 0) {
            throw std::runtime_error(
                "LLMBackbone: llama_token_to_piece failed for token " + std::to_string(ids[i]));
        }
        out.append(buf, (size_t)len);
    }
    return out;
}

}  // namespace nemo_speech::s2s
