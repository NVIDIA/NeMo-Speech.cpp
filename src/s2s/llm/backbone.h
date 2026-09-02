// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// llama.cpp LLM backbone (NemotronH 9B) for the S2S voicechat pipeline.
//
// Embedding-passthrough wrapper: the orchestrator feeds combined input
// embeddings (f32, n_embd=4480) via batch.embd; each step() returns the text
// logits (for text-token sampling) and the final hidden state. The
// function-head matmul is NOT done here — the caller runs it via LLMHeads on
// the returned hidden state.
//
// prefill(), step(), step_batch(), and cleanup() are internally serialized;
// step_batch() owns the shared llama context lock.
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama.h"

namespace nemo_speech::s2s {

class LLMBackbone {
   public:
    explicit LLMBackbone(const std::string& gguf_path, int max_streams = 1);
    ~LLMBackbone();

    LLMBackbone(const LLMBackbone&) = delete;
    LLMBackbone& operator=(const LLMBackbone&) = delete;

    int n_embd() const { return n_embd_; }  // 4480 for NemotronH 9B
    int n_vocab() const { return n_vocab_; }

    // Decode `n_tokens` input embeddings (row-major [n_tokens, n_embd] f32)
    // into the sequence's KV state, chunked at 512 tokens per llama_decode.
    // Logits are requested only on the last row of each chunk.
    void prefill(int64_t ext_seq_id, const float* embeds, int n_tokens);

    // Single-token decode. `embed` is n_embd() f32. Copies the resulting
    // text logits (n_vocab() f32) into out_text_logits and the final hidden
    // state (n_embd() f32) into out_hidden before returning — the underlying
    // llama.cpp buffers are overwritten by the next decode.
    void step(int64_t ext_seq_id, const float* embed, float* out_text_logits, float* out_hidden);

    // One token for each independent sequence in a single llama_decode.
    // Embeddings/results are row-major [batch, ...]. Dynamic batching above
    // this class keeps calls aligned; this method owns the llama context lock.
    void step_batch(
        const int64_t* ext_seq_ids, const float* embeds, int batch, float* out_text_logits,
        float* out_hidden);

    // Remove the sequence's KV state and release its slot. No-op if the
    // external id was never mapped.
    void cleanup(int64_t ext_seq_id);

    // Tokenizer access for system-prompt encoding and output text.
    const llama_vocab* vocab() const { return vocab_; }

    // llama_tokenize with add_special=false, parse_special=true (special
    // tokens written literally in `text` are parsed into their ids).
    std::vector<int32_t> text_to_ids(const std::string& text) const;

    // Inverse: special tokens are rendered as their literal text.
    std::string ids_to_text(const int32_t* ids, int n) const;

   private:
    // Map external sequence id -> internal llama seq slot, allocating from
    // the pool on first use. Throws when the pool is exhausted.
    int map_slot(int64_t ext_seq_id);

    static constexpr int kMaxBatchTokens = 512;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    llama_batch batch_{};

    int n_embd_ = 0;
    int n_vocab_ = 0;
    int n_seq_max_ = 1;

    std::unordered_map<int64_t, int> ext_to_slot_;
    std::unordered_map<int, llama_pos> slot_pos_;  // next position per slot
    std::vector<int> slot_pool_;                   // descending: pop hands out slot 0 first
    std::mutex mu_;
};

}  // namespace nemo_speech::s2s
