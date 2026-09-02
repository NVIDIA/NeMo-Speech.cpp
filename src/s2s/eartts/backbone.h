// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// llama.cpp TTS backbone (Gemma3) for the S2S voicechat pipeline.
//
// Embedding-passthrough wrapper for the EarTTS Gemma3 backbone. CFG runs two
// llama seq slots per logical stream (cond + uncond): the input embedder
// (outside this class) produces both embeddings per step, both ride one
// llama_decode, and the sampler mixes the two returned hidden states.
//
// Position budget: each slot gets n_ctx / n_seq_max = 4500 positions; the
// prompt prefill (~37 frames) leaves roughly a 4463-step budget per stream.
// No overflow handling — llama_decode fails when a slot's KV range is
// exhausted.
//
// Threading contract: single-threaded. One shared llama_batch is reused
// across prefill_pair()/step_pair() calls.
#pragma once

#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "llama.h"

namespace nemo_speech::s2s {

class EarTTSBackbone {
   public:
    explicit EarTTSBackbone(const std::string& gguf_path, int max_streams = 1);
    ~EarTTSBackbone();

    EarTTSBackbone(const EarTTSBackbone&) = delete;
    EarTTSBackbone& operator=(const EarTTSBackbone&) = delete;

    int n_embd() const { return n_embd_; }  // 1152 for Gemma3 EarTTS

    // Number of single-frame decode steps left after the immutable prompt.
    int step_capacity() const;

    // Decode the immutable voice prompt once into two reserved template
    // sequences. Every allocate_pair() clones this exact KV state.
    void initialize_prompt_template(
        const float* cond_emb, const float* uncond_emb, int n_tokens, float* out_h_cond,
        float* out_h_uncond);

    // Allocate a consecutive-ascending (cond, uncond) slot pair and clone the
    // prefilled prompt template into it.
    //
    // llama.cpp's batch splitter split_equal(sequential=true)
    // (llama-batch.cpp:536) keeps the two CFG tokens in ONE ubatch only when
    // batch.seq_id[1] == batch.seq_id[0] + 1. Otherwise it splits into two
    // ubatches with separate sub-cgraphs whose nodes get reordered across
    // calls, preventing llama.cpp's CUDA-graph warmup from completing.
    std::pair<int, int> allocate_pair();

    // Remove both slots' KV state and return them to the pool.
    void free_pair(int cond_slot, int uncond_slot);

    // Restore an allocated pair to the immutable voice-prompt state.
    void reset_pair(int cond_slot, int uncond_slot);

    // Prompt prefill: decode `n_tokens` cond embeddings into cond_slot, then
    // n_tokens uncond embeddings into uncond_slot (two separate chunked
    // decodes, <=512 tokens each, positions 0..n-1 per slot). The last
    // hidden state of each branch is fetched immediately after its decode
    // (before the other branch's decode overwrites the ctx buffer) into
    // out_h_cond / out_h_uncond (n_embd() f32 each).
    void prefill_pair(
        int cond_slot, int uncond_slot, const float* cond_emb, const float* uncond_emb,
        int n_tokens, float* out_h_cond, float* out_h_uncond);

    // Single-frame CFG step: one 2-token llama_decode (row 0 cond, row 1
    // uncond). cond_emb / uncond_emb are n_embd() f32 each; the hidden
    // states are copied into out_h_cond / out_h_uncond. Advances both
    // positions.
    void step_pair(
        int cond_slot, int uncond_slot, const float* cond_emb, const float* uncond_emb,
        float* out_h_cond, float* out_h_uncond);

    // Batched form: each row is one logical CFG pair. Embeddings and outputs
    // are row-major [batch, n_embd].
    void step_pairs(
        const int* cond_slots, const int* uncond_slots, const float* cond_emb,
        const float* uncond_emb, int batch, float* out_h_cond, float* out_h_uncond);

   private:
    // Chunked decode of [n_tokens, n_embd] f32 rows into one slot; logits
    // requested only on the final chunk's last row.
    void decode_chunked(int slot, const float* embs, int n_tokens);

    // Copy the last logits-marked position's hidden state. Must run before
    // the next llama_decode (which overwrites the buffer).
    void fetch_last_hidden(float* out) const;

    static constexpr int kMaxBatchTokens = 512;
    static constexpr int kPositionsPerSequence = 4500;

    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    llama_batch batch_{};

    int n_embd_ = 0;
    int n_seq_max_ = 0;
    int template_cond_slot_ = 0;
    int template_uncond_slot_ = 1;
    bool prompt_template_ready_ = false;

    std::vector<llama_pos> slot_pos_;  // next position per slot, from 0
    std::vector<int> slot_pool_;       // pop from back; handed out in pairs
    std::mutex mu_;
};

}  // namespace nemo_speech::s2s
