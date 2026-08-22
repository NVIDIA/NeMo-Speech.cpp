// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Cache-aware streaming encoder: one shared, fixed-shape FastConformer encoder
// Session whose per-layer K/V caches are two planes of one persistent arena
// and whose convolution cache is a persistent matrix, with one indexed row per
// stream. A microbatch supplies stream IDs, gathers those rows, and scatters
// updated cache rows in-graph. State stays device-resident with no host hand-off.
//
// An optional output tail is composed into the same graph (RNNT uses prompt
// fusion + joint.enc), avoiding an intermediate host round trip.
#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "batching.h"
#include "fastconformer.h"
#include "runtime.h"

namespace nemo_speech::asr {

class CacheAwareEncoder {
   public:
    class State {
       public:
        State() = default;
        ~State();
        State(State&&) noexcept;
        State& operator=(State&&) noexcept;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        bool valid() const { return owner_ != nullptr && slot_ >= 0; }

       private:
        friend class CacheAwareEncoder;
        State(CacheAwareEncoder* owner, int slot, bool needs_reset)
            : owner_(owner), slot_(slot), needs_reset_(needs_reset) {}
        CacheAwareEncoder* owner_ = nullptr;
        int slot_ = -1;
        int ring_head_ = 0;
        bool needs_reset_ = false;
    };
    // Borrows `bm` + `loader` (owned by AsrModel for this encoder's lifetime).
    // `base_cfg` is the model's non-cache EncoderConfig; `n_mels` sizes the mel
    // input. `output_tail`, when present, consumes the encoder output and may
    // consume one auxiliary `[tail_input_dim, T]` input. The Session is built
    // lazily so set_right_ctx() can apply before construction.
    CacheAwareEncoder(
        ggml_runtime::BackendManager& bm, ggml_runtime::GGUFLoader* loader,
        const EncoderConfig& base_cfg, int n_mels, ggml_runtime::Module* output_tail = nullptr,
        int output_dim = 0, const BatchingConfig& batching = {});
    ~CacheAwareEncoder();

    // Override the model's trained right-context (R) before the Session is built.
    // Throws if changed after the Session is built (all streams share one).
    void set_right_ctx(int R);

    // Allocate a fresh, zero-initialised per-stream cache handle. Each stream
    // (runner) holds one and passes it to encode(); builds the Session if needed.
    State make_state();
    void reset_state(State& state);

    // Fixed number of encoder frames produced per cache-aware step (1 + R).
    int chunk_frames();

    // One cache-aware encoder step for the stream owning `state`. `mel` is
    // (n_mels x n_mel_frames) frame-major (= pre_encode_cache_size + sub*(1+R)
    // frames). The stream's K/V/conv cache lives in `state` (device-resident,
    // updated in-graph). `enc_out` receives (output_dim x T_enc) flat-packed
    // (d_model without a tail); T_enc = 1 + R. Thread-safe.
    void encode(
        State& state, const float* mel, int n_mel_frames, const float* attn_mask, int attn_mask_len,
        std::vector<float>& enc_out, int& T_enc, const float* tail_input = nullptr,
        int tail_input_dim = 0, ggml_runtime::DeviceTensor* device_output = nullptr);

    // Null before the first use builds the session.
    ggml_runtime::Session* session() const { return session_.get(); }
    BatchMetrics batch_metrics() const;

   private:
    void ensure_session();  // build the Session on first use; called under mu_

    ggml_runtime::BackendManager* bm_ = nullptr;
    ggml_runtime::GGUFLoader* loader_ = nullptr;
    int n_mels_ = 0;
    ggml_runtime::Module* output_tail_ = nullptr;  // borrowed from RnntModel
    int output_dim_ = 0;
    BatchingConfig batching_;
    int arena_slots_ = 0;

    // Guards the lazy Session build + right-context config. The per-run cache
    // binding + compute are serialized by the session's compute_mutex.
    std::mutex mu_;
    EncoderConfig cfg_;  // cache-aware variant of base_cfg (derived in ensure_session)
    class EncoderRoot;
    std::unique_ptr<FastConformerEncoder> encoder_;
    std::unique_ptr<EncoderRoot> root_;
    std::unique_ptr<ggml_runtime::Session> session_;
    class EncoderBatcher;
    std::unique_ptr<EncoderBatcher> batcher_;
    std::mutex slots_mu_;
    std::vector<bool> slots_used_;
    std::vector<bool> slots_need_reset_;
    void release_slot(int slot);
    void zero_slot(int slot);
    void zero_slots(std::vector<int> slots);
    bool ring_cache_enabled_ = false;
    int right_ctx_ = -1;  // -1 = use the model's stored train_right_ctx
};

}  // namespace nemo_speech::asr
