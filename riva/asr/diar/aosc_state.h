// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Arrival-Order Speaker Cache (AOSC) streaming state for Sortformer.
//
// Host-side port of NeMo's synchronous `SortformerModules.streaming_update`
// and `_compress_spkcache` inference behavior.
//
// Per chunk the model re-predicts the whole [spkcache | fifo | chunk]
// concatenation; only the embeddings are persisted (fifo preds are refreshed
// from the current output each step, spkcache preds persist once seeded at
// the first compression - NeMo's `spkcache_preds is None` sentinel).
#pragma once

#include <string>
#include <vector>

#include "sortformer_model.h"  // DiarScoringConfig

namespace nemo_speech::asr {

// Streaming geometry in 80 ms encoder frames. Prefer the named presets over
// setting individual fields.
struct DiarGeometry {
    int spkcache_len = 160;           // speaker cache capacity
    int fifo_len = 80;                // FIFO capacity
    int chunk_len = 20;               // new frames consumed per step (1.6 s)
    int spkcache_update_period = 80;  // FIFO pop on overflow (riva refresh_rate=0)
    int chunk_left_context = 0;
    int chunk_right_context = 0;

    // The "offline" preset still uses AOSC streaming with larger chunks and caches.
    static DiarGeometry riva_streaming() { return {}; }
    static DiarGeometry riva_offline() { return {312, 100, 100, 100, 0, 0}; }
    // Throws std::invalid_argument for unknown names.
    static DiarGeometry preset(const std::string& name);

    // Validates positive chunk/update sizes, non-negative contexts and FIFO,
    // the minimum speaker-cache budget, and the positional-encoding limit.
    void validate(int n_spk, int sil_frames_per_spk, int pos_emb_max_len) const;
};

class AoscState {
   public:
    AoscState(const DiarGeometry& geo, const DiarScoringConfig& scoring, int n_spk, int emb_dim);

    // One streaming update after a model chunk.
    //   chunk_embs: (t3, emb_dim) pre-encode embeddings of the whole window
    //               (including lc/rc frames; trimmed internally).
    //   preds:      ((spkcache_frames + fifo_frames + t3), n_spk) model output.
    //   lc, rc:     this window's left/right context in encoder frames.
    // Returns the emitted chunk predictions ((t3 - lc - rc), n_spk).
    std::vector<float> update(const float* chunk_embs, int t3, const float* preds, int lc, int rc);

    int spkcache_frames() const { return spk_frames_; }
    int fifo_frames() const { return fifo_frames_; }
    const std::vector<float>& spkcache() const { return spkcache_; }
    const std::vector<float>& fifo() const { return fifo_; }

    // True once the cache preds were seeded at the first compression (NeMo's
    // `spkcache_preds is None` sentinel; emptiness carries the same bit).
    bool spkcache_preds_valid() const { return !spkcache_preds_.empty(); }
    const std::vector<float>& spkcache_preds() const { return spkcache_preds_; }
    const std::vector<float>& mean_sil_emb() const { return mean_sil_emb_; }
    int64_t n_sil_frames() const { return n_sil_frames_; }

   private:
    void accumulate_silence(const float* embs, const float* preds, int n);
    void compress(const std::vector<float>& seed_preds);

    DiarGeometry geo_;
    DiarScoringConfig sc_;
    int n_spk_;
    int emb_dim_;

    std::vector<float> spkcache_;        // spk_frames_ x emb_dim
    std::vector<float> spkcache_preds_;  // spk_frames_ x n_spk (empty until seeded)
    int spk_frames_ = 0;
    std::vector<float> fifo_;  // fifo_frames_ x emb_dim
    int fifo_frames_ = 0;
    std::vector<float> mean_sil_emb_;  // emb_dim
    int64_t n_sil_frames_ = 0;
};

}  // namespace nemo_speech::asr
