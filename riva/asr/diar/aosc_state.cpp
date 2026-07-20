// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "aosc_state.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace nemo_speech::asr;

namespace {
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf = std::numeric_limits<float>::infinity();
// NeMo's placeholder for disabled top-k picks (sortformer_modules.max_index).
constexpr int64_t kMaxIndex = 99999;

// Indices of the k largest values in column `spk` of `scores` (n x n_spk).
// Ties break toward the lower frame index (deterministic; torch's order for
// exact ties is unspecified, and exact float ties among finite scores are
// vanishingly rare).
std::vector<int>
topk_column(const std::vector<float>& scores, int n, int n_spk, int spk, int k) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min(k, n);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        const float sa = scores[static_cast<size_t>(a) * n_spk + spk];
        const float sb = scores[static_cast<size_t>(b) * n_spk + spk];
        if (sa != sb)
            return sa > sb;
        return a < b;
    });
    idx.resize(k);
    return idx;
}
}  // namespace

DiarGeometry
DiarGeometry::preset(const std::string& name) {
    if (name == "streaming")
        return riva_streaming();
    if (name == "offline")
        return riva_offline();
    throw std::invalid_argument(
        "unknown diarizer geometry preset '" + name + "' (expected streaming | offline)");
}

void
DiarGeometry::validate(int n_spk, int sil_frames_per_spk, int pos_emb_max_len) const {
    auto fail = [](const std::string& msg) {
        throw std::invalid_argument("diar geometry: " + msg);
    };
    if (chunk_len < 1)
        fail("chunk_len must be >= 1 (got " + std::to_string(chunk_len) + ")");
    if (spkcache_update_period < 1)
        fail(
            "spkcache_update_period must be >= 1 (got " + std::to_string(spkcache_update_period) +
            ")");
    if (fifo_len < 0 || chunk_left_context < 0 || chunk_right_context < 0)
        fail("fifo_len and contexts must be >= 0");
    const int min_spkcache = (1 + sil_frames_per_spk) * n_spk;
    if (spkcache_len < min_spkcache)
        fail(
            "spkcache_len must be >= (1 + sil_frames_per_spk) * n_spk = " +
            std::to_string(min_spkcache) + " (got " + std::to_string(spkcache_len) + ")");
    const int total =
        spkcache_len + fifo_len + chunk_left_context + chunk_len + chunk_right_context;
    if (total > pos_emb_max_len)
        fail(
            "total sequence length " + std::to_string(total) +
            " exceeds the encoder rel-pos table (" + std::to_string(pos_emb_max_len) + ")");
}

AoscState::AoscState(
    const DiarGeometry& geo, const DiarScoringConfig& scoring, int n_spk, int emb_dim)
    : geo_(geo), sc_(scoring), n_spk_(n_spk), emb_dim_(emb_dim) {
    mean_sil_emb_.assign(emb_dim_, 0.f);
}

// NeMo `_get_silence_profile`: running mean embedding over frames whose
// summed speaker activity is below sil_threshold.
void
AoscState::accumulate_silence(const float* embs, const float* preds, int n) {
    int cnt = 0;
    std::vector<double> sum(emb_dim_, 0.0);
    for (int f = 0; f < n; f++) {
        float act = 0.f;
        for (int s = 0; s < n_spk_; s++) act += preds[static_cast<size_t>(f) * n_spk_ + s];
        if (act < sc_.sil_threshold) {
            cnt++;
            for (int d = 0; d < emb_dim_; d++)
                sum[d] += embs[static_cast<size_t>(f) * emb_dim_ + d];
        }
    }
    if (cnt == 0)
        return;
    const int64_t n_new = n_sil_frames_ + cnt;
    const double denom = static_cast<double>(std::max<int64_t>(n_new, 1));
    for (int d = 0; d < emb_dim_; d++) {
        mean_sil_emb_[d] = static_cast<float>(
            (static_cast<double>(mean_sil_emb_[d]) * n_sil_frames_ + sum[d]) / denom);
    }
    n_sil_frames_ = n_new;
}

std::vector<float>
AoscState::update(const float* chunk_embs, int t3, const float* preds, int lc, int rc) {
    const int chunk_valid = t3 - lc - rc;
    if (chunk_valid <= 0)
        return {};

    const int l1 = spk_frames_;
    const int l2 = fifo_frames_;

    // Slice this step's outputs. preds rows: [0,l1) spkcache, [l1,l1+l2) fifo
    // (fresh re-prediction), then the window frames.
    const float* fifo_preds = preds + static_cast<size_t>(l1) * n_spk_;
    const float* chunk_preds = preds + static_cast<size_t>(l1 + l2 + lc) * n_spk_;
    const float* chunk_valid_embs = chunk_embs + static_cast<size_t>(lc) * emb_dim_;

    std::vector<float> emitted(
        chunk_preds, chunk_preds + static_cast<size_t>(chunk_valid) * n_spk_);

    // Append chunk to FIFO (embeddings persist; preds are assembled fresh).
    fifo_.insert(
        fifo_.end(), chunk_valid_embs,
        chunk_valid_embs + static_cast<size_t>(chunk_valid) * emb_dim_);
    std::vector<float> fifo_preds_full(static_cast<size_t>(l2 + chunk_valid) * n_spk_);
    std::memcpy(fifo_preds_full.data(), fifo_preds, static_cast<size_t>(l2) * n_spk_ * 4);
    std::memcpy(
        fifo_preds_full.data() + static_cast<size_t>(l2) * n_spk_, chunk_preds,
        static_cast<size_t>(chunk_valid) * n_spk_ * 4);
    fifo_frames_ = l2 + chunk_valid;

    if (fifo_frames_ > geo_.fifo_len) {
        // Pop the head of the FIFO into the speaker cache
        // (NeMo streaming_update lines 570-601).
        int pop = geo_.spkcache_update_period;
        pop = std::max(pop, chunk_valid - geo_.fifo_len + l2);
        pop = std::min(pop, fifo_frames_);

        const float* pop_embs = fifo_.data();
        const float* pop_preds = fifo_preds_full.data();
        accumulate_silence(pop_embs, pop_preds, pop);

        spkcache_.insert(spkcache_.end(), pop_embs, pop_embs + static_cast<size_t>(pop) * emb_dim_);
        if (spkcache_preds_valid()) {
            spkcache_preds_.insert(
                spkcache_preds_.end(), pop_preds, pop_preds + static_cast<size_t>(pop) * n_spk_);
        }
        spk_frames_ += pop;

        if (spk_frames_ > geo_.spkcache_len && !spkcache_preds_valid()) {
            // First compression: seed cache preds from the CURRENT model
            // output for the spkcache region + the popped preds (NeMo's
            // `spkcache_preds is None` bootstrap).
            spkcache_preds_.resize(static_cast<size_t>(spk_frames_) * n_spk_);
            std::memcpy(spkcache_preds_.data(), preds, static_cast<size_t>(l1) * n_spk_ * 4);
            std::memcpy(
                spkcache_preds_.data() + static_cast<size_t>(l1) * n_spk_, pop_preds,
                static_cast<size_t>(pop) * n_spk_ * 4);
        }
        fifo_.erase(fifo_.begin(), fifo_.begin() + static_cast<size_t>(pop) * emb_dim_);
        fifo_frames_ -= pop;
        if (spk_frames_ > geo_.spkcache_len)
            compress(spkcache_preds_);
    }
    return emitted;
}

// NeMo `_compress_spkcache` at inference (permute_spk=False, no noise):
// score every (frame, speaker), disable non-speech / overlapped frames,
// boost the newest and per-speaker-strongest frames, append per-speaker
// silence sentinels, then keep the spkcache_len globally-highest entries in
// speaker-major order (original frame order within each speaker block).
void
AoscState::compress(const std::vector<float>& cache_preds) {
    const int n = spk_frames_;  // > geo_.spkcache_len
    const int cap = geo_.spkcache_len;
    const int per_spk = cap / n_spk_ - sc_.sil_frames_per_spk;
    const int strong_k = static_cast<int>(std::floor(per_spk * sc_.strong_boost_rate));
    const int weak_k = static_cast<int>(std::floor(per_spk * sc_.weak_boost_rate));
    const int min_pos = static_cast<int>(std::floor(per_spk * sc_.min_pos_scores_rate));
    const float log_half = std::log(0.5f);

    // Scores have shape (n, n_spk).
    std::vector<float> scores(static_cast<size_t>(n) * n_spk_);
    for (int f = 0; f < n; f++) {
        float sum_log1p = 0.f;
        for (int s = 0; s < n_spk_; s++) {
            const float p = cache_preds[static_cast<size_t>(f) * n_spk_ + s];
            sum_log1p += std::log(std::max(1.f - p, sc_.pred_score_threshold));
        }
        for (int s = 0; s < n_spk_; s++) {
            const float p = cache_preds[static_cast<size_t>(f) * n_spk_ + s];
            const float logp = std::log(std::max(p, sc_.pred_score_threshold));
            const float log1p = std::log(std::max(1.f - p, sc_.pred_score_threshold));
            scores[static_cast<size_t>(f) * n_spk_ + s] = logp - log1p + sum_log1p - log_half;
        }
    }

    // Disable non-speech and, when enough clean frames exist, overlaps.
    // (NeMo `_disable_low_scores`)
    std::vector<int> pos_count(n_spk_, 0);
    for (int f = 0; f < n; f++)
        for (int s = 0; s < n_spk_; s++) {
            const size_t i = static_cast<size_t>(f) * n_spk_ + s;
            const bool is_speech = cache_preds[i] > 0.5f;
            if (!is_speech)
                scores[i] = kNegInf;
            if (scores[i] > 0.f)
                pos_count[s]++;
        }
    for (int s = 0; s < n_spk_; s++) {
        if (pos_count[s] < min_pos)
            continue;
        for (int f = 0; f < n; f++) {
            const size_t i = static_cast<size_t>(f) * n_spk_ + s;
            const bool is_speech = cache_preds[i] > 0.5f;
            if (is_speech && !(scores[i] > 0.f))
                scores[i] = kNegInf;
        }
    }

    // Boost frames newer than the cache capacity.
    if (sc_.scores_boost_latest > 0.f)
        for (int f = cap; f < n; f++)
            for (int s = 0; s < n_spk_; s++)
                scores[static_cast<size_t>(f) * n_spk_ + s] += sc_.scores_boost_latest;

    // Apply strong and weak per-speaker top-k boosts.
    for (int s = 0; s < n_spk_; s++)
        for (int f : topk_column(scores, n, n_spk_, s, strong_k))
            scores[static_cast<size_t>(f) * n_spk_ + s] -= 2.f * log_half;
    for (int s = 0; s < n_spk_; s++)
        for (int f : topk_column(scores, n, n_spk_, s, weak_k))
            scores[static_cast<size_t>(f) * n_spk_ + s] -= log_half;

    // Append per-speaker silence sentinels; +inf keeps them in the cache.
    const int n_pad = n + sc_.sil_frames_per_spk;

    // Select the global top `cap` speaker-major scores.
    // flat index = s * n_pad + f, matching NeMo's permute(0,2,1).reshape.
    std::vector<int64_t> flat(static_cast<size_t>(n_spk_) * n_pad);
    std::iota(flat.begin(), flat.end(), 0);
    auto flat_score = [&](int64_t i) -> float {
        const int f = static_cast<int>(i % n_pad);
        if (f >= n)
            return kPosInf;  // silence sentinel rows
        const int s = static_cast<int>(i / n_pad);
        return scores[static_cast<size_t>(f) * n_spk_ + s];
    };
    std::partial_sort(flat.begin(), flat.begin() + cap, flat.end(), [&](int64_t a, int64_t b) {
        const float sa = flat_score(a), sb = flat_score(b);
        if (sa != sb)
            return sa > sb;
        return a < b;
    });
    std::vector<int64_t> picked(flat.begin(), flat.begin() + cap);
    // -inf picks become the disabled placeholder, then indices are sorted
    // ascending: speaker-major blocks, frame order preserved within a block,
    // disabled entries at the end (NeMo `_get_topk_indices`).
    for (auto& i : picked)
        if (flat_score(i) == kNegInf)
            i = kMaxIndex * static_cast<int64_t>(n_pad) + kMaxIndex;
    std::sort(picked.begin(), picked.end());

    // Gather the selected embeddings and predictions.
    std::vector<float> new_cache(static_cast<size_t>(cap) * emb_dim_);
    std::vector<float> new_preds(static_cast<size_t>(cap) * n_spk_, 0.f);
    for (int j = 0; j < cap; j++) {
        const int64_t i = picked[j];
        const int f = static_cast<int>(i % n_pad);
        const bool disabled = i >= static_cast<int64_t>(n_spk_) * n_pad || f >= n;
        if (disabled) {
            std::memcpy(
                new_cache.data() + static_cast<size_t>(j) * emb_dim_, mean_sil_emb_.data(),
                static_cast<size_t>(emb_dim_) * 4);
        } else {
            std::memcpy(
                new_cache.data() + static_cast<size_t>(j) * emb_dim_,
                spkcache_.data() + static_cast<size_t>(f) * emb_dim_,
                static_cast<size_t>(emb_dim_) * 4);
            std::memcpy(
                new_preds.data() + static_cast<size_t>(j) * n_spk_,
                cache_preds.data() + static_cast<size_t>(f) * n_spk_,
                static_cast<size_t>(n_spk_) * 4);
        }
    }
    spkcache_ = std::move(new_cache);
    spkcache_preds_ = std::move(new_preds);
    spk_frames_ = cap;
}
