// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "audio_pp.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "nvtx_utils.h"

namespace nemo_speech::tts {

AudioPostProcessor::AudioPostProcessor(int samples_per_frame, int future_frames, int window_samples)
    : samples_per_frame_(samples_per_frame), future_frames_(future_frames),
      window_samples_(window_samples) {}

float
AudioPostProcessor::hanningOverlapWeight(size_t i, size_t count) {
    if (count <= 1) {
        return 1.0f;
    }
    static constexpr float pi = 3.14159265358979323846f;
    const float phase = (float)i / (float)(count - 1);
    return 0.5f - 0.5f * std::cos(pi * phase);
}

bool
AudioPostProcessor::writeDecodedAudio(
    const std::vector<float>& audio, int history_frames, bool final_chunk,
    const WriteCallback& write_audio) {
    const ggml_nvtx::range nvtx_range("magpietts_stream_overlap_write_audio");
    (void)final_chunk;
    if (audio.empty()) {
        return true;
    }
    if (!write_audio) {
        return false;
    }
    if (future_frames_ <= 0 && window_samples_ <= 0) {
        return write_audio(audio);
    }
    if (samples_per_frame_ <= 0) {
        return write_audio(audio);
    }

    const int64_t history_samples = (int64_t)history_frames * samples_per_frame_;
    const int64_t future_samples = (int64_t)future_frames_ * samples_per_frame_;
    const int64_t main_begin =
        std::min<int64_t>(std::max<int64_t>(history_samples, 0), (int64_t)audio.size());
    int64_t main_end = (int64_t)audio.size() - std::max<int64_t>(future_samples, 0);
    main_end = std::min<int64_t>(std::max<int64_t>(main_end, main_begin), (int64_t)audio.size());

    const int64_t window = std::max<int64_t>(window_samples_, 0);
    const int64_t raw_begin = main_begin - window;
    const int64_t raw_end = main_end + window;
    const int64_t clamped_begin =
        std::min<int64_t>(std::max<int64_t>(raw_begin, 0), (int64_t)audio.size());
    const int64_t clamped_end =
        std::min<int64_t>(std::max<int64_t>(raw_end, clamped_begin), (int64_t)audio.size());

    const size_t leading_overlap = (size_t)std::min<int64_t>(
        std::max<int64_t>(main_begin - clamped_begin, 0), clamped_end - clamped_begin);
    const size_t trailing_overlap = (size_t)std::min<int64_t>(
        std::max<int64_t>(clamped_end - main_end, 0), clamped_end - clamped_begin);
    std::vector<float> next_audio(
        audio.begin() + (std::ptrdiff_t)clamped_begin, audio.begin() + (std::ptrdiff_t)clamped_end);

    if (!has_pending_audio_) {
        pending_audio_.assign(
            audio.begin() + (std::ptrdiff_t)main_begin, audio.begin() + (std::ptrdiff_t)main_end);
        has_pending_audio_ = true;
        return true;
    }

    std::vector<float> curr_audio = pending_audio_;
    size_t overlap = std::min((size_t)window, leading_overlap);
    overlap = std::min(overlap, curr_audio.size());
    overlap = std::min(overlap, next_audio.size());
    if (overlap > 0) {
        const size_t curr_offset = curr_audio.size() - overlap;
        for (size_t i = 0; i < overlap; ++i) {
            const float w_next = hanningOverlapWeight(i, overlap);
            const float w_curr = 1.0f - w_next;
            curr_audio[curr_offset + i] =
                curr_audio[curr_offset + i] * w_curr + next_audio[i] * w_next;
        }
    }

    if (!write_audio(curr_audio)) {
        return false;
    }

    const size_t pending_begin = leading_overlap;
    size_t pending_end = next_audio.size();
    if (pending_end >= trailing_overlap) {
        pending_end -= trailing_overlap;
    } else {
        pending_end = 0;
    }
    if (pending_end < pending_begin) {
        pending_end = pending_begin;
    }
    pending_audio_.assign(
        next_audio.begin() + (std::ptrdiff_t)pending_begin,
        next_audio.begin() + (std::ptrdiff_t)pending_end);
    has_pending_audio_ = true;
    return true;
}

bool
AudioPostProcessor::flush(const WriteCallback& write_audio) {
    if (!has_pending_audio_) {
        return true;
    }
    if (!write_audio) {
        return false;
    }
    const bool ok = write_audio(pending_audio_);
    pending_audio_.clear();
    has_pending_audio_ = false;
    return ok;
}

}  // namespace nemo_speech::tts
