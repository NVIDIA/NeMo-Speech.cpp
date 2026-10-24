// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "decoder.h"

namespace nemo_speech::asr {

struct BoundedWordTime {
    int32_t start_ms = 0;
    int32_t end_ms = 0;
    bool has_audio_position = false;
};

// Project a result copy, never the decoder's frame clock or recurrent state.
// The exclusive frame limit counts causal outputs covering real Mel frames.
inline void
bound_word_frames(WordTiming& word, int64_t frame_limit) {
    word.has_audio_position = word.has_audio_position && frame_limit > 0 &&
                              word.start_frame < frame_limit && word.end_frame > 0;
    if (frame_limit <= 0) {
        word.start_frame = word.end_frame = 0;
        return;
    }
    word.start_frame = std::clamp<int64_t>(word.start_frame, 0, frame_limit - 1);
    word.end_frame = std::clamp<int64_t>(word.end_frame, word.start_frame + 1, frame_limit);
}

inline int32_t
real_audio_extent_ms(uint64_t samples, int rate) {
    if (rate <= 0)
        return 0;
    const long double ms = static_cast<long double>(samples) * 1000.0L / rate;
    return static_cast<int32_t>(
        std::min(ms, static_cast<long double>(std::numeric_limits<int32_t>::max())));
}

// The final encoder interval can straddle EOF even after frame bounding.
// Bound milliseconds to original-rate input samples too. Preserve legitimate
// zero-width ITN spans; this projection does not invent phonetic alignment.
inline BoundedWordTime
bound_word_time(
    int64_t start_frame, int64_t end_frame, double frame_ms, uint64_t input_samples,
    int input_rate) {
    if (input_rate <= 0 || input_samples == 0 || frame_ms <= 0.0 || !std::isfinite(frame_ms))
        return {};
    const double audio_end =
        static_cast<double>(static_cast<long double>(input_samples) * 1000.0L / input_rate);
    const double limit = real_audio_extent_ms(input_samples, input_rate);
    const double start = start_frame * frame_ms;
    const double end = end_frame * frame_ms;
    const auto project = [limit](double t) {
        return static_cast<int32_t>(std::clamp(std::floor(t + 0.5), 0.0, limit));
    };
    const int32_t bounded_start = project(start);
    return {
        bounded_start, std::max(bounded_start, project(end)),
        end_frame >= start_frame && start < audio_end && (end > 0.0 || start >= 0.0)};
}

}  // namespace nemo_speech::asr
