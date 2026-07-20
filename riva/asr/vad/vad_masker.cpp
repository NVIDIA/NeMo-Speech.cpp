// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "vad_masker.h"

#include <algorithm>

namespace nemo_speech::asr {

namespace {
int
ms_to_frames(float ms, int sample_rate, int hop) {
    return static_cast<int>(ms / 1000.0f * static_cast<float>(sample_rate) / hop + 0.5f);
}
}  // namespace

VadMasker::VadMasker(SileroVad* vad, int n_mels, VadMaskerCfg cfg)
    : vad_(vad), n_mels_(n_mels), cfg_(cfg) {
    const int sr = vad_->config().sample_rate;
    const int hop = vad_->hop_length();  // single source: set by set_binarizer
    pad_onset_frames_ = ms_to_frames(cfg_.pad_onset_ms, sr, hop);
    pad_offset_frames_ = ms_to_frames(cfg_.pad_offset_ms, sr, hop);
    merge_off_frames_ = ms_to_frames(cfg_.min_duration_off_ms, sr, hop);
    // decide_masked scans for the nearest speech frame each side; it needs to
    // reach a pad edge or detect that the enclosing gap exceeds the merge
    // threshold, so max(merge, pad) + 1 frames is sufficient.
    max_scan_ = std::max({merge_off_frames_, pad_onset_frames_, pad_offset_frames_}) + 1;
}

bool
VadMasker::decide_masked(int64_t g) const {
    if (vad_->frame_speech(g))
        return false;  // speech frames are never masked
    // Find the nearest speech frame on each side (bounded scan).
    int64_t left = -1, right = -1;
    for (int d = 1; d <= max_scan_; d++) {
        if (vad_->frame_speech(g - d)) {
            left = g - d;
            break;
        }
    }
    for (int d = 1; d <= max_scan_; d++) {
        if (vad_->frame_speech(g + d)) {
            right = g + d;
            break;
        }
    }
    // pad: within pad_offset of the previous segment's end, or pad_onset of the
    // next segment's start -> keep as speech (riva pad_onset/pad_offset).
    if (left >= 0 && (g - left) <= pad_offset_frames_)
        return false;
    if (right >= 0 && (right - g) <= pad_onset_frames_)
        return false;
    // merge: silence gap bounded by speech on both sides and short enough ->
    // keep (riva MergeShortNonSpeechSegments, gap <= min_duration_off).
    if (left >= 0 && right >= 0 && (right - left - 1) <= merge_off_frames_)
        return false;
    return true;
}

void
VadMasker::apply(float* mel, int n_mel_frames, int64_t first_global_frame) {
    for (int i = 0; i < n_mel_frames; i++) {
        if (decide_masked(first_global_frame + i)) {
            float* f = mel + static_cast<size_t>(i) * n_mels_;
            std::fill(f, f + n_mels_, cfg_.mask_value);
        }
    }
}

}  // namespace nemo_speech::asr
