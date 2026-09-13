// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Regression test for DiarStream::speaker_for_frames() under timeline
// compaction: a caller that tags words long after they were spoken (e.g. a
// whole-file batch pass, see Recognizer::recognize()) must still get the
// correct speaker for a range that predates the retained live window,
// instead of the frontier-extrapolation clamp silently substituting whatever
// sits at the live window's edge.
#include <cstdio>
#include <vector>

#include "diar_pipeline.h"

using namespace nemo_speech::asr;

namespace {

bool
test_frozen_prefix_used_for_compacted_words() {
    // probs_base=100: frames [0,100) were already compacted into `frozen`.
    // The live window (frame 100) leans speaker 0; the true history at frame
    // 20 (frozen segment [0,50) -> speaker 1) disagrees. A caller tagging
    // this word long after the fact must get the frozen answer, not the
    // live window's edge.
    const std::vector<DiarSegment> frozen = {{0.0, 50.0, 1}};
    const std::vector<float> live = {0.9f, 0.1f};  // one frame, n_spk=2, favors speaker 0
    const int got = speaker_for_frame_range(
        live, /*probs_base=*/100, /*n_spk=*/2,
        /*sec_per_frame=*/1.0, frozen,
        /*start_frame=*/20, /*end_frame=*/22);
    if (got != 1) {
        std::fprintf(stderr, "[FAIL] expected frozen speaker 1, got %d\n", got);
        return false;
    }
    return true;
}

bool
test_live_window_unaffected() {
    // A range fully inside the live window must still resolve exactly as
    // before: mean probability across the range, argmax.
    const std::vector<DiarSegment> frozen = {{0.0, 50.0, 1}};
    const std::vector<float> live = {
        0.1f, 0.9f, 0.2f, 0.8f};  // 2 frames, n_spk=2, both favor speaker 1
    const int got = speaker_for_frame_range(
        live, /*probs_base=*/100, /*n_spk=*/2,
        /*sec_per_frame=*/1.0, frozen,
        /*start_frame=*/100, /*end_frame=*/102);
    if (got != 1) {
        std::fprintf(stderr, "[FAIL] expected live-window speaker 1, got %d\n", got);
        return false;
    }
    return true;
}

bool
test_gap_between_frozen_segments_falls_back_to_nearest() {
    // A query landing in a gap between frozen segments (e.g. a silence the
    // segmenter's hysteresis dropped) has no direct overlap; fall back to
    // whichever neighbor is temporally closer.
    const std::vector<DiarSegment> frozen = {{0.0, 10.0, 0}, {20.0, 30.0, 1}};
    const std::vector<float> live = {0.5f, 0.5f};
    const int got = speaker_for_frame_range(
        live, /*probs_base=*/100, /*n_spk=*/2,
        /*sec_per_frame=*/1.0, frozen,
        /*start_frame=*/12, /*end_frame=*/13);
    if (got != 0) {
        std::fprintf(stderr, "[FAIL] expected nearest frozen speaker 0, got %d\n", got);
        return false;
    }
    return true;
}

}  // namespace

int
main() {
    bool ok = true;
    ok &= test_frozen_prefix_used_for_compacted_words();
    ok &= test_live_window_unaffected();
    ok &= test_gap_between_frozen_segments_falls_back_to_nearest();
    if (!ok) {
        std::fprintf(stderr, "[FAIL] speaker_for_frame_range compaction regression\n");
        return 1;
    }
    std::printf("[PASS] speaker_for_frame_range resolves compacted ranges via frozen segments\n");
    return 0;
}
