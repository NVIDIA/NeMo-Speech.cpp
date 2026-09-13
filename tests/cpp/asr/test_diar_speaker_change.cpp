// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Unit tests for detect_speaker_change(): the pure comparison logic behind
// the realtime WebSocket's speaker_diarization.changed event. Runs entirely
// on synthetic DiarSegment lists, no model required -- mirrors the
// speaker_for_frame_range test pattern in test_diar_frame_lookup.cpp.
#include <cstdio>
#include <optional>
#include <vector>

#include "diar_pipeline.h"

using namespace nemo_speech::asr;

namespace {

bool
test_first_segment_ever_is_reported() {
    // Nothing reported yet (nullopt): the very first segment always counts
    // as a change, since there's no prior speaker to compare against.
    const std::vector<DiarSegment> segs = {{0.0, 2.0, 0}};
    const auto got = detect_speaker_change(segs, std::nullopt);
    if (!got || got->speaker != 0 || got->start_time != 0.0) {
        std::fprintf(stderr, "[FAIL] expected first segment reported as speaker 0 at t=0.0\n");
        return false;
    }
    return true;
}

bool
test_same_speaker_continuing_is_not_reported() {
    // The most recent segment's speaker matches what was already reported:
    // this is the same speaker's turn continuing, not a change.
    const std::vector<DiarSegment> segs = {{0.0, 2.0, 0}, {2.0, 5.0, 0}};
    const auto got = detect_speaker_change(segs, /*last_reported=*/0);
    if (got) {
        std::fprintf(stderr, "[FAIL] expected no change when speaker is unchanged\n");
        return false;
    }
    return true;
}

bool
test_real_change_is_reported_with_new_segments_start_time() {
    const std::vector<DiarSegment> segs = {{0.0, 2.0, 0}, {2.0, 5.0, 1}};
    const auto got = detect_speaker_change(segs, /*last_reported=*/0);
    if (!got || got->speaker != 1 || got->start_time != 2.0) {
        std::fprintf(stderr, "[FAIL] expected speaker 1 reported at t=2.0\n");
        return false;
    }
    return true;
}

bool
test_empty_segments_is_not_reported() {
    // No confirmed segments yet at all (e.g. right at stream start, before
    // onset hysteresis confirms anything).
    const std::vector<DiarSegment> segs = {};
    const auto got = detect_speaker_change(segs, std::nullopt);
    if (got) {
        std::fprintf(stderr, "[FAIL] expected no change when there are no segments\n");
        return false;
    }
    return true;
}

bool
test_third_speaker_after_two_reported_changes() {
    // Guards against an implementation that only ever compares against the
    // *first* reported speaker instead of the *last* one.
    const std::vector<DiarSegment> segs = {{0.0, 2.0, 0}, {2.0, 4.0, 1}, {4.0, 6.0, 2}};
    const auto got = detect_speaker_change(segs, /*last_reported=*/1);
    if (!got || got->speaker != 2 || got->start_time != 4.0) {
        std::fprintf(stderr, "[FAIL] expected speaker 2 reported at t=4.0\n");
        return false;
    }
    return true;
}

bool
test_tracker_first_segment_ever_is_not_reported() {
    SpeakerChangeTracker tracker;
    const std::vector<DiarSegment> segs = {{0.0, 2.0, 0}};
    const auto got = tracker.observe(segs);
    if (got) {
        std::fprintf(stderr, "[FAIL] expected no event on the tracker's first observation\n");
        return false;
    }
    return true;
}

bool
test_tracker_second_call_same_speaker_is_not_reported() {
    SpeakerChangeTracker tracker;
    tracker.observe({{0.0, 2.0, 0}});  // establishes baseline, silently
    const auto got = tracker.observe({{0.0, 2.0, 0}, {2.0, 5.0, 0}});
    if (got) {
        std::fprintf(stderr, "[FAIL] expected no event when speaker is unchanged after baseline\n");
        return false;
    }
    return true;
}

bool
test_tracker_genuine_change_after_baseline_is_reported() {
    SpeakerChangeTracker tracker;
    tracker.observe({{0.0, 2.0, 0}});  // establishes baseline, silently
    const auto got = tracker.observe({{0.0, 2.0, 0}, {2.0, 5.0, 1}});
    if (!got || got->speaker != 1 || got->start_time != 2.0) {
        std::fprintf(stderr, "[FAIL] expected speaker 1 reported at t=2.0 after baseline\n");
        return false;
    }
    return true;
}

bool
test_tracker_second_genuine_change_is_also_reported() {
    SpeakerChangeTracker tracker;
    tracker.observe({{0.0, 2.0, 0}});                                        // baseline
    tracker.observe({{0.0, 2.0, 0}, {2.0, 4.0, 1}});                         // first real change
    const auto got = tracker.observe({{0.0, 2.0, 0}, {2.0, 4.0, 1}, {4.0, 6.0, 2}});
    if (!got || got->speaker != 2 || got->start_time != 4.0) {
        std::fprintf(stderr, "[FAIL] expected speaker 2 reported at t=4.0 as a second real change\n");
        return false;
    }
    return true;
}

bool
test_tracker_empty_segments_never_reported_and_does_not_corrupt_state() {
    SpeakerChangeTracker tracker;
    const auto got1 = tracker.observe({});
    if (got1) {
        std::fprintf(stderr, "[FAIL] expected no event on empty segments\n");
        return false;
    }
    // Still no baseline established (empty segments didn't count as the
    // "first observation") -- the real first segment, arriving later,
    // should still be treated as the baseline and not reported.
    const auto got2 = tracker.observe({{0.0, 2.0, 0}});
    if (got2) {
        std::fprintf(stderr, "[FAIL] expected the real first segment to still establish baseline silently\n");
        return false;
    }
    return true;
}

}  // namespace

int
main() {
    bool ok = true;
    ok &= test_first_segment_ever_is_reported();
    ok &= test_same_speaker_continuing_is_not_reported();
    ok &= test_real_change_is_reported_with_new_segments_start_time();
    ok &= test_empty_segments_is_not_reported();
    ok &= test_third_speaker_after_two_reported_changes();
    ok &= test_tracker_first_segment_ever_is_not_reported();
    ok &= test_tracker_second_call_same_speaker_is_not_reported();
    ok &= test_tracker_genuine_change_after_baseline_is_reported();
    ok &= test_tracker_second_genuine_change_is_also_reported();
    ok &= test_tracker_empty_segments_never_reported_and_does_not_corrupt_state();
    if (!ok) {
        std::fprintf(stderr, "[FAIL] detect_speaker_change / SpeakerChangeTracker\n");
        return 1;
    }
    std::printf(
        "[PASS] detect_speaker_change and SpeakerChangeTracker (10 tests) report real speaker "
        "transitions only\n");
    return 0;
}
