// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstdio>
#include <vector>

#include "aosc_state.h"

using namespace nemo_speech::asr;

namespace {

bool
near(float a, float b) {
    return std::fabs(a - b) < 1e-6f;
}

bool
test_channel_birth_gate() {
    ChannelBirthGate gate(4);
    std::vector<float> timeline;

    gate.append(
        {0.99f, 0.01f, 0.01f, 0.01f, 0.99f, 0.01f, 0.01f, 0.01f, 0.99f, 0.01f, 0.01f, 0.01f, 0.99f,
         0.01f, 0.01f, 0.01f},
        timeline);
    if (!gate.is_established(0))
        return false;

    std::vector<float> redraw;
    for (int i = 0; i < 20; i++) redraw.insert(redraw.end(), {0.4f, 0.01f, 0.01f, 0.8f});
    const size_t redraw_offset = timeline.size();
    gate.append(redraw, timeline);
    if (gate.is_established(3))
        return false;
    for (size_t i = redraw_offset; i < timeline.size(); i += 4)
        if (!near(timeline[i], 0.8f) || !near(timeline[i + 3], 0.0f))
            return false;

    gate.append({0.01f, 0.99f, 0.01f, 0.01f, 0.01f, 0.99f, 0.01f, 0.01f}, timeline);
    const size_t handoff_offset = timeline.size() - 8;
    gate.append({0.01f, 0.99f, 0.01f, 0.01f, 0.01f, 0.99f, 0.01f, 0.01f}, timeline);
    if (!gate.is_established(1))
        return false;
    for (size_t i = handoff_offset; i < timeline.size(); i += 4)
        if (!near(timeline[i + 1], 0.99f))
            return false;

    const size_t revision_offset = timeline.size();
    gate.append({0.01f, 0.01f, 0.99f, 0.01f, 0.01f, 0.01f, 0.99f, 0.01f}, timeline);
    if (gate.is_established(2))
        return false;
    gate.append({0.01f, 0.01f, 0.99f, 0.01f, 0.01f, 0.01f, 0.99f, 0.01f}, timeline);
    if (!gate.is_established(2))
        return false;
    for (size_t i = revision_offset; i < timeline.size(); i += 4)
        if (!near(timeline[i + 2], 0.99f))
            return false;

    gate.append(
        {0.01f, 0.01f, 0.01f, 0.99f, 0.01f, 0.01f, 0.01f, 0.99f, 0.01f, 0.01f, 0.01f, 0.99f, 0.01f,
         0.01f, 0.01f, 0.99f},
        timeline);
    if (!gate.is_established(3))
        return false;
    return true;
}

}  // namespace

int
main() {
    if (!test_channel_birth_gate()) {
        std::fprintf(stderr, "[FAIL] transient speaker channel was established\n");
        return 1;
    }
    std::printf("[PASS] transient speaker channels are relabeled\n");
    return 0;
}
