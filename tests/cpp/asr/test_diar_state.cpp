// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "aosc_state.h"
#include "diar_pipeline.h"
#include "numeric_parity.h"

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

void
require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

std::vector<float>
repeat_frame(int count, const std::vector<float>& frame) {
    std::vector<float> out;
    for (int i = 0; i < count; i++) out.insert(out.end(), frame.begin(), frame.end());
    return out;
}

void
test_birth_gate_settled_frames() {
    ChannelBirthGate gate(4);
    std::vector<float> timeline;
    gate.append(repeat_frame(4, {0.99f, 0.01f, 0.01f, 0.01f}), timeline);
    require(gate.settled_frames() == 4, "gate: established speech settles immediately");
    gate.append(repeat_frame(10, {0.02f, 0.01f, 0.25f, 0.01f}), timeline);
    require(gate.settled_frames() == 14, "gate: frames below speech probability are settled");
    gate.append(repeat_frame(2, {0.30f, 0.01f, 0.01f, 0.80f}), timeline);
    require(
        !gate.is_established(3) && gate.settled_frames() == 14,
        "gate: an unborn winning channel holds the frontier");
    gate.append(repeat_frame(20, {0.99f, 0.01f, 0.01f, 0.01f}), timeline);
    require(gate.settled_frames() == 14, "gate: the frontier waits for the revision window");
    gate.append(
        repeat_frame(ChannelBirthGate::revision_frames, {0.99f, 0.01f, 0.01f, 0.01f}), timeline);
    require(
        gate.settled_frames() == 36 + ChannelBirthGate::revision_frames,
        "gate: the frontier advances once the revision window passes");
}

void
test_geometry_and_word_cadence() {
    DiarConfig config;
    nemo_speech::common::ParameterParser parser;
    parser.Register("diar", config);
    auto resolve = [&](bool v3) { return config.resolved_geometry().resolved(v3); };
    auto g = resolve(false);
    require(g.chunk_len == 20 && g.spkcache_len == 160 && g.fifo_len == 80, "V2 defaults");
    g = resolve(true);
    require(g.chunk_len == 13 && g.spkcache_len == 264 && g.chunk_left_context == 0, "V3 defaults");
    for (int chunk : {20, 9}) {
        bool consumed = false;
        const auto value = std::to_string(chunk);
        require(
            parser.ParseCliArg("--diar-chunk", value.c_str(), &consumed) && consumed,
            "parse chunk");
        g = resolve(true);
        require(g.chunk_len == chunk, "explicit chunk must survive resolution");
        require(
            g.spkcache_len == 264 && g.fifo_len == 80 && g.spkcache_update_period == 40 &&
                g.chunk_left_context == 0 && g.chunk_right_context == 1,
            "inherit V3 fields");
    }
    bool consumed = false;
    require(parser.ParseCliArg("--diar-lc", "0", &consumed), "parse zero left context");
    require(parser.ParseCliArg("--diar-rc", "0", &consumed), "parse zero right context");
    require(parser.ParseCliArg("--diar-fifo", "0", &consumed), "parse zero fifo");
    g = resolve(true);
    require(
        g.chunk_left_context == 0 && g.chunk_right_context == 0 && g.fifo_len == 0,
        "explicit zeros");
    config.preset = "streaming";
    g = resolve(true);
    require(g.chunk_len == 20 && g.fifo_len == 80, "explicit preset replaces individual keys");
    require(g.resolved(true).chunk_len == 20, "resolution is idempotent");
    auto direct = DiarGeometry{};
    direct.chunk_len = 20;
    require(
        direct.resolved(true).chunk_len == 20 && direct.resolved(true).fifo_len == 80,
        "direct overrides");
    SortformerModelConfig model;
    require(model.word_anchor_frames() == 2, "V2 word anchor must remain two frames");
    model.output_subsampling_factor = 1;
    require(model.word_anchor_frames() == 16, "V3 word anchor must be sixteen frames");
}

void
test_finite_parity() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    require(finite_max_abs_diff({1.0f}, {0.5f}) == 0.5f, "finite comparison");
    for (float bad : {nan, inf, -inf}) {
        require(std::isinf(finite_max_abs_diff({bad}, {0.0f})), "reject invalid actual");
        require(std::isinf(finite_max_abs_diff({0.0f}, {bad})), "reject invalid reference");
        require(std::isinf(finite_max_abs_diff({bad}, {bad})), "reject matching invalid values");
        require(std::isinf(finite_max_abs_diff({bad}, {0.0f}, 1)), "reject invalid prefix");
    }
    for (int which = 0; which < 3; ++which) {
        bool mismatch = false;
        try {
            if (which == 0)
                finite_max_abs_diff({}, {0.0f});
            else if (which == 1)
                finite_max_abs_diff({}, {0.0f}, 1);
            else
                finite_max_abs_diff({0.0f}, {}, 1);
        }
        catch (const std::runtime_error&) {
            mismatch = true;
        }
        require(mismatch, "reject mismatched shapes");
    }
}

void
test_invalid_rotary_geometry() {
    for (int which = 0; which < 4; ++which) {
        RopeTransformerConfig cfg;
        if (which == 0)
            cfg.n_heads = 0;
        if (which == 1)
            cfg.rotary_fraction = std::numeric_limits<float>::quiet_NaN();
        if (which == 2)
            cfg.rotary_fraction = 2.0f;
        if (which == 3)
            cfg.n_layers = -1;
        bool rejected = false;
        try {
            RopeTransformerEncoder encoder("invalid", cfg);
        }
        catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "reject invalid rotary geometry before allocation");
    }
}

bool
test_learned_silence_embedding() {
    DiarGeometry geo{8, 0, 1, 1, 0, 0};
    DiarScoringConfig scoring;
    const std::vector<float> learned{2.0f, -3.0f};
    AoscState state(geo, scoring, 2, 2, learned);
    const float embs[] = {10.0f, 20.0f};
    const float preds[] = {0.0f, 0.0f};
    state.update(embs, 1, preds, 0, 0);
    return state.n_sil_frames() == 0 && state.mean_sil_emb() == learned;
}

}  // namespace

int
main() {
    if (!test_channel_birth_gate()) {
        std::fprintf(stderr, "[FAIL] transient speaker channel was established\n");
        return 1;
    }
    test_birth_gate_settled_frames();
    test_geometry_and_word_cadence();
    test_finite_parity();
    test_invalid_rotary_geometry();
    if (!test_learned_silence_embedding()) {
        std::fprintf(stderr, "[FAIL] learned silence embedding was not retained\n");
        return 1;
    }
    std::printf("[PASS] channel state, silence, geometry, word cadence, and finite parity\n");
    return 0;
}
