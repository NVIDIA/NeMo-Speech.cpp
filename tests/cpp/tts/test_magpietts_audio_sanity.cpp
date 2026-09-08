// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "tts/magpietts/audio_pp.h"
#include "tts/magpietts/magpietts.h"

namespace tts = nemo_speech::tts;

static int failures = 0;

// The function under test logs its own rejection, which would make a passing
// run look like a failing one. Keep that output out of the test transcript.
static bool
valid_quietly(const std::vector<float>& audio) {
    std::fflush(stderr);
    const int saved = dup(STDERR_FILENO);
    FILE* devnull = std::fopen("/dev/null", "w");
    if (devnull != nullptr) {
        dup2(fileno(devnull), STDERR_FILENO);
    }
    const bool ok = tts::magpie_require_valid_audio(audio);
    std::fflush(stderr);
    dup2(saved, STDERR_FILENO);
    close(saved);
    if (devnull != nullptr) {
        std::fclose(devnull);
    }
    return ok;
}

static void
expect(const char* name, const std::vector<float>& audio, bool want_rejected) {
    if (valid_quietly(audio) == want_rejected) {
        std::fprintf(stderr, "%s: expected %s\n", name, want_rejected ? "rejection" : "acceptance");
        ++failures;
    }
}

int
main() {
    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    const float inf_value = std::numeric_limits<float>::infinity();

    expect("speech", {0.0f, 0.31f, -0.22f, 0.75f, -0.61f}, false);
    expect("quiet", {0.0f, 0.0f, 0.0f, 0.0f}, false);
    expect("digital silence", std::vector<float>(4096, 0.0f), false);
    expect("clipped but varying", {1.0f, 1.0f, 0.42f, -1.0f, -1.0f}, false);
    expect("single sample", {0.5f}, false);
    expect("empty", {}, false);

    // Loud and clipped audio stays valid: a saturated waveform still crosses zero.
    {
        std::vector<float> clipped(4096);
        for (size_t i = 0; i < clipped.size(); ++i) {
            const float x = 4.0f * std::sin(0.03f * (float)i);
            clipped[i] = x > 1.0f ? 1.0f : (x < -1.0f ? -1.0f : x);
        }
        expect("hard-clipped loud speech", clipped, false);
        std::vector<float> square(4096);
        for (size_t i = 0; i < square.size(); ++i) {
            square[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        }
        expect("fully saturated square wave", square, false);
    }

    expect("all negative full scale", std::vector<float>(1024, -1.0f), true);
    expect("all positive full scale", std::vector<float>(1024, 1.0f), true);
    expect("one sample at full scale", {-1.0f}, true);

    expect("nan", {0.2f, nan_value, 0.4f}, true);
    expect("nan at end", {0.2f, 0.4f, nan_value}, true);
    expect("positive infinity", {0.2f, inf_value}, true);
    expect("negative infinity", {0.2f, -inf_value}, true);
    expect("all nan", std::vector<float>(64, nan_value), true);

    expect("uniform mid scale", std::vector<float>(1024, 0.5f), false);
    expect("uniform near full scale", std::vector<float>(1024, 0.999f), false);

    // Why the check runs on the decoded chunk: overlap-add crossfades an invalid
    // chunk against its neighbour, and it no longer looks uniform afterwards.
    {
        const std::vector<float> invalid(1024, -1.0f);
        std::vector<float> valid(1024);
        for (size_t i = 0; i < valid.size(); ++i) {
            valid[i] = 0.5f * std::sin(0.05f * (float)i);
        }
        expect("decoded chunk, before post-processing", invalid, true);

        tts::AudioPostProcessor pp(
            /*samples_per_frame=*/128, /*future_frames=*/0,
            /*window_samples=*/64);
        std::vector<std::vector<float>> written;
        auto sink = [&](const std::vector<float>& chunk) {
            written.push_back(chunk);
            return true;
        };
        const bool wrote_invalid = pp.writeDecodedAudio(invalid, /*history_frames=*/2, false, sink);
        const bool wrote_valid = pp.writeDecodedAudio(valid, /*history_frames=*/2, false, sink);
        const bool flushed = pp.flush(sink);
        if (!wrote_invalid || !wrote_valid || !flushed) {
            std::fprintf(
                stderr, "crossfade case: post-processor write failed (%d %d %d)\n",
                (int)wrote_invalid, (int)wrote_valid, (int)flushed);
            ++failures;
        }

        if (written.empty()) {
            std::fprintf(stderr, "crossfade case: post-processor produced no chunks\n");
            ++failures;
        } else {
            size_t full_scale = 0;
            for (float x : written[0]) {
                full_scale += (x == -1.0f) ? 1 : 0;
            }
            if (full_scale * 2 <= written[0].size()) {
                std::fprintf(
                    stderr, "crossfade case: expected a mostly full-scale chunk, got %zu of %zu\n",
                    full_scale, written[0].size());
                ++failures;
            }
            expect("same chunk, after overlap-add", written[0], false);
        }
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d audio sanity case(s) failed\n", failures);
        return 1;
    }
    std::printf("audio sanity: all cases passed\n");
    return 0;
}
