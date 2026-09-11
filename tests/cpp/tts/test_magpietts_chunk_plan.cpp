// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// plan_text_chunk decides one long-form chunk's text window. It is pure -- no
// model, no device -- and both the sequential loop and the wave scheduler run
// it, so a mistake here is a mistake everywhere. Two copies of this logic had
// already diverged once before they were merged, and a miscomputed window
// killed a run outright on ordinary prose.
#include <cstdio>
#include <numeric>
#include <vector>

#include "tts/magpietts/magpietts.h"

namespace tts = nemo_speech::tts;

namespace {

int failures = 0;

void
expect(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL %s\n", msg);
        ++failures;
    }
}

void
expect_eq(int got, int want, const char* msg) {
    if (got != want) {
        std::fprintf(stderr, "FAIL %s: got %d, want %d\n", msg, got, want);
        ++failures;
    }
}

tts::magpietts_hparams
hparams(int n_ctx = 2048) {
    tts::magpietts_hparams h;
    h.n_ctx = n_ctx;
    return h;
}

// Adaptive history: longform_history_tokens < 0.
tts::magpie_stream_params
adaptive() {
    tts::magpie_stream_params p;
    p.longform_history_tokens = -1;
    return p;
}

tts::magpie_stream_params
pinned(int n) {
    tts::magpie_stream_params p;
    p.longform_history_tokens = n;
    return p;
}

std::vector<int32_t>
tokens(int n, int32_t base = 0) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    std::iota(v.begin(), v.end(), base);
    return v;
}

}  // namespace

int
main() {
    const tts::magpietts_hparams h = hparams();

    // The first chunk has no history to splice, whatever is asked for.
    {
        const auto p = tts::plan_text_chunk(h, adaptive(), {}, tokens(30), 0, 0, 0);
        expect_eq(p.history_len, 0, "first chunk takes no history");
        expect_eq(p.text_len, 30, "first chunk window is its own tokens");
        expect_eq(p.left_offset, 0, "first chunk starts at absolute zero");
    }

    // Steady state: the adaptive rule caps history at 20.
    {
        const auto p = tts::plan_text_chunk(h, adaptive(), tokens(500), tokens(40), 500, 0, 200);
        expect_eq(p.history_len, 20, "adaptive history caps at 20");
        expect_eq(p.text_len, 60, "window is history plus current");
        expect_eq(p.left_offset, 480, "left offset walks back by the history");
        expect_eq(static_cast<int>(p.text_window.size()), 60, "window vector matches text_len");
    }

    // The regression this file exists for. History is spliced from the previous
    // chunk's encoder output, so it cannot be longer than that chunk's window --
    // a fact unrelated to how many tokens have been seen in total. Asking for
    // more than the cache holds used to abort the run with
    //   "longform history context cache is too short: need 20 token(s), have 16"
    {
        const auto p = tts::plan_text_chunk(h, adaptive(), tokens(2000), tokens(40), 2000, 0, 16);
        expect(p.history_len <= 16, "adaptive history never exceeds the previous chunk");
        expect_eq(p.history_len, 16, "adaptive history takes all 16 available");
        expect_eq(p.text_len, 56, "window reflects the clamped history");
    }
    {
        const auto p = tts::plan_text_chunk(h, pinned(20), tokens(2000), tokens(40), 2000, 0, 16);
        expect(p.history_len <= 16, "pinned history never exceeds the previous chunk");
    }

    // A previous chunk that supplied nothing leaves this one standing alone.
    {
        const auto p = tts::plan_text_chunk(h, adaptive(), tokens(2000), tokens(40), 2000, 0, 0);
        expect_eq(p.history_len, 0, "no available history means no history");
        expect_eq(p.text_len, 40, "window is the current chunk only");
    }

    // required_history asks for more than the default 20 so chunk N can still
    // see where chunk N-1's attention stopped. It is still bounded by what the
    // previous chunk holds.
    {
        const auto a = tts::plan_text_chunk(h, adaptive(), tokens(500), tokens(80), 500, 45, 200);
        expect_eq(a.history_len, 45, "required_history raises the adaptive window");
        const auto b = tts::plan_text_chunk(h, adaptive(), tokens(500), tokens(80), 500, 45, 30);
        expect_eq(b.history_len, 30, "required_history still clamps to what is available");
    }

    // Pinning overrides the adaptive cap, and ignores required_history.
    {
        const auto p = tts::plan_text_chunk(h, pinned(40), tokens(500), tokens(10), 500, 0, 200);
        expect_eq(p.history_len, 40, "pinned history overrides the 20-token default");
        expect_eq(p.text_len, 50, "pinned window is history plus current");
    }

    // Pinning cannot ask for more history than has ever been seen.
    {
        const auto p = tts::plan_text_chunk(h, pinned(40), tokens(12), tokens(10), 12, 0, 200);
        expect_eq(p.history_len, 12, "pinned history clamps to the tokens seen so far");
    }

    // The model context bounds the whole window: a chunk that nearly fills it
    // leaves no room for history.
    {
        const tts::magpietts_hparams small = hparams(64);
        const auto p =
            tts::plan_text_chunk(small, pinned(20), tokens(500), tokens(60), 500, 0, 200);
        expect(p.text_len <= 64, "window never exceeds the model context");
        expect_eq(p.history_len, 4, "history shrinks to fit the context");
    }

    // The window is history followed by the current chunk, in order.
    {
        const auto prior = tokens(50, 1000);
        const auto cur = tokens(5, 7000);
        const auto p = tts::plan_text_chunk(h, pinned(3), prior, cur, 50, 0, 200);
        expect_eq(static_cast<int>(p.text_window.size()), 8, "window length");
        expect_eq(p.text_window[0], 1047, "history is the tail of what came before");
        expect_eq(p.text_window[2], 1049, "history runs up to the boundary");
        expect_eq(p.text_window[3], 7000, "current chunk follows the history");
        expect_eq(p.text_window[7], 7004, "current chunk is intact");
    }

    if (failures == 0) {
        std::fprintf(stderr, "plan_text_chunk: all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
