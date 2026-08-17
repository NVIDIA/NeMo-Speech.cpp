// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <vector>

#include "tts/magpietts/model.h"

namespace tts = nemo_speech::tts;

int
main() {
    tts::magpietts_hparams h;
    h.audio_codebooks = 2;
    h.frame_stacking_factor = 2;
    h.audio_eos_id = 99;

    const std::vector<int32_t> stacked = {10, 11, 20, 21};
    std::vector<std::vector<int32_t>> frames;
    if (!tts::magpietts_unstack_codes(stacked, h, frames) || frames.size() != 2 ||
        frames[0] != std::vector<int32_t>({10, 11}) ||
        frames[1] != std::vector<int32_t>({20, 21})) {
        std::fprintf(stderr, "stacked-frame reconstruction failed\n");
        return 1;
    }

    std::vector<int32_t> greedy = stacked;
    greedy[3] = h.audio_eos_id;
    if (tts::magpietts_first_eos_lane(stacked, greedy, h) != 1) {
        std::fprintf(stderr, "EOS lane detection failed\n");
        return 1;
    }
    if (tts::magpietts_first_eos_lane(stacked, stacked, h) != -1) {
        std::fprintf(stderr, "unexpected EOS lane\n");
        return 1;
    }
    return 0;
}
