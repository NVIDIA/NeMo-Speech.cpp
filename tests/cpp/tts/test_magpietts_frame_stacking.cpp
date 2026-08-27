// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <vector>

#include "tts/magpietts/lt.h"
#include "tts/magpietts/model.h"

namespace tts = nemo_speech::tts;

int
main() {
    tts::magpietts_hparams v2602;
    v2602.text_vocab_size = 2362;
    v2602.frame_stacking_factor = 1;
    tts::magpietts_hparams v2607;
    v2607.text_vocab_size = 3359;
    v2607.frame_stacking_factor = 2;
    if (tts::magpietts_infer_tokenizer_profile(v2602) != "v2602" ||
        tts::magpietts_infer_tokenizer_profile(v2607) != "v2607" ||
        !tts::magpietts_tokenizer_profile_matches("v2602", v2602) ||
        !tts::magpietts_tokenizer_profile_matches("v2607", v2607) ||
        tts::magpietts_tokenizer_profile_matches("v2602", v2607) ||
        tts::magpietts_tokenizer_profile_matches("v2607", v2602)) {
        std::fprintf(stderr, "tokenizer profile compatibility check failed\n");
        return 1;
    }
    v2607.text_vocab_size = 2362;
    if (!tts::magpietts_infer_tokenizer_profile(v2607).empty()) {
        std::fprintf(stderr, "unknown tokenizer dimensions were accepted\n");
        return 1;
    }

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

    const std::vector<std::vector<int32_t>> forced_frames = {
        {10, 11}, {20, 21}, {30, 31}, {40, 41}};
    std::vector<int32_t> stacked_forced;
    if (!tts::magpietts_stack_forced_code_frames(forced_frames, 0, h, stacked_forced) ||
        stacked_forced != stacked) {
        std::fprintf(stderr, "forced-code frame stacking failed\n");
        return 1;
    }
    const std::vector<std::vector<int32_t>> incomplete_forced_frames = {
        {10, 11}, {20, 21}, {30, 31}};
    if (tts::magpietts_stack_forced_code_frames(
            incomplete_forced_frames, 0, h, stacked_forced) ||
        !stacked_forced.empty()) {
        std::fprintf(stderr, "incomplete forced-code frame group was accepted\n");
        return 1;
    }
    const std::vector<std::vector<int32_t>> malformed_forced_frames = {{10}, {20, 21}};
    if (tts::magpietts_stack_forced_code_frames(
            malformed_forced_frames, 0, h, stacked_forced)) {
        std::fprintf(stderr, "malformed forced-code frame was accepted\n");
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

    h.max_decoder_steps = 5;
    int emitted_frames = 0;
    int final_position_frames = 0;
    const int decoder_positions =
        (h.max_decoder_steps + h.frame_stacking_factor - 1) / h.frame_stacking_factor;
    for (int step = 0; step < decoder_positions; ++step) {
        const int frames_remaining =
            h.max_decoder_steps - step * h.frame_stacking_factor;
        final_position_frames = tts::magpietts_frames_to_emit(
            frames_remaining, h.frame_stacking_factor, -1);
        emitted_frames += final_position_frames;
    }
    if (emitted_frames != h.max_decoder_steps || final_position_frames != 1) {
        std::fprintf(stderr, "non-divisible decoder frame budget was exceeded\n");
        return 1;
    }
    if (tts::magpietts_frames_to_emit(1, h.frame_stacking_factor, 0) != 0 ||
        tts::magpietts_frames_to_emit(1, h.frame_stacking_factor, 1) != 1) {
        std::fprintf(stderr, "partial final stacked frame did not preserve EOS handling\n");
        return 1;
    }
    return 0;
}
