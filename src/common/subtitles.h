// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>

namespace nemo_speech::subtitle {

struct Word {
    std::string text;
    int start_ms = 0;
    int end_ms = 0;
    float confidence = 0.0f;
    int speaker = 0;
};

struct Cue {
    int start_ms = 0;
    int end_ms = 0;
    std::string text;
};

std::vector<Cue> make_cues(
    const std::vector<Word>& words, const std::string& fallback_text, int audio_ms);

}  // namespace nemo_speech::subtitle
