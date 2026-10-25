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

struct SpeakerTurn {
    int speaker = 0;
    int start_ms = 0;
    int end_ms = 0;
    std::string text;
};

// Punctuation that joins the preceding token. Shared by transcript, subtitle,
// and live-turn code.
bool attaches_to_previous(const std::string& text);

std::vector<SpeakerTurn> make_speaker_turns(const std::vector<Word>& words);

std::vector<Cue> make_cues(
    const std::vector<Word>& words, const std::string& fallback_text, int audio_ms);

}  // namespace nemo_speech::subtitle
