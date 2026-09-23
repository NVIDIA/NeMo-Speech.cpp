// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "subtitles.h"

namespace nemo_speech::cli {

// stderr renderer for `transcribe --live`: in-place partials and speaker blocks
// on a TTY, one plain line per final otherwise.
class LiveTerminal {
   public:
    explicit LiveTerminal(bool visible, bool diarize = false);

    void start(const std::string& device, int sample_rate);
    void partial(const std::string& text);
    void clear_interim() {
        clear_partial();
        clear_current_turn();
    }
    void partial_turns(const std::vector<subtitle::SpeakerTurn>& turns, double fallback_seconds);
    void current_turn(const subtitle::SpeakerTurn& turn, double fallback_seconds);
    void current_turns(const std::vector<subtitle::SpeakerTurn>& turns, double fallback_seconds);
    void final_turn(const subtitle::SpeakerTurn& turn, double fallback_seconds);
    // Redraw the last final block with punctuation that arrived after it was
    // printed. False when the block is not the latest output or has scrolled.
    bool amend_last_final(const std::string& punctuation);
    void stopped(double audio_seconds);

   private:
    const char* style(const char* code) const { return color_ ? code : ""; }
    const char* speaker_style(int speaker) const;
    std::vector<std::string> turn_lines(
        const subtitle::SpeakerTurn& turn, double fallback_seconds, bool provisional) const;
    void preview_turns(
        const std::vector<subtitle::SpeakerTurn>& turns, double fallback_seconds, bool provisional);
    void clear_current_turn();
    void clear_partial();

    bool visible_ = false;
    bool diarize_ = false;
    bool interactive_ = false;
    bool color_ = false;
    bool partial_visible_ = false;
    std::string partial_text_;
    // Display-only continuity while the diarizer catches up. Never used to
    // assign final tags, exported words, or recognition state.
    int current_speaker_ = 0;
    size_t current_turn_lines_ = 0;
    std::vector<std::string> current_preview_;
    // Latest printed final block; only the replaceable region follows it.
    bool last_final_open_ = false;
    subtitle::SpeakerTurn last_final_;
    double last_final_seconds_ = 0.0;
    size_t last_final_lines_ = 0;
};

}  // namespace nemo_speech::cli
