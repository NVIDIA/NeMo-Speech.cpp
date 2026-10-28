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
    // Running text when it differs from `text` (a Chinese, Japanese or Thai character keeps the
    // punctuation around it), and whether it follows the previous word without a space.
    std::string display;
    bool joins_previous = false;
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

// ---- subtitle cues of the transcription API ----
// Cues of at most 7 s and two lines of at most 42 extended grapheme clusters, closed at a
// sentence end once the cue is at least 0.833 s long; every word keeps its timing. Times are
// seconds; the SRT/VTT renderers round to whole milliseconds (half away from zero).

struct TimedWord {
    std::string text;               // as shown: the running text
    double start = 0.0, end = 0.0;  // seconds
    bool joins_previous = false;    // shown without a space before it (Chinese, Japanese, Thai)
};

struct WordCue {
    size_t first_word = 0, last_word = 0;  // inclusive positions in the word list
    double start = 0.0, end = 0.0;         // seconds: first word's start, its words' latest end
    std::string text;                      // one line, or two joined by '\n'
};

// Words come in the order of their starts, as NeMo's aligner gives them: a word can have no
// duration, and the words of different speakers overlap. Each needs non-blank text and finite
// times with end >= start >= 0; otherwise throws std::invalid_argument ("word N has ...",
// "word N starts before the preceding word").
std::vector<WordCue> word_cues(const std::vector<TimedWord>& words);

// format "srt" or "vtt". The cues of overlapping speakers can overlap, which both formats allow; a
// cue that rounds to no duration ends 1 ms after its start. Throws std::invalid_argument for an
// unknown format, an empty cue, or (vtt) cue text that contains "-->" or an empty line.
std::string render_word_cues(const std::vector<WordCue>& cues, const std::string& format);

// Extended grapheme clusters (UAX #29 rules GB1-GB13; the Unicode 15.1 Indic conjunct rule GB9c
// is not applied) of UTF-8 text; invalid bytes count as one cluster each.
size_t grapheme_count(const std::string& text);

}  // namespace nemo_speech::subtitle
