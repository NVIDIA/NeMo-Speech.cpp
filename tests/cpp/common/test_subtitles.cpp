// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "subtitles.h"

using nemo_speech::subtitle::Cue;
using nemo_speech::subtitle::Word;

namespace {

void
require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

size_t
characters(const std::string& text) {
    size_t count = 0;
    for (const unsigned char byte : text)
        if ((byte & 0xc0) != 0x80)
            ++count;
    return count;
}

std::vector<std::string>
lines(const std::string& text) {
    const auto split = text.find('\n');
    if (split == std::string::npos)
        return {text};
    return {text.substr(0, split), text.substr(split + 1)};
}

void
test_clause_wrapping_and_readable_duration() {
    const std::vector<Word> words{{"When", 0, 300},        {"we", 300, 500},
                                  {"arrived,", 500, 900},  {"the", 900, 1100},
                                  {"station", 1100, 1500}, {"was", 1500, 1700},
                                  {"already", 1700, 2100}, {"closed.", 2100, 2500}};
    const auto cues = nemo_speech::subtitle::make_cues(words, "", 4000);
    require(cues.size() == 1, "one grammatical cue");
    require(
        cues[0].text == "When we arrived,\nthe station was already closed.",
        "prefer clause punctuation when wrapping");
    require(cues[0].start_ms == 0 && cues[0].end_ms == 2667, "180 WPM display time");
}

void
test_sentence_and_pause_boundaries() {
    const std::vector<Word> words{
        {"First", 0, 300},
        {"sentence.", 300, 600},
        {"After", 1600, 1900},
        {"a", 1900, 2000},
        {"pause", 2000, 2300}};
    const auto cues = nemo_speech::subtitle::make_cues(words, "", 3000);
    require(cues.size() == 2, "sentence and pause produce two cues");
    require(cues[0].text == "First sentence.", "first sentence text");
    require(cues[0].end_ms <= cues[1].start_ms, "cues do not overlap");
}

void
test_adjacent_short_sentences_share_a_cue() {
    const std::vector<Word> words{{"Yes.", 0, 250}, {"No.", 250, 1000}};
    const auto cues = nemo_speech::subtitle::make_cues(words, "", 1000);
    require(cues.size() == 1, "adjacent short sentences share one cue");
    require(cues[0].text == "Yes. No.", "short sentence text is preserved");
    require(cues[0].end_ms - cues[0].start_ms >= 834, "combined cue meets minimum duration");
}

void
test_short_sentence_with_time_stays_separate() {
    const std::vector<Word> words{{"Yes.", 0, 250}, {"No.", 900, 1200}};
    const auto cues = nemo_speech::subtitle::make_cues(words, "", 1800);
    require(cues.size() == 2, "short sentence stays separate when display time is available");
    require(cues[0].text == "Yes." && cues[0].end_ms == 834, "first cue uses minimum duration");
    require(cues[1].start_ms == 900 && cues[1].end_ms == 1734, "second cue uses minimum duration");
}

void
test_title_abbreviation_does_not_split_sentence() {
    const std::vector<Word> words{{"Dr.", 0, 220}, {"Smith", 220, 600}, {"arrived.", 600, 1054}};
    const auto cues = nemo_speech::subtitle::make_cues(words, "", 1054);
    require(cues.size() == 1, "title abbreviation stays with its sentence");
    require(cues[0].text == "Dr. Smith arrived.", "abbreviation sentence text");
}

void
test_suffix_abbreviation_can_end_sentence() {
    const std::vector<Word> words{
        {"John", 0, 220}, {"Jr.", 220, 450}, {"He", 900, 1050}, {"left.", 1050, 1400}};
    const auto cues = nemo_speech::subtitle::make_cues(words, "", 1800);
    require(cues.size() == 2, "suffix abbreviation may end a sentence");
    require(cues[0].text == "John Jr." && cues[1].text == "He left.", "suffix sentence text");
}

void
test_two_lines_and_utf8_character_count() {
    const std::vector<Word> words{
        {"Друзья,", 0, 300},      {"сегодня", 300, 600},       {"мы", 600, 800},
        {"обсуждаем", 800, 1200}, {"современное", 1200, 1600}, {"образование", 1600, 2000},
        {"и", 2000, 2100},        {"его", 2100, 2300},         {"будущее.", 2300, 2700}};
    const auto cues = nemo_speech::subtitle::make_cues(words, "", 3500);
    require(cues.size() == 1, "UTF-8 words stay in one cue");
    const auto wrapped = lines(cues[0].text);
    require(wrapped.size() == 2, "at most two subtitle lines");
    require(
        characters(wrapped[0]) <= 42 && characters(wrapped[1]) <= 42, "42-character outer bound");
}

void
test_duration_and_length_split_are_ordered() {
    std::vector<Word> words;
    words.reserve(30);
    for (int i = 0; i < 30; ++i)
        words.push_back({"word" + std::to_string(i), i * 400, i * 400 + 300});
    const auto cues = nemo_speech::subtitle::make_cues(words, "", 12500);
    require(cues.size() >= 2, "long transcript splits into cues");
    int previous_end = 0;
    for (const Cue& cue : cues) {
        require(cue.start_ms >= previous_end, "ordered non-overlapping cues");
        require(cue.end_ms > cue.start_ms, "positive cue duration");
        require(cue.end_ms - cue.start_ms <= 7000, "seven-second cue maximum");
        require(lines(cue.text).size() <= 2, "two-line cue maximum");
        previous_end = cue.end_ms;
    }
}

void
test_fallback_text() {
    const auto cues = nemo_speech::subtitle::make_cues({}, "Fallback transcript.", 1200);
    require(cues.size() == 1, "fallback cue");
    require(cues[0].start_ms == 0 && cues[0].end_ms == 1200, "fallback timing");
    require(cues[0].text == "Fallback transcript.", "fallback text");
}

}  // namespace

int
main() {
    test_clause_wrapping_and_readable_duration();
    test_sentence_and_pause_boundaries();
    test_adjacent_short_sentences_share_a_cue();
    test_short_sentence_with_time_stays_separate();
    test_title_abbreviation_does_not_split_sentence();
    test_suffix_abbreviation_can_end_sentence();
    test_two_lines_and_utf8_character_count();
    test_duration_and_length_split_are_ordered();
    test_fallback_text();
    std::cout << "subtitle tests passed\n";
    return 0;
}
