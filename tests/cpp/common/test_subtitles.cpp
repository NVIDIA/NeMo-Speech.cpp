// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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

// Character units (Chinese): cues show the running text, without spaces inside a word; the
// sentence end comes from the punctuation the unit carries.
void
test_character_units_keep_running_text() {
    std::vector<Word> words{{"我", 0, 200},   {"爱", 200, 400},   {"北", 400, 600},
                            {"京", 600, 900}, {"你", 2500, 2700}, {"好", 2700, 3000}};
    words[3].display = "京。";
    words[5].display = "好！";
    for (size_t i : {1, 2, 3, 5}) words[i].joins_previous = true;
    const auto cues = nemo_speech::subtitle::make_cues(words, "", 4000);
    require(cues.size() == 2, "character units split at the pause");
    require(cues[0].text == "我爱北京。" && cues[1].text == "你好！", "character cue text");
    words[4].speaker = 1;
    words[5].speaker = 1;
    const auto turns = nemo_speech::subtitle::make_speaker_turns(words);
    require(turns.size() == 2 && turns[0].text == "我爱北京。", "character speaker turn text");
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

void
test_speaker_turns() {
    const std::vector<Word> words{{"Hello", 0, 200, 0.0f, 1},  {"there", 200, 400, 0.0f, 1},
                                  {".", 400, 450, 0.0f, 2},    {"How", 700, 900, 0.0f, 2},
                                  {"are", 900, 1050, 0.0f, 2}, {"you?", 1050, 1300, 0.0f, 2}};
    const auto turns = nemo_speech::subtitle::make_speaker_turns(words);
    require(turns.size() == 2, "speaker changes produce separate turns");
    require(turns[0].speaker == 1 && turns[0].text == "Hello there.", "first speaker turn");
    require(turns[0].start_ms == 0 && turns[0].end_ms == 450, "first speaker turn timing");
    require(turns[1].speaker == 2 && turns[1].text == "How are you?", "second speaker turn");
    require(turns[1].start_ms == 700 && turns[1].end_ms == 1300, "second speaker turn timing");
}

Word
spoken(const std::string& text, int start_ms, int end_ms, int speaker) {
    Word word;
    word.text = text;
    word.start_ms = start_ms;
    word.end_ms = end_ms;
    word.speaker = speaker;
    return word;
}

void
require_ordered(const std::vector<Cue>& cues, int minimum_ms, const char* message) {
    for (size_t i = 0; i < cues.size(); ++i) {
        require(cues[i].end_ms - cues[i].start_ms >= minimum_ms, message);
        require(i + 1 == cues.size() || cues[i].end_ms <= cues[i + 1].start_ms, message);
    }
}

// Without speakers, cues and their SRT text are those of the subtitles before speaker labels.
void
test_untagged_subtitles_are_unchanged() {
    std::vector<Word> words;
    int start = 0;
    for (const char* text :
         {"When",    "we", "arrived,", "the",    "station", "was",    "already",
          "closed,", "so", "we",       "walked", "back",    "along",  "the",
          "river",   "to", "the",      "old",    "town.",   "Nobody", "answered."}) {
        words.push_back(spoken(text, start, start + 240, 0));
        start += 320;
    }
    words.push_back(spoken("Later", start + 1500, start + 1800, 0));
    require(
        nemo_speech::subtitle::render_cues(
            nemo_speech::subtitle::make_cues(words, "", 10000), false) ==
            "1\n00:00:00,000 --> 00:00:02,560\nWhen we arrived,\nthe station was already "
            "closed,\n\n"
            "2\n00:00:02,560 --> 00:00:06,080\nso we walked back along\nthe river to the old "
            "town.\n\n"
            "3\n00:00:06,080 --> 00:00:06,914\nNobody answered.\n\n"
            "4\n00:00:08,220 --> 00:00:09,054\nLater\n\n",
        "untagged SRT matches the unlabelled layout");
}

// Issue #86: ten words of speaker 1 and two of speaker 2 fit one cue by length.
void
test_cue_never_spans_a_speaker_change() {
    std::vector<Word> words;
    int start = 57040;
    for (const char* text :
         {"speech", "recognition", "is", "most", "important", "as", "far", "as", "your",
          "concerned", "this", "could"}) {
        words.push_back(spoken(text, start, start + 250, words.size() < 10 ? 1 : 2));
        start += 300;
    }
    const auto cues = nemo_speech::subtitle::make_cues(words, "", 61000);
    require(cues.size() == 2, "a speaker change starts a cue");
    require(cues[0].speaker == 1 && cues[1].speaker == 2, "cue speakers");
    require(cues[1].text == "this could" && cues[1].start_ms == 60040, "second speaker's cue");
    require_ordered(cues, 1, "speaker cues do not overlap");
    // The label shares the first line's 37-character target.
    const auto wrapped = lines(cues[0].text);
    require(
        wrapped.size() == 2 && characters("Speaker 1: " + wrapped[0]) <= 37 &&
            wrapped[0] + " " + wrapped[1] ==
                "speech recognition is most important as far as your concerned",
        "labelled first line");
}

void
test_speaker_labels_in_subtitles_and_text() {
    using nemo_speech::subtitle::make_cues;
    using nemo_speech::subtitle::render_cues;
    const std::vector<Word> words{
        spoken("Hello", 0, 400, 1), spoken("there.", 400, 900, 1), spoken("Hi.", 1000, 1400, 2),
        spoken("um", 3000, 3200, 0)};
    const auto cues = make_cues(words, "", 4000);
    require(
        render_cues(cues, false) ==
            "1\n00:00:00,000 --> 00:00:00,900\nSpeaker 1: Hello there.\n\n"
            "2\n00:00:01,000 --> 00:00:01,834\nSpeaker 2: Hi.\n\n"
            "3\n00:00:03,000 --> 00:00:03,834\num\n\n",
        "SRT speaker prefixes");
    require(
        render_cues(cues, true) ==
            "WEBVTT\n\n00:00:00.000 --> 00:00:00.900\n<v Speaker 1>Hello there.\n\n"
            "00:00:01.000 --> 00:00:01.834\n<v Speaker 2>Hi.\n\n"
            "00:00:03.000 --> 00:00:03.834\num\n\n",
        "WebVTT voice spans");
    require(
        nemo_speech::subtitle::render_speaker_turns(nemo_speech::subtitle::make_speaker_turns(
            words)) == "Speaker 1: Hello there.\n\nSpeaker 2: Hi.\n\num\n",
        "one paragraph per speaker turn");
}

// Untagged words (0 or a negative tag) form their own unlabelled turn and cue; punctuation stays
// with the word before it.
void
test_untagged_words_between_speakers() {
    const std::vector<Word> words{
        spoken("so", 0, 200, 0), spoken("we", 200, 400, -1), spoken("start", 400, 700, 1),
        spoken(".", 700, 720, 0), spoken("okay", 800, 1000, 0)};
    const auto turns = nemo_speech::subtitle::make_speaker_turns(words);
    require(
        turns.size() == 3 && turns[0].speaker == 0 && turns[0].text == "so we" &&
            turns[1].speaker == 1 && turns[1].text == "start." && turns[2].speaker == 0,
        "untagged turns");
    const auto cues = nemo_speech::subtitle::make_cues(words, "", 2000);
    require(
        cues.size() == 3 && cues[0].speaker == 0 && cues[0].text == "so we" &&
            cues[1].speaker == 1 && cues[1].text == "start." && cues[2].text == "okay",
        "untagged cues");
    require_ordered(cues, 1, "untagged cues do not overlap");
    require(
        nemo_speech::subtitle::render_speaker_turns(turns) ==
            "so we\n\nSpeaker 1: start.\n\nokay\n",
        "untagged turn has no label");
}

// Words with one interval (end-of-stream times clamped to the audio) never split into flashing or
// overlapping cues.
void
test_identical_word_times() {
    using nemo_speech::subtitle::make_cues;
    std::vector<Word> words;
    for (int i = 0; i < 12; ++i)
        words.push_back(spoken("abcdefghi" + std::to_string(i % 10), 560, 640, 0));
    auto cues = make_cues(words, "", 640);
    require(cues.size() == 1, "words at one time share a cue past the length limit");
    require(
        cues[0].start_ms == 560 && cues[0].end_ms == 640, "shared cue keeps the words' interval");
    require(lines(cues[0].text).size() == 2, "shared cue has two lines");

    words.push_back(spoken("later", 2000, 2300, 0));
    cues = make_cues(words, "", 3000);
    require(
        cues.size() == 2 && cues[0].end_ms == 2000 && cues[1].start_ms == 2000,
        "next word starts a cue");

    // Zero-length words at the end of the audio stay with the cue before them.
    std::vector<Word> tail;
    for (int i = 0; i < 5; ++i)
        tail.push_back(spoken("wordnumbr" + std::to_string(i), i * 600, i * 600 + 300, 1));
    for (int i = 0; i < 6; ++i)
        tail.push_back(spoken("clampedwr" + std::to_string(i), 3000, 3000, 1));
    cues = make_cues(tail, "", 3000);
    require(
        cues.size() == 1 && cues[0].start_ms == 0 && cues[0].end_ms == 3000,
        "zero-length tail joins the cue");

    // Across a speaker change words cannot share a cue: a cue that only its start time places waits
    // for the minimum display time of the one before.
    cues = make_cues(
        {spoken("Yes.", 560, 640, 1), spoken("No.", 560, 640, 2), spoken("Maybe.", 600, 640, 1)},
        "", 640);
    require(cues.size() == 3, "one cue per speaker turn");
    require(
        cues[0].start_ms == 560 && cues[1].start_ms == 1394 && cues[2].start_ms == 2228,
        "delayed starts");
    require_ordered(cues, 834, "delayed cues get the minimum duration");
    cues = make_cues({spoken("Thanks", 0, 500, 1), spoken("bye", 1000, 1000, 2)}, "", 1000);
    require(
        cues.size() == 2 && cues[1].start_ms == 1000 && cues[1].end_ms == 1834,
        "last cue at the audio end");
    require_ordered(cues, 834, "last cue does not flash");
}


// ---- transcription API cue builder and renderers ----
using nemo_speech::subtitle::TimedWord;
using nemo_speech::subtitle::WordCue;

template <typename F>
bool
throws_with(F&& f, const std::string& needle) {
    try {
        f();
    }
    catch (const std::invalid_argument& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}

void
test_word_cues_short_phrase_and_renders() {
    const std::vector<TimedWord> words{{"Hello", 0.42, 0.81}, {"world.", 0.88, 1.31}};
    const auto cues = nemo_speech::subtitle::word_cues(words);
    require(
        cues.size() == 1 && cues[0].first_word == 0 && cues[0].last_word == 1, "one canonical cue");
    require(
        cues[0].start == 0.42 && cues[0].end == 1.31 && cues[0].text == "Hello world.",
        "cue fields");
    require(
        nemo_speech::subtitle::render_word_cues(cues, "srt") ==
            "1\n00:00:00,420 --> 00:00:01,310\nHello world.\n\n",
        "srt body");
    require(
        nemo_speech::subtitle::render_word_cues(cues, "vtt") ==
            "WEBVTT\n\n00:00:00.420 --> 00:00:01.310\nHello world.\n\n",
        "vtt body");
    require(
        nemo_speech::subtitle::render_word_cues({}, "srt").empty() &&
            nemo_speech::subtitle::render_word_cues({}, "vtt") == "WEBVTT\n\n",
        "no cues");
}

void
test_word_cues_two_lines_by_graphemes() {
    const std::vector<TimedWord> words{
        {"cafe\xcc\x81", 0.0, 0.3}, {"characters", 0.31, 0.7}, {"remain", 0.71, 1.1},
        {"whole", 1.11, 1.4},       {"while", 1.41, 1.7},      {"caption", 1.71, 2.0},
        {"lines", 2.01, 2.3},       {"stay", 2.31, 2.6},       {"readable.", 2.61, 3.0}};
    const auto cues = nemo_speech::subtitle::word_cues(words);
    require(cues.size() == 1, "one two-line cue");
    const auto split = cues[0].text.find('\n');
    require(split != std::string::npos, "two lines");
    std::string joined = cues[0].text;
    joined[split] = ' ';
    require(
        joined == "cafe\xcc\x81 characters remain whole while caption lines stay readable.",
        "words kept whole");
    require(
        nemo_speech::subtitle::grapheme_count(cues[0].text.substr(0, split)) <= 42 &&
            nemo_speech::subtitle::grapheme_count(cues[0].text.substr(split + 1)) <= 42,
        "lines within 42 graphemes");
    // 63 graphemes: splits after "whole" (28 | 34) and "while" (34 | 28) tie on balance; the
    // byte-wise greater first line wins the tie.
    require(
        cues[0].text == "cafe\xcc\x81 characters remain whole while\ncaption lines stay readable.",
        "layout choice");
}

void
test_word_cues_sentence_break_and_duration() {
    // A sentence end closes the cue once the cue is >= 0.833 s; shorter sentences keep going.
    const std::vector<TimedWord> words{
        {"Hi.", 0.0, 0.3},  {"How", 0.4, 0.6},    {"are", 0.6, 0.8},
        {"you?", 0.8, 1.0}, {"Fine", 1.2, 1.5},   {"thanks.", 1.5, 1.9},
        {"Long", 8.0, 8.5}, {"pause", 9.0, 15.0}, {"after", 15.6, 16.0}};
    const auto cues = nemo_speech::subtitle::word_cues(words);
    require(cues.size() == 3, "three cues");
    require(
        cues[0].first_word == 0 && cues[0].last_word == 5 &&
            cues[0].text == "Hi. How are you? Fine thanks.",
        "last sentence end within 7 s");
    require(
        cues[1].first_word == 6 && cues[1].last_word == 7 && cues[1].end == 15.0,
        "7 s bound is inclusive");
    require(cues[2].first_word == 8, "next cue");
}

void
test_word_cues_oversized_word() {
    const std::string long_word(60, 'a');
    const std::vector<TimedWord> words{
        {"hello", 0.0, 0.4}, {long_word, 0.5, 9.0}, {"world", 9.1, 9.5}};
    const auto cues = nemo_speech::subtitle::word_cues(words);
    require(
        cues.size() == 3 && cues[1].first_word == 1 && cues[1].last_word == 1 &&
            cues[1].text == long_word && cues[2].text == "world",
        "oversized word is its own cue");
}

void
test_word_cues_render_rounding_and_hours() {
    const std::vector<TimedWord> words{
        {"One.", 0.9995, 1.25}, {"Two?", 59.9995, 60.25}, {"Three!", 299.9995, 300.25}};
    const auto cues = nemo_speech::subtitle::word_cues(words);
    require(
        nemo_speech::subtitle::render_word_cues(cues, "srt") ==
            "1\n00:00:01,000 --> 00:00:01,250\nOne.\n\n2\n00:01:00,000 --> 00:01:00,250\nTwo?\n\n"
            "3\n00:05:00,000 --> 00:05:00,250\nThree!\n\n",
        "millisecond rounding rolls over seconds and minutes");
    const auto hour = nemo_speech::subtitle::word_cues({{"Later.", 3600.42, 3601.31}});
    require(
        nemo_speech::subtitle::render_word_cues(hour, "srt")
                    .find("01:00:00,420 --> 01:00:01,310") != std::string::npos &&
            nemo_speech::subtitle::render_word_cues(hour, "vtt")
                    .find("01:00:00.420 --> 01:00:01.310") != std::string::npos,
        "hour-scale times");
}

void
test_word_cues_overlap_and_joined_text() {
    using nemo_speech::subtitle::render_word_cues;
    using nemo_speech::subtitle::word_cues;
    // NeMo's aligner: the words of two speakers overlap and a word can have no duration; the cue
    // ends at its words' latest end.
    const auto cues = word_cues(
        {{"Hello", 0.40, 0.90},
         {"Yes.", 0.60, 1.00},
         {"there.", 0.95, 0.95},
         {"Fine.", 1.10, 1.50}});
    require(
        cues.size() == 1 && cues[0].last_word == 3 && cues[0].end == 1.5 &&
            cues[0].text == "Hello Yes. there. Fine.",
        "overlapping and zero-length words");
    // A cue inside another (two speakers) renders as it is; one that rounds to no duration ends
    // 1 ms after its start.
    const auto nested = word_cues({{"Hello", 0.4, 9.0}, {"Yes.", 1.0, 1.0005}});
    require(
        render_word_cues(nested, "srt") ==
            "1\n00:00:00,400 --> 00:00:09,000\nHello\n\n2\n00:00:01,000 --> 00:00:01,001\nYes.\n\n",
        "overlapping cues and a 1 ms minimum");
    // Running text of Chinese, Japanese and Thai characters joins without a space.
    const auto joined = word_cues(
        {{"\xe4\xbd\xa0", 0.0, 0.2},
         {"\xe5\xa5\xbd\xe3\x80\x82", 0.2, 0.4, true},
         {"OK.", 0.5, 0.9}});
    require(
        joined.size() == 1 && joined[0].text == "\xe4\xbd\xa0\xe5\xa5\xbd\xe3\x80\x82 OK.",
        "joined running text");
}

void
test_word_cues_rejections() {
    using nemo_speech::subtitle::render_word_cues;
    using nemo_speech::subtitle::word_cues;
    require(
        throws_with(
            [] { word_cues({{"a", 0.0, 0.5}, {" \t", 0.5, 1.0}}); },
            "word 1 has empty display text"),
        "blank word");
    require(
        throws_with([] { word_cues({{"a", 0.5, 0.4}}); }, "word 0 has invalid timestamps"),
        "word ending before its start");
    require(
        throws_with(
            [] { word_cues({{"a", 0.5, 0.6}, {"b", 0.4, 1.0}}); },
            "word 1 starts before the preceding word"),
        "words out of order");
    const auto unsafe = word_cues({{"unsafe --> cue", 0.0, 0.5}});
    require(
        throws_with(
            [&] { render_word_cues(unsafe, "vtt"); },
            "WebVTT cue text contains an invalid separator"),
        "vtt separator");
    require(
        render_word_cues(unsafe, "srt").find("unsafe --> cue") != std::string::npos,
        "srt keeps the text");
    require(throws_with([&] { render_word_cues(unsafe, "txt"); }, "srt or vtt"), "unknown format");
}

void
test_grapheme_counts_match_icu() {
    // Expected counts from ICU 74's character break iterator.
    const std::vector<std::pair<std::string, size_t>> cases{
        {"\x70\x6c\x61\x69\x6e\x20\x61\x73\x63\x69\x69\x20\x77\x6f\x72\x64\x73",
         17},  // plain ascii words
        {"\x63\x61\x66\x65\xcc\x81\x20\x63\x68\x61\x72\x61\x63\x74\x65\x72\x73",
         15},                                                      // cafe\u0301 characters
        {"\x61\x0d\x0a\x62\x0d\x63\x0a\x64", 7},                   // a\r\nb\rc\nd
        {"\x74\x61\x62\x09\x68\x65\x72\x65\x01\x63\x74\x6c", 12},  // tab\there\x01ctl
        {"\xf0\x9f\x91\xa8\xe2\x80\x8d\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x91\xa7\x20\x66\x61\x6d"
         "\x69\x6c\x79",
         8},  // \U0001f468\u200d\U0001f469\u200d\U0001f467 family
        {"\xf0\x9f\x87\xba\xf0\x9f\x87\xb8\xf0\x9f\x87\xac\xf0\x9f\x87\xa7\x20\x66\x6c\x61\x67\x73",
         8},  // \U0001f1fa\U0001f1f8\U0001f1ec\U0001f1e7 flags
        {"\xf0\x9f\x87\xba\xf0\x9f\x87\xb8\xf0\x9f\x87\xac\x20\x6f\x64\x64",
         6},  // \U0001f1fa\U0001f1f8\U0001f1ec odd
        {"\xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd\x20\x74\x68\x75\x6d\x62\x73",
         8},  // \U0001f44d\U0001f3fd thumbs
        {"\x31\xef\xb8\x8f\xe2\x83\xa3\x20\x6b\x65\x79\x63\x61\x70", 8},  // 1\ufe0f\u20e3 keycap
        {"\xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8\x20\x6a\x61\x6d\x6f\x20\xea\xb0\x80\xe1\x86\xa8\x20"
         "\x6c\x76\x74",
         12},  // \u1100\u1161\u11a8 jamo \uac00\u11a8 lvt
        {"\xe0\xa4\x95\xe0\xa4\xbf\x20\x73\x70\x61\x63\x69\x6e\x67", 9},  // \u0915\u093f spacing
        {"\xd8\x80\xd9\xa1\x20\x70\x72\x65\x70\x65\x6e\x64", 9},          // \u0600\u0661 prepend
        {"\x61\xe2\x80\x8d\x62\x20\x7a\x77\x6a", 6},                      // a\u200db zwj
        {"\xe2\x9d\xa4\xef\xb8\x8f\x20\x68\x65\x61\x72\x74", 7},          // \u2764\ufe0f heart
        {"\xe0\xb8\x81\xe0\xb8\xb3\x20\x74\x68\x61\x69", 6},              // \u0e01\u0e33 thai
        {"\x65\xcc\x81\xcc\x82\xcc\x83\x20\x73\x74\x61\x63\x6b\x65\x64",
         9},  // e\u0301\u0302\u0303 stacked
        {"\xc3\xa9\x74\xc3\xa9\x20\x70\x72\x65\x63\x6f\x6d\x70\x6f\x73\x65\x64",
         15},  // \xe9t\xe9 precomposed
        {"\x20\xcc\x81\x20\x6c\x6f\x6e\x65\x20\x6d\x61\x72\x6b\x20\x61\x66\x74\x65\x72\x20\x73\x70"
         "\x61\x63\x65",
         23},  //  \u0301 lone mark after space
        {"\xf0\x9f\x98\x80\xe2\x80\x8d\xe2\x80\x8d\xf0\x9f\x98\x80\x20\x64\x6f\x75\x62\x6c\x65\x20"
         "\x7a\x77\x6a",
         13},  // \U0001f600\u200d\u200d\U0001f600 double zwj
        {"\xf0\x9f\x8f\xb3\xef\xb8\x8f\xe2\x80\x8d\xf0\x9f\x8c\x88\x20\x72\x61\x69\x6e\x62\x6f\x77"
         "\x20\x66\x6c\x61\x67",
         14},  // \U0001f3f3\ufe0f\u200d\U0001f308 rainbow flag
    };
    for (const auto& c : cases)
        require(
            nemo_speech::subtitle::grapheme_count(c.first) == c.second,
            "grapheme count differs from ICU");
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
    test_character_units_keep_running_text();
    test_two_lines_and_utf8_character_count();
    test_duration_and_length_split_are_ordered();
    test_fallback_text();
    test_speaker_turns();
    test_untagged_subtitles_are_unchanged();
    test_cue_never_spans_a_speaker_change();
    test_speaker_labels_in_subtitles_and_text();
    test_untagged_words_between_speakers();
    test_identical_word_times();
    test_word_cues_short_phrase_and_renders();
    test_word_cues_two_lines_by_graphemes();
    test_word_cues_sentence_break_and_duration();
    test_word_cues_oversized_word();
    test_word_cues_render_rounding_and_hours();
    test_word_cues_overlap_and_joined_text();
    test_word_cues_rejections();
    test_grapheme_counts_match_icu();
    std::cout << "subtitle tests passed\n";
    return 0;
}
