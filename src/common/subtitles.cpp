// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "subtitles.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>

namespace nemo_speech::subtitle {
namespace {

// BBC broadcast subtitles target two 37-character lines and 180 words per
// minute. Netflix permits a 42-character outer line bound and cue durations
// from 5/6 second through 7 seconds.
constexpr int kTargetLineCharacters = 37;
constexpr int kMaximumCueCharacters = 2 * kTargetLineCharacters;
constexpr int kMaximumCueDurationMs = 7000;
constexpr int kMinimumCueDurationMs = 834;
constexpr int kPauseBoundaryMs = 800;
constexpr int kWordsPerMinute = 180;

bool
starts_with_any(const std::string& value, const std::vector<std::string>& prefixes) {
    for (const auto& prefix : prefixes)
        if (value.rfind(prefix, 0) == 0)
            return true;
    return false;
}

bool
ends_with_any(const std::string& value, const std::vector<std::string>& suffixes) {
    for (const auto& suffix : suffixes)
        if (value.size() >= suffix.size() &&
            value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0)
            return true;
    return false;
}

bool
attaches_to_previous(const std::string& word) {
    static const std::vector<std::string> punctuation{
        ".",  ",",  "!",  "?",  ";",  ":",  "%",  ")",  "]", "}", "\xE2\x80\x9D", "\xE2\x80\x99",
        "。", "，", "！", "？", "；", "：", "）", "】", "》"};
    return starts_with_any(word, punctuation);
}

bool
sentence_end(const std::string& word) {
    static const std::vector<std::string> punctuation{".",  "!",  "?", "\xE2\x80\xA6",
                                                      "。", "！", "？"};
    return ends_with_any(word, punctuation);
}

bool
clause_end(const std::string& word) {
    static const std::vector<std::string> punctuation{
        ",", ";", ":", "\xE2\x80\x94", "\xE2\x80\x93", "，", "；", "："};
    return ends_with_any(word, punctuation);
}

size_t
character_count(const std::string& text) {
    size_t count = 0;
    for (const unsigned char byte : text)
        if ((byte & 0xc0) != 0x80)
            ++count;
    return count;
}

std::string
join_words(const std::vector<Word>& words, size_t begin, size_t end) {
    std::ostringstream output;
    for (size_t i = begin; i < end; ++i) {
        if (output.tellp() > 0 && !attaches_to_previous(words[i].text))
            output << ' ';
        output << words[i].text;
    }
    return output.str();
}

std::string
format_lines(const std::vector<Word>& words, size_t begin, size_t end) {
    std::string text = join_words(words, begin, end);
    if (end - begin <= 1 || character_count(text) <= kTargetLineCharacters)
        return text;

    size_t best_break = begin + 1;
    bool best_punctuation = false;
    size_t best_overflow = std::numeric_limits<size_t>::max();
    size_t best_imbalance = std::numeric_limits<size_t>::max();
    for (size_t split = begin + 1; split < end; ++split) {
        const std::string left = join_words(words, begin, split);
        const std::string right = join_words(words, split, end);
        const size_t left_length = character_count(left);
        const size_t right_length = character_count(right);
        const size_t overflow =
            (left_length > kTargetLineCharacters ? left_length - kTargetLineCharacters : 0) +
            (right_length > kTargetLineCharacters ? right_length - kTargetLineCharacters : 0);
        const bool punctuation =
            sentence_end(words[split - 1].text) || clause_end(words[split - 1].text);
        const size_t imbalance =
            left_length > right_length ? left_length - right_length : right_length - left_length;
        if (overflow < best_overflow ||
            (overflow == best_overflow && punctuation && !best_punctuation) ||
            (overflow == best_overflow && punctuation == best_punctuation &&
             imbalance < best_imbalance)) {
            best_break = split;
            best_punctuation = punctuation;
            best_overflow = overflow;
            best_imbalance = imbalance;
        }
    }
    return join_words(words, begin, best_break) + '\n' + join_words(words, best_break, end);
}

struct Group {
    size_t begin;
    size_t end;
    Cue cue;
};

}  // namespace

std::vector<Cue>
make_cues(const std::vector<Word>& words, const std::string& fallback_text, int audio_ms) {
    if (words.empty()) {
        if (fallback_text.empty())
            return {};
        return {{0, std::max(1, audio_ms), fallback_text}};
    }

    std::vector<Group> groups;
    for (size_t begin = 0; begin < words.size();) {
        size_t end = begin + 1;
        size_t last_clause = begin;
        bool has_clause = false;
        for (; end <= words.size(); ++end) {
            const size_t current = end - 1;
            if (end > begin + 1) {
                const bool pause =
                    words[current].start_ms - words[current - 1].end_ms > kPauseBoundaryMs;
                const bool too_long =
                    words[current].end_ms - words[begin].start_ms > kMaximumCueDurationMs;
                const bool too_many_characters =
                    character_count(join_words(words, begin, end)) > kMaximumCueCharacters;
                if (pause || too_long || too_many_characters) {
                    --end;
                    if (!pause && has_clause)
                        end = last_clause;
                    break;
                }
            }
            if (clause_end(words[current].text)) {
                last_clause = end;
                has_clause = true;
            }
            if (sentence_end(words[current].text))
                break;
        }
        end = std::min(std::max(end, begin + 1), words.size());
        groups.push_back(
            {begin,
             end,
             {std::max(0, words[begin].start_ms),
              std::min(words[end - 1].end_ms, words[begin].start_ms + kMaximumCueDurationMs),
              format_lines(words, begin, end)}});
        begin = end;
    }

    std::vector<Cue> cues;
    cues.reserve(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        auto cue = groups[i].cue;
        const int available_end =
            i + 1 < groups.size() ? groups[i + 1].cue.start_ms : std::max(cue.end_ms, audio_ms);
        const int words_in_cue = static_cast<int>(groups[i].end - groups[i].begin);
        const int readable_ms = std::max(
            kMinimumCueDurationMs, (words_in_cue * 60000 + kWordsPerMinute - 1) / kWordsPerMinute);
        const int desired_end = cue.start_ms + std::min(readable_ms, kMaximumCueDurationMs);
        cue.end_ms = std::min(
            std::max(cue.end_ms, desired_end),
            std::min(available_end, cue.start_ms + kMaximumCueDurationMs));
        if (cue.end_ms <= cue.start_ms)
            cue.end_ms = cue.start_ms + 1;
        cues.push_back(std::move(cue));
    }
    return cues;
}

}  // namespace nemo_speech::subtitle
