// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "subtitles.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace nemo_speech::subtitle {
namespace {

// BBC broadcast subtitles target two 37-character lines and 180 words per
// minute. Netflix permits a 42-character outer line bound and cue durations
// from 5/6 second through 7 seconds.
// References:
// https://www.bbc.co.uk/accessibility/forproducts/guides/subtitles/
// https://partnerhelp.netflixstudios.com/hc/en-us/articles/215758617-Timed-Text-Style-Guide-General-Requirements
// https://partnerhelp.netflixstudios.com/hc/en-us/articles/360051554394-Timed-Text-Style-Guide-Subtitle-Timing-Guidelines
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
is_title_abbreviation(const std::string& word) {
    std::string lower;
    lower.reserve(word.size());
    for (const unsigned char character : word)
        lower.push_back(static_cast<char>(std::tolower(character)));
    static const std::vector<std::string> titles{"capt.", "cmdr.", "col.", "dr.",  "gen.",
                                                 "gov.",  "lt.",   "mr.",  "mrs.", "ms.",
                                                 "prof.", "rep.",  "rev.", "sen.", "sgt."};
    return std::find(titles.begin(), titles.end(), lower) != titles.end();
}

bool
sentence_end(const std::string& word) {
    static const std::vector<std::string> punctuation{".",  "!",  "?", "\xE2\x80\xA6",
                                                      "。", "！", "？"};
    return !is_title_abbreviation(word) && ends_with_any(word, punctuation);
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

const std::string&
shown(const Word& word) {
    return word.display.empty() ? word.text : word.display;
}

std::string
join_words(const std::vector<Word>& words, size_t begin, size_t end) {
    std::ostringstream output;
    for (size_t i = begin; i < end; ++i) {
        if (output.tellp() > 0 && !words[i].joins_previous &&
            !attaches_to_previous(shown(words[i])))
            output << ' ';
        output << shown(words[i]);
    }
    return output.str();
}

// `prefix`: characters shown before the first line (the speaker label).
std::string
format_lines(const std::vector<Word>& words, size_t begin, size_t end, size_t prefix) {
    std::string text = join_words(words, begin, end);
    if (end - begin <= 1 || prefix + character_count(text) <= kTargetLineCharacters)
        return text;

    size_t best_break = begin + 1;
    bool best_punctuation = false;
    size_t best_overflow = std::numeric_limits<size_t>::max();
    size_t best_imbalance = std::numeric_limits<size_t>::max();
    for (size_t split = begin + 1; split < end; ++split) {
        const std::string left = join_words(words, begin, split);
        const std::string right = join_words(words, split, end);
        const size_t left_length = prefix + character_count(left);
        const size_t right_length = character_count(right);
        const size_t overflow =
            (left_length > kTargetLineCharacters ? left_length - kTargetLineCharacters : 0) +
            (right_length > kTargetLineCharacters ? right_length - kTargetLineCharacters : 0);
        const bool punctuation =
            sentence_end(shown(words[split - 1])) || clause_end(shown(words[split - 1]));
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

bool
has_minimum_display_time(const std::vector<Word>& words, size_t begin, size_t end, int audio_ms) {
    const int cue_start = std::max(0, words[begin].start_ms);
    const int available_end =
        end < words.size() ? words[end].start_ms : std::max(words[end - 1].end_ms, audio_ms);
    return available_end - cue_start >= kMinimumCueDurationMs;
}

struct Group {
    size_t begin;
    size_t end;
    Cue cue;
};

// Each word's speaker: untagged is 0, and punctuation shown with the previous word keeps its
// speaker.
std::vector<int>
word_speakers(const std::vector<Word>& words) {
    std::vector<int> speakers;
    speakers.reserve(words.size());
    for (const auto& word : words)
        speakers.push_back(
            !speakers.empty() && (word.joins_previous || attaches_to_previous(shown(word)))
                ? speakers.back()
                : std::max(0, word.speaker));
    return speakers;
}

// What a speaker's cue shows before its text in SRT: "Speaker N: ".
std::string
cue_label(int speaker) {
    return speaker > 0 ? speaker_label(speaker) + ": " : std::string();
}

std::string
timecode(int milliseconds, bool vtt) {
    milliseconds = std::max(0, milliseconds);
    char result[32];
    std::snprintf(
        result, sizeof(result), "%02d:%02d:%02d%c%03d", milliseconds / 3600000,
        milliseconds / 60000 % 60, milliseconds / 1000 % 60, vtt ? '.' : ',', milliseconds % 1000);
    return result;
}

}  // namespace

bool
attaches_to_previous(const std::string& text) {
    static const std::vector<std::string> punctuation{
        ".",  ",",  "!",  "?",  ";",  ":",  "%",  ")",  "]", "}", "\xE2\x80\x9D", "\xE2\x80\x99",
        "。", "，", "！", "？", "；", "：", "）", "】", "》"};
    return starts_with_any(text, punctuation);
}

std::string
speaker_label(int speaker) {
    return speaker > 0 ? "Speaker " + std::to_string(speaker) : std::string();
}

std::vector<SpeakerTurn>
make_speaker_turns(const std::vector<Word>& words) {
    const auto speakers = word_speakers(words);
    std::vector<SpeakerTurn> turns;
    for (size_t i = 0; i < words.size(); ++i) {
        const auto& word = words[i];
        const int speaker = speakers[i];
        if (turns.empty() || turns.back().speaker != speaker)
            turns.push_back({speaker, word.start_ms, word.end_ms, {}});
        if (!turns.back().text.empty() && !word.joins_previous &&
            !attaches_to_previous(shown(word)))
            turns.back().text += ' ';
        turns.back().text += shown(word);
        turns.back().end_ms = std::max(turns.back().end_ms, word.end_ms);
    }
    return turns;
}

std::vector<Cue>
make_cues(const std::vector<Word>& words, const std::string& fallback_text, int audio_ms) {
    if (words.empty()) {
        if (fallback_text.empty())
            return {};
        return {{0, std::max(1, audio_ms), fallback_text}};
    }

    const auto speakers = word_speakers(words);
    int timeline_end = audio_ms;
    for (const auto& word : words) timeline_end = std::max(timeline_end, word.end_ms);
    std::vector<Group> groups;
    for (size_t begin = 0; begin < words.size();) {
        const int speaker = speakers[begin];
        const size_t label = character_count(cue_label(speaker));
        // A cue closed for length keeps some display time, and the words after it get some
        // before the audio ends; words that start together stay together.
        const auto can_split_before = [&](size_t word) {
            return words[word].start_ms > words[begin].start_ms &&
                   words[word].start_ms < timeline_end;
        };
        size_t end = begin + 1;
        size_t last_clause = begin;
        bool has_clause = false;
        for (; end <= words.size(); ++end) {
            const size_t current = end - 1;
            if (end > begin + 1) {
                if (speakers[current] != speaker) {
                    --end;
                    break;
                }
                const bool pause =
                    words[current].start_ms - words[current - 1].end_ms > kPauseBoundaryMs;
                const bool too_long =
                    words[current].end_ms - words[begin].start_ms > kMaximumCueDurationMs;
                const bool too_many_characters =
                    label + character_count(join_words(words, begin, end)) > kMaximumCueCharacters;
                if (pause) {
                    --end;
                    break;
                }
                if (too_long || too_many_characters) {
                    if (has_clause && can_split_before(last_clause)) {
                        end = last_clause;
                        break;
                    }
                    if (can_split_before(current)) {
                        end = current;
                        break;
                    }
                }
            }
            if (clause_end(shown(words[current]))) {
                last_clause = end;
                has_clause = true;
            }
            if (sentence_end(shown(words[current])) &&
                has_minimum_display_time(words, begin, end, audio_ms))
                break;
        }
        end = std::min(std::max(end, begin + 1), words.size());
        groups.push_back(
            {begin,
             end,
             {std::max(0, words[begin].start_ms),
              std::min(words[end - 1].end_ms, words[begin].start_ms + kMaximumCueDurationMs),
              format_lines(words, begin, end, label), speaker}});
        begin = end;
    }

    // A cue whose words start no later than the previous cue's (identical times across a speaker
    // change), or that follows a delayed cue too closely, waits until the previous cue has been
    // shown for the minimum duration.
    std::vector<int> starts(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        starts[i] = groups[i].cue.start_ms;
        if (i == 0)
            continue;
        const bool previous_delayed = starts[i - 1] > groups[i - 1].cue.start_ms;
        if (starts[i] <= starts[i - 1] ||
            (previous_delayed && starts[i] < starts[i - 1] + kMinimumCueDurationMs))
            starts[i] = starts[i - 1] + kMinimumCueDurationMs;
    }

    std::vector<Cue> cues;
    cues.reserve(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        auto cue = groups[i].cue;
        const bool delayed = starts[i] > cue.start_ms;
        cue.start_ms = starts[i];
        int available_end = i + 1 < groups.size() ? starts[i + 1] : std::max(cue.end_ms, audio_ms);
        // The last cue may run past the audio rather than flash when its words start at its end.
        if (i + 1 == groups.size() && (delayed || available_end <= cue.start_ms))
            available_end = std::max(available_end, cue.start_ms + kMinimumCueDurationMs);
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

std::string
render_cues(const std::vector<Cue>& cues, bool vtt) {
    std::string output = vtt ? "WEBVTT\n\n" : "";
    for (size_t i = 0; i < cues.size(); ++i) {
        const auto& cue = cues[i];
        if (!vtt)
            output += std::to_string(i + 1) + "\n";
        output += timecode(cue.start_ms, vtt) + " --> " +
                  timecode(std::max(cue.start_ms + 1, cue.end_ms), vtt) + "\n";
        if (cue.speaker > 0)
            output += vtt ? "<v " + speaker_label(cue.speaker) + ">" : cue_label(cue.speaker);
        output += cue.text + "\n\n";
    }
    return output;
}

std::string
render_speaker_turns(const std::vector<SpeakerTurn>& turns) {
    std::string output;
    for (const auto& turn : turns) {
        if (!output.empty())
            output += '\n';
        output += cue_label(turn.speaker) + turn.text + "\n";
    }
    return output;
}

// ---- subtitle cues of the transcription API ----

namespace {

enum GraphemeBreak : uint8_t {
    kOther = 0,
    kCR,
    kLF,
    kControl,
    kExtend,
    kZWJ,
    kRegionalIndicator,
    kPrepend,
    kSpacingMark,
    kL,
    kV,
    kT,
    kLV,
    kLVT
};
struct BreakRange {
    uint32_t first, last;
    uint8_t value;  // GraphemeBreak | 0x80 for Extended_Pictographic
};
constexpr BreakRange kBreakRanges[] = {
#include "grapheme_break_table.inc"
};

uint8_t
break_value(uint32_t cp) {
    size_t lo = 0, hi = sizeof(kBreakRanges) / sizeof(kBreakRanges[0]);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (kBreakRanges[mid].last < cp)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < sizeof(kBreakRanges) / sizeof(kBreakRanges[0]) && kBreakRanges[lo].first <= cp
               ? kBreakRanges[lo].value
               : uint8_t(0);
}

// Decodes one UTF-8 code point at text[i], advancing i; an invalid or truncated sequence yields
// U+FFFD for its first byte.
uint32_t
next_code_point(const std::string& text, size_t& i) {
    const auto byte = [&](size_t k) { return static_cast<unsigned char>(text[k]); };
    const unsigned char b0 = byte(i);
    int length = 0;
    uint32_t cp = 0;
    if (b0 < 0x80) {
        ++i;
        return b0;
    } else if ((b0 & 0xe0) == 0xc0 && b0 >= 0xc2) {
        length = 2, cp = b0 & 0x1f;
    } else if ((b0 & 0xf0) == 0xe0) {
        length = 3, cp = b0 & 0x0f;
    } else if ((b0 & 0xf8) == 0xf0 && b0 <= 0xf4) {
        length = 4, cp = b0 & 0x07;
    } else {
        ++i;
        return 0xfffd;
    }
    if (i + length > text.size()) {
        ++i;
        return 0xfffd;
    }
    for (int k = 1; k < length; ++k) {
        const unsigned char b = byte(i + k);
        if ((b & 0xc0) != 0x80) {
            ++i;
            return 0xfffd;
        }
        cp = (cp << 6) | (b & 0x3f);
    }
    if ((length == 3 && (cp < 0x800 || (cp >= 0xd800 && cp <= 0xdfff))) ||
        (length == 4 && (cp < 0x10000 || cp > 0x10ffff))) {
        ++i;
        return 0xfffd;
    }
    i += length;
    return cp;
}

// Unicode White_Space.
bool
rust_whitespace(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0d) || cp == 0x20 || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f ||
           cp == 0x205f || cp == 0x3000;
}

std::vector<uint32_t>
code_points(const std::string& text) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < text.size();) out.push_back(next_code_point(text, i));
    return out;
}

bool
blank(const std::string& text) {
    for (uint32_t cp : code_points(text))
        if (!rust_whitespace(cp))
            return false;
    return true;
}

// The last character after trailing whitespace is one of . ! ? …
bool
cue_sentence_end(const std::string& text) {
    const auto cps = code_points(text);
    for (size_t k = cps.size(); k-- > 0;) {
        if (rust_whitespace(cps[k]))
            continue;
        return cps[k] == '.' || cps[k] == '!' || cps[k] == '?' || cps[k] == 0x2026;
    }
    return false;
}

constexpr double kMaxCueSeconds = 7.0;
constexpr size_t kMaxLineGraphemes = 42;
constexpr double kMinSentenceBreakSeconds = 0.833;

std::string
join_texts(const std::vector<TimedWord>& words, size_t begin, size_t end) {
    std::string out;
    for (size_t k = begin; k < end; ++k) {
        if (k > begin && !words[k].joins_previous)
            out += ' ';
        out += words[k].text;
    }
    return out;
}

// One line when it fits, else the best two-line split (sentence end at the
// split first, then balance, then the byte-wise greater first line); false when nothing fits.
bool
cue_layout(
    const std::vector<TimedWord>& words, size_t begin, size_t end_inclusive, std::string& out) {
    const size_t end = end_inclusive + 1;
    const std::string single = join_texts(words, begin, end);
    if (grapheme_count(single) <= kMaxLineGraphemes) {
        out = single;
        return true;
    }
    bool have = false;
    size_t best_punct = 0, best_balance = 0;
    std::string best_first, best_second;
    for (size_t split = begin + 1; split < end; ++split) {
        std::string first = join_texts(words, begin, split), second = join_texts(words, split, end);
        const size_t first_length = grapheme_count(first), second_length = grapheme_count(second);
        if (first_length > kMaxLineGraphemes || second_length > kMaxLineGraphemes)
            continue;
        const size_t punct = cue_sentence_end(words[split - 1].text) ? 1 : 0;
        const size_t balance = std::numeric_limits<size_t>::max() -
                               (first_length > second_length ? first_length - second_length
                                                             : second_length - first_length);
        const auto candidate = std::tie(punct, balance, first, second);
        if (!have || candidate > std::tie(best_punct, best_balance, best_first, best_second)) {
            have = true;
            best_punct = punct, best_balance = balance;
            best_first = std::move(first), best_second = std::move(second);
        }
    }
    if (!have)
        return false;
    out = best_first + "\n" + best_second;
    return true;
}

uint64_t
subtitle_milliseconds(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0)
        throw std::invalid_argument("subtitle cue times must be finite and non-negative");
    const double milliseconds = std::round(seconds * 1000.0);
    if (milliseconds > static_cast<double>(std::numeric_limits<uint64_t>::max()))
        throw std::invalid_argument("subtitle cue time is outside the supported range");
    return static_cast<uint64_t>(milliseconds);
}

std::string
format_milliseconds(uint64_t milliseconds, char separator) {
    char buffer[64];
    std::snprintf(
        buffer, sizeof(buffer), "%02llu:%02llu:%02llu%c%03llu",
        static_cast<unsigned long long>(milliseconds / 3600000),
        static_cast<unsigned long long>((milliseconds / 60000) % 60),
        static_cast<unsigned long long>((milliseconds / 1000) % 60), separator,
        static_cast<unsigned long long>(milliseconds % 1000));
    return buffer;
}

}  // namespace

size_t
grapheme_count(const std::string& text) {
    const auto cps = code_points(text);
    size_t clusters = 0;
    uint8_t previous = 0;
    bool previous_is_zwj_after_pictographic = false;  // GB11: ExtPict Extend* ZWJ
    bool in_pictographic_run = false;                 // ExtPict Extend* seen so far
    size_t regional_run = 0;                          // RIs ending at the previous code point
    for (size_t k = 0; k < cps.size(); ++k) {
        const uint8_t value = break_value(cps[k]);
        const uint8_t cls = value & 0x7f;
        const bool pictographic = (value & 0x80) != 0;
        bool boundary = true;
        if (k > 0) {
            const uint8_t prev = previous & 0x7f;
            if (prev == kCR && cls == kLF)
                boundary = false;  // GB3
            else if (prev == kControl || prev == kCR || prev == kLF)
                boundary = true;  // GB4
            else if (cls == kControl || cls == kCR || cls == kLF)
                boundary = true;  // GB5
            else if (prev == kL && (cls == kL || cls == kV || cls == kLV || cls == kLVT))
                boundary = false;  // GB6
            else if ((prev == kLV || prev == kV) && (cls == kV || cls == kT))
                boundary = false;  // GB7
            else if ((prev == kLVT || prev == kT) && cls == kT)
                boundary = false;  // GB8
            else if (cls == kExtend || cls == kZWJ)
                boundary = false;  // GB9
            else if (cls == kSpacingMark)
                boundary = false;  // GB9a
            else if (prev == kPrepend)
                boundary = false;  // GB9b
            else if (previous_is_zwj_after_pictographic && pictographic)
                boundary = false;  // GB11
            else if (
                prev == kRegionalIndicator && cls == kRegionalIndicator && regional_run % 2 == 1)
                boundary = false;  // GB12/GB13
        }
        if (boundary)
            ++clusters;
        // State for GB11 and GB12/13.
        previous_is_zwj_after_pictographic = cls == kZWJ && in_pictographic_run;
        if (pictographic)
            in_pictographic_run = true;
        else if (cls != kExtend)
            in_pictographic_run = false;
        regional_run = cls == kRegionalIndicator ? regional_run + 1 : 0;
        previous = value;
    }
    return clusters;
}

std::vector<WordCue>
word_cues(const std::vector<TimedWord>& words) {
    for (size_t index = 0; index < words.size(); ++index) {
        const auto& word = words[index];
        if (blank(word.text))
            throw std::invalid_argument(
                "word " + std::to_string(index) + " has empty display text");
        if (!std::isfinite(word.start) || !std::isfinite(word.end) || word.start < 0.0 ||
            word.end < word.start)
            throw std::invalid_argument(
                "word " + std::to_string(index) + " has invalid timestamps");
        if (index > 0 && word.start < words[index - 1].start)
            throw std::invalid_argument(
                "word " + std::to_string(index) + " starts before the preceding word");
    }
    // Another speaker's word can end after the last word of a cue.
    const auto latest_end = [&](size_t begin, size_t end_inclusive) {
        double end = words[begin].end;
        for (size_t k = begin + 1; k <= end_inclusive; ++k) end = std::max(end, words[k].end);
        return end;
    };
    std::vector<WordCue> cues;
    size_t first = 0;
    while (first < words.size()) {
        bool have_valid = false, have_sentence = false;
        size_t last_valid = 0, last_sentence = 0;
        std::string layout;
        double end = words[first].end;
        for (size_t candidate = first; candidate < words.size(); ++candidate) {
            end = std::max(end, words[candidate].end);
            const double duration = end - words[first].start;
            if (duration > kMaxCueSeconds)
                break;
            if (!cue_layout(words, first, candidate, layout))
                break;
            have_valid = true, last_valid = candidate;
            if (duration >= kMinSentenceBreakSeconds && cue_sentence_end(words[candidate].text))
                have_sentence = true, last_sentence = candidate;
        }
        WordCue cue;
        cue.first_word = first;
        if (have_sentence || have_valid) {
            cue.last_word = have_sentence ? last_sentence : last_valid;
            cue_layout(words, first, cue.last_word, cue.text);
        } else {
            // A single word that cannot meet the layout or duration bounds is its own cue.
            cue.last_word = first;
            cue.text = words[first].text;
        }
        cue.start = words[first].start;
        cue.end = latest_end(first, cue.last_word);
        first = cue.last_word + 1;
        cues.push_back(std::move(cue));
    }
    return cues;
}

std::string
render_word_cues(const std::vector<WordCue>& cues, const std::string& format) {
    const bool vtt = format == "vtt";
    if (!vtt && format != "srt")
        throw std::invalid_argument("subtitle format must be srt or vtt");
    std::string body = vtt ? "WEBVTT\n\n" : "";
    for (size_t index = 0; index < cues.size(); ++index) {
        const std::string& text = cues[index].text;
        if (blank(text))
            throw std::invalid_argument(
                "subtitle cue " + std::to_string(index) + " has empty text");
        if (vtt) {
            bool empty_line = text.front() == '\n' || text.back() == '\n' ||
                              text.find("\n\n") != std::string::npos;
            if (text.find("-->") != std::string::npos || empty_line)
                throw std::invalid_argument("WebVTT cue text contains an invalid separator");
        } else {
            body += std::to_string(index + 1) + "\n";
        }
        const uint64_t start = subtitle_milliseconds(cues[index].start);
        const uint64_t end = std::max(start + 1, subtitle_milliseconds(cues[index].end));
        body += format_milliseconds(start, vtt ? '.' : ',');
        body += " --> ";
        body += format_milliseconds(end, vtt ? '.' : ',');
        body += "\n" + text + "\n\n";
    }
    return body;
}

}  // namespace nemo_speech::subtitle
