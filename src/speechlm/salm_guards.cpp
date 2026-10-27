// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "salm_guards.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <string>

namespace nemo_speech::speechlm::guards {
namespace {

struct Range {
    char32_t first, last;
};

constexpr Range kLetterMark[] = {
#include "letter_mark_table.inc"
};

constexpr Range kLatin[] = {{0x41, 0x5A},     {0x61, 0x7A},     {0xAA, 0xAA},     {0xBA, 0xBA},
                            {0xC0, 0x24F},    {0x250, 0x2AF},   {0x1E00, 0x1EFF}, {0x2C60, 0x2C7F},
                            {0xA720, 0xA7FF}, {0xAB30, 0xAB6F}, {0xFB00, 0xFB06}};

bool
in_ranges(const Range* begin, const Range* end, char32_t cp) {
    const Range* it =
        std::upper_bound(begin, end, cp, [](char32_t c, const Range& r) { return c < r.first; });
    return it != begin && cp <= (it - 1)->last;
}

// Strict UTF-8 decode of one code point at `i`; false for an invalid or incomplete sequence.
bool
next_code_point(std::string_view s, size_t& i, char32_t& cp) {
    const auto b = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
    const unsigned char c = b(i);
    size_t len = 0;
    char32_t min = 0;
    if (c < 0x80) {
        cp = c;
        i += 1;
        return true;
    } else if ((c & 0xE0) == 0xC0) {
        len = 2, cp = c & 0x1F, min = 0x80;
    } else if ((c & 0xF0) == 0xE0) {
        len = 3, cp = c & 0x0F, min = 0x800;
    } else if ((c & 0xF8) == 0xF0) {
        len = 4, cp = c & 0x07, min = 0x10000;
    } else {
        return false;
    }
    if (i + len > s.size())
        return false;
    for (size_t k = 1; k < len; ++k) {
        if ((b(i + k) & 0xC0) != 0x80)
            return false;
        cp = (cp << 6) | (b(i + k) & 0x3F);
    }
    if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
        return false;
    i += len;
    return true;
}

bool
periodic(const std::vector<int>& t, size_t base, size_t n, size_t p) {
    for (size_t i = base; i + p < n; ++i)
        if (t[i] != t[i + p])
            return false;
    return true;
}

constexpr size_t kPeriodMax = 32, kSpan = 32, kMinRepeats = 4;
// Four repeats for short phrases: speech repeats a phrase three times ("I'm like X, I'm like X, I'm
// like X"), a loop does not stop.
constexpr size_t kShortPeriodMin = 4, kShortPeriodMax = 12, kShortSpan = 15, kShortMinRepeats = 4;
constexpr size_t kMinWindow = kShortSpan, kEvery = 4;
constexpr size_t kReplayNgram = 16, kReplayWindow = 256;
constexpr double kReplayFraction = 0.9;

}  // namespace

bool
latin_script_language(std::string_view language) {
    static const std::array<std::string_view, 29> kLatinLanguages = {
        "en", "fr", "de", "it", "es", "pt", "nl", "pl", "tr", "sv", "da", "fi", "nb", "nn", "no",
        "cs", "hr", "sk", "sl", "et", "lv", "lt", "hu", "ro", "mt", "id", "vi", "tl", "fil"};
    std::string tag(language);
    for (char& c : tag)
        c = c == '_' ? '-' : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const size_t dash = tag.find('-');
    if (dash != std::string::npos && tag.find('-', dash + 1) != std::string::npos)
        return false;  // > 2 subtags
    const std::string primary = tag.substr(0, dash);
    return !primary.empty() && std::find(kLatinLanguages.begin(), kLatinLanguages.end(), primary) !=
                                   kLatinLanguages.end();
}

bool
script_guard(std::string_view output_language, bool client_instruction) {
    return !client_instruction && latin_script_language(output_language);
}

bool
is_latin_script(std::string_view utf8) {
    for (size_t i = 0; i < utf8.size();) {
        char32_t cp = 0;
        if (!next_code_point(utf8, i, cp) || cp == 0xFFFD)
            return false;
        if (in_ranges(std::begin(kLetterMark), std::end(kLetterMark), cp) &&
            !in_ranges(std::begin(kLatin), std::end(kLatin), cp))
            return false;
    }
    return true;
}

int
loop_period(const std::vector<int>& t, size_t n) {
    if (n < kMinWindow)
        return 0;
    const int last = t[n - 1];
    for (size_t p = 1; p <= kPeriodMax; ++p) {
        const bool short_rule = p >= kShortPeriodMin && p <= kShortPeriodMax;
        const size_t need = short_rule ? std::max(kShortSpan, p * kShortMinRepeats)
                                       : std::max(kSpan, p * kMinRepeats);
        if (n < need || t[n - 1 - p] != last)
            continue;
        const size_t base = n - need;
        if (!periodic(t, base, n, p))
            continue;
        if (short_rule) {
            bool divisor = false;
            for (size_t q = 1; q < p && !divisor; ++q)
                divisor = p % q == 0 && periodic(t, base, n, q);
            if (divisor)
                continue;  // a shorter period explains the window and its own rule declined it
        }
        return static_cast<int>(p);
    }
    return 0;
}

bool
LoopGuard::replay(size_t n) {
    for (size_t i = scanned_; i < n; ++i) {
        if (i + 1 < kReplayNgram) {
            hashes_.push_back(0);
            continue;
        }
        uint64_t h = 1469598103934665603ull;  // FNV-1a over the n-gram's token ids
        for (size_t k = i + 1 - kReplayNgram; k <= i; ++k) {
            h ^= static_cast<uint32_t>(tokens_[k]);
            h *= 1099511628211ull;
        }
        hashes_.push_back(h);
        if (i >= kReplayNgram &&
            i - kReplayNgram + 1 >= kReplayNgram)  // the last n-gram not overlapping this one
            seen_.insert(hashes_[i - kReplayNgram]);
        const char flag = seen_.count(h) ? 1 : 0;
        recent_.push_back(flag);
        repeats_ += flag;
        if (recent_.size() > kReplayWindow) {
            repeats_ -= recent_.front();
            recent_.pop_front();
        }
    }
    scanned_ = std::max(scanned_, n);
    return recent_.size() >= kReplayWindow &&
           static_cast<double>(repeats_) / kReplayWindow >= kReplayFraction;
}

bool
LoopGuard::fires(size_t n) {
    if (n > scanned_ && replay(n))
        return true;
    if (n < kMinWindow || n - last_check_ < kEvery)
        return false;
    last_check_ = n;
    return loop_period(tokens_, n) > 0;
}

bool
LoopGuard::push(int t) {
    if (!fired_)
        fired_ = fires(tokens_.size());
    tokens_.push_back(t);
    return fired_;
}

void
LoopGuard::reset() {
    *this = LoopGuard{};
}

}  // namespace nemo_speech::speechlm::guards
