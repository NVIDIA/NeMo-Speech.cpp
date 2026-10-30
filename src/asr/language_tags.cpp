// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "language_tags.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <regex>
#include <utility>

namespace nemo_speech::asr {

namespace {
const std::regex&
tag_regex() {
    static const std::regex tag_re(R"(<([A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,4})?)>)");
    return tag_re;
}

void
add_language(std::vector<std::string>& languages, std::string code) {
    if (std::find(languages.begin(), languages.end(), code) == languages.end())
        languages.push_back(std::move(code));
}
}  // namespace

std::vector<std::string>
extract_language_tags(std::string& text, std::vector<std::string>* leading) {
    const auto& tag_re = tag_regex();
    std::vector<std::string> langs;
    bool after_word = leading == nullptr;
    size_t scanned = 0;
    bool found = false;
    for (std::sregex_iterator it(text.begin(), text.end(), tag_re), end; it != end; ++it) {
        const size_t pos = static_cast<size_t>(it->position(0));
        after_word = after_word ||
                     std::any_of(text.begin() + scanned, text.begin() + pos, [](unsigned char c) {
                         return c >= 0x80 || std::isalnum(c);
                     });
        scanned = pos + static_cast<size_t>(it->length(0));
        add_language(after_word ? langs : *leading, (*it)[1].str());
        found = true;
    }
    if (found) {
        text = std::regex_replace(text, tag_re, "");
        text = std::regex_replace(text, std::regex(R"( {2,})"), " ");
        const size_t a = text.find_first_not_of(' ');
        const size_t b = text.find_last_not_of(' ');
        text = (a == std::string::npos) ? std::string() : text.substr(a, b - a + 1);
    }
    return langs;
}

void
extract_appended_language_tags(
    std::string& text, size_t old_size, std::vector<std::string>& languages,
    std::vector<std::string>& leading) {
    // A tag can start in the text before the new pieces.
    constexpr size_t kMaxLanguageTagBytes = 10;
    const size_t scan_from = old_size > kMaxLanguageTagBytes ? old_size - kMaxLanguageTagBytes : 0;
    if (text.find('<', scan_from) == std::string::npos)
        return;
    for (auto& code : extract_language_tags(text, &leading))
        add_language(languages, std::move(code));
}

bool
is_language_tag(const std::string& word) {
    return std::regex_match(word, tag_regex());
}

std::vector<LanguageSpan>
language_spans(const std::vector<WordTiming>& words) {
    std::vector<LanguageSpan> spans;
    std::string pending;
    for (const auto& w : words) {
        if (is_language_tag(w.word)) {
            if (pending.empty())
                pending = w.word.substr(1, w.word.size() - 2);
        } else if (!pending.empty()) {
            spans.push_back({w.start_frame, std::exchange(pending, {})});
        }
    }
    if (!pending.empty())
        spans.push_back({std::numeric_limits<int64_t>::max(), std::move(pending)});
    return spans;
}

std::string
word_language(const std::vector<LanguageSpan>& spans, int64_t start_frame) {
    for (const auto& span : spans)
        if (start_frame < span.end_frame)
            return span.code;
    return {};
}

}  // namespace nemo_speech::asr
