// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "language_tags.h"

#include <algorithm>
#include <regex>
#include <sstream>
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

// Punctuation is what the decoder calls punctuation; everything else is a word.
bool
has_word(const std::string& text) {
    std::istringstream pieces(text);
    std::string piece;
    while (pieces >> piece)
        if (!sp_is_punctuation(piece))
            return true;
    return false;
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
        after_word = after_word || has_word(text.substr(scanned, pos - scanned));
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

std::vector<WordTiming>
label_word_languages(const std::vector<WordTiming>& words) {
    std::vector<WordTiming> labeled;
    labeled.reserve(words.size());
    size_t unlabeled = 0;
    // A tag token has no word boundary, so it can join the open word, with text on either side.
    // Each piece of text is a word of its own, with the frames of the decoder word.
    auto add_text = [&](const WordTiming& word, size_t pos, size_t count) {
        if (count == 0)
            return;
        labeled.push_back(word);
        labeled.back().word = word.word.substr(pos, count);
    };
    for (const auto& w : words) {
        size_t pos = 0;
        for (std::sregex_iterator it(w.word.begin(), w.word.end(), tag_regex()), end; it != end;
             ++it) {
            const size_t tag_pos = static_cast<size_t>(it->position(0));
            add_text(w, pos, tag_pos - pos);
            pos = tag_pos + static_cast<size_t>(it->length(0));
            for (; unlabeled < labeled.size(); ++unlabeled)
                labeled[unlabeled].language = (*it)[1].str();
        }
        add_text(w, pos, w.word.size() - pos);
    }
    return labeled;
}

}  // namespace nemo_speech::asr
