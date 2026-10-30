// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Language tags ("<fr-FR>") of prompt-conditioned multilingual models. With language "auto" the
// model writes a tag after the text it labels, following the terminal punctuation; in a stream,
// that can be after the endpoint, at the start of the next utterance.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "decoder.h"

namespace nemo_speech::asr {

// Removes the tags from `text` and returns their codes, first occurrence first. With `leading`
// set, the codes of tags before the first word of `text` go there instead: they label the
// previous utterance.
std::vector<std::string> extract_language_tags(
    std::string& text, std::vector<std::string>* leading = nullptr);

// extract_language_tags() for an utterance transcript that grew from `old_size` bytes; earlier
// tags are already removed. Adds codes to `languages` and to `leading` (see above).
void extract_appended_language_tags(
    std::string& text, size_t old_size, std::vector<std::string>& languages,
    std::vector<std::string>& leading);

// Whether a decoder word is a language tag.
bool is_language_tag(const std::string& word);

// A tag labels the words that start before the first word after it. Frame times survive ITN
// merging words; the words after the last tag stay unlabeled until the next tag.
struct LanguageSpan {
    int64_t end_frame;  // words that start before this frame
    std::string code;
};

std::vector<LanguageSpan> language_spans(const std::vector<WordTiming>& words);

// The code of the first span that holds a word starting at `start_frame`; empty if none.
std::string word_language(const std::vector<LanguageSpan>& spans, int64_t start_frame);

}  // namespace nemo_speech::asr
