// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Language tags ("<fr-FR>") of prompt-conditioned multilingual models. With language "auto" the
// model writes a tag after the text it labels, following the terminal punctuation; in a stream,
// that can be after the endpoint, at the start of the next utterance.
#pragma once

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

// Decoder words without their language tags, in decoder order. A tag sets `language` on the
// unlabeled words before it; the words after the last tag stay unlabeled. A tag inside a decoder
// word splits it: "Hello<en-US>bonjour" gives "Hello" (en-US) and "bonjour" (unlabeled).
std::vector<WordTiming> label_word_languages(const std::vector<WordTiming>& words);

}  // namespace nemo_speech::asr
