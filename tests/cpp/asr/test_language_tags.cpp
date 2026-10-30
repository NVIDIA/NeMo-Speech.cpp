// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Language tags of auto-language multilingual models: which text a tag labels.
#include <cstdio>
#include <string>
#include <vector>

#include "language_tags.h"

using namespace nemo_speech::asr;

static int g_fail = 0;
static void
check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        g_fail++;
    }
}

using Codes = std::vector<std::string>;

void
test_extract_strips_tags() {
    std::string text = "Bonjour. <fr-FR> Hello. <en-US>";
    check(extract_language_tags(text) == Codes{"fr-FR", "en-US"}, "codes in order");
    check(text == "Bonjour. Hello.", "tags removed from text");

    text = "no tag";
    check(extract_language_tags(text).empty() && text == "no tag", "text without tags unchanged");
}

void
test_leading_tag_labels_previous_utterance() {
    Codes leading;
    std::string text = "<en-US> Il y a";
    check(extract_language_tags(text, &leading).empty(), "a tag before any word is not this one's");
    check(leading == Codes{"en-US"}, "a tag before any word goes to the previous utterance");
    check(text == "Il y a", "leading tag removed");

    leading.clear();
    text = ". <en-US> Il y a. <fr-FR>";
    check(extract_language_tags(text, &leading) == Codes{"fr-FR"}, "punctuation is not a word");
    check(leading == Codes{"en-US"}, "a tag after leading punctuation is still leading");

    leading.clear();
    text = "now. <en-US> Widely";
    check(
        extract_language_tags(text, &leading) == Codes{"en-US"},
        "a tag after a word is this one's");
    check(leading.empty(), "no leading tag");
}

void
test_appended_tags() {
    Codes languages, leading;
    std::string text = "<fr-FR>";
    extract_appended_language_tags(text, 0, languages, leading);
    check(languages.empty() && leading == Codes{"fr-FR"}, "a tag alone in a new utterance is late");
    check(text.empty(), "late tag removed");

    text += " Vous";
    extract_appended_language_tags(text, 0, languages, leading);
    const size_t old_size = text.size();
    text += " États. <fr-FR>";
    extract_appended_language_tags(text, old_size, languages, leading);
    check(languages == Codes{"fr-FR"}, "a tag after words labels this utterance");
    check(text == "Vous États.", "appended tag removed");
}

void
test_word_languages_follow_tag_position() {
    const std::vector<WordTiming> words = {
        {"Hello.", 0, 4},  {"<en-US>", 5, 6},   {"Bonjour", 8, 10},
        {"tout.", 11, 12}, {"<fr-FR>", 13, 14}, {"Again", 20, 22},
    };
    check(is_language_tag("<en-US>") && !is_language_tag("Hello."), "tag shape");
    const auto spans = language_spans(words);
    check(word_language(spans, 0) == "en-US", "a tag labels the words before it");
    check(word_language(spans, 8) == "fr-FR", "the next tag labels the words since the last one");
    check(word_language(spans, 11) == "fr-FR", "all words up to the tag");
    check(word_language(spans, 20).empty(), "words after the last tag stay unlabeled");

    const std::vector<WordTiming> late = {{"<en-US>", 0, 1}, {"Il", 2, 3}};
    check(word_language(language_spans(late), 2).empty(), "a leading tag labels no word");
    check(language_spans({{"Hi", 0, 1}}).empty(), "no tags, no spans");
}

int
main() {
    test_extract_strips_tags();
    test_leading_tag_labels_previous_utterance();
    test_appended_tags();
    test_word_languages_follow_tag_position();
    std::fprintf(stdout, g_fail ? "FAILED (%d)\n" : "ALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
