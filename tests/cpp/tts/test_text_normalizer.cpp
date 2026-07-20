// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// TTS text normalization smoke test. With NEMO_SPEECH_WITH_NORM=ON, this loads
// the Sparrowhawk TN FARs and verifies written-form text is rewritten to spoken
// form before TTS tokenization.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "tts/preproc/text_normalizer.h"
#include "tts/tokenizer/tokenizer.h"

namespace {

void
check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
    std::cerr << "OK: " << message << "\n";
}

#ifdef NEMO_SPEECH_WITH_NORM
bool
contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

bool
contains_language(const std::vector<std::string>& languages, const std::string& language) {
    return std::find(languages.begin(), languages.end(), language) != languages.end();
}

bool
reaches_language_lookup(
    const nemo_speech::tts::preproc::TextNormalizer& tn, const std::string& text,
    const std::string& missing_language) {
    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    try {
        (void)tn.normalize(text, missing_language);
    }
    catch (...) {
        std::cerr.rdbuf(previous);
        throw;
    }
    std::cerr.rdbuf(previous);
    return contains(captured.str(), "no TN grammar loaded for language");
}
#endif

std::string
model_dir_from_args(int argc, char** argv) {
    if (argc > 1 && argv[1][0] != '\0') {
        return argv[1];
    }
    if (const char* env = std::getenv("TN_MODEL_DIR")) {
        return env;
    }
    return "models/tn_configs";
}

}  // namespace

int
main(int argc, char** argv) {
    const std::string model_dir = model_dir_from_args(argc, argv);
    nemo_speech::tts::preproc::TextNormalizer tn(model_dir);

#ifdef NEMO_SPEECH_WITH_NORM
    check(tn.enabled(), "TN grammar loaded");

    const std::string input = "I have 2 apples";
    const std::string output = tn.normalize(input);
    std::cerr << "TN: '" << input << "' -> '" << output << "'\n";
    check(output != input, "TN rewrites written-form input");
    check(contains(output, "two"), "TN verbalizes numeral 2");
    check(!contains(output, "2"), "TN removes written numeral");
    check(tn.normalize("$") == "$", "TN skips text without alphanumeric characters");
    check(
        reaches_language_lookup(tn, "१२३", "x-native-numeral"),
        "TN detects Hindi Devanagari numerals");
    check(
        reaches_language_lookup(tn, "一二三", "y-native-numeral"), "TN detects Mandarin numerals");
    check(
        reaches_language_lookup(tn, "壱弐参", "z-native-numeral"),
        "TN detects Japanese formal numerals");
    check(
        reaches_language_lookup(tn, "１２３", "w-native-numeral"),
        "TN detects full-width numerals");

    const auto languages = tn.languages();
    check(contains_language(languages, "en"), "TN reports English grammar");

    const std::string french_input = "J'ai 2 pommes";
    if (contains_language(languages, "fr")) {
        const std::string french_output = tn.normalize(french_input, "fr-FR");
        std::cerr << "TN fr: '" << french_input << "' -> '" << french_output << "'\n";
        check(french_output != french_input, "TN routes French request to French grammar");
        check(!contains(french_output, "2"), "French TN removes written numeral");
    } else {
        check(
            tn.normalize(french_input, "fr-FR") == french_input,
            "TN does not use English grammar for French request");
    }

    if (contains_language(languages, "de")) {
        const std::string german_input =
            "die preise für eine einzelfahrkarte zwischen den beiden städten beginnen bei €11";
        const std::string german_normalized = tn.normalize(german_input, "de-DE");
        const std::string german_processed =
            nemo_speech::tts::ensure_terminal_punctuation(german_normalized, "de-DE");
        check(
            contains(german_normalized, "elf euro"),
            "German TN verbalizes sentence-final currency before punctuation is added");
        check(
            !contains(german_normalized, "ein euro erste"),
            "German TN does not interpret a synthetic period as an ordinal");
        check(
            german_processed == german_normalized + ".",
            "terminal punctuation is added after German TN");
        for (const char punctuation : {',', ':', ';'}) {
            const std::string punctuated =
                tn.normalize(german_input + std::string(1, punctuation), "de-DE");
            check(
                nemo_speech::tts::ensure_terminal_punctuation(punctuated, "de-DE") ==
                    german_processed,
                "German trailing comma/colon/semicolon becomes a sentence terminal after TN");
        }
    }
#else
    check(!tn.enabled(), "TN disabled without Sparrowhawk build flag");
    check(tn.normalize("I have 2 apples") == "I have 2 apples", "disabled TN is pass-through");
#endif

    return 0;
}
