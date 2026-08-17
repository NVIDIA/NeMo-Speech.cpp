// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tts/tokenizer/tokenizer.h"

namespace tts = nemo_speech::tts;

namespace {

bool
check_tokens(
    const tts::MagpieNativeTokenizer& tokenizer, const std::string& text,
    const std::string& language, const std::vector<int32_t>& expected) {
    const auto result = tokenizer.tokenize(text, language);
    if (result.tokens == expected) {
        return true;
    }

    std::fprintf(
        stderr, "token mismatch for text='%s' language='%s'\nexpected:", text.c_str(),
        language.c_str());
    for (const int32_t token : expected) {
        std::fprintf(stderr, " %d", token);
    }
    std::fprintf(stderr, "\nactual:");
    for (const int32_t token : result.tokens) {
        std::fprintf(stderr, " %d", token);
    }
    std::fprintf(stderr, "\n");
    return false;
}

#if !defined(NEMO_SPEECH_TTS_WITH_JA) || !defined(NEMO_SPEECH_TTS_WITH_ZH)
bool
check_unsupported(
    const tts::MagpieNativeTokenizer& tokenizer, const std::string& text,
    const std::string& language) {
    try {
        static_cast<void>(tokenizer.tokenize(text, language));
    }
    catch (const std::invalid_argument& error) {
        return std::string(error.what()).find("unsupported language_code") != std::string::npos;
    }
    return false;
}
#endif

bool
check_chunk_count(
    const tts::MagpieNativeTokenizer& tokenizer, const std::string& text,
    const std::string& language, size_t expected_min_chunks) {
    const auto result = tokenizer.tokenize(text, language);
    std::vector<int32_t> flattened;
    for (const auto& chunk : result.chunks) {
        flattened.insert(flattened.end(), chunk.tokens.begin(), chunk.tokens.end());
    }
    if (flattened != result.tokens) {
        std::fprintf(stderr, "chunk tokens do not flatten to result tokens\n");
        return false;
    }
    if (result.chunks.size() >= expected_min_chunks) {
        return true;
    }
    std::fprintf(
        stderr, "expected at least %zu chunks for language='%s', got %zu\n", expected_min_chunks,
        language.c_str(), result.chunks.size());
    return false;
}

int
count_words_ascii_space(const std::string& text) {
    int count = 0;
    bool in_word = false;
    for (const char c : text) {
        const bool is_space = c == ' ' || c == '\t' || c == '\r' || c == '\n';
        if (is_space) {
            if (in_word) {
                ++count;
                in_word = false;
            }
        } else {
            in_word = true;
        }
    }
    return count + (in_word ? 1 : 0);
}

bool
check_chunks(
    const tts::MagpieNativeTokenizer& tokenizer, const std::string& text,
    const std::string& language, const std::vector<std::string>& expected_texts,
    int max_words_per_chunk) {
    const auto result = tokenizer.tokenize(text, language);
    if (result.chunks.size() != expected_texts.size()) {
        std::fprintf(
            stderr, "expected %zu chunks for language='%s', got %zu\n", expected_texts.size(),
            language.c_str(), result.chunks.size());
        return false;
    }
    for (size_t i = 0; i < result.chunks.size(); ++i) {
        const int words = count_words_ascii_space(result.chunks[i].text);
        if (words > max_words_per_chunk) {
            std::fprintf(
                stderr, "chunk %zu has %d words, expected at most %d: '%s'\n", i, words,
                max_words_per_chunk, result.chunks[i].text.c_str());
            return false;
        }
        if (result.chunks[i].text != expected_texts[i]) {
            std::fprintf(
                stderr, "chunk %zu mismatch\nexpected: '%s'\nactual:   '%s'\n", i,
                expected_texts[i].c_str(), result.chunks[i].text.c_str());
            return false;
        }
    }
    return true;
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s TOKENIZER_MODEL_DIR\n", argv[0]);
        return 2;
    }

    tts::MagpieNativeTokenizer tokenizer(argv[1]);
    std::ifstream tokenizer_config(std::string(argv[1]) + "/model_config.yaml");
    const std::string config_text(
        (std::istreambuf_iterator<char>(tokenizer_config)), std::istreambuf_iterator<char>());
    const bool v2607 = config_text.find("portuguese_Brazilian_phoneme") != std::string::npos;
    if (v2607) {
        bool v2607_ok = true;
        v2607_ok &= check_tokens(tokenizer, "A", "en-US", {90, 94, 94, 94, 88, 3358});
        v2607_ok &= check_tokens(tokenizer, "مرحبا", "ar-AE", {1329, 1405, 1391, 1387, 1382,
                                                                    1381, 1329, 3358});
        v2607_ok &= check_tokens(tokenizer, "مرحبا", "ar-SA", {1493, 1569, 1555, 1551, 1546,
                                                                    1545, 1493, 3358});
        v2607_ok &= check_tokens(tokenizer, "مرحبا", "ar-MSA", {1657, 1733, 1719, 1715, 1710,
                                                                     1709, 1657, 3358});
        v2607_ok &= check_tokens(tokenizer, "안녕하세요.", "ko-KR", {3212, 3125, 3112, 3211, 3109,
                                                                          3125, 3213, 3125, 3128, 3212,
                                                                          3108, 3160, 3212, 3130, 3124,
                                                                          3022, 2974, 3358});
        v2607_ok &= check_tokens(tokenizer, "Olá!", "pt-BR", {1125, 1082, 1079, 1119, 1070,
                                                                    1017, 1125, 3358});
        return v2607_ok ? 0 : 1;
    }
    bool ok = true;

    ok &= check_tokens(tokenizer, "A", "en-US", {90, 94, 94, 53, 84, 2361});
    ok &= check_tokens(tokenizer, "T", "en-US", {90, 94, 94, 65, 56, 2361});
    ok &= check_tokens(tokenizer, "S", "en-US", {90, 94, 94, 81, 64, 2361});
    ok &= check_tokens(tokenizer, "T T S", "en-US", {41, 93, 41, 93, 40, 2361});
    ok &= check_tokens(tokenizer, "TTS", "en-US", {90, 65, 56, 90, 65, 56, 90, 81, 64, 2361});
    ok &= check_tokens(
        tokenizer, "I don’t know what to do!", "en-US",
        {90, 50, 84, 93, 90, 52, 62, 87, 61, 65, 93, 90, 61, 62,  87,
         93, 90, 68, 88, 65, 93, 90, 65, 66, 93, 25, 36, 0,  2361});
    ok &= check_tokens(
        tokenizer, "I’m so scared! What’s happening? I don’t know what to do!", "en-US",
        {90, 50, 84, 60, 93, 90, 64, 62, 87, 93, 90, 64, 58, 81, 85, 52, 0,  93, 90, 68, 88,
         65, 64, 93, 90, 55, 74, 63, 79, 61, 84, 76, 21, 93, 90, 50, 84, 93, 90, 52, 62, 87,
         61, 65, 93, 90, 61, 62, 87, 93, 90, 68, 88, 65, 93, 90, 65, 66, 93, 25, 36, 0,  2361});

    tts::MagpieTokenizerConfig low_limit_config;
    low_limit_config.sentence_limit.en = 1;
    tts::MagpieNativeTokenizer low_limit_tokenizer(argv[1], low_limit_config);
    ok &= check_tokens(
        low_limit_tokenizer, "I’m so scared! What’s happening? I don’t know what to do!", "en-US",
        {90, 50, 84, 60, 93, 90, 64, 62, 87, 93, 90, 64, 58,   81, 85, 52, 0,  2361, 90, 68, 88,
         65, 64, 93, 90, 55, 74, 63, 79, 61, 84, 76, 21, 2361, 90, 50, 84, 93, 90,   52, 62, 87,
         61, 65, 93, 90, 61, 62, 87, 93, 90, 68, 88, 65, 93,   90, 65, 66, 93, 25,   36, 0,  2361});
    ok &= check_chunk_count(
        low_limit_tokenizer, "I’m so scared! What’s happening? I don’t know what to do!", "en-US",
        3);
    ok &= check_chunks(
        low_limit_tokenizer,
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree "
        "twentyfour twentyfive twentysix twentyseven twentyeight twentynine thirty "
        "thirtyone thirtytwo thirtythree thirtyfour thirtyfive thirtysix thirtyseven "
        "thirtyeight thirtynine forty",
        "en-US",
        {"One two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree "
         "twentyfour twentyfive twentysix twentyseven twentyeight twentynine thirty "
         "thirtyone thirtytwo thirtythree thirtyfour thirtyfive.",
         "Thirtysix thirtyseven thirtyeight thirtynine forty"},
        35);
    int transformed_chunks = 0;
    const auto transformed =
        low_limit_tokenizer.tokenize("a 2. b 3.", "en-US", [&](const std::string& chunk) {
            ++transformed_chunks;
            std::string out = chunk;
            const size_t two = out.find("2");
            if (two != std::string::npos) {
                out.replace(two, 1, "two");
            }
            const size_t three = out.find("3");
            if (three != std::string::npos) {
                out.replace(three, 1, "three");
            }
            return out;
        });
    if (transformed_chunks != 2 || transformed.chunks.size() != 2 ||
        transformed.chunks[0].text != "A two." || transformed.chunks[1].text != "B three.") {
        std::fprintf(stderr, "chunk text transform did not run once per chunk after splitting\n");
        ok = false;
    }
    std::vector<bool> final_chunk_flags;
    const auto positioned = low_limit_tokenizer.tokenize(
        "a 2. b 3.", "en-US", [&](const std::string& chunk, bool is_final) {
            final_chunk_flags.push_back(is_final);
            return chunk;
        });
    if (positioned.chunks.size() != 2 || final_chunk_flags != std::vector<bool>{false, true}) {
        std::fprintf(stderr, "positioned chunk transform did not identify only the final chunk\n");
        ok = false;
    }
    ok &= check_tokens(tokenizer, "T", "es-ES", {196, 197, 197, 127, 196, 2361});
    ok &= check_tokens(tokenizer, "año", "es-ES", {196, 187, 136, 180, 148, 196, 2361});
    ok &=
        check_tokens(tokenizer, "última", "es-ES", {196, 170, 119, 127, 116, 120, 108, 196, 2361});
    ok &= check_tokens(
        tokenizer, "Qué quieres", "es-ES",
        {196, 144, 187, 139, 196, 124, 128, 116, 112, 125, 112, 126, 196, 2361});
    ok &= check_tokens(
        tokenizer, "¡Qué alegría verte!", "es-ES",
        {196, 158, 144, 187, 139, 196, 188, 136, 145, 139, 178, 181, 187,
         142, 136, 196, 137, 187, 139, 181, 152, 139, 96,  196, 2361});
    ok &= check_tokens(tokenizer, "T", "de-DE", {346, 301, 332, 288, 334, 346, 2361});
    ok &= check_tokens(
        tokenizer, "er ist sehr gut in physik und mathematik", "de-DE",
        {346, 231, 244, 346, 235, 245, 246, 346, 307, 332, 288, 334, 327, 346, 233, 247, 246,
         346, 332, 325, 296, 346, 289, 306, 334, 307, 332, 291, 334, 293, 346, 247, 240, 230,
         346, 295, 333, 285, 301, 288, 334, 295, 332, 318, 301, 291, 334, 293, 346, 2361});
    ok &= check_tokens(
        tokenizer, "नमस्ते दुनिया।", "hi-IN",
        {1017, 1124, 1129, 1136, 1157, 1120, 1148, 1017, 1122, 1144, 1124, 1142, 1130, 1141, 1163,
         1017, 2361});
    ok &= check_tokens(
        tokenizer, "hello भारत।", "hi-IN",
        {1017, 1171, 1168, 1175, 1175, 1178, 1017, 1128, 1141, 1131, 1120, 1163, 1017, 2361});
    ok &= check_tokens(tokenizer, "अ", "hi-IN", {1017, 1206, 1206, 1091, 1017, 2361});
#ifdef NEMO_SPEECH_TTS_WITH_JA
    ok &= check_tokens(tokenizer, "こんにちは世界。", "ja-JP", {458, 459, 479, 460, 541, 460, 503,
                                                                460, 493, 460, 539, 460, 487, 459,
                                                                471, 459, 464, 597, 458, 2361});
    ok &= check_tokens(tokenizer, "あ", "ja-JP", {458, 631, 459, 462, 458, 2361});
#else
    ok &= check_unsupported(tokenizer, "こんにちは世界。", "ja-JP");
#endif
#ifndef NEMO_SPEECH_TTS_WITH_ZH
    ok &= check_unsupported(tokenizer, "你好。", "zh-CN");
#endif
    ok &= check_tokens(tokenizer, "A", "fr-FR", {701, 633, 633, 633, 634, 2361});
    ok &= check_tokens(tokenizer, "A", "it-IT", {1276, 1208, 1208, 1208, 1209, 2361});
    ok &= check_tokens(tokenizer, "A", "vi-VN", {1660, 1592, 1592, 1592, 1593, 2361});
    ok &= check_chunk_count(tokenizer, "Short sentence. Still short.", "en-US", 1);
    ok &= check_chunk_count(
        tokenizer,
        "One two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty twenty one twenty two. "
        "Twenty three twenty four twenty five twenty six twenty seven twenty eight twenty nine "
        "thirty thirty one thirty two thirty three thirty four thirty five.",
        "en-US", 2);
    ok &= check_chunks(
        tokenizer,
        "One two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty, alpha beta gamma delta epsilon "
        "zeta eta theta iota kappa lambda mu nu xi omicron, red orange yellow green blue "
        "purple silver gold black white brown gray pink violet cyan teal indigo, north south "
        "east west center local global public private final.",
        "en-US",
        {"One two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen twenty, alpha beta gamma delta epsilon "
         "zeta eta theta iota kappa lambda mu nu xi omicron,",
         "Red orange yellow green blue purple silver gold black white brown gray pink violet "
         "cyan teal indigo, north south east west center local global public private final."},
        35);

    return ok ? 0 : 1;
}
