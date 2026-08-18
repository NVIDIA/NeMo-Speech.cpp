// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nemo_speech::common {
class ParameterParser;
}

namespace nemo_speech::tts {

std::string ensure_terminal_punctuation(const std::string& text, const std::string& language_code);

struct MagpieTokenizerSentenceLimits {
    int en = 45;
    int es = 73;
    int fr = 69;
    int vi = 50;
    int it = 53;
    int de = 50;
    int zh = 100;
    int hi = 40;
    int ja = 40;
    int ar = 45;
    int ko = 45;
    int pt = 45;

    void Register(common::ParameterParser& parser);
};

struct MagpieTokenizerConfig {
    MagpieTokenizerSentenceLimits sentence_limit;

    void Register(common::ParameterParser& parser);
};

struct MagpieTokenChunk {
    std::string text;
    std::vector<int32_t> tokens;
};

struct MagpieTokenizationResult {
    std::string language;
    std::string tokenizer_name;
    std::vector<int32_t> tokens;
    std::vector<MagpieTokenChunk> chunks;
};

struct MagpieSupportedLanguage {
    std::string language_code;
    std::string normalized_language;
};

const std::vector<MagpieSupportedLanguage>& supported_languages();
std::vector<std::string> supported_language_codes();

class MagpieNativeTokenizer {
   public:
    using ChunkTextTransform = std::function<std::string(const std::string&)>;
    using PositionedChunkTextTransform =
        std::function<std::string(const std::string&, bool is_final)>;

    explicit MagpieNativeTokenizer(std::string model_dir);
    MagpieNativeTokenizer(std::string model_dir, MagpieTokenizerConfig config);
    ~MagpieNativeTokenizer();

    MagpieNativeTokenizer(const MagpieNativeTokenizer&) = delete;
    MagpieNativeTokenizer& operator=(const MagpieNativeTokenizer&) = delete;

    MagpieTokenizationResult tokenize(
        const std::string& text, const std::string& language_code) const;
    MagpieTokenizationResult tokenize(
        const std::string& text, const std::string& language_code,
        const ChunkTextTransform& chunk_text_transform) const;
    MagpieTokenizationResult tokenize(
        const std::string& text, const std::string& language_code,
        const PositionedChunkTextTransform& chunk_text_transform) const;

    static std::string normalize_language_code(const std::string& language_code);
    std::vector<std::string> supported_language_codes() const;

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts
