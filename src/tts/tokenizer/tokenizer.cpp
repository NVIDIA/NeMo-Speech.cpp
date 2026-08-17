// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "tts/tokenizer/tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "tokenizer_impl.cpp"

namespace nemo_speech::tts {
namespace {

std::string
trim_language(std::string value) {
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char c) {
                    return !std::isspace(c);
                }));
    value.erase(
        std::find_if(value.rbegin(), value.rend(), [](unsigned char c) { return !std::isspace(c); })
            .base(),
        value.end());
    return value;
}

int
sentence_limit_for_language(
    const MagpieTokenizerSentenceLimits& limits, const std::string& language) {
    if (language == "en")
        return limits.en;
    if (language == "es")
        return limits.es;
    if (language == "fr")
        return limits.fr;
    if (language == "de")
        return limits.de;
    if (language == "it")
        return limits.it;
    if (language == "vi")
        return limits.vi;
    if (language == "zh")
        return limits.zh;
    if (language == "hi")
        return limits.hi;
    if (language == "ja")
        return limits.ja;
    if (language == "ar" || language.rfind("ar-", 0) == 0)
        return limits.ar;
    if (language == "ko")
        return limits.ko;
    if (language == "pt" || language == "pt-br")
        return limits.pt;
    return limits.en;
}

int
count_utf8_chars(const std::string& text) {
    int count = 0;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = (unsigned char)text[i];
        size_t n = 1;
        if ((c & 0xe0) == 0xc0) {
            n = 2;
        } else if ((c & 0xf0) == 0xe0) {
            n = 3;
        } else if ((c & 0xf8) == 0xf0) {
            n = 4;
        }
        if (i + n > text.size()) {
            n = 1;
        }
        ++count;
        i += n;
    }
    return count;
}

int
count_words_ascii_space(const std::string& text) {
    int count = 0;
    bool in_word = false;
    for (unsigned char c : text) {
        if (std::isspace(c)) {
            if (in_word) {
                ++count;
                in_word = false;
            }
        } else {
            in_word = true;
        }
    }
    if (in_word) {
        ++count;
    }
    return count;
}

bool
should_tokenize_by_sentence(
    const std::string& text, const std::string& language,
    const MagpieTokenizerSentenceLimits& limits) {
    const int threshold = sentence_limit_for_language(limits, language);
    if (threshold < 0) {
        return false;
    }
    const int measured = (language == "zh" || language == "ja") ? count_utf8_chars(text)
                                                                : count_words_ascii_space(text);
    return measured >= threshold;
}

}  // namespace

const std::vector<MagpieSupportedLanguage>&
supported_languages() {
    static const std::vector<MagpieSupportedLanguage> languages = {
        {"en-US", "en"}, {"es-ES", "es"}, {"de-DE", "de"},
        {"fr-FR", "fr"}, {"it-IT", "it"}, {"vi-VN", "vi"},
#ifdef NEMO_SPEECH_TTS_WITH_ZH
        {"zh-CN", "zh"},
#endif
        {"hi-IN", "hi"},
#ifdef NEMO_SPEECH_TTS_WITH_JA
        {"ja-JP", "ja"},
#endif
    };
    return languages;
}

std::vector<std::string>
supported_language_codes() {
    std::vector<std::string> codes;
    const auto& languages = supported_languages();
    codes.reserve(languages.size());
    for (const auto& language : languages) {
        codes.push_back(language.language_code);
    }
    return codes;
}

class MagpieNativeTokenizer::Impl {
   public:
    Impl(std::string model_dir, MagpieTokenizerConfig config)
        : model_dir_(std::move(model_dir)), config_(config) {
        if (model_dir_.empty()) {
            throw std::invalid_argument("tokenizer model directory is required");
        }
        std::ifstream config_file(fs::path(model_dir_) / "model_config.yaml");
        std::string contents((std::istreambuf_iterator<char>(config_file)), {});
        v2607_ = contents.find("portuguese_Brazilian_phoneme") != std::string::npos;
    }

    MagpieTokenizationResult tokenize(
        const std::string& text, const std::string& language_code,
        const MagpieNativeTokenizer::PositionedChunkTextTransform& chunk_text_transform) const {
        if (text.empty()) {
            throw std::invalid_argument("text is empty");
        }

        params p;
        p.model = model_dir_;
        p.text = text;
        p.language = MagpieNativeTokenizer::normalize_language_code(language_code);
        p.v2607 = v2607_;
        p.chunk_text_transform = chunk_text_transform;

        const std::string tokenizer_name = tokenizer_for_language(p.language);
        if (tokenizer_name.empty()) {
            throw std::invalid_argument("unsupported language_code '" + language_code + "'");
        }
        if (!supports_native(p)) {
            throw std::invalid_argument(
                "native Magpie tokenizer is not available for language_code '" + language_code +
                "'");
        }
        p.sentence_chunking = should_tokenize_by_sentence(text, p.language, config_.sentence_limit);

        tokenizer_result native = tokenize_native(p);
        if (v2607_) {
            int offset_delta = 0;
            if (p.language == "es" || p.language == "de" || p.language == "zh" ||
                p.language == "ja")
                offset_delta = 384;
            else if (p.language == "fr")
                offset_delta = 1188;
            else if (p.language == "it" || p.language == "vi")
                offset_delta = 997;
            else if (p.language == "hi")
                offset_delta = 111;
            for (auto& chunk : native.chunks)
                for (int& token : chunk.tokens) {
                    if (token == native.eos_id)
                        token = 3358;
                    else
                        token += offset_delta;
                }
        }
        MagpieTokenizationResult out;
        out.language = native.language;
        out.tokenizer_name = native.tokenizer_name;
        out.chunks.reserve(native.chunks.size());

        for (const auto& ch : native.chunks) {
            MagpieTokenChunk out_chunk;
            out_chunk.text = ch.text;
            out_chunk.tokens.assign(ch.tokens.begin(), ch.tokens.end());
            out.tokens.insert(out.tokens.end(), out_chunk.tokens.begin(), out_chunk.tokens.end());
            out.chunks.push_back(std::move(out_chunk));
        }
        if (out.tokens.empty()) {
            throw std::invalid_argument("tokenizer produced no text tokens");
        }
        return out;
    }

    std::vector<std::string> supported_language_codes() const {
        std::vector<std::string> languages = {"en-US", "es-ES", "de-DE", "fr-FR",
                                              "it-IT", "vi-VN", "hi-IN"};
        if (v2607_) {
            languages.insert(
                languages.end(), {"ar-AE", "ar-SA", "ar-MSA", "ko-KR", "pt-BR"});
        }
#ifdef NEMO_SPEECH_TTS_WITH_ZH
        languages.emplace_back("zh-CN");
#endif
#ifdef NEMO_SPEECH_TTS_WITH_JA
        languages.emplace_back("ja-JP");
#endif
        return languages;
    }

   private:
    tokenizer_result tokenize_native(const params& p) const {
        if (p.language == "en" || p.language == "es" || p.language == "de") {
            return tokenize_ipa(p);
        }
#ifdef NEMO_SPEECH_TTS_WITH_ZH
        if (p.language == "zh") {
            return tokenize_mandarin(p);
        }
#endif
        return run_native(p);
    }

#ifdef NEMO_SPEECH_TTS_WITH_ZH
    tokenizer_result tokenize_mandarin(const params& p) const {
        const mandarin_tokenizer& tok = mandarin_tokenizer_for_model();
        return run_mandarin_native(p, tok);
    }

    const mandarin_tokenizer& mandarin_tokenizer_for_model() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (!mandarin_cache_) {
            mandarin_cache_ = std::make_unique<mandarin_tokenizer>(model_dir_);
        }
        return *mandarin_cache_;
    }
#endif

    tokenizer_result tokenize_ipa(const params& p) const {
        if (!fs::is_directory(p.model)) {
            throw std::runtime_error(
                "native IPA tokenization requires an extracted Magpie .nemo directory");
        }
        const ipa_config cfg = ipa_config_for_language(p.language);
        const ipa_tokenizer& tok = ipa_tokenizer_for(p.language, cfg);

        tokenizer_result result;
        result.language = p.language;
        result.tokenizer_name = cfg.tokenizer_name;
        const int pad_id = ipa_pad_id_for_config(cfg);
        for (const std::string& sentence : tokenizer_input_units(p, p.text)) {
            chunk ch;
            ch.text = sentence;
            ch.tokens = tok.encode(sentence);
            ch.tokens.push_back(result.eos_id);
            pad_short_text_chunk_before_eos(ch, result.eos_id, pad_id);
            result.chunks.push_back(std::move(ch));
        }
        return result;
    }

    const ipa_tokenizer& ipa_tokenizer_for(
        const std::string& language, const ipa_config& cfg) const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = ipa_cache_.find(language);
        if (it == ipa_cache_.end()) {
            it = ipa_cache_.emplace(language, std::make_unique<ipa_tokenizer>(model_dir_, cfg))
                     .first;
        }
        return *it->second;
    }

    std::string model_dir_;
    MagpieTokenizerConfig config_;
    bool v2607_ = false;
    mutable std::mutex cache_mutex_;
    mutable std::map<std::string, std::unique_ptr<ipa_tokenizer>> ipa_cache_;
#ifdef NEMO_SPEECH_TTS_WITH_ZH
    mutable std::unique_ptr<mandarin_tokenizer> mandarin_cache_;
#endif
};

MagpieNativeTokenizer::MagpieNativeTokenizer(std::string model_dir)
    : MagpieNativeTokenizer(std::move(model_dir), MagpieTokenizerConfig{}) {}

MagpieNativeTokenizer::MagpieNativeTokenizer(std::string model_dir, MagpieTokenizerConfig config)
    : impl_(std::make_unique<Impl>(std::move(model_dir), config)) {}

MagpieNativeTokenizer::~MagpieNativeTokenizer() = default;

MagpieTokenizationResult
MagpieNativeTokenizer::tokenize(const std::string& text, const std::string& language_code) const {
    return impl_->tokenize(text, language_code, PositionedChunkTextTransform{});
}

MagpieTokenizationResult
MagpieNativeTokenizer::tokenize(
    const std::string& text, const std::string& language_code,
    const ChunkTextTransform& chunk_text_transform) const {
    if (!chunk_text_transform) {
        return impl_->tokenize(text, language_code, PositionedChunkTextTransform{});
    }
    return impl_->tokenize(
        text, language_code, [&chunk_text_transform](const std::string& chunk, bool /*is_final*/) {
            return chunk_text_transform(chunk);
        });
}

MagpieTokenizationResult
MagpieNativeTokenizer::tokenize(
    const std::string& text, const std::string& language_code,
    const PositionedChunkTextTransform& chunk_text_transform) const {
    return impl_->tokenize(text, language_code, chunk_text_transform);
}

std::string
MagpieNativeTokenizer::normalize_language_code(const std::string& language_code) {
    std::string lang = trim_language(language_code);
    if (lang.empty()) {
        return "en";
    }
    std::replace(lang.begin(), lang.end(), '_', '-');
    std::transform(lang.begin(), lang.end(), lang.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    if (lang == "ar-ae" || lang == "ar-sa" || lang == "ar-msa" || lang == "pt-br")
        return lang;
    const size_t dash = lang.find('-');
    if (dash != std::string::npos)
        lang.resize(dash);
    return lang;
}

std::vector<std::string>
MagpieNativeTokenizer::supported_language_codes() const {
    return impl_->supported_language_codes();
}

std::string
ensure_terminal_punctuation(const std::string& text, const std::string& language_code) {
    const size_t last = text.find_last_not_of(" \t\r\n");
    if (last == std::string::npos) {
        return text;
    }

    const std::string language = MagpieNativeTokenizer::normalize_language_code(language_code);
    std::string terminal;
    if (language == "hi") {
        terminal = "।";
    } else if (language == "zh" || language == "ja") {
        terminal = "。";
    } else if (
        language == "en" || language == "es" || language == "fr" || language == "de" ||
        language == "it" || language == "vi" || language == "pt-br" || language == "ko" ||
        language.rfind("ar-", 0) == 0) {
        terminal = ".";
    } else {
        return text;
    }

    auto remove_space_before = [](std::string result, size_t marker_start) {
        size_t first_space = marker_start;
        while (first_space > 0 &&
               (result[first_space - 1] == ' ' || result[first_space - 1] == '\t' ||
                result[first_space - 1] == '\r' || result[first_space - 1] == '\n')) {
            --first_space;
        }
        result.erase(first_space, marker_start - first_space);
        return result;
    };

    static const std::array<std::string, 7> terminals = {".", "?", "!", "？", "！", "。", "।"};
    for (const std::string& existing : terminals) {
        if (last + 1 >= existing.size()) {
            const size_t marker_start = last + 1 - existing.size();
            if (text.compare(marker_start, existing.size(), existing) == 0) {
                return remove_space_before(text, marker_start);
            }
        }
    }

    if (text[last] == ',' || text[last] == ':' || text[last] == ';') {
        std::string result = text;
        result.replace(last, 1, terminal);
        return remove_space_before(std::move(result), last);
    }

    std::string result = text;
    result.insert(last + 1, terminal);
    return result;
}

}  // namespace nemo_speech::tts
