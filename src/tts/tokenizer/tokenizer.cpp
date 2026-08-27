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

enum class tokenizer_kind { ipa, byt5, hindi_chars, mandarin, japanese, arabic };

struct tokenizer_entry {
    std::string name;
    tokenizer_kind kind = tokenizer_kind::byt5;
    int offset = 0;
    int size = 0;
    std::string locale;
    std::string grapheme_case;
    std::string grapheme_prefix;
    std::string ascii_letter_case;
    std::string phoneme_dict;
    std::string heteronyms;
    bool apostrophe = true;
    bool pad_with_space = false;
};

struct tokenizer_profile {
    std::string id;
    std::string nemo_version;
    int text_vocab_size = 0;
    int eos_id = 0;
    std::vector<tokenizer_entry> entries;
    std::map<std::string, std::string> language_mapping;

    const tokenizer_entry& entry_for_language(const std::string& language) const {
        const auto mapping = language_mapping.find(language);
        if (mapping == language_mapping.end()) {
            throw std::invalid_argument("unsupported tokenizer language '" + language + "'");
        }
        const auto entry = std::find_if(
            entries.begin(), entries.end(),
            [&](const tokenizer_entry& value) { return value.name == mapping->second; });
        if (entry == entries.end()) {
            throw std::runtime_error(
                "tokenizer profile '" + id + "' maps language '" + language +
                "' to missing tokenizer '" + mapping->second + "'");
        }
        return *entry;
    }
};

std::string
trim(std::string value) {
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char c) {
                    return !std::isspace(c);
                }));
    value.erase(
        std::find_if(value.rbegin(), value.rend(), [](unsigned char c) { return !std::isspace(c); })
            .base(),
        value.end());
    return value;
}

std::string
unquote(std::string value) {
    value = trim(std::move(value));
    if (value.size() >= 2 && ((value.front() == '\'' && value.back() == '\'') ||
                              (value.front() == '"' && value.back() == '"'))) {
        value = value.substr(1, value.size() - 2);
    }
    return value;
}

int
yaml_indent(const std::string& line) {
    int indent = 0;
    while (indent < static_cast<int>(line.size()) && line[static_cast<size_t>(indent)] == ' ') {
        ++indent;
    }
    return indent;
}

std::vector<std::string>
yaml_lines(const std::string& contents) {
    std::vector<std::string> lines;
    std::istringstream input(contents);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(std::move(line));
    }
    return lines;
}

std::string
top_level_scalar(const std::vector<std::string>& lines, const std::string& key) {
    const std::string prefix = key + ":";
    for (const auto& line : lines) {
        if (yaml_indent(line) != 0 || line.rfind(prefix, 0) != 0) {
            continue;
        }
        return unquote(line.substr(prefix.size()));
    }
    return {};
}

std::map<std::string, std::string>
yaml_child_blocks(
    const std::vector<std::string>& lines, const std::string& section,
    std::vector<std::string>& order) {
    std::map<std::string, std::string> blocks;
    bool active = false;
    std::string current;
    for (const auto& line : lines) {
        if (!active) {
            if (line == section + ":") {
                active = true;
            }
            continue;
        }
        if (!line.empty() && yaml_indent(line) == 0) {
            break;
        }
        if (yaml_indent(line) == 2) {
            const std::string value = trim(line);
            if (!value.empty() && value.back() == ':') {
                current = value.substr(0, value.size() - 1);
                order.push_back(current);
                blocks[current] = {};
                continue;
            }
        }
        if (!current.empty()) {
            blocks[current] += line;
            blocks[current].push_back('\n');
        }
    }
    return blocks;
}

std::string
block_scalar(const std::string& block, const std::string& key) {
    std::istringstream input(block);
    std::string line;
    const std::string prefix = key + ":";
    while (std::getline(input, line)) {
        const std::string value = trim(line);
        if (value.rfind(prefix, 0) == 0) {
            return unquote(value.substr(prefix.size()));
        }
    }
    return {};
}

std::string
artifact_name(const std::string& value) {
    static const std::string prefix = "nemo:";
    if (value.rfind(prefix, 0) == 0) {
        return value.substr(prefix.size());
    }
    return value;
}

void
require_block_value(
    const std::map<std::string, std::string>& blocks, const std::string& tokenizer,
    const std::string& key, const std::string& expected) {
    const auto it = blocks.find(tokenizer);
    if (it == blocks.end()) {
        throw std::runtime_error("missing tokenizer config block '" + tokenizer + "'");
    }
    const std::string actual = block_scalar(it->second, key);
    if (actual != expected) {
        throw std::runtime_error(
            "unsupported tokenizer config: '" + tokenizer + "." + key + "' is '" + actual +
            "', expected '" + expected + "'");
    }
}

void
require_common_tokenizer_values(
    const std::map<std::string, std::string>& blocks, const std::string& tokenizer,
    const std::string& target, const std::string& apostrophe, const std::string& pad_with_space) {
    require_block_value(blocks, tokenizer, "_target_", target);
    require_block_value(blocks, tokenizer, "punct", "true");
    require_block_value(blocks, tokenizer, "apostrophe", apostrophe);
    require_block_value(blocks, tokenizer, "pad_with_space", pad_with_space);
}

void
require_ipa_values(
    const std::map<std::string, std::string>& blocks, const tokenizer_entry& item,
    const std::string& configured_locale = {}) {
    require_common_tokenizer_values(
        blocks, item.name,
        "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.IPATokenizer",
        item.apostrophe ? "true" : "false", item.pad_with_space ? "true" : "false");
    require_block_value(blocks, item.name, "locale", configured_locale);
    const std::string grapheme_case = block_scalar(blocks.at(item.name), "grapheme_case");
    const std::string effective_case = grapheme_case.empty() ? "upper" : grapheme_case;
    if (effective_case != item.grapheme_case) {
        throw std::runtime_error(
            "unsupported tokenizer config: '" + item.name + ".grapheme_case' is '" +
            effective_case + "', expected '" + item.grapheme_case + "'");
    }
    require_block_value(blocks, item.name, "grapheme_prefix", item.grapheme_prefix);
}

void
require_byt5_values(
    const std::map<std::string, std::string>& blocks, const std::string& tokenizer) {
    require_block_value(blocks, tokenizer, "_target_", "AutoTokenizer");
    require_block_value(blocks, tokenizer, "pretrained_model", "google/byt5-small");
}

std::map<std::string, std::string>
parse_language_mapping(const std::vector<std::string>& lines) {
    std::map<std::string, std::string> result;
    bool active = false;
    for (const auto& line : lines) {
        if (!active) {
            if (line == "language_to_tokenizer_mapping:") {
                active = true;
            }
            continue;
        }
        if (!line.empty() && yaml_indent(line) == 0) {
            break;
        }
        if (yaml_indent(line) != 2) {
            continue;
        }
        const std::string value = trim(line);
        const size_t colon = value.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        std::string tokenizer = trim(value.substr(colon + 1));
        if (tokenizer.size() < 2 || tokenizer.front() != '[' || tokenizer.back() != ']') {
            throw std::runtime_error(
                "unsupported language_to_tokenizer_mapping value '" + tokenizer + "'");
        }
        tokenizer = trim(tokenizer.substr(1, tokenizer.size() - 2));
        if (tokenizer.find(',') != std::string::npos) {
            throw std::runtime_error("multiple tokenizer candidates are not supported");
        }
        result.emplace(value.substr(0, colon), unquote(std::move(tokenizer)));
    }
    return result;
}

tokenizer_entry
entry(
    std::string name, tokenizer_kind kind, int offset, int size, std::string locale = {},
    std::string grapheme_case = {}, std::string grapheme_prefix = {}, bool apostrophe = true,
    bool pad_with_space = false) {
    tokenizer_entry value;
    value.name = std::move(name);
    value.kind = kind;
    value.offset = offset;
    value.size = size;
    value.locale = std::move(locale);
    value.grapheme_case = std::move(grapheme_case);
    value.grapheme_prefix = std::move(grapheme_prefix);
    value.apostrophe = apostrophe;
    value.pad_with_space = pad_with_space;
    return value;
}

void
attach_artifacts(
    tokenizer_profile& profile, const std::map<std::string, std::string>& blocks,
    const fs::path& root) {
    for (auto& item : profile.entries) {
        const auto block = blocks.find(item.name);
        if (block == blocks.end()) {
            continue;
        }
        item.phoneme_dict = artifact_name(block_scalar(block->second, "phoneme_dict"));
        item.heteronyms = artifact_name(block_scalar(block->second, "heteronyms"));
        for (const std::string* asset : {&item.phoneme_dict, &item.heteronyms}) {
            if (!asset->empty() && !fs::is_regular_file(root / *asset)) {
                throw std::runtime_error(
                    "tokenizer profile '" + profile.id + "' references missing asset '" +
                    (root / *asset).string() + "'");
            }
        }
        if ((item.kind == tokenizer_kind::ipa || item.kind == tokenizer_kind::mandarin) &&
            item.phoneme_dict.empty()) {
            throw std::runtime_error(
                "tokenizer profile '" + profile.id + "' has no phoneme_dict for '" + item.name +
                "'");
        }
    }
}

void
validate_profile_layout(const tokenizer_profile& profile) {
    int offset = 0;
    for (const auto& item : profile.entries) {
        if (item.offset != offset || item.size <= 0) {
            throw std::runtime_error(
                "invalid tokenizer offset table for profile '" + profile.id + "'");
        }
        offset += item.size;
    }
    if (offset + 2 != profile.text_vocab_size || profile.eos_id != offset + 1) {
        throw std::runtime_error(
            "invalid tokenizer vocabulary size for profile '" + profile.id + "'");
    }
}

tokenizer_profile
load_tokenizer_profile(const fs::path& root) {
    const fs::path config_path = root / "model_config.yaml";
    const std::string contents = read_file(config_path);
    const auto lines = yaml_lines(contents);
    std::vector<std::string> order;
    const auto blocks = yaml_child_blocks(lines, "text_tokenizers", order);
    const std::vector<std::string> v2602_order = {
        "english_phoneme",    "spanish_phoneme",      "german_phoneme",      "mandarin_phoneme",
        "japanese_phoneme",   "french_chartokenizer", "hindi_chartokenizer", "italian_phoneme",
        "vietnamese_phoneme", "text_ce_tokenizer",
    };
    const std::vector<std::string> v2607_order = {
        "english_phoneme",
        "text_ce_tokenizer",
        "spanish_phoneme",
        "german_phoneme",
        "mandarin_phoneme",
        "japanese_phoneme",
        "portuguese_Brazilian_phoneme",
        "hindi_phoneme",
        "arabic_AE_chartokenizer",
        "arabic_SA_chartokenizer",
        "arabic_MSA_chartokenizer",
        "french_chartokenizer",
        "italian_chartokenizer",
        "vietnamese_chartokenizer",
        "korean_chartokenizer",
    };

    tokenizer_profile profile;
    profile.nemo_version = top_level_scalar(lines, "nemo_version");
    if (order == v2602_order) {
        profile.id = "v2602";
        profile.text_vocab_size = 2362;
        profile.eos_id = 2361;
        profile.entries = {
            entry("english_phoneme", tokenizer_kind::ipa, 0, 96, "en-US", "upper", "", true, false),
            entry(
                "spanish_phoneme", tokenizer_kind::ipa, 96, 103, "es-ES", "upper", "", true, true),
            entry(
                "german_phoneme", tokenizer_kind::ipa, 199, 150, "de-DE", "mixed", "#", true, true),
            entry("mandarin_phoneme", tokenizer_kind::mandarin, 349, 109),
            entry("japanese_phoneme", tokenizer_kind::japanese, 458, 175),
            entry("french_chartokenizer", tokenizer_kind::byt5, 633, 384),
            entry("hindi_chartokenizer", tokenizer_kind::hindi_chars, 1017, 191),
            entry("italian_phoneme", tokenizer_kind::byt5, 1208, 384),
            entry("vietnamese_phoneme", tokenizer_kind::byt5, 1592, 384),
            entry("text_ce_tokenizer", tokenizer_kind::byt5, 1976, 384),
        };
        profile.language_mapping = {
            {"en", "english_phoneme"},  {"es", "spanish_phoneme"},
            {"de", "german_phoneme"},   {"fr", "french_chartokenizer"},
            {"it", "italian_phoneme"},  {"vi", "vietnamese_phoneme"},
            {"zh", "mandarin_phoneme"}, {"hi", "hindi_chartokenizer"},
            {"ja", "japanese_phoneme"},
        };
        if (profile.nemo_version != "2.6.0rc0") {
            throw std::runtime_error(
                "unsupported v2602 nemo_version '" + profile.nemo_version + "'");
        }
    } else if (order == v2607_order) {
        profile.id = "v2607";
        profile.text_vocab_size = 3359;
        profile.eos_id = 3358;
        profile.entries = {
            entry("english_phoneme", tokenizer_kind::ipa, 0, 96, "en-US", "upper", "", true, false),
            entry("text_ce_tokenizer", tokenizer_kind::byt5, 96, 384),
            entry(
                "spanish_phoneme", tokenizer_kind::ipa, 480, 103, "es-ES", "upper", "", true, true),
            entry(
                "german_phoneme", tokenizer_kind::ipa, 583, 150, "de-DE", "mixed", "#", true, true),
            entry("mandarin_phoneme", tokenizer_kind::mandarin, 733, 109),
            entry("japanese_phoneme", tokenizer_kind::japanese, 842, 175),
            entry(
                "portuguese_Brazilian_phoneme", tokenizer_kind::ipa, 1017, 111, "pt-BR", "upper",
                "#", true, true),
            entry(
                "hindi_phoneme", tokenizer_kind::ipa, 1128, 201, "hi-IN", "upper", "", true, true),
            entry("arabic_AE_chartokenizer", tokenizer_kind::arabic, 1329, 164),
            entry("arabic_SA_chartokenizer", tokenizer_kind::arabic, 1493, 164),
            entry("arabic_MSA_chartokenizer", tokenizer_kind::arabic, 1657, 164),
            entry("french_chartokenizer", tokenizer_kind::byt5, 1821, 384),
            entry("italian_chartokenizer", tokenizer_kind::byt5, 2205, 384),
            entry("vietnamese_chartokenizer", tokenizer_kind::byt5, 2589, 384),
            entry("korean_chartokenizer", tokenizer_kind::byt5, 2973, 384),
        };
        profile.language_mapping = parse_language_mapping(lines);
        const std::map<std::string, std::string> expected_mapping = {
            {"en", "english_phoneme"},
            {"de", "german_phoneme"},
            {"es", "spanish_phoneme"},
            {"fr", "french_chartokenizer"},
            {"it", "italian_chartokenizer"},
            {"vi", "vietnamese_chartokenizer"},
            {"zh", "mandarin_phoneme"},
            {"hi", "hindi_phoneme"},
            {"ja", "japanese_phoneme"},
            {"pt-BR", "portuguese_Brazilian_phoneme"},
            {"ko", "korean_chartokenizer"},
            {"ar-AE", "arabic_AE_chartokenizer"},
            {"ar-SA", "arabic_SA_chartokenizer"},
            {"ar-MSA", "arabic_MSA_chartokenizer"},
        };
        if (profile.language_mapping != expected_mapping) {
            throw std::runtime_error("unsupported v2607 language_to_tokenizer_mapping");
        }
        profile.language_mapping = {
            {"en", "english_phoneme"},
            {"de", "german_phoneme"},
            {"es", "spanish_phoneme"},
            {"fr", "french_chartokenizer"},
            {"it", "italian_chartokenizer"},
            {"vi", "vietnamese_chartokenizer"},
            {"zh", "mandarin_phoneme"},
            {"hi", "hindi_phoneme"},
            {"ja", "japanese_phoneme"},
            {"pt-br", "portuguese_Brazilian_phoneme"},
            {"ko", "korean_chartokenizer"},
            {"ar-ae", "arabic_AE_chartokenizer"},
            {"ar-sa", "arabic_SA_chartokenizer"},
            {"ar-msa", "arabic_MSA_chartokenizer"},
        };
        if (profile.nemo_version != "2.8.0rc0") {
            throw std::runtime_error(
                "unsupported v2607 nemo_version '" + profile.nemo_version + "'");
        }
    } else {
        std::ostringstream found;
        for (size_t i = 0; i < order.size(); ++i) {
            if (i != 0)
                found << ", ";
            found << order[i];
        }
        throw std::runtime_error(
            "unsupported Magpie tokenizer layout in " + config_path.string() + ": [" + found.str() +
            "]");
    }

    for (auto& item : profile.entries) {
        if (item.kind == tokenizer_kind::japanese) {
            item.ascii_letter_case = block_scalar(blocks.at(item.name), "ascii_letter_case");
        }
    }

    for (const auto& item : profile.entries) {
        switch (item.kind) {
            case tokenizer_kind::ipa:
                require_ipa_values(
                    blocks, item, item.locale == "en-US" ? std::string{} : item.locale);
                if (item.name == "portuguese_Brazilian_phoneme") {
                    require_block_value(blocks, item.name, "locale_specific_punct", "false");
                }
                break;
            case tokenizer_kind::byt5:
                require_byt5_values(blocks, item.name);
                break;
            case tokenizer_kind::hindi_chars:
                require_common_tokenizer_values(
                    blocks, item.name,
                    "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers."
                    "HindiCharsTokenizer",
                    "true", "true");
                break;
            case tokenizer_kind::mandarin:
                require_common_tokenizer_values(
                    blocks, item.name,
                    "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers."
                    "ChinesePhonemesTokenizer",
                    "true", "true");
                require_block_value(blocks, item.name, "ascii_letter_case", "upper");
                break;
            case tokenizer_kind::japanese:
                require_common_tokenizer_values(
                    blocks, item.name,
                    "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers."
                    "JapanesePhonemeTokenizer",
                    "false", "true");
                require_block_value(blocks, item.name, "ascii_letter_case", item.ascii_letter_case);
                break;
            case tokenizer_kind::arabic:
                require_common_tokenizer_values(
                    blocks, item.name,
                    "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers."
                    "ArabicCharsTokenizer",
                    "true", "true");
                require_block_value(blocks, item.name, "charset_version", "1");
                break;
        }
    }
    attach_artifacts(profile, blocks, root);
    validate_profile_layout(profile);
    return profile;
}

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
        profile_ = load_tokenizer_profile(model_dir_);
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
        p.chunk_text_transform = chunk_text_transform;

        const tokenizer_entry* selected = nullptr;
        try {
            selected = &profile_.entry_for_language(p.language);
        }
        catch (const std::invalid_argument&) {
            throw std::invalid_argument("unsupported language_code '" + language_code + "'");
        }
        if (!supports(*selected)) {
            throw std::invalid_argument(
                "native Magpie tokenizer is not available for language_code '" + language_code +
                "'");
        }
        p.tokenizer_name = selected->name;
        p.locale = selected->locale;
        p.grapheme_case = selected->grapheme_case;
        p.grapheme_prefix = selected->grapheme_prefix;
        p.ascii_letter_case = selected->ascii_letter_case;
        p.phoneme_dict = selected->phoneme_dict;
        p.heteronyms = selected->heteronyms;
        p.offset = selected->offset;
        p.eos_id = profile_.eos_id;
        p.expected_vocab_size = selected->size;
        p.apostrophe = selected->apostrophe;
        p.pad_with_space = selected->pad_with_space;
        p.sentence_chunking = should_tokenize_by_sentence(text, p.language, config_.sentence_limit);

        tokenizer_result native = tokenize_native(p, *selected);
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
        static const std::vector<std::pair<std::string, std::string>> canonical = {
            {"en", "en-US"}, {"es", "es-ES"},    {"de", "de-DE"},    {"fr", "fr-FR"},
            {"it", "it-IT"}, {"vi", "vi-VN"},    {"zh", "zh-CN"},    {"hi", "hi-IN"},
            {"ja", "ja-JP"}, {"ar-ae", "ar-AE"}, {"ar-sa", "ar-SA"}, {"ar-msa", "ar-MSA"},
            {"ko", "ko-KR"}, {"pt-br", "pt-BR"},
        };
        std::vector<std::string> languages;
        for (const auto& [normalized, code] : canonical) {
            const auto mapping = profile_.language_mapping.find(normalized);
            if (mapping == profile_.language_mapping.end()) {
                continue;
            }
            const auto item = std::find_if(
                profile_.entries.begin(), profile_.entries.end(),
                [&](const tokenizer_entry& entry) { return entry.name == mapping->second; });
            if (item != profile_.entries.end() && supports(*item)) {
                languages.push_back(code);
            }
        }
        return languages;
    }

    const std::string& profile_id() const { return profile_.id; }
    int text_vocab_size() const { return profile_.text_vocab_size; }

   private:
    bool supports(const tokenizer_entry& entry) const {
        if (entry.kind == tokenizer_kind::japanese) {
#ifdef NEMO_SPEECH_TTS_WITH_JA
            return !find_openjtalk_dictionary_dir(model_dir_).empty();
#else
            return false;
#endif
        }
        if (entry.kind == tokenizer_kind::mandarin) {
#ifdef NEMO_SPEECH_TTS_WITH_ZH
            return mandarin_tokenizer_available(model_dir_);
#else
            return false;
#endif
        }
        return true;
    }

    tokenizer_result tokenize_native(const params& p, const tokenizer_entry& entry) const {
        switch (entry.kind) {
            case tokenizer_kind::ipa:
                return tokenize_ipa(p);
            case tokenizer_kind::byt5:
                return run_byt5_native(p);
            case tokenizer_kind::hindi_chars:
                return run_hindi_native(p);
            case tokenizer_kind::arabic:
                return run_arabic_native(p);
            case tokenizer_kind::mandarin:
#ifdef NEMO_SPEECH_TTS_WITH_ZH
                return tokenize_mandarin(p);
#else
                break;
#endif
            case tokenizer_kind::japanese:
#ifdef NEMO_SPEECH_TTS_WITH_JA
                return run_japanese_native(p);
#else
                break;
#endif
        }
        throw std::invalid_argument(
            "native Magpie tokenizer is not available for language '" + p.language + "'");
    }

#ifdef NEMO_SPEECH_TTS_WITH_ZH
    tokenizer_result tokenize_mandarin(const params& p) const {
        const mandarin_tokenizer& tok = mandarin_tokenizer_for_model();
        return run_mandarin_native(p, tok);
    }

    const mandarin_tokenizer& mandarin_tokenizer_for_model() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (!mandarin_cache_) {
            const auto& entry = profile_.entry_for_language("zh");
            mandarin_cache_ =
                std::make_unique<mandarin_tokenizer>(model_dir_, entry.offset, entry.phoneme_dict);
        }
        return *mandarin_cache_;
    }
#endif

    tokenizer_result tokenize_ipa(const params& p) const {
        if (!fs::is_directory(p.model)) {
            throw std::runtime_error(
                "native IPA tokenization requires an extracted Magpie .nemo directory");
        }
        const ipa_config cfg = ipa_config_for_params(p);
        const ipa_tokenizer& tok = ipa_tokenizer_for(p.language, cfg);
        if (tok.vocab_size() != p.expected_vocab_size) {
            throw std::runtime_error(
                "tokenizer '" + p.tokenizer_name + "' vocabulary has " +
                std::to_string(tok.vocab_size()) + " entries; expected " +
                std::to_string(p.expected_vocab_size));
        }

        tokenizer_result result;
        result.language = p.language;
        result.tokenizer_name = cfg.tokenizer_name;
        result.eos_id = p.eos_id;
        const int pad_id = tok.pad_id();
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
    tokenizer_profile profile_;
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

const std::string&
MagpieNativeTokenizer::profile_id() const {
    return impl_->profile_id();
}

int
MagpieNativeTokenizer::text_vocab_size() const {
    return impl_->text_vocab_size();
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

    static const std::array<std::string, 8> terminals = {".", "?", "!", "？", "！", "。", "।", "؟"};
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
