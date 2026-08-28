// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "mandarin_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "cppjieba/MixSegment.hpp"

namespace fs = std::filesystem;

#ifndef NEMO_SPEECH_MANDARIN_DATA_DIR
#define NEMO_SPEECH_MANDARIN_DATA_DIR ""
#endif
#ifndef NEMO_SPEECH_MANDARIN_INSTALLED_DATA_DIR
#define NEMO_SPEECH_MANDARIN_INSTALLED_DATA_DIR ""
#endif

namespace {

struct utf8_char {
    uint32_t codepoint = 0;
    std::string text;
};

std::vector<utf8_char>
decode_utf8(const std::string& text) {
    std::vector<utf8_char> result;
    for (size_t i = 0; i < text.size();) {
        const unsigned char first = static_cast<unsigned char>(text[i]);
        size_t length = 1;
        uint32_t codepoint = first;
        if ((first & 0xe0) == 0xc0 && i + 1 < text.size()) {
            length = 2;
            codepoint = first & 0x1f;
        } else if ((first & 0xf0) == 0xe0 && i + 2 < text.size()) {
            length = 3;
            codepoint = first & 0x0f;
        } else if ((first & 0xf8) == 0xf0 && i + 3 < text.size()) {
            length = 4;
            codepoint = first & 0x07;
        }
        bool valid = length > 1;
        for (size_t j = 1; j < length && valid; ++j) {
            const unsigned char continuation = static_cast<unsigned char>(text[i + j]);
            valid = (continuation & 0xc0) == 0x80;
            codepoint = (codepoint << 6) | (continuation & 0x3f);
        }
        if (!valid && length > 1) {
            length = 1;
            codepoint = first;
        }
        result.push_back({codepoint, text.substr(i, length)});
        i += length;
    }
    return result;
}

bool
is_han(uint32_t value) {
    return value == 0x3007 || (value >= 0xe815 && value <= 0xe864) || value == 0xfa18 ||
           (value >= 0x3400 && value <= 0x4dbf) || (value >= 0x4e00 && value <= 0x9fff) ||
           (value >= 0xf900 && value <= 0xfaff) || (value >= 0x20000 && value <= 0x2a6df) ||
           (value >= 0x2a703 && value <= 0x2b73f) || (value >= 0x2b740 && value <= 0x2b81d) ||
           (value >= 0x2b825 && value <= 0x2bf6e) || (value >= 0x2c029 && value <= 0x2ce93) ||
           value == 0x2d016 || (value >= 0x2d11b && value <= 0x2ebd9) ||
           (value >= 0x2f80a && value <= 0x2fa1f) || (value >= 0x30000 && value <= 0x3134a) ||
           (value >= 0x300f7 && value <= 0x31288) || value == 0x30edd || value == 0x30ede ||
           (value >= 0x31350 && value <= 0x32389);
}

std::vector<std::string>
split_spaces(const std::string& text) {
    std::vector<std::string> result;
    std::istringstream input(text);
    std::string value;
    while (input >> value) {
        result.push_back(value);
    }
    return result;
}

bool
is_data_dir(const fs::path& path) {
    return fs::is_regular_file(path / "jieba.dict.utf8") &&
           fs::is_regular_file(path / "hmm_model.utf8") &&
           fs::is_regular_file(path / "pinyin_chars.tsv") &&
           fs::is_regular_file(path / "pinyin_phrases.tsv");
}

fs::path
find_data_dir(const fs::path& model_dir) {
    std::vector<fs::path> candidates;
    if (const char* configured = std::getenv("MAGPIE_MANDARIN_G2P_DIR")) {
        if (*configured != '\0') {
            candidates.emplace_back(configured);
        }
    }
    candidates.push_back(model_dir / "mandarin_g2p");
    candidates.push_back(model_dir / "mandarin_data");
    if (std::string(NEMO_SPEECH_MANDARIN_DATA_DIR).empty() == false) {
        candidates.emplace_back(NEMO_SPEECH_MANDARIN_DATA_DIR);
    }
    if (std::string(NEMO_SPEECH_MANDARIN_INSTALLED_DATA_DIR).empty() == false) {
        candidates.emplace_back(NEMO_SPEECH_MANDARIN_INSTALLED_DATA_DIR);
    }
    for (const auto& candidate : candidates) {
        if (is_data_dir(candidate)) {
            return candidate;
        }
    }
    return {};
}

fs::path
find_phoneme_dict(const fs::path& model_dir) {
    if (!fs::is_directory(model_dir)) {
        return {};
    }
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.find("ipa_dict") != std::string::npos &&
            name.find("cmudict") == std::string::npos) {
            return entry.path();
        }
    }
    return {};
}

std::string
ascii_upper(std::string text) {
    for (char& value : text) {
        const unsigned char byte = static_cast<unsigned char>(value);
        if (byte >= 'a' && byte <= 'z') {
            value = static_cast<char>(byte - 'a' + 'A');
        }
    }
    return text;
}

}  // namespace

class mandarin_tokenizer::impl {
   public:
    explicit impl(const fs::path& model_dir, int offset, const std::string& phoneme_dict)
        : offset_(offset) {
        data_dir_ = find_data_dir(model_dir);
        if (data_dir_.empty()) {
            throw std::runtime_error(
                "failed to find Mandarin G2P data; set MAGPIE_MANDARIN_G2P_DIR");
        }
        const bool configured_phoneme_dict = !phoneme_dict.empty();
        const fs::path phoneme_path =
            configured_phoneme_dict ? model_dir / phoneme_dict : find_phoneme_dict(model_dir);
        if (!fs::is_regular_file(phoneme_path)) {
            throw std::runtime_error(
                "failed to find Mandarin pinyin-to-phoneme dictionary at '" +
                (phoneme_path.empty() ? std::string("<not found>") : phoneme_path.string()) +
                "' (" +
                (configured_phoneme_dict ? "configured phoneme_dict" : "automatic discovery") +
                ")");
        }

        segmenter_ = std::make_unique<cppjieba::MixSegment>(
            (data_dir_ / "jieba.dict.utf8").string(), (data_dir_ / "hmm_model.utf8").string());
        load_chars(data_dir_ / "pinyin_chars.tsv");
        load_phrases(data_dir_ / "pinyin_phrases.tsv");
        load_phonemes(phoneme_path);
        build_vocabulary();
    }

    std::vector<int> encode(const std::string& raw_text) const {
        const std::string text = ascii_upper(raw_text);
        std::vector<std::string> words;
        segmenter_->Cut(text, words, true);

        std::vector<std::string> pinyin;
        for (const auto& word : words) {
            append_word_pinyin(word, pinyin);
        }

        std::vector<std::string> symbols;
        for (const auto& value : pinyin) {
            if (!value.empty() && value.back() >= '1' && value.back() <= '5') {
                const std::string syllable = value.substr(0, value.size() - 1);
                const auto found = phonemes_.find(syllable);
                if (found == phonemes_.end()) {
                    continue;
                }
                symbols.insert(symbols.end(), found->second.begin(), found->second.end());
                symbols.push_back("#" + value.substr(value.size() - 1));
            } else if (value.size() == 1 && value[0] >= 'A' && value[0] <= 'Z') {
                symbols.push_back(value);
            } else {
                symbols.push_back(value);
            }
        }

        std::vector<std::string> kept;
        for (const auto& symbol : symbols) {
            if (symbol == " ") {
                if (!kept.empty() && kept.back() != " ") {
                    kept.push_back(symbol);
                }
            } else if (token_to_id_.count(symbol) != 0) {
                kept.push_back(symbol);
            }
        }
        while (!kept.empty() && kept.back() == " ") {
            kept.pop_back();
        }
        kept.insert(kept.begin(), " ");
        kept.push_back(" ");

        std::vector<int> result;
        result.reserve(kept.size());
        for (const auto& symbol : kept) {
            result.push_back(token_to_id_.at(symbol));
        }
        return result;
    }

    int pad_id() const { return token_to_id_.at("<pad>"); }

   private:
    fs::path data_dir_;
    std::unique_ptr<cppjieba::MixSegment> segmenter_;
    std::unordered_map<uint32_t, std::string> character_pinyin_;
    std::unordered_map<std::string, std::vector<std::string>> phrase_pinyin_;
    size_t max_phrase_chars_ = 0;
    std::unordered_map<std::string, std::vector<std::string>> phonemes_;
    std::unordered_map<std::string, int> token_to_id_;

    void load_chars(const fs::path& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("failed to read " + path.string());
        }
        std::string line;
        while (std::getline(input, line)) {
            const size_t tab = line.find('\t');
            if (tab == std::string::npos) {
                continue;
            }
            const uint32_t codepoint =
                static_cast<uint32_t>(std::stoul(line.substr(0, tab), nullptr, 16));
            character_pinyin_[codepoint] = line.substr(tab + 1);
        }
    }

    void load_phrases(const fs::path& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("failed to read " + path.string());
        }
        std::string line;
        while (std::getline(input, line)) {
            const size_t tab = line.find('\t');
            if (tab == std::string::npos) {
                continue;
            }
            const std::string phrase = line.substr(0, tab);
            phrase_pinyin_[phrase] = split_spaces(line.substr(tab + 1));
            max_phrase_chars_ = std::max(max_phrase_chars_, decode_utf8(phrase).size());
        }
    }

    void load_phonemes(const fs::path& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("failed to read " + path.string());
        }
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line.rfind(";;;", 0) == 0) {
                continue;
            }
            const size_t tab = line.find('\t');
            if (tab == std::string::npos) {
                continue;
            }
            std::string syllable = line.substr(0, tab);
            std::transform(syllable.begin(), syllable.end(), syllable.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            std::string pronunciation = line.substr(tab + 1);
            std::transform(
                pronunciation.begin(), pronunciation.end(), pronunciation.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            phonemes_[syllable] = split_spaces(pronunciation);
        }
    }

    void build_vocabulary() {
        std::unordered_set<std::string> unique_phonemes;
        for (const auto& [syllable, pronunciation] : phonemes_) {
            (void)syllable;
            unique_phonemes.insert(pronunciation.begin(), pronunciation.end());
        }
        std::vector<std::string> tokens = {" "};
        std::vector<std::string> sorted_phonemes(unique_phonemes.begin(), unique_phonemes.end());
        std::sort(sorted_phonemes.begin(), sorted_phonemes.end());
        tokens.insert(tokens.end(), sorted_phonemes.begin(), sorted_phonemes.end());
        for (int tone = 1; tone <= 5; ++tone) {
            tokens.push_back("#" + std::to_string(tone));
        }
        for (char letter = 'A'; letter <= 'Z'; ++letter) {
            tokens.emplace_back(1, letter);
        }
        tokens.push_back("'");
        for (const auto& punctuation : decode_utf8("，。？！；：、‘’“”（）【】「」《》")) {
            tokens.push_back(punctuation.text);
        }
        for (const auto& punctuation : decode_utf8(",.!?-:;/\"()[]{}")) {
            tokens.push_back(punctuation.text);
        }
        tokens.push_back("<pad>");
        tokens.push_back("<oov>");
        if (tokens.size() != 109) {
            throw std::runtime_error(
                "Mandarin vocabulary does not match the Magpie model (expected 109 tokens)");
        }
        for (size_t index = 0; index < tokens.size(); ++index) {
            token_to_id_[tokens[index]] = offset_ + static_cast<int>(index);
        }
    }

    int offset_ = 349;

    void append_word_pinyin(const std::string& word, std::vector<std::string>& output) const {
        const auto chars = decode_utf8(word);
        for (size_t begin = 0; begin < chars.size();) {
            const bool han = is_han(chars[begin].codepoint);
            size_t end = begin + 1;
            while (end < chars.size() && is_han(chars[end].codepoint) == han) {
                ++end;
            }
            if (han) {
                append_han_pinyin(chars, begin, end, output);
            } else {
                for (size_t index = begin; index < end; ++index) {
                    output.push_back(chars[index].text);
                }
            }
            begin = end;
        }
    }

    void append_han_pinyin(
        const std::vector<utf8_char>& chars, size_t begin, size_t end,
        std::vector<std::string>& output) const {
        for (size_t position = begin; position < end;) {
            size_t matched_length = 0;
            const std::vector<std::string>* matched = nullptr;
            std::string candidate;
            const size_t limit = std::min(end, position + max_phrase_chars_);
            for (size_t cursor = position; cursor < limit; ++cursor) {
                candidate += chars[cursor].text;
                const auto found = phrase_pinyin_.find(candidate);
                if (found != phrase_pinyin_.end()) {
                    matched_length = cursor - position + 1;
                    matched = &found->second;
                }
            }
            if (matched) {
                output.insert(output.end(), matched->begin(), matched->end());
                position += matched_length;
                continue;
            }
            const auto found = character_pinyin_.find(chars[position].codepoint);
            output.push_back(
                found == character_pinyin_.end() ? chars[position].text : found->second);
            ++position;
        }
    }
};

mandarin_tokenizer::mandarin_tokenizer(
    const fs::path& model_dir, int offset, std::string phoneme_dict)
    : impl_(std::make_unique<impl>(model_dir, offset, phoneme_dict)) {}

mandarin_tokenizer::~mandarin_tokenizer() = default;

std::vector<int>
mandarin_tokenizer::encode(const std::string& text) const {
    return impl_->encode(text);
}

int
mandarin_tokenizer::pad_id() const {
    return impl_->pad_id();
}

bool
mandarin_tokenizer_available(const fs::path& model_dir) {
    return !find_data_dir(model_dir).empty() && !find_phoneme_dict(model_dir).empty();
}
