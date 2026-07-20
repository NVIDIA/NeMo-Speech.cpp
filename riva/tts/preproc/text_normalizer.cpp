// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "text_normalizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#ifdef NEMO_SPEECH_WITH_NORM
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <string_view>
#include <system_error>
#include <vector>

#include "fst_normalizer.h"
#endif

namespace nemo_speech::tts::preproc {

namespace {

bool
decode_utf8_codepoint(const std::string& text, size_t& offset, uint32_t& codepoint) {
    const unsigned char first = static_cast<unsigned char>(text[offset]);
    if (first < 0x80) {
        codepoint = first;
        ++offset;
        return true;
    }

    size_t length = 0;
    uint32_t minimum = 0;
    if ((first & 0xe0) == 0xc0) {
        length = 2;
        minimum = 0x80;
        codepoint = first & 0x1f;
    } else if ((first & 0xf0) == 0xe0) {
        length = 3;
        minimum = 0x800;
        codepoint = first & 0x0f;
    } else if ((first & 0xf8) == 0xf0) {
        length = 4;
        minimum = 0x10000;
        codepoint = first & 0x07;
    } else {
        ++offset;
        return false;
    }

    if (offset + length > text.size()) {
        ++offset;
        return false;
    }
    for (size_t index = 1; index < length; ++index) {
        const unsigned char continuation = static_cast<unsigned char>(text[offset + index]);
        if ((continuation & 0xc0) != 0x80) {
            ++offset;
            return false;
        }
        codepoint = (codepoint << 6) | (continuation & 0x3f);
    }
    offset += length;
    return codepoint >= minimum && codepoint <= 0x10ffff &&
           !(codepoint >= 0xd800 && codepoint <= 0xdfff);
}

bool
is_native_numeral(uint32_t codepoint) {
    if ((codepoint >= 0x0966 && codepoint <= 0x096f) ||
        (codepoint >= 0xff10 && codepoint <= 0xff19)) {
        return true;
    }

    switch (codepoint) {
        // Common Chinese and Japanese numerals, including financial/formal variants.
        case 0x3007:  // 〇
        case 0x96f6:  // 零
        case 0x4e00:  // 一
        case 0x4e8c:  // 二
        case 0x4e09:  // 三
        case 0x56db:  // 四
        case 0x4e94:  // 五
        case 0x516d:  // 六
        case 0x4e03:  // 七
        case 0x516b:  // 八
        case 0x4e5d:  // 九
        case 0x5341:  // 十
        case 0x767e:  // 百
        case 0x5343:  // 千
        case 0x4e07:  // 万
        case 0x842c:  // 萬
        case 0x4ebf:  // 亿
        case 0x5104:  // 億
        case 0x5146:  // 兆
        case 0x58f9:  // 壹
        case 0x58f1:  // 壱
        case 0x8d30:  // 贰
        case 0x8cb3:  // 貳
        case 0x5f10:  // 弐
        case 0x53c1:  // 叁
        case 0x53c3:  // 參
        case 0x53c2:  // 参
        case 0x8086:  // 肆
        case 0x4f0d:  // 伍
        case 0x9646:  // 陆
        case 0x9678:  // 陸
        case 0x67d2:  // 柒
        case 0x634c:  // 捌
        case 0x7396:  // 玖
        case 0x62fe:  // 拾
        case 0x4f70:  // 佰
        case 0x4edf:  // 仟
        case 0x5eff:  // 廿
        case 0x5345:  // 卅
        case 0x534c:  // 卌
            return true;
        default:
            return false;
    }
}

bool
has_alnum(const std::string& text) {
    for (size_t offset = 0; offset < text.size();) {
        const unsigned char byte = static_cast<unsigned char>(text[offset]);
        if (byte < 0x80 && std::isalnum(byte)) {
            return true;
        }
        uint32_t codepoint = 0;
        if (decode_utf8_codepoint(text, offset, codepoint) && is_native_numeral(codepoint)) {
            return true;
        }
    }
    return false;
}

#ifdef NEMO_SPEECH_WITH_NORM
std::string
trim_ascii(std::string value) {
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
normalize_language_code(std::string language_code) {
    std::string language = trim_ascii(std::move(language_code));
    if (language.empty()) {
        return "en";
    }
    std::replace(language.begin(), language.end(), '_', '-');
    const size_t dash = language.find('-');
    if (dash != std::string::npos) {
        language.resize(dash);
    }
    std::transform(language.begin(), language.end(), language.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return language;
}
#endif

}  // namespace

#ifdef NEMO_SPEECH_WITH_NORM
namespace {

std::string
resolve_abs_dir(const std::string& model_dir) {
    char abs[PATH_MAX];
    if (!realpath(model_dir.c_str(), abs)) {
        throw std::runtime_error("TN: cannot resolve model dir: " + model_dir);
    }
    return abs;
}

std::string
lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool
contains(std::string_view value, std::string_view needle) {
    return value.find(needle) != std::string_view::npos;
}

bool
exists_path(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

bool
is_tn_grammar_dir(const std::filesystem::path& model_dir) {
    const bool flat = exists_path(model_dir / "tokenize_and_classify.far") &&
                      exists_path(model_dir / "verbalize.far");
    const bool split = exists_path(model_dir / "classify" / "tokenize_and_classify.far") &&
                       exists_path(model_dir / "verbalize" / "verbalize.far");
    return flat || split;
}

std::string
language_from_tn_dir(const std::filesystem::path& path) {
    const std::string name = lower_ascii(path.filename().string());
    const size_t tn = name.find("_tn_");
    if (tn == std::string::npos || tn == 0) {
        return {};
    }
    return normalize_language_code(name.substr(0, tn));
}

std::filesystem::path
find_required(
    const std::filesystem::path& model_dir, const std::vector<std::string>& candidates,
    const std::string& label) {
    for (const auto& rel : candidates) {
        std::error_code ec;
        std::filesystem::path path = model_dir / rel;
        if (std::filesystem::exists(path, ec)) {
            return std::filesystem::canonical(path);
        }
    }
    throw std::runtime_error("TN: missing " + label + " FAR under " + model_dir.string());
}

std::filesystem::path
find_optional(const std::filesystem::path& model_dir, const std::vector<std::string>& candidates) {
    for (const auto& rel : candidates) {
        std::error_code ec;
        std::filesystem::path path = model_dir / rel;
        if (std::filesystem::exists(path, ec)) {
            return std::filesystem::canonical(path);
        }
    }
    return {};
}

std::filesystem::path
find_tokenizer(const std::filesystem::path& model_dir) {
    const auto split = find_optional(
        model_dir, {"tokenize_and_classify.far", "classify/tokenize_and_classify.far",
                    "en_tn_True_deterministic_cased__tokenize.far"});
    if (!split.empty()) {
        return split;
    }

    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(model_dir, ec)) {
        if (ec) {
            break;
        }
        const auto path = entry.path();
        if (path.extension() == ".far" && contains(path.filename().string(), "tokenize")) {
            return std::filesystem::canonical(path);
        }
    }
    throw std::runtime_error("TN: missing tokenizer FAR under " + model_dir.string());
}

std::filesystem::path
make_temp_dir() {
    std::string templ = (std::filesystem::temp_directory_path() / "nemo-speech-tn-XXXXXX").string();
    std::vector<char> buf(templ.begin(), templ.end());
    buf.push_back('\0');
    char* created = mkdtemp(buf.data());
    if (!created) {
        throw std::runtime_error("TN: failed to create temporary grammar directory");
    }
    return created;
}

void
stage_file(
    const std::filesystem::path& src, const std::filesystem::path& dst, const std::string& label) {
    std::error_code ec;
    std::filesystem::create_symlink(src, dst, ec);
    if (ec) {
        ec.clear();
        std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing, ec);
    }
    if (ec) {
        throw std::runtime_error(
            "TN: failed to stage " + label + " FAR from " + src.string() + " to " + dst.string() +
            ": " + ec.message());
    }
}

}  // namespace

struct TextNormalizer::Impl {
    struct LanguageNormalizer {
        std::unique_ptr<text_normalization::FstNormalizer> normalizer;
        std::filesystem::path stage_dir;

        explicit LanguageNormalizer(const std::filesystem::path& source_dir) {
            const std::filesystem::path tokenizer = find_tokenizer(source_dir);
            const std::filesystem::path verbalizer = find_required(
                source_dir, {"verbalize.far", "verbalize/verbalize.far"}, "verbalizer");
            const std::filesystem::path post_processor =
                find_optional(source_dir, {"post_process.far", "verbalize/post_process.far"});

            stage_dir = make_temp_dir();
            try {
                stage_file(tokenizer, stage_dir / "tokenize_and_classify.far", "tokenizer");
                stage_file(verbalizer, stage_dir / "verbalize.far", "verbalizer");
                if (!post_processor.empty()) {
                    stage_file(post_processor, stage_dir / "post_process.far", "postprocessor");
                }

                normalizer =
                    std::make_unique<text_normalization::FstNormalizer>(stage_dir.string());
            }
            catch (...) {
                cleanup();
                throw;
            }
        }

        ~LanguageNormalizer() { cleanup(); }

        void cleanup() {
            normalizer.reset();
            if (!stage_dir.empty()) {
                std::error_code ec;
                std::filesystem::remove_all(stage_dir, ec);
                stage_dir.clear();
            }
        }
    };

    std::map<std::string, std::unique_ptr<LanguageNormalizer>> normalizers;
    std::string default_language = "en";
    mutable std::mutex warnings_mu;
    mutable std::set<std::string> missing_language_warnings;

    explicit Impl(const std::string& model_dir) {
        const std::filesystem::path source_dir(resolve_abs_dir(model_dir));
        std::vector<std::pair<std::string, std::filesystem::path>> candidates;

        if (is_tn_grammar_dir(source_dir)) {
            std::string language = language_from_tn_dir(source_dir);
            if (language.empty()) {
                language = default_language;
            }
            candidates.emplace_back(std::move(language), source_dir);
        } else {
            std::error_code ec;
            for (const auto& entry : std::filesystem::directory_iterator(source_dir, ec)) {
                if (ec || !entry.is_directory()) {
                    continue;
                }
                const std::filesystem::path path = entry.path();
                if (!is_tn_grammar_dir(path)) {
                    continue;
                }
                std::string language = language_from_tn_dir(path);
                if (language.empty()) {
                    language = normalize_language_code(path.filename().string());
                }
                if (!language.empty()) {
                    candidates.emplace_back(language, std::filesystem::canonical(path));
                }
            }
        }

        if (candidates.empty()) {
            throw std::runtime_error(
                "TN: no TTS TN grammar directories found under " + source_dir.string());
        }

        std::sort(candidates.begin(), candidates.end());
        for (const auto& [language, path] : candidates) {
            if (normalizers.find(language) != normalizers.end()) {
                throw std::runtime_error(
                    "TN: duplicate normalized language directory '" + language + "' under " +
                    source_dir.string());
            }
            normalizers.emplace(language, std::make_unique<LanguageNormalizer>(path));
        }
        if (normalizers.find(default_language) == normalizers.end()) {
            default_language = normalizers.begin()->first;
        }
    }

    const LanguageNormalizer* find(const std::string& language_code) const {
        const std::string language = normalize_language_code(language_code);
        auto it = normalizers.find(language);
        if (it != normalizers.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    std::vector<std::string> languages() const {
        std::vector<std::string> out;
        out.reserve(normalizers.size());
        for (const auto& [language, _] : normalizers) {
            out.push_back(language);
        }
        return out;
    }

    void warn_missing_language_once(const std::string& language_code) const {
        const std::string language = normalize_language_code(language_code);
        std::lock_guard<std::mutex> lock(warnings_mu);
        if (missing_language_warnings.insert(language).second) {
            std::cerr << "[tn] WARNING: no TN grammar loaded for language '" << language
                      << "'; returning text unchanged\n";
        }
    }
};
#else
struct TextNormalizer::Impl {};
#endif

TextNormalizer::TextNormalizer(const std::string& model_dir) {
    if (model_dir.empty()) {
        return;
    }
#ifdef NEMO_SPEECH_WITH_NORM
    impl_ = std::make_unique<Impl>(model_dir);
    enabled_ = true;
#else
    std::cerr << "[tn] WARNING: --tts.tn-model-dir set (" << model_dir
              << ") but this build has no Sparrowhawk support "
                 "(configure with -DNEMO_SPEECH_WITH_NORM=ON). TN disabled.\n";
#endif
}

TextNormalizer::~TextNormalizer() = default;

std::string
TextNormalizer::normalize(const std::string& text) const {
#ifdef NEMO_SPEECH_WITH_NORM
    return normalize(text, enabled_ && impl_ ? impl_->default_language : "en");
#else
    return normalize(text, "en");
#endif
}

std::string
TextNormalizer::normalize(const std::string& text, const std::string& language_code) const {
    if (!enabled_ || !has_alnum(text)) {
        return text;
    }
#ifdef NEMO_SPEECH_WITH_NORM
    const Impl::LanguageNormalizer* normalizer = impl_->find(language_code);
    if (!normalizer) {
        impl_->warn_missing_language_once(language_code);
        return text;
    }

    return normalizer->normalizer->normalize(text);
#else
    (void)language_code;
#endif
    return text;
}

std::vector<std::string>
TextNormalizer::languages() const {
#ifdef NEMO_SPEECH_WITH_NORM
    if (enabled_ && impl_) {
        return impl_->languages();
    }
#endif
    return {};
}

}  // namespace nemo_speech::tts::preproc
