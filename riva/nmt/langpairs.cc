// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "langpairs.h"

#include <algorithm>
#include <cctype>
#include <set>
#include <unordered_map>

namespace nemo_speech::nmt::langpairs {

namespace {

// Tag -> target-language display name, from the model's chat template.
const std::unordered_map<std::string, std::string>&
table() {
    static const std::unordered_map<std::string, std::string> t = {
        {"en-zh-cn", "Simplified Chinese"},
        {"en-zh", "Simplified Chinese"},
        {"en-zh-tw", "Traditional Chinese"},
        {"en-ar", "Arabic"},
        {"en-de", "German"},
        {"en-es", "European Spanish"},
        {"en-es-es", "European Spanish"},
        {"en-es-us", "Latin American Spanish"},
        {"en-fr", "French"},
        {"en-ja", "Japanese"},
        {"en-ko", "Korean"},
        {"en-ru", "Russian"},
        {"en-pt", "Brazilian Portuguese"},
        {"en-pt-br", "Brazilian Portuguese"},
        {"en-pt-pt", "European Portuguese"},
        {"zh-en", "English"},
        {"zh-cn-en", "English"},
        {"zh-tw-en", "English"},
        {"ar-en", "English"},
        {"de-en", "English"},
        {"es-en", "English"},
        {"es-es-en", "English"},
        {"es-us-en", "English"},
        {"fr-en", "English"},
        {"ja-en", "English"},
        {"ko-en", "English"},
        {"ru-en", "English"},
        {"pt-en", "English"},
        {"pt-br-en", "English"},
        {"en-it", "Italian"},
        {"it-en", "English"},
        {"en-nl", "Dutch"},
        {"nl-en", "English"},
        {"en-pl", "Polish"},
        {"pl-en", "English"},
        {"en-cs", "Czech"},
        {"cs-en", "English"},
        {"en-sv", "Swedish"},
        {"sv-en", "English"},
        {"en-da", "Danish"},
        {"da-en", "English"},
        {"en-fi", "Finnish"},
        {"fi-en", "English"},
        {"en-no", "Norwegian"},
        {"no-en", "English"},
        {"en-hu", "Hungarian"},
        {"hu-en", "English"},
        {"en-ro", "Romanian"},
        {"ro-en", "English"},
        {"en-bg", "Bulgarian"},
        {"bg-en", "English"},
        {"en-uk", "Ukrainian"},
        {"uk-en", "English"},
        {"en-sk", "Slovak"},
        {"sk-en", "English"},
        {"en-hr", "Croatian"},
        {"hr-en", "English"},
        {"en-sl", "Slovenian"},
        {"sl-en", "English"},
        {"en-et", "Estonian"},
        {"et-en", "English"},
        {"en-lv", "Latvian"},
        {"lv-en", "English"},
        {"en-lt", "Lithuanian"},
        {"lt-en", "English"},
        {"en-el", "Greek"},
        {"el-en", "English"},
        {"en-tr", "Turkish"},
        {"tr-en", "English"},
        {"en-id", "Indonesian"},
        {"id-en", "English"},
        {"en-vi", "Vietnamese"},
        {"vi-en", "English"},
        {"en-th", "Thai"},
        {"th-en", "English"},
        {"en-hi", "Hindi"},
        {"hi-en", "English"},
    };
    return t;
}

}  // namespace

std::string
normalize_language_code(std::string code) {
    std::transform(code.begin(), code.end(), code.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (code.empty() || is_supported(code))
        return code;
    static const std::set<std::string> codes = [] {
        std::set<std::string> result;
        for (const auto& pair : supported_pairs()) {
            result.insert(pair.first);
            result.insert(pair.second);
        }
        return result;
    }();
    if (codes.count(code) != 0)
        return code;
    const size_t dash = code.find('-');
    if (dash != std::string::npos) {
        const std::string base = code.substr(0, dash);
        if (codes.count(base) != 0)
            return base;
    }
    return code;
}

// Split a tag into (source_code, target_code). Every pair has English on one
// side, so the other side is the remainder after the "en-" prefix or "-en"
// suffix.
std::pair<std::string, std::string>
split_tag(const std::string& tag) {
    if (tag.rfind("en-", 0) == 0)
        return {"en", tag.substr(3)};
    if (tag.size() > 3 && tag.compare(tag.size() - 3, 3, "-en") == 0)
        return {tag.substr(0, tag.size() - 3), "en"};
    return {tag, ""};
}

bool
is_supported(const std::string& tag) {
    return table().count(tag) != 0;
}

std::string
resolve_tag(const std::string& source_language, const std::string& target_language) {
    const std::string source = normalize_language_code(source_language);
    const std::string target = normalize_language_code(target_language);
    const bool have_src = !source.empty();
    const bool have_tgt = !target.empty();

    // Both codes given: the unambiguous reading is the combined "<src>-<tgt>"
    // pair, so that takes priority.
    if (have_src && have_tgt) {
        const std::string pair = source + "-" + target;
        if (is_supported(pair))
            return pair;
        // Tolerate a ready tag (e.g. "en-de") in one field only when the other
        // field is a consistent split of it; otherwise the two fields disagree,
        // so reject rather than silently honor one and ignore the other.
        if (is_supported(source) && split_tag(source).second == target)
            return source;
        if (is_supported(target) && split_tag(target).first == source)
            return target;
        return "";
    }

    // Exactly one field set: accept a ready tag passed in it.
    if (have_tgt && is_supported(target))
        return target;
    if (have_src && is_supported(source))
        return source;
    return "";
}

std::string
build_prompt(const std::string& tag, const std::string& text) {
    const auto it = table().find(tag);
    if (it == table().end())
        return "";
    const auto pair = split_tag(tag);
    const auto source =
        pair.first == "en" ? std::string("English") : table().at("en-" + pair.first);
    return "<s>System\nYou are an expert at translating text from " + source + " to " + it->second +
           ".</s>\n<s>User\nWhat is the " + it->second + " translation of the sentence: " + text +
           "</s>\n<s>Assistant\n";
}

const std::vector<std::pair<std::string, std::string>>&
supported_pairs() {
    static const std::vector<std::pair<std::string, std::string>> pairs = [] {
        std::vector<std::pair<std::string, std::string>> v;
        v.reserve(table().size());
        for (const auto& [tag, _] : table()) v.push_back(split_tag(tag));
        return v;
    }();
    return pairs;
}

}  // namespace nemo_speech::nmt::langpairs
