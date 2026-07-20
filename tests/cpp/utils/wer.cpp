// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "utils/wer.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

namespace nemo_speech::tests {
namespace {

std::vector<std::string>
split_words(const std::string& text) {
    std::istringstream in(text);
    std::vector<std::string> words;
    std::string word;
    while (in >> word) {
        words.push_back(word);
    }
    return words;
}

template <typename T>
size_t
levenshtein_distance(const std::vector<T>& reference, const std::vector<T>& hypothesis) {
    std::vector<size_t> prev(hypothesis.size() + 1);
    std::vector<size_t> curr(hypothesis.size() + 1);

    for (size_t j = 0; j <= hypothesis.size(); ++j) {
        prev[j] = j;
    }
    for (size_t i = 1; i <= reference.size(); ++i) {
        curr[0] = i;
        for (size_t j = 1; j <= hypothesis.size(); ++j) {
            const size_t cost = reference[i - 1] == hypothesis[j - 1] ? 0 : 1;
            curr[j] = std::min({
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            });
        }
        prev.swap(curr);
    }
    return prev[hypothesis.size()];
}

double
error_rate(size_t edits, size_t reference_units, size_t hypothesis_units) {
    if (reference_units == 0) {
        return hypothesis_units == 0 ? 0.0 : 1.0;
    }
    return (double)edits / (double)reference_units;
}

}  // namespace

std::string
lowercase_text(const std::string& text) {
    std::string out = text;
    std::transform(
        out.begin(), out.end(), out.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return out;
}

ErrorRateResult
word_error_rate(const std::string& reference, const std::string& hypothesis) {
    const std::vector<std::string> ref_words = split_words(lowercase_text(reference));
    const std::vector<std::string> hyp_words = split_words(lowercase_text(hypothesis));
    const size_t edits = levenshtein_distance(ref_words, hyp_words);
    return {
        edits,
        ref_words.size(),
        hyp_words.size(),
        error_rate(edits, ref_words.size(), hyp_words.size()),
    };
}

ErrorRateResult
char_error_rate(const std::string& reference, const std::string& hypothesis) {
    const std::string ref = lowercase_text(reference);
    const std::string hyp = lowercase_text(hypothesis);
    const std::vector<char> ref_chars(ref.begin(), ref.end());
    const std::vector<char> hyp_chars(hyp.begin(), hyp.end());
    const size_t edits = levenshtein_distance(ref_chars, hyp_chars);
    return {
        edits,
        ref_chars.size(),
        hyp_chars.size(),
        error_rate(edits, ref_chars.size(), hyp_chars.size()),
    };
}

}  // namespace nemo_speech::tests
