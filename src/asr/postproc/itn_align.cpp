// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "itn_align.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nemo_speech::asr::postproc {
namespace {

// Lowercase + strip trailing punctuation, for matching a token's spoken-text
// words against the ASR timestamp words.
std::string
clean(const std::string& w) {
    std::string c = w;
    while (!c.empty() && !std::isalnum(static_cast<unsigned char>(c.back()))) c.pop_back();
    std::transform(c.begin(), c.end(), c.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return c;
}

bool
probable_abbreviation(const std::string& abbreviation, const std::string& word) {
    if (abbreviation.size() < 2 || word.empty() ||
        abbreviation.size() >= static_cast<size_t>(word.size() * 0.7) ||
        abbreviation.front() != word.front())
        return false;
    size_t pos = 0;
    size_t matches = 0;
    for (char c : abbreviation) {
        pos = word.find(c, pos);
        if (pos == std::string::npos)
            break;
        ++matches;
        ++pos;
    }
    return matches * 5 >= abbreviation.size() * 4;
}

bool
words_match(const std::string& normalized, const std::string& spoken) {
    const std::string norm = clean(normalized);
    const std::string source = clean(spoken);
    return norm == source || (!norm.empty() && source.find(norm) != std::string::npos) ||
           probable_abbreviation(norm, source);
}

bool
starts_with_digit(const std::string& word) {
    return !word.empty() && (std::isdigit(static_cast<unsigned char>(word.front())) ||
                             (word.front() == '-' && word.size() > 1 &&
                              std::isdigit(static_cast<unsigned char>(word[1]))));
}

float
average_confidence(const std::vector<WordTiming>& timings, const std::vector<int>& indices) {
    if (indices.empty())
        return 0.0f;
    float sum = 0.0f;
    for (int index : indices) sum += timings[index].confidence;
    return sum / static_cast<float>(indices.size());
}

std::vector<std::string>
split_ws(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string w;
    while (iss >> w) out.push_back(w);
    return out;
}

// ShowLinks fields are tab-separated; a Token's spoken-text field may itself
// contain spaces, so split on tabs (not whitespace).
std::vector<std::string>
split_tab(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == '\t') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    out.push_back(cur);
    return out;
}

struct Token {
    int id;
    std::string name;  // spoken text this token matched
};

}  // namespace

void
update_word_timings(std::vector<WordTiming>& timings, const std::string& alignment_links) {
    if (timings.empty() || alignment_links.empty())
        return;

    // Parse the ShowLinks output:
    //   Token:\t<id>\t<spoken text>\t<char start,end>\t<first,last daughter>
    //   Word:\t<id>\t<normalized spelling>\t<parent token id>
    std::vector<Token> tokens;
    std::unordered_map<int, std::vector<std::string>> out_by_token;  // token id -> written words
    {
        std::istringstream ss(alignment_links);
        std::string line;
        while (std::getline(ss, line)) {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (line.rfind("Token:", 0) == 0) {
                const auto f = split_tab(line);
                if (f.size() < 3)
                    continue;
                try {
                    tokens.push_back({std::stoi(f[1]), f[2]});
                }
                catch (const std::exception&) {
                }
            } else if (line.rfind("Word:", 0) == 0) {
                const auto f = split_tab(line);
                if (f.size() < 4)
                    continue;
                try {
                    out_by_token[std::stoi(f[3])].push_back(f[2]);
                }
                catch (const std::exception&) {
                }
            }
        }
    }
    if (tokens.empty())
        return;

    std::vector<WordTiming> result;
    int cursor = 0;
    const int n = static_cast<int>(timings.size());

    // Tokens are in input order. Match exact/abbreviated written words first so
    // unchanged units keep their own timestamps, then merge numeric rewrites and
    // distribute any remaining spans. This follows Riva's alignment utility.
    for (const auto& tok : tokens) {
        std::vector<int> spoken;
        for (const auto& nw : split_ws(tok.name)) {
            const std::string c = clean(nw);
            int j = cursor;
            while (j < n && clean(timings[j].word) != c) ++j;
            if (j < n) {
                spoken.push_back(j);
                cursor = j + 1;
            }
        }
        auto it = out_by_token.find(tok.id);
        if (it == out_by_token.end() || it->second.empty() || spoken.empty())
            continue;  // no written words, or no spoken span matched (skip, like riva)

        const auto& outs = it->second;
        std::vector<bool> spoken_used(spoken.size(), false);
        std::vector<bool> output_used(outs.size(), false);

        for (int oi = static_cast<int>(outs.size()) - 1; oi >= 0; --oi) {
            for (int si = static_cast<int>(spoken.size()) - 1; si >= 0; --si) {
                if (spoken_used[si] || !words_match(outs[oi], timings[spoken[si]].word))
                    continue;
                WordTiming exact = timings[spoken[si]];
                exact.word = outs[oi];
                result.push_back(std::move(exact));
                spoken_used[si] = true;
                output_used[oi] = true;
                break;
            }
        }

        std::vector<int> remaining_spoken;
        for (size_t i = 0; i < spoken.size(); ++i)
            if (!spoken_used[i])
                remaining_spoken.push_back(spoken[i]);
        std::vector<int> remaining_output;
        for (size_t i = 0; i < outs.size(); ++i)
            if (!output_used[i])
                remaining_output.push_back(static_cast<int>(i));
        if (remaining_spoken.empty() || remaining_output.empty())
            continue;

        if (remaining_output.size() == 1 && starts_with_digit(outs[remaining_output.front()])) {
            result.push_back(
                {outs[remaining_output.front()], timings[remaining_spoken.front()].start_frame,
                 timings[remaining_spoken.back()].end_frame,
                 average_confidence(timings, remaining_spoken)});
            continue;
        }

        // Evenly partition the remaining spoken words. If a grammar expands one
        // spoken word into several written words, split that word's frame span.
        if (remaining_spoken.size() < remaining_output.size()) {
            const int64_t start = timings[remaining_spoken.front()].start_frame;
            const int64_t end = timings[remaining_spoken.back()].end_frame;
            const int64_t span = end - start;
            const float confidence = average_confidence(timings, remaining_spoken);
            for (size_t i = 0; i < remaining_output.size(); ++i) {
                result.push_back(
                    {outs[remaining_output[i]],
                     start + span * static_cast<int64_t>(i) /
                                 static_cast<int64_t>(remaining_output.size()),
                     start + span * static_cast<int64_t>(i + 1) /
                                 static_cast<int64_t>(remaining_output.size()),
                     confidence});
            }
            continue;
        }

        size_t offset = 0;
        for (size_t i = 0; i < remaining_output.size(); ++i) {
            const size_t base = remaining_spoken.size() / remaining_output.size();
            const size_t extra = remaining_spoken.size() % remaining_output.size();
            const size_t take = base + (i < extra ? 1 : 0);
            std::vector<int> slice(
                remaining_spoken.begin() + offset, remaining_spoken.begin() + offset + take);
            result.push_back(
                {outs[remaining_output[i]], timings[slice.front()].start_frame,
                 timings[slice.back()].end_frame, average_confidence(timings, slice)});
            offset += take;
        }
    }

    if (!result.empty()) {
        std::sort(result.begin(), result.end(), [](const WordTiming& a, const WordTiming& b) {
            return a.start_frame < b.start_frame;
        });
        timings = std::move(result);
    }
}

}  // namespace nemo_speech::asr::postproc
