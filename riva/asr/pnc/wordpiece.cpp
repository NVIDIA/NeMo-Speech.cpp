// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "wordpiece.h"

#include <sstream>
#include <utility>

namespace nemo_speech::asr::pnc {

WordPieceTokenizer::WordPieceTokenizer(
    std::vector<std::string> vocab, int cls_id, int sep_id, int pad_id, int unk_id,
    int max_chars_per_word)
    : vocab_(std::move(vocab)), cls_id_(cls_id), sep_id_(sep_id), pad_id_(pad_id), unk_id_(unk_id),
      max_chars_(max_chars_per_word) {
    idx_.reserve(vocab_.size() * 2);
    for (int i = 0; i < static_cast<int>(vocab_.size()); ++i) idx_.emplace(vocab_[i], i);
}

std::vector<WordPieceTokenizer::Token>
WordPieceTokenizer::tokenize(const std::string& text) const {
    std::vector<Token> out;
    std::istringstream iss(text);
    std::string word;
    int wi = 0;
    while (iss >> word) {
        if (static_cast<int>(word.size()) > max_chars_) {
            out.push_back({unk_id_, wi, false});
            ++wi;
            continue;
        }
        // Greedy longest-match from the front; continuation pieces get "##".
        std::vector<Token> pieces;
        size_t start = 0;
        bool bad = false;
        while (start < word.size()) {
            size_t end = word.size();
            int found = -1;
            while (start < end) {
                std::string sub = (start > 0 ? "##" : "") + word.substr(start, end - start);
                auto it = idx_.find(sub);
                if (it != idx_.end()) {
                    found = it->second;
                    break;
                }
                --end;
            }
            if (found < 0) {
                bad = true;
                break;
            }
            pieces.push_back({found, wi, start > 0});
            start = end;
        }
        if (bad)
            out.push_back({unk_id_, wi, false});
        else
            out.insert(out.end(), pieces.begin(), pieces.end());
        ++wi;
    }
    return out;
}

}  // namespace nemo_speech::asr::pnc
