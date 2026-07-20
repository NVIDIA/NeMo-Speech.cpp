// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Greedy longest-match WordPiece tokenizer for the BERT PnC head.
//
// Input is the ASR transcript (lowercased, no punctuation), so BERT's
// BasicTokenizer reduces to whitespace splitting. Each word is greedily
// segmented into vocab pieces; pieces after the first carry the "##" prefix.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace nemo_speech::asr::pnc {

class WordPieceTokenizer {
   public:
    struct Token {
        int id;
        int word_index;        // source whitespace-word index (for boundary merging)
        bool is_continuation;  // piece begins with "##"
    };

    WordPieceTokenizer(
        std::vector<std::string> vocab, int cls_id, int sep_id, int pad_id, int unk_id,
        int max_chars_per_word = 200);

    // WordPiece tokens for `text` (no [CLS]/[SEP]; the runner adds those).
    std::vector<Token> tokenize(const std::string& text) const;

    int cls_id() const { return cls_id_; }
    int sep_id() const { return sep_id_; }
    int pad_id() const { return pad_id_; }
    int unk_id() const { return unk_id_; }

   private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> idx_;
    int cls_id_, sep_id_, pad_id_, unk_id_, max_chars_;
};

}  // namespace nemo_speech::asr::pnc
