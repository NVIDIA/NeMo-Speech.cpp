// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Drives PnC over a transcript string: WordPiece-tokenize, run the BERT model
// over sliding 128-token windows, merge per-token punct/capit labels, and
// reconstruct the capitalized + punctuated text.
#pragma once

#include <string>

#include "pnc_model.h"
#include "wordpiece.h"

namespace nemo_speech::asr::pnc {

class PncRunner {
   public:
    explicit PncRunner(PncModel* model);

    // Lowercased, unpunctuated input -> capitalized + punctuated text.
    std::string postprocess(const std::string& text) const;

   private:
    PncModel* model_;
    WordPieceTokenizer tok_;
    int capit_upper_;  // index of the "U" capit label
};

}  // namespace nemo_speech::asr::pnc
