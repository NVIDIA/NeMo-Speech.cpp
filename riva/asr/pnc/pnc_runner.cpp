// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "pnc_runner.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <vector>

namespace nemo_speech::asr::pnc {
namespace {

std::vector<std::string>
split_ws(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string w;
    while (iss >> w) out.push_back(w);
    return out;
}

}  // namespace

PncRunner::PncRunner(PncModel* model)
    : model_(model), tok_(
                         model->config().vocab, model->config().cls_id, model->config().sep_id,
                         model->config().pad_id, model->config().unk_id),
      capit_upper_(-1) {
    const auto& caps = model_->config().capit_labels;
    for (int i = 0; i < static_cast<int>(caps.size()); ++i)
        if (caps[i] == "U")
            capit_upper_ = i;
}

std::string
PncRunner::postprocess(const std::string& text) const {
    const std::vector<std::string> words = split_ws(text);
    if (words.empty())
        return text;
    const std::vector<WordPieceTokenizer::Token> tokens = tok_.tokenize(text);
    if (tokens.empty())
        return text;

    const int T = static_cast<int>(tokens.size());
    const int Wc = std::max(1, model_->config().max_seq_length - 2);  // reserve [CLS]/[SEP]
    const int margin = (Wc > 4 * 32) ? 32 : 0;
    const int stride = std::max(1, Wc - 2 * margin);

    // Per-token argmax label index, merged across overlapping windows.
    std::vector<int> punct(T, 0), capit(T, 0);
    for (int s = 0; s < T; s += stride) {
        const int lo = s;
        const int hi = std::min(s + Wc, T);

        std::vector<int32_t> ids;
        ids.reserve(hi - lo + 2);
        ids.push_back(model_->config().cls_id);
        for (int j = lo; j < hi; ++j) ids.push_back(tokens[j].id);
        ids.push_back(model_->config().sep_id);

        std::vector<int> wp, wc;
        model_->infer(ids.data(), static_cast<int>(ids.size()), wp, wc);

        // Drop [CLS] (index 0) and [SEP] (last); keep non-margin content.
        const int keep_lo = (lo == 0) ? 0 : margin;
        int keep_hi = (hi == T) ? (hi - lo) : (Wc - margin);
        // Clamp to the real window length and to the label vector sizes so
        // that (1 + j) never reads past wp/wc (final window may be < Wc).
        keep_hi = std::min(keep_hi, hi - lo);
        keep_hi = std::min(keep_hi, static_cast<int>(std::min(wp.size(), wc.size())) - 1);
        for (int j = keep_lo; j < keep_hi && lo + j < T; ++j) {
            punct[lo + j] = wp[1 + j];
            capit[lo + j] = wc[1 + j];
        }
        if (hi == T)
            break;
    }

    const auto& punct_labels = model_->config().punct_labels;

    // Reconstruct: per word, capitalize if its head subword voted "U"; append
    // the punctuation symbol from its last subword's label.
    std::string out;
    out.reserve(text.size() + words.size());
    for (int wi = 0; wi < static_cast<int>(words.size()); ++wi) {
        int head = -1, last = -1;
        for (int t = 0; t < T; ++t) {
            if (tokens[t].word_index == wi) {
                if (head < 0)
                    head = t;
                last = t;
            }
        }
        std::string w = words[wi];
        if (head >= 0 && capit[head] == capit_upper_ && !w.empty())
            w[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(w[0])));
        if (last >= 0) {
            const std::string& sym = punct_labels[punct[last]];
            if (sym != "O")
                w += sym;
        }
        if (wi)
            out += ' ';
        out += w;
    }
    return out;
}

}  // namespace nemo_speech::asr::pnc
