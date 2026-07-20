// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// BERT punctuation+capitalization model on the ggml runtime.
//
// Loads a `pnc.*` GGUF (see convert_model.py): a BERT-base
// encoder (embeddings + N post-LN transformer blocks) plus two token-level FC
// heads (punctuation, capitalization). infer() runs one token segment and
// returns the per-token argmax label index for each head.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "batching.h"
#include "runtime.h"

namespace nemo_speech::asr::pnc {

struct PncConfig {
    int hidden = 768;
    int n_layers = 12;
    int n_heads = 12;
    int intermediate = 3072;
    int max_position = 512;
    int max_seq_length = 128;
    int type_vocab_size = 2;
    int cls_id = 101;
    int sep_id = 102;
    int pad_id = 0;
    int unk_id = 100;
    std::vector<std::string> punct_labels;  // e.g. {"O", ",", ".", "?"}
    std::vector<std::string> capit_labels;  // e.g. {"O", "U"}
    std::vector<std::string> vocab;
};

class PncModel {
   public:
    PncModel(
        ggml_runtime::BackendManager& bm, const std::string& gguf_path,
        const BatchingConfig& batching = {});
    ~PncModel();

    // Run one segment of `n` token ids (including [CLS]/[SEP]). Fills `punct`
    // and `capit` (length n) with per-token argmax label indices.
    void infer(const int32_t* ids, int n, std::vector<int>& punct, std::vector<int>& capit);

    const PncConfig& config() const { return cfg_; }
    BatchMetrics batch_metrics() const;

   private:
    ggml_runtime::BackendManager* bm_;
    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    std::unique_ptr<ggml_runtime::Module> module_;
    std::unique_ptr<ggml_runtime::Session> session_;
    PncConfig cfg_;
    class PncBatcher;
    std::unique_ptr<PncBatcher> batcher_;
};

}  // namespace nemo_speech::asr::pnc
