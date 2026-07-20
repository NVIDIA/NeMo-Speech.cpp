// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// CTC head for FastConformer ASR.
//
// Two layers:
//   - CtcHeadModule: a ggml Module that takes encoder features
//     (d_model, T, 1, 1) and produces log-softmax over classes
//     (num_classes+1, T, 1, 1). Glued onto the encoder graph.
//   - GreedyCtcDecoder: the Head interface implementation. Receives log-probs
//     for each chunk, runs greedy argmax + collapse-rule, emits tokens.
#pragma once

#include <string>
#include <vector>

#include "decoder.h"
#include "nn.h"
#include "runtime.h"

namespace nemo_speech::asr {

struct CtcConfig {
    int d_model = 1024;
    int num_classes = 1024;  // excluding blank
    int blank_id = 1024;
};

// ggml Module: Conv1D(d_model -> num_classes+1, k=1) -> log_softmax.
// Input  : (d_model,        T, 1, 1)
// Output : (num_classes+1,  T, 1, 1)  log_softmax along dim 0.
class CtcHeadModule : public ggml_runtime::Module {
   public:
    CtcHeadModule(const std::string& name, const CtcConfig& cfg);
    ~CtcHeadModule();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    // Same projection/softmax, but reduce on-device to one class id and one
    // winning probability per frame for greedy decoding.
    ggml_runtime::TensorBag build_greedy_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc);
    void set_data(ggml_runtime::Session* session) override;

   private:
    std::string name_;
    CtcConfig cfg_;
    ggml_runtime::Conv1D* proj_;  // decoder.decoder_layers.0
    std::string argmax_eye_name_;
};

// Head impl. Greedy collapse-rule across chunk boundaries.
class GreedyCtcDecoder : public Decoder {
   public:
    GreedyCtcDecoder(CtcConfig cfg, std::vector<std::string> vocab)
        : cfg_(cfg), vocab_(std::move(vocab)), prev_id_(-1) {}

    void reset() override;
    // EOU: drop accumulated word timings + confidence and clear the cross-chunk
    // collapse state. A natural silence EOU already landed prev_id_ on blank
    // (so this is a no-op there), but a VAD-driven or client-forced EOU can
    // split mid-token with no intervening blank; clearing prev_id_ ensures the
    // next utterance's first token is emitted even when it repeats the last one.
    void reset_utterance() override {
        words_.clear();
        cur_ = WordTiming{};
        cur_open_ = false;
        conf_sum_ = 0.0;
        conf_n_ = 0;
        prev_id_ = -1;
        last_committed_id_ = -1;
        last_committed_frame_ = -kSeamMergeMaxGap - 1;
    }
    std::vector<int> step(
        const float* log_probs, int n_classes, int T, int64_t frame_offset) override;
    std::vector<int> step_compact(
        const int32_t* best_ids, const float* best_probs, int T, int64_t frame_offset);
    int blank_id() const override { return cfg_.blank_id; }
    const std::vector<std::string>& vocab() const override { return vocab_; }

    void set_compute_timestamps(bool on) override { compute_ts_ = on; }
    const std::vector<WordTiming>& word_timings() const override { return words_; }
    void finalize() override;  // flush the trailing in-progress word
    int64_t last_emit_frame() const override { return last_emit_frame_; }
    float confidence() const override {
        return conf_n_ > 0 ? static_cast<float>(conf_sum_ / conf_n_) : 1.0f;
    }

   private:
    void flush_word();  // push the in-progress word (if any) into words_

    CtcConfig cfg_;
    std::vector<std::string> vocab_;
    int prev_id_;  // last emitted (or blank) id, for cross-chunk collapse
    // Buffered-window seam dedup state: the last token actually committed and
    // its frame. Overlapping windows encode the seam frames with slightly
    // different context, so a word straddling an emit boundary can spike once
    // in each window (blank-separated — the collapse rule correctly keeps
    // both). A window's FIRST emission is therefore suppressed when it
    // repeats the last committed token within kSeamMergeMaxGap frames; real
    // repeated words are far further apart than one chunk seam.
    static constexpr int64_t kSeamMergeMaxGap = 2;
    int last_committed_id_ = -1;
    int64_t last_committed_frame_ = -kSeamMergeMaxGap - 1;
    // Last non-blank argmax frame (speech evidence for token-silence EOU).
    int64_t last_emit_frame_ = -1;
    // Utterance confidence: running mean of emitted tokens' softmax prob.
    double conf_sum_ = 0.0;
    int64_t conf_n_ = 0;

    // Word-timestamp accumulation (only when compute_ts_). A word opens on a
    // token whose SentencePiece piece starts with ▁ (the boundary marker the
    // runner's detokenizer splits on) and stays open until the next boundary.
    bool compute_ts_ = false;
    std::vector<WordTiming> words_;
    bool cur_open_ = false;
    WordTiming cur_;
};

}  // namespace nemo_speech::asr
