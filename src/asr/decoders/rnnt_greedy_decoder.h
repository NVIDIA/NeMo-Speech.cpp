// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// RNNT head: 2-layer LSTM predictor + joint network + greedy emit loop.
//
// Architecture:
//   predictor: prev_token -> embed -> LSTM[0] -> LSTM[1] -> dec_out (pred_hidden)
//   joint:     joint_net(enc.proj(enc_frame) + pred.proj(dec_out)) -> logits over vocab
//   greedy:    per encoder frame, inner loop: emit symbols until blank or
//              MAX_SYMBOLS_PER_STEP reached; update LSTM state only on non-blank.
//
// Tensor naming (NeMo convention):
//   decoder.prediction.embed.weight                 [embed_dim, vocab_size]
//   decoder.prediction.dec_rnn.lstm.weight_ih_l{0,1} [input_dim, 4*hidden]
//   decoder.prediction.dec_rnn.lstm.weight_hh_l{0,1} [hidden, 4*hidden]
//   decoder.prediction.dec_rnn.lstm.bias_ih_l{0,1}   [4*hidden]
//   decoder.prediction.dec_rnn.lstm.bias_hh_l{0,1}   [4*hidden]
//   joint.enc.{weight,bias}                          d_model -> joint_dim
//   joint.pred.{weight,bias}                         pred_hidden -> joint_dim
//   joint.joint_net.2.{weight,bias}                  joint_dim -> vocab_size
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "context_biasing.h"  // ContextBiasingTree (RNNT word boosting)
#include "decoder.h"          // Decoder, WordTiming, DecoderConfig (and runtime.h transitively)

namespace nemo_speech::asr {

struct RnntConfig {
    int d_model = 1024;     // must match encoder
    int vocab_size = 1025;  // includes blank
    int blank_id = 1024;
    int pred_embed_dim = 640;
    int pred_hidden = 640;
    int pred_num_layers = 2;
    int joint_dim = 640;
    int max_symbols_per_step = 10;
    // Non-empty for Token-and-Duration Transducers. The joint output is then
    // [token classes including blank | duration classes].
    std::vector<int32_t> durations;

    bool is_tdt() const { return !durations.empty(); }
    int joint_output_size() const { return vocab_size + static_cast<int>(durations.size()); }
};

// Opaque per-stream predictor state. The model implementation keeps its
// ping-pong LSTM banks and joint projection device-resident.
class RnntStreamState {
   public:
    virtual ~RnntStreamState() = default;
};

// Per-stream counters for validating the staged decoder's execution pattern.
// They are intentionally maintained by the host state machine (rather than
// atomics in the shared model) so diagnostics add no cross-stream contention.
struct RnntDecodeStats {
    uint64_t predictor_calls = 0;
    uint64_t joint_calls = 0;
    uint64_t joint_frames = 0;  // includes speculative frames after an emit
    uint64_t encoder_frames = 0;
    uint64_t emitted_tokens = 0;
};

// AsrModel exposes this opaque interface; RnntGreedyDecoder calls the two
// independently cached decoder stages through it. The encoder projection runs
// once by RnntModel::encode_cache_aware for the whole chunk. The predictor is
// run only when its cached output is stale (initially and after a non-blank
// emission). The joint tail may evaluate several consecutive encoder frames at
// once while the predictor output is unchanged. Thread-safe.
class RnntEngine {
   public:
    virtual ~RnntEngine() = default;

    virtual const RnntConfig& rnnt_config() const = 0;
    virtual const std::vector<std::string>& vocab() const = 0;

    virtual std::unique_ptr<RnntStreamState> make_rnnt_stream_state() = 0;

    // Bracket one host greedy step. Production uses this scope to align an
    // admission wave and account for decoder work that can become ready; test
    // engines can keep the no-op defaults.
    virtual void begin_decode_step() {}
    virtual void end_decode_step() {}

    // Advance the prediction LSTM once. It reads `active_bank`, writes the
    // other bank, and stores the predictor projection in the stream state.
    // The caller flips banks only after a non-blank token, making commit a
    // zero-copy host-side index update.
    virtual void predict_rnnt(RnntStreamState& state, int prev_token, int active_bank) = 0;

    // Evaluate the joint tail for T already-projected encoder frames using one
    // cached predictor projection. enc_proj is [joint_dim, T], column-major;
    // token_ids receives the device-side argmax for each frame. The caller uses
    // results only up to the first non-blank (later results are speculative).
    virtual void joint_argmax(
        RnntStreamState& state, const float* enc_proj, int joint_dim, int T, int32_t* token_ids,
        const float* logit_bias = nullptr) = 0;

    virtual void joint_argmax_device(
        RnntStreamState&, const ggml_runtime::DeviceTensor&, int, int, int, int32_t*,
        const float* = nullptr) {
        throw std::runtime_error("device RNNT joint is not supported by this engine");
    }

    virtual void predict_and_joint_rnnt_argmax_device(
        RnntStreamState& state, int prev_token, int active_bank,
        const ggml_runtime::DeviceTensor& enc_proj, int frame_offset, int joint_dim, int T,
        int32_t* token_ids, const float* logit_bias = nullptr) {
        predict_rnnt(state, prev_token, active_bank);
        joint_argmax_device(state, enc_proj, frame_offset, joint_dim, T, token_ids, logit_bias);
    }

    // TDT splits the joint output into independent token and duration slices.
    virtual void joint_tdt_argmax(
        RnntStreamState& state, const float* enc_proj, int joint_dim, int T, int32_t* token_ids,
        int32_t* duration_ids) = 0;

    // Implementations may fuse this common predictor-plus-scalar-joint path;
    // the default composes the two operations.
    virtual void predict_and_joint_tdt_argmax(
        RnntStreamState& state, int prev_token, int active_bank, const float* enc_proj,
        int joint_dim, int32_t* token_id, int32_t* duration_id) {
        predict_rnnt(state, prev_token, active_bank);
        joint_tdt_argmax(state, enc_proj, joint_dim, 1, token_id, duration_id);
    }

    // Encode a word-boosting phrase into subword token ids using the model's
    // tokenizer (for RNNT context biasing). Returns empty when no tokenizer is
    // available, in which case boosting is silently skipped.
    virtual std::vector<int> encode_phrase(const std::string& /*text*/) const { return {}; }
};

// Head impl for RNNT models.
class RnntGreedyDecoder : public Decoder {
   public:
    RnntGreedyDecoder(RnntEngine* engine, const DecoderConfig& cfg = DecoderConfig{});

    void reset() override;
    // Soft utterance reset for callers that intentionally preserve predictor
    // context. CacheStreamRunner uses reset() at an endpoint boundary.
    // Also resets the context-biasing match position so phrase
    // matching restarts cleanly.
    void reset_utterance() override {
        words_.clear();
        cur_ = WordTiming{};
        cur_open_ = false;
        bias_node_ = ContextBiasingTree::kRoot;
    }
    // Build the per-request context-biasing tree from speech_contexts.
    void set_request_options(const AsrRequestOptions& opts) override;
    // enc_out: joint encoder projection (joint_dim, T) flat-packed. The model
    // computes it once per chunk, after optional language-prompt fusion.
    std::vector<int> step(const float* enc_out, int d_model, int T, int64_t frame_offset) override;
    std::vector<int> step_device(
        const ggml_runtime::DeviceTensor& enc_out, int d_model, int T,
        int64_t frame_offset) override;
    int blank_id() const override { return engine_->rnnt_config().blank_id; }
    const std::vector<std::string>& vocab() const override { return engine_->vocab(); }

    // The next step() is an end-of-utterance (is_last) chunk: apply the EOU
    // punctuation floor (see the punct-bias note in the .cpp).
    void set_finalizing(bool on) override { finalizing_ = on; }

    void set_compute_timestamps(bool on) override { compute_ts_ = on; }
    const std::vector<WordTiming>& word_timings() const override { return words_; }
    void finalize() override;  // flush the trailing in-progress word
    int64_t last_emit_frame() const override { return last_emit_frame_; }
    const RnntDecodeStats& stats() const { return stats_; }

   private:
    std::vector<int> step_impl(
        const float* enc_out, const ggml_runtime::DeviceTensor* device_enc_out, int d_model, int T,
        int64_t frame_offset);
    void flush_word();
    void build_punct_bias();

    RnntEngine* engine_;
    std::unique_ptr<RnntStreamState> stream_state_;
    int prev_token_ = -1;
    int active_bank_ = 0;

    // Predictor cache. For a fixed (prev_token, h, c), all blank encoder
    // frames use the same predictor result and candidate next state. It becomes
    // stale only after a non-blank commits the candidate and changes
    // prev_token. Kept across step() calls and soft reset_utterance() calls.
    bool predictor_valid_ = false;
    std::vector<int32_t> token_ids_;
    RnntDecodeStats stats_;
    // Last token-emitting encoder frame (speech evidence for token-silence EOU).
    int64_t last_emit_frame_ = -1;

    // End-of-utterance punctuation floor (adapted from riva's PunctBiasHelper).
    // The RNNT model self-punctuates internally, but the terminal '.'/'?' is
    // marginal - on the trailing frames it sits as the #2 logit just behind
    // blank, so greedy never commits it. On the is_last flush we add a floor to
    // those tokens' logits so they win. Empty for models whose vocab carries no
    // sentence terminators (then it is a no-op).
    // Dense vocab-sized bias so the joint graph can broadcast it over all
    // remaining encoder frames before device-side argmax.
    std::vector<float> punct_bias_;
    bool has_punct_bias_ = false;
    bool finalizing_ = false;  // next step() is an end-of-utterance chunk

    // Word-timestamp accumulation (only when compute_ts_); grouped on the ▁
    // boundary, same convention as GreedyCtcDecoder.
    bool compute_ts_ = false;
    std::vector<WordTiming> words_;
    bool cur_open_ = false;
    WordTiming cur_;

    // RNNT context biasing (word boosting), built per request from
    // speech_contexts. bias_node_ is the per-stream Aho-Corasick match state.
    ContextBiasingTree bias_tree_;
    int bias_node_ = ContextBiasingTree::kRoot;
    bool bias_on_ = false;
    // --boosting-tree-alpha: an extra global multiplier (default 1.0, i.e. no
    // change). The effective alpha is this times the per-request weight derived
    // from the speech_contexts boost (see set_request_options).
    float bias_alpha_cfg_ = 1.0f;
    float bias_eff_alpha_ = 1.0f;  // effective alpha for the current request
    float bias_depth_scaling_ = 2.0f;
    double bias_max_boost_ = 5.0;           // greedy-safe upper clamp on the request boost
    std::vector<float> bias_scratch_;       // dense per-round joint-logit bias
    std::vector<int32_t> boost_token_ids_;  // blank-excluded re-rank pass output
};

// Token-and-Duration Transducer greedy decoder. Predictor state commits only
// on non-blank tokens; the independently predicted duration advances the
// encoder clock and duration zero permits another symbol at the same frame.
class TdtGreedyDecoder : public Decoder {
   public:
    explicit TdtGreedyDecoder(RnntEngine* engine);

    void reset() override;
    void reset_utterance() override;
    std::vector<int> step(const float* enc_out, int d_model, int T, int64_t frame_offset) override;
    int blank_id() const override { return engine_->rnnt_config().blank_id; }
    const std::vector<std::string>& vocab() const override { return engine_->vocab(); }
    void set_compute_timestamps(bool on) override { compute_ts_ = on; }
    const std::vector<WordTiming>& word_timings() const override { return words_; }
    void finalize() override;
    int64_t last_emit_frame() const override { return last_emit_frame_; }
    const RnntDecodeStats& stats() const { return stats_; }

   private:
    void flush_word();

    RnntEngine* engine_;
    std::unique_ptr<RnntStreamState> stream_state_;
    int prev_token_ = -1;
    int active_bank_ = 0;
    bool predictor_valid_ = false;
    // A duration may advance past the end of one encoder block. Carry that
    // remainder into the next step() so segmented and contiguous decoding
    // present the same encoder clock to TDT.
    int pending_skip_ = 0;
    RnntDecodeStats stats_;
    int64_t last_emit_frame_ = -1;
    bool compute_ts_ = false;
    std::vector<WordTiming> words_;
    bool cur_open_ = false;
    WordTiming cur_;
};

}  // namespace nemo_speech::asr
