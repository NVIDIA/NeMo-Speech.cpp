// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Host-side RNNT/TDT greedy decoding over the staged predictor and joint graphs.
#include "rnnt_greedy_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime.h"

namespace nemo_speech::asr {

namespace {
class DecodeStepScope {
   public:
    explicit DecodeStepScope(RnntEngine* engine) : engine_(engine) { engine_->begin_decode_step(); }
    ~DecodeStepScope() { engine_->end_decode_step(); }

   private:
    RnntEngine* engine_;
};
}  // namespace

RnntGreedyDecoder::RnntGreedyDecoder(RnntEngine* engine) : engine_(engine) {
    build_punct_bias();
    reset();
}

// Identify the sentence-terminator tokens and their end-of-utterance floor.
// riva's PunctBiasHelper uses bump 1.5 for '.'/'?' (and ▁-prefixed) with an EOU
// floor of PUNCT_BIAS_EOS_STALL_FLOOR (5) * bump = 7.5; we reuse that floor.
// Comma is mid-sentence (the model emits it on its own) and is not floored.
void
RnntGreedyDecoder::build_punct_bias() {
    const auto& vocab = engine_->vocab();
    // The vocabulary vector may omit the terminal blank entry even though the
    // joint output includes it, so size the graph input from RNNT metadata.
    punct_bias_.assign(static_cast<size_t>(engine_->rnnt_config().vocab_size), 0.0f);
    has_punct_bias_ = false;
    constexpr float kFloor = 5.0f * 1.5f;
    for (int id = 0; id < static_cast<int>(vocab.size()); ++id) {
        const std::string& p = vocab[id];
        // Sentence terminators get the EOU floor so the marginal trailing one
        // wins over blank on the flush. Multilingual models also terminate with
        // '!' and the Devanagari danda U+0964 (Hindi etc.); floor those too so
        // the model's natural terminal wins instead of a spurious '.'/'?'.
        if (p == "." || p == "?" || p == "!" || p == "\xE0\xA5\xA4" || p == "\xE2\x96\x81." ||
            p == "\xE2\x96\x81?") {
            punct_bias_[static_cast<size_t>(id)] = kFloor;
            has_punct_bias_ = true;
        }
    }
}

void
RnntGreedyDecoder::reset() {
    const auto& cfg = engine_->rnnt_config();
    stream_state_ = engine_->make_rnnt_stream_state();
    if (!stream_state_)
        throw std::runtime_error("RnntEngine returned a null stream state");
    prev_token_ = cfg.blank_id;
    active_bank_ = 0;
    predictor_valid_ = false;
    token_ids_.clear();
    stats_ = {};
    last_emit_frame_ = -1;
    words_.clear();
    cur_open_ = false;
    cur_ = WordTiming{};
    finalizing_ = false;
}

void
RnntGreedyDecoder::flush_word() {
    if (cur_open_ && !cur_.word.empty())
        words_.push_back(cur_);
    cur_open_ = false;
    cur_ = WordTiming{};
}

void
RnntGreedyDecoder::finalize() {
    flush_word();
}

std::vector<int>
RnntGreedyDecoder::step(const float* enc_out, int d_model, int T, int64_t frame_offset) {
    std::vector<int> emitted;
    const auto& vocab = engine_->vocab();
    const auto& cfg = engine_->rnnt_config();
    const int blank_id = cfg.blank_id;
    const int max_sym = cfg.max_symbols_per_step;
    if (d_model != cfg.joint_dim) {
        throw std::runtime_error(
            "RnntGreedyDecoder: encoder input is not joint-projected (got " +
            std::to_string(d_model) + ", expected " + std::to_string(cfg.joint_dim) + ")");
    }
    if (T <= 0)
        return emitted;
    DecodeStepScope decode_scope(engine_);
    stats_.encoder_frames += static_cast<uint64_t>(T);

    // With a fixed predictor output, joint evaluations for all remaining
    // encoder frames are independent. Evaluate them in one graph and consume
    // the leading blank run. If a non-blank appears, later results are ignored:
    // committing that token changes the predictor and requires a fresh round
    // beginning at the same frame.
    token_ids_.resize(static_cast<size_t>(T));
    int t = 0;
    int symbols_at_frame = 0;
    while (t < T) {
        if (!predictor_valid_) {
            engine_->predict_rnnt(*stream_state_, prev_token_, active_bank_);
            ++stats_.predictor_calls;
            predictor_valid_ = true;
        }

        const int remaining = T - t;
        // Apply the release branch's end-of-utterance punctuation floor in the
        // staged joint graph, before its device-side argmax. Keeping this as an
        // optional dense bias preserves the vectorized blank-run path and avoids
        // copying full logits back to the host.
        const float* logit_bias = (finalizing_ && has_punct_bias_) ? punct_bias_.data() : nullptr;
        engine_->joint_argmax(
            *stream_state_, enc_out + static_cast<size_t>(t) * d_model, d_model, remaining,
            token_ids_.data(), logit_bias);
        ++stats_.joint_calls;
        stats_.joint_frames += static_cast<uint64_t>(remaining);

        int first_emit = -1;
        for (int i = 0; i < remaining; ++i) {
            if (token_ids_[static_cast<size_t>(i)] != blank_id) {
                first_emit = i;
                break;
            }
        }
        if (first_emit < 0) {
            // Every remaining frame is blank under this predictor state.
            break;
        }

        // A blank advances the encoder clock and ends any symbol run at the
        // previous frame. The first non-blank is handled at its own frame.
        if (first_emit > 0)
            symbols_at_frame = 0;
        t += first_emit;
        const int token = token_ids_[static_cast<size_t>(first_emit)];

        emitted.push_back(token);
        ++stats_.emitted_tokens;
        last_emit_frame_ = frame_offset + t;
        if (compute_ts_ && token >= 0 && token < (int)vocab.size()) {
            const int64_t f = frame_offset + t;
            const std::string& piece = vocab[token];
            if (sp_starts_new_word(piece) || !cur_open_) {
                flush_word();
                cur_open_ = true;
                cur_.start_frame = f;
                cur_.confidence = 1.0f;  // RNNT greedy emits no per-token posterior
            }
            cur_.word += sp_piece_text(piece);
            cur_.end_frame = f + 1;
        }

        // A non-blank commits the candidate prediction state. The newly
        // emitted token becomes the next predictor input, invalidating every
        // cached predictor-side value. Blank paths never reach this block and
        // retain the cache across frames and step() calls.
        prev_token_ = token;
        active_bank_ ^= 1;
        predictor_valid_ = false;

        if (++symbols_at_frame >= max_sym) {
            ++t;
            symbols_at_frame = 0;
        }
    }
    return emitted;
}

TdtGreedyDecoder::TdtGreedyDecoder(RnntEngine* engine) : engine_(engine) {
    if (!engine_ || !engine_->rnnt_config().is_tdt())
        throw std::invalid_argument("TdtGreedyDecoder requires TDT durations");
    reset();
}

void
TdtGreedyDecoder::reset() {
    stream_state_ = engine_->make_rnnt_stream_state();
    if (!stream_state_)
        throw std::runtime_error("RnntEngine returned a null stream state");
    prev_token_ = engine_->rnnt_config().blank_id;
    active_bank_ = 0;
    predictor_valid_ = false;
    stats_ = {};
    last_emit_frame_ = -1;
    words_.clear();
    cur_open_ = false;
    cur_ = WordTiming{};
}

void
TdtGreedyDecoder::reset_utterance() {
    words_.clear();
    cur_open_ = false;
    cur_ = WordTiming{};
}

void
TdtGreedyDecoder::flush_word() {
    if (cur_open_ && !cur_.word.empty())
        words_.push_back(cur_);
    cur_open_ = false;
    cur_ = WordTiming{};
}

void
TdtGreedyDecoder::finalize() {
    flush_word();
}

std::vector<int>
TdtGreedyDecoder::step(const float* enc_out, int d_model, int T, int64_t frame_offset) {
    const auto& cfg = engine_->rnnt_config();
    if (d_model != cfg.joint_dim)
        throw std::runtime_error("TdtGreedyDecoder: invalid encoder projection width");
    std::vector<int> emitted;
    if (T <= 0)
        return emitted;

    DecodeStepScope decode_scope(engine_);

    stats_.encoder_frames += static_cast<uint64_t>(T);
    int t = 0;
    while (t < T) {
        int symbols_added = 0;
        bool need_loop = true;
        int skip = 0;
        while (need_loop && symbols_added < cfg.max_symbols_per_step) {
            int32_t token = cfg.blank_id;
            int32_t duration_index = 0;
            if (!predictor_valid_) {
                engine_->predict_and_joint_tdt_argmax(
                    *stream_state_, prev_token_, active_bank_,
                    enc_out + static_cast<size_t>(t) * d_model, d_model, &token, &duration_index);
                ++stats_.predictor_calls;
                ++stats_.joint_calls;
                ++stats_.joint_frames;
                predictor_valid_ = true;
            } else {
                engine_->joint_tdt_argmax(
                    *stream_state_, enc_out + static_cast<size_t>(t) * d_model, d_model, 1, &token,
                    &duration_index);
                ++stats_.joint_calls;
                ++stats_.joint_frames;
            }
            if (duration_index < 0 || duration_index >= static_cast<int>(cfg.durations.size()))
                throw std::runtime_error("TDT joint returned an invalid duration index");
            skip = cfg.durations[static_cast<size_t>(duration_index)];
            if (skip < 0)
                throw std::runtime_error("TDT duration values must be non-negative");

            // NeMo special-cases blank + duration zero: no decoder state has
            // changed and revisiting the same encoder frame could only repeat
            // that decision, so advance one frame immediately.
            if (token == cfg.blank_id && skip == 0)
                skip = 1;

            if (token != cfg.blank_id) {
                emitted.push_back(token);
                ++stats_.emitted_tokens;
                last_emit_frame_ = frame_offset + t;
                if (compute_ts_ && token >= 0 &&
                    token < static_cast<int>(engine_->vocab().size())) {
                    const int64_t f = frame_offset + t;
                    const auto& piece = engine_->vocab()[static_cast<size_t>(token)];
                    if (sp_is_word_boundary(piece) || !cur_open_) {
                        flush_word();
                        cur_open_ = true;
                        cur_.start_frame = f;
                        cur_.confidence = 1.0f;
                    }
                    cur_.word += sp_piece_text(piece);
                    cur_.end_frame = f + std::max(1, skip);
                }
                prev_token_ = token;
                active_bank_ ^= 1;
                predictor_valid_ = false;
            }

            ++symbols_added;
            t += skip;
            need_loop = skip == 0;
        }

        // NeMo's infinite-loop guard: duration zero can otherwise revisit the
        // same frame forever (blank or a chain of zero-duration tokens).
        if (symbols_added == cfg.max_symbols_per_step && skip == 0)
            ++t;
    }
    return emitted;
}

}  // namespace nemo_speech::asr
