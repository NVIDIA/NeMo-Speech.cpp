// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Host-side RNNT/TDT greedy decoding over the staged predictor and joint graphs.
#include "rnnt_greedy_decoder.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

namespace {
// Riva treats phrases matching ^\$[A-Z_]+$ as class hints (e.g. "$NAME"), not
// literal boost phrases; skip them like parse_riva_speech_contexts does.
bool
is_class_hint(const std::string& s) {
    if (s.size() < 2 || s[0] != '$')
        return false;
    for (size_t i = 1; i < s.size(); ++i) {
        const char c = s[i];
        if (!((c >= 'A' && c <= 'Z') || c == '_'))
            return false;
    }
    return true;
}
}  // namespace

// ===========================================================================
// RnntGreedyDecoder
// ===========================================================================
RnntGreedyDecoder::RnntGreedyDecoder(RnntEngine* engine, const DecoderConfig& cfg)
    : engine_(engine), bias_alpha_cfg_(static_cast<float>(cfg.boosting_tree_alpha)),
      bias_depth_scaling_(static_cast<float>(cfg.boosting_depth_scaling)),
      bias_max_boost_(cfg.boosting_max_boost) {
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
RnntGreedyDecoder::set_request_options(const AsrRequestOptions& opts) {
    set_compute_timestamps(opts.needs_word_timings());
    bias_on_ = false;
    bias_node_ = ContextBiasingTree::kRoot;
    bias_eff_alpha_ = 1.0f;
    bias_tree_.clear();
    if (opts.speech_contexts.empty() || bias_alpha_cfg_ <= 0.0f)
        return;

    // Mirror riva's parse_riva_speech_contexts: collect every phrase (skipping
    // class hints like "$NAME"), tokenize with the model's tokenizer, build the
    // tree with a fixed per-token context_score of 1.0, and fold the request's
    // boost into the fusion alpha (clamped to [0.1, max_boost]). The single
    // max boost across the request drives one alpha for all phrases.
    std::vector<std::vector<int>> phrases;
    float max_boost = 0.0f;
    for (const auto& sc : opts.speech_contexts) {
        if (sc.boost > 0.0f)
            max_boost = std::max(max_boost, sc.boost);
        for (const auto& phrase : sc.phrases) {
            if (is_class_hint(phrase))
                continue;
            std::vector<int> ids = engine_->encode_phrase(phrase);
            if (ids.empty())
                continue;
            phrases.push_back(std::move(ids));
        }
    }
    if (phrases.empty()) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            std::fprintf(
                stderr,
                "[asr] RNNT word boosting requested but no phrases could be tokenized "
                "(no embedded SentencePiece tokenizer in the GGUF?); boosting disabled.\n");
        return;
    }

    const float req_alpha =
        (max_boost > 0.0f)
            ? std::max(0.1f, std::min(max_boost, static_cast<float>(bias_max_boost_)))
            : 1.0f;
    bias_eff_alpha_ = bias_alpha_cfg_ * req_alpha;

    const std::vector<float> scores(phrases.size(), 1.0f);  // context_score = 1.0, as in riva
    bias_tree_.build(phrases, scores, bias_depth_scaling_);
    bias_on_ = !bias_tree_.empty();
}

void
RnntGreedyDecoder::reset() {
    const auto& cfg = engine_->rnnt_config();
    // Release the old slot before acquiring the new one; plain assignment would
    // transiently hold two slots per stream and exhaust the arena under churn.
    stream_state_.reset();
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
    bias_node_ = ContextBiasingTree::kRoot;
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
    return step_impl(enc_out, nullptr, d_model, T, frame_offset);
}

std::vector<int>
RnntGreedyDecoder::step_device(
    const ggml_runtime::DeviceTensor& enc_out, int d_model, int T, int64_t frame_offset) {
    if (!enc_out.valid())
        throw std::runtime_error("RnntGreedyDecoder: invalid device encoder output");
    return step_impl(nullptr, &enc_out, d_model, T, frame_offset);
}

std::vector<int>
RnntGreedyDecoder::step_impl(
    const float* enc_out, const ggml_runtime::DeviceTensor* device_enc_out, int d_model, int T,
    int64_t frame_offset) {
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
        const int remaining = T - t;
        const float* logit_bias = (finalizing_ && has_punct_bias_) ? punct_bias_.data() : nullptr;
        const float* boost_bias = nullptr;
        if (bias_on_) {
            // NeMo/riva semantics (rnnt_label_looping.py): blank vs non-blank is
            // decided on the unboosted logits; boosting only re-ranks which
            // non-blank label wins. Build a second bias for a blank-excluded
            // argmax pass. A uniform backoff base on every non-blank token
            // cannot change that ranking, so only the per-token deltas from the
            // current match state are applied. Valid for the whole speculative
            // frame run: the match state only changes on a non-blank commit,
            // which starts a fresh round.
            const ContextBiasingTree::StepBoost sb = bias_tree_.step_boost(bias_node_);
            if (logit_bias != nullptr)
                bias_scratch_.assign(punct_bias_.begin(), punct_bias_.end());
            else
                bias_scratch_.assign(static_cast<size_t>(cfg.vocab_size), 0.0f);
            for (const auto& sp : sb.specials) {
                if (sp.first != blank_id && sp.first >= 0 && sp.first < cfg.vocab_size)
                    bias_scratch_[static_cast<size_t>(sp.first)] +=
                        bias_eff_alpha_ * (sp.second - sb.base);
            }
            bias_scratch_[static_cast<size_t>(blank_id)] = -1e30f;  // exclude blank
            boost_bias = bias_scratch_.data();
        }
        bool joint_complete = false;
        if (!predictor_valid_ && device_enc_out != nullptr && boost_bias == nullptr) {
            engine_->predict_and_joint_rnnt_argmax_device(
                *stream_state_, prev_token_, active_bank_, *device_enc_out, t, d_model, remaining,
                token_ids_.data(), logit_bias);
            ++stats_.predictor_calls;
            ++stats_.joint_calls;
            stats_.joint_frames += static_cast<uint64_t>(remaining);
            predictor_valid_ = true;
            joint_complete = true;
        } else if (!predictor_valid_) {
            engine_->predict_rnnt(*stream_state_, prev_token_, active_bank_);
            ++stats_.predictor_calls;
            predictor_valid_ = true;
        }

        // Apply the release branch's end-of-utterance punctuation floor in the
        // staged joint graph, before its device-side argmax. Keeping this as an
        // optional dense bias preserves the vectorized blank-run path and avoids
        // copying full logits back to the host.
        if (!joint_complete) {
            if (device_enc_out != nullptr) {
                engine_->joint_argmax_device(
                    *stream_state_, *device_enc_out, t, d_model, remaining, token_ids_.data(),
                    logit_bias);
            } else {
                engine_->joint_argmax(
                    *stream_state_, enc_out + static_cast<size_t>(t) * d_model, d_model, remaining,
                    token_ids_.data(), logit_bias);
            }
            ++stats_.joint_calls;
            stats_.joint_frames += static_cast<uint64_t>(remaining);
        }
        if (boost_bias != nullptr) {
            boost_token_ids_.resize(static_cast<size_t>(remaining));
            if (device_enc_out != nullptr) {
                engine_->joint_argmax_device(
                    *stream_state_, *device_enc_out, t, d_model, remaining, boost_token_ids_.data(),
                    boost_bias);
            } else {
                engine_->joint_argmax(
                    *stream_state_, enc_out + static_cast<size_t>(t) * d_model, d_model, remaining,
                    boost_token_ids_.data(), boost_bias);
            }
            ++stats_.joint_calls;
            stats_.joint_frames += static_cast<uint64_t>(remaining);
            for (int i = 0; i < remaining; ++i) {
                if (token_ids_[static_cast<size_t>(i)] != blank_id)
                    token_ids_[static_cast<size_t>(i)] = boost_token_ids_[static_cast<size_t>(i)];
            }
        }

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
        // Advance the context-biasing match on the emitted token.
        if (bias_on_)
            bias_node_ = bias_tree_.advance(bias_node_, token);

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
    // Release the old slot before acquiring the new one; plain assignment would
    // transiently hold two slots per stream and exhaust the arena under churn.
    stream_state_.reset();
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
