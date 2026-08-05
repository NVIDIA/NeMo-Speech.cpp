// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "greedy_ctc_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "runtime.h"

namespace nemo_speech::asr {

CtcHeadModule::CtcHeadModule(const std::string& name, const CtcConfig& cfg)
    : name_(name), cfg_(cfg), argmax_eye_name_(name + ".argmax_eye") {
    // NeMo wraps the CTC projection in Sequential at index 0 -> "decoder.decoder_layers.0".
    proj_ = new ggml_runtime::Conv1D(
        "decoder.decoder_layers.0", cfg.d_model, cfg.num_classes + 1,
        /*kernel=*/1, /*stride=*/1, /*padding=*/0, /*dilation=*/1,
        /*use_bias=*/true, /*is_dw=*/false);
}

CtcHeadModule::~CtcHeadModule() {
    delete proj_;
}


void
CtcHeadModule::define_tensors(ggml_runtime::Session* session) {
    proj_->define_tensors(session);
    const int C = cfg_.num_classes + 1;
    session->model_tensor_container->create_tensor_2d(argmax_eye_name_, GGML_TYPE_F32, C, C);
}

ggml_runtime::TensorBag
CtcHeadModule::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    // Input: encoder features (d_model=C, T, 1, 1).
    auto x = input_tensors.get_tensor(0);
    auto bf_ctx = tc->get_ctx_of_buffer_type(x.buft);

    // Conv1D expects input shape (T, C, B, 1). Transpose dims 0 and 1.
    auto x_t = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, x.tensor, 1, 0, 2, 3));
    ggml_runtime::TensorBag head_in;
    head_in.add_tensor(ggml_runtime::ggml_bf_tensor(x_t, x.buft));
    auto head_out = proj_->build_graph(session, head_in, tc);
    auto logits = head_out.get_tensor(0);

    // (T, num_classes+1, 1, 1) -> (num_classes+1, T, 1, 1)
    auto logits_t = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, logits.tensor, 1, 0, 2, 3));
    auto sm = ggml_soft_max(bf_ctx.ctx, logits_t);
    auto log_sm = ggml_log(bf_ctx.ctx, sm);

    ggml_runtime::TensorBag out;
    out.add_tensor(ggml_runtime::ggml_bf_tensor(log_sm, x.buft));
    return out;
}

ggml_runtime::TensorBag
CtcHeadModule::build_greedy_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    auto full = build_graph(session, input_tensors, tc);
    auto log_probs = full.get_tensor(0);
    auto bf_ctx = tc->get_ctx_of_buffer_type(log_probs.buft);

    auto eye = session->model_tensor_container->get_tensor_by_name(argmax_eye_name_);
    const int64_t C = log_probs.tensor->ne[0];
    const int64_t T = log_probs.tensor->ne[1];
    const int64_t B = log_probs.tensor->ne[2];
    // Argmax reduces each column independently, so flatten the outer axes and
    // avoid backend-specific handling of integer concat and strided views.
    auto flat = ggml_reshape_2d(bf_ctx.ctx, ggml_cont(bf_ctx.ctx, log_probs.tensor), C, T * B);
    auto ids = ggml_argmax(bf_ctx.ctx, flat);
    auto onehot = ggml_get_rows(bf_ctx.ctx, eye.tensor, ids);
    auto winning_prob =
        ggml_exp(bf_ctx.ctx, ggml_sum_rows(bf_ctx.ctx, ggml_mul(bf_ctx.ctx, flat, onehot)));
    ids = ggml_reshape_2d(bf_ctx.ctx, ids, T, B);
    winning_prob = ggml_reshape_2d(bf_ctx.ctx, winning_prob, T, B);

    ggml_runtime::TensorBag out;
    out.add_tensor(ggml_runtime::ggml_bf_tensor(ids, log_probs.buft));
    out.add_tensor(ggml_runtime::ggml_bf_tensor(winning_prob, log_probs.buft));
    return out;
}

void
CtcHeadModule::set_data(ggml_runtime::Session* session) {
    proj_->set_data(session);
    const int C = cfg_.num_classes + 1;
    std::vector<float> eye(static_cast<size_t>(C) * C, 0.0f);
    for (int c = 0; c < C; ++c) eye[static_cast<size_t>(c) * C + c] = 1.0f;
    auto t = session->model_tensor_container->get_tensor_by_name(argmax_eye_name_);
    ggml_backend_tensor_set(t.tensor, eye.data(), 0, eye.size() * sizeof(float));
}

void
GreedyCtcDecoder::reset() {
    prev_id_ = -1;
    last_committed_id_ = -1;
    last_committed_frame_ = -kSeamMergeMaxGap - 1;
    last_emit_frame_ = -1;
    conf_sum_ = 0.0;
    conf_n_ = 0;
    words_.clear();
    cur_open_ = false;
    cur_ = WordTiming{};
}

void
GreedyCtcDecoder::flush_word() {
    if (cur_open_ && !cur_.word.empty())
        words_.push_back(cur_);
    cur_open_ = false;
    cur_ = WordTiming{};
}

void
GreedyCtcDecoder::finalize() {
    flush_word();
}

std::vector<int>
GreedyCtcDecoder::step(const float* log_probs, int n_classes, int T, int64_t frame_offset) {
    std::vector<int32_t> ids(static_cast<size_t>(T));
    std::vector<float> probs(static_cast<size_t>(T));
    for (int t = 0; t < T; t++) {
        int best = 0;
        float best_lp = log_probs[t * n_classes];
        for (int c = 1; c < n_classes; c++) {
            const float v = log_probs[t * n_classes + c];
            if (v > best_lp) {
                best_lp = v;
                best = c;
            }
        }
        ids[static_cast<size_t>(t)] = best;
        probs[static_cast<size_t>(t)] = std::exp(best_lp);
    }
    return step_compact(ids.data(), probs.data(), T, frame_offset);
}

std::vector<int>
GreedyCtcDecoder::step_compact(
    const int32_t* best_ids, const float* best_probs, int T, int64_t frame_offset) {
    static const bool dbg = std::getenv("NEMO_SPEECH_CTC_DEBUG") != nullptr;
    std::vector<int> emitted;
    int prev = prev_id_;
    for (int t = 0; t < T; ++t) {
        const int best = best_ids[t];
        const float best_prob = best_probs[t];
        if (dbg) {
            fprintf(
                stderr, "[ctc-dbg] frame=%lld id=%d piece=%s prob=%.3f prev=%d\n",
                (long long)(frame_offset + t), best,
                (best >= 0 && best < (int)vocab_.size()) ? vocab_[best].c_str() : "?", best_prob,
                prev);
        }
        const int64_t frame = frame_offset + t;
        // Seam dedup (see header): the same word straddling a buffered-window
        // emit boundary can spike once per window with a blank between; treat
        // a window-leading repeat of the last committed token within
        // kSeamMergeMaxGap frames as a continuation, not a new emission.
        const bool seam_repeat = emitted.empty() && best == last_committed_id_ &&
                                 frame - last_committed_frame_ <= kSeamMergeMaxGap;
        if (best != cfg_.blank_id)
            last_emit_frame_ = frame;
        if (best != cfg_.blank_id && best != prev && !seam_repeat) {
            emitted.push_back(best);
            last_committed_id_ = best;
            conf_sum_ += best_prob;
            conf_n_++;
            if (compute_ts_ && best >= 0 && best < (int)vocab_.size()) {
                const int64_t f = frame_offset + t;
                const std::string& piece = vocab_[best];
                // Open a new word on a word-starting ▁ piece (or the first token).
                if (sp_starts_new_word(piece) || !cur_open_) {
                    flush_word();
                    cur_open_ = true;
                    cur_.start_frame = f;
                    cur_.confidence = 1.0f;
                }
                cur_.word += sp_piece_text(piece);
                cur_.end_frame = f + 1;
                cur_.confidence = std::min(cur_.confidence, best_prob);
            }
        }
        // The committed token's reach extends through collapse repeats and
        // seam-suppressed re-spikes, so a word spanning several windows keeps
        // chaining the seam-gap check off its latest evidence.
        if (best != cfg_.blank_id && best == last_committed_id_)
            last_committed_frame_ = frame;
        prev = best;
    }
    prev_id_ = prev;
    return emitted;
}

void
append_sentencepiece_tokens(
    std::string& out, const std::vector<int>& ids, const std::vector<std::string>& vocab) {
    const bool trim_leading_space = out.empty();
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(vocab.size()))
            continue;
        const std::string& p = vocab[id];
        for (size_t i = 0; i < p.size();) {
            // U+2581 = '▁' = 0xE2 0x96 0x81 in UTF-8.
            if (i + 2 < p.size() && (uint8_t)p[i] == 0xE2 && (uint8_t)p[i + 1] == 0x96 &&
                (uint8_t)p[i + 2] == 0x81) {
                if (i != 0 || !sp_is_sentence_terminator(p))
                    out.push_back(' ');
                i += 3;
            } else {
                out.push_back(p[i]);
                i += 1;
            }
        }
    }
    if (trim_leading_space) {
        const size_t s = out.find_first_not_of(' ');
        if (s == std::string::npos)
            out.clear();
        else if (s > 0)
            out.erase(0, s);
    }
}

std::string
detokenize_sentencepiece(const std::vector<int>& ids, const std::vector<std::string>& vocab) {
    std::string out;
    append_sentencepiece_tokens(out, ids, vocab);
    return out;
}

}  // namespace nemo_speech::asr
