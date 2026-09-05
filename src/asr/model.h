// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Shared ASR model resources. AsrModel::load() selects the concrete head from
// GGUF metadata; per-stream recurrent state remains isolated.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "batching.h"
#include "cache_aware_encoder.h"
#include "fastconformer.h"
#include "fe.h"
#include "greedy_ctc_decoder.h"
#include "parameter_parser.h"
#include "rnnt_greedy_decoder.h"
#include "runtime.h"

// Forward-declared so model.h consumers don't pull in the SentencePiece header;
// only model.cpp (which loads the embedded tokenizer) needs the complete type.
namespace sentencepiece {
class SentencePieceProcessor;
}

namespace nemo_speech::asr {

class RnntPredictorModule;
class RnntJointModule;
class PromptFusionModule;

struct ModelConfig {
    std::string path;
    std::string name;  // empty = derived from the model
    void Register(common::ParameterParser& p) {
        p.Register("path", &path, "Model GGUF path");
        p.Register("name", &name, "Model name override (default: derived from model)");
    }
};

enum class HeadKind { Ctc, Rnnt, Tdt };

class AsrModel {
   public:
    virtual ~AsrModel();

    AsrModel(const AsrModel&) = delete;
    AsrModel& operator=(const AsrModel&) = delete;

    // The BackendManager is borrowed for the model's lifetime.
    static std::unique_ptr<AsrModel> load(
        ggml_runtime::BackendManager& bm, const std::string& model_path,
        const BatchingConfig& batching = {});

    virtual HeadKind head_kind() const = 0;

    int sample_rate() const { return fe_cfg_.sample_rate; }
    int subsampling_factor() const { return enc_cfg_.subsampling_factor; }
    // Duration of one encoder output frame.
    double ms_per_enc_frame() const {
        return static_cast<double>(enc_cfg_.subsampling_factor) * fe_->hop_length() * 1000.0 /
               static_cast<double>(fe_cfg_.sample_rate);
    }
    const EncoderConfig& encoder_config() const { return enc_cfg_; }
    virtual const std::vector<std::string>& vocab() const { return vocab_; }
    const std::string& model_name() const { return model_name_; }

    ggml_runtime::BackendManager& backend_manager() { return *bm_; }

    // Diagnostic sessions may be null until their lazy path has run.
    struct DiagSession {
        std::string label;
        ggml_runtime::Session* session;
    };
    virtual std::vector<DiagSession> diagnostic_sessions() const = 0;

    // Shared, stream-independent frontend.
    MelSpectrogramExtractor& fe() { return *fe_; }
    const MelSpectrogramExtractor& fe() const { return *fe_; }
    const MelSpecConfig& fe_config() const { return fe_cfg_; }

    // Prompt-conditioned multilingual support.
    virtual bool has_prompt() const { return false; }
    virtual int prompt_index_for_lang(const std::string& /*lang*/) const { return -1; }
    virtual std::vector<std::string> prompt_languages() const { return {}; }

    // Embedded SentencePiece tokenizer, shared by both word-boosting paths
    // (flashlight OOV phrases, RNNT context biasing). Null if not embedded.
    const sentencepiece::SentencePieceProcessor* embedded_tokenizer() const { return spm_.get(); }

   protected:
    struct Common {
        std::unique_ptr<ggml_runtime::GGUFLoader> loader;
        std::string ns;  // metadata namespace ("asr" / "parakeet-ctc")
        std::string model_name;
        EncoderConfig enc_cfg;
        MelSpecConfig fe_cfg;
        std::vector<std::string> vocab;
    };
    AsrModel(ggml_runtime::BackendManager& bm, Common&& c, const BatchingConfig& batching = {});

    // Apply the model's serialized mel filterbank to an additional frontend
    // (the base owns the streaming FE; CTC also owns an offline GPU FE).
    void apply_model_mel_basis(MelSpectrogramExtractor& extractor);

    // Shared with derived constructors without reopening the GGUF.
    ggml_runtime::GGUFLoader* loader() { return loader_.get(); }

    ggml_runtime::BackendManager* bm_ = nullptr;
    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    std::string ns_;
    std::string model_name_;
    EncoderConfig enc_cfg_;
    MelSpecConfig fe_cfg_;
    std::vector<std::string> vocab_;
    // SentencePiece tokenizer loaded from the GGUF's embedded model proto
    // (asr.tokenizer.spm_model); tokenizes word-boosting phrases exactly as the
    // model does. Null when the GGUF predates the embedded tokenizer.
    std::unique_ptr<sentencepiece::SentencePieceProcessor> spm_;
    std::unique_ptr<MelSpectrogramExtractor> fe_;
};

// Full-context encoder and CTC classifier in one Session.
class CtcModel final : public AsrModel {
   public:
    CtcModel(ggml_runtime::BackendManager& bm, Common&& c, const BatchingConfig& batching = {});
    ~CtcModel() override;

    HeadKind head_kind() const override { return HeadKind::Ctc; }
    const CtcConfig& ctc_config() const { return ctc_cfg_; }

    // Run the encoder + CTC head graph on one audio window. `audio` is float32
    // mono at sample_rate(). Returns log-probabilities of shape (n_classes,
    // T_out) flat-packed (frame t starts at t*n_classes). Thread-safe.
    void infer_ctc(
        const float* audio, size_t n_samples, std::vector<float>& out_log_probs, int& T_out,
        int& n_classes);

    void infer_ctc_greedy(
        const float* audio, size_t n_samples, std::vector<int32_t>& best_ids,
        std::vector<float>& best_probs, int& T_out);

    // `feats` is frame-major (n_mels, n_frames).
    void infer_ctc_from_mel(
        const float* feats, int n_frames, std::vector<float>& out_log_probs, int& T_out,
        int& n_classes);

    // Greedy-only compact result: device argmax + winning softmax probability
    // per encoder frame. Flashlight continues to use the full CTC distribution.
    void infer_ctc_greedy_from_mel(
        const float* feats, int n_frames, std::vector<int32_t>& best_ids,
        std::vector<float>& best_probs, int& T_out);

    std::vector<DiagSession> diagnostic_sessions() const override;
    BatchMetrics batch_metrics() const;
    BatchMetrics offline_frontend_batch_metrics() const;
    bool offline_frontend_uses_gpu() const;

   private:
    class CTCEncoderClassifier;
    class CtcBatcher;
    CtcConfig ctc_cfg_;
    std::unique_ptr<MelSpectrogramExtractor> offline_fe_;
    std::unique_ptr<FastConformerEncoder> encoder_;
    std::unique_ptr<CtcHeadModule> ctc_head_;
    std::unique_ptr<CTCEncoderClassifier> ctc_enc_classifier_;
    std::unique_ptr<ggml_runtime::Session> session_;
    std::unique_ptr<CtcBatcher> batcher_;
};

// Cache-aware encoder with RNNT/TDT predictor and joint networks.
class RnntModel final : public AsrModel, public RnntEngine {
   public:
    RnntModel(ggml_runtime::BackendManager& bm, Common&& c, const BatchingConfig& batching = {});
    ~RnntModel() override;

    HeadKind head_kind() const override {
        return rnnt_cfg_.is_tdt() ? HeadKind::Tdt : HeadKind::Rnnt;
    }

    const RnntConfig& rnnt_config() const override { return rnnt_cfg_; }
    const std::vector<std::string>& vocab() const override { return AsrModel::vocab(); }
    // Predictor and joint-tail stages. The predictor result is cached by the
    // greedy decoder across blank frames; joint_argmax evaluates a whole run of
    // already-projected encoder frames in one graph.
    std::unique_ptr<RnntStreamState> make_rnnt_stream_state() override;
    void begin_decode_step() override;
    void end_decode_step() override;
    void predict_rnnt(RnntStreamState& state, int prev_token, int active_bank) override;
    void joint_argmax(
        RnntStreamState& state, const float* enc_proj, int joint_dim, int T, int32_t* token_ids,
        const float* logit_bias = nullptr) override;
    void joint_argmax_device(
        RnntStreamState& state, const ggml_runtime::DeviceTensor& enc_proj, int frame_offset,
        int joint_dim, int T, int32_t* token_ids, const float* logit_bias = nullptr) override;
    void predict_and_joint_rnnt_argmax_device(
        RnntStreamState& state, int prev_token, int active_bank,
        const ggml_runtime::DeviceTensor& enc_proj, int frame_offset, int joint_dim, int T,
        int32_t* token_ids, const float* logit_bias = nullptr) override;
    void joint_tdt_argmax(
        RnntStreamState& state, const float* enc_proj, int joint_dim, int T, int32_t* token_ids,
        int32_t* duration_ids) override;
    void predict_and_joint_tdt_argmax(
        RnntStreamState& state, int prev_token, int active_bank, const float* enc_proj,
        int joint_dim, int32_t* token_id, int32_t* duration_id) override;

    std::unique_ptr<Decoder> make_transducer_decoder(const DecoderConfig& cfg = {});

    // Tokenize a word-boosting phrase with the embedded SentencePiece model.
    // Empty when the GGUF carries no tokenizer model (boosting then no-ops).
    std::vector<int> encode_phrase(const std::string& text) const override;

    // Full-utterance frontend + non-cache-aware encoder + joint encoder
    // projection. Compatible requests batch as [feature,time,batch].
    void infer_offline(
        const float* audio, size_t n_samples, std::vector<float>& enc_out, int& T_enc,
        int prompt_index = -1);
    void infer_offline_from_mel(
        const float* feats, int n_frames, std::vector<float>& enc_out, int& T_enc,
        int prompt_index = -1);
    // S2S models carry an additional encoder projection (`proj.*`) alongside
    // the RNNT joint encoder projection. Run both tails from one shared,
    // dynamically batched FastConformer graph so perception is never encoded
    // twice and no raw encoder activation is staged through host memory.
    bool has_s2s_projection() const { return s2s_projection_dim_ > 0; }
    int s2s_projection_dim() const { return s2s_projection_dim_; }
    void infer_s2s_from_mel(
        const float* feats, int n_frames, std::vector<float>& s2s_out, std::vector<float>& rnnt_out,
        int& T_enc);
    BatchMetrics offline_encoder_batch_metrics() const;
    BatchMetrics offline_frontend_batch_metrics() const;
    bool offline_frontend_uses_gpu() const;
    bool supports_cache_streaming() const { return enc_cfg_.cache_supported; }

    // Must be called before cache state creation or the first encode.
    void set_cache_right_ctx(int R);

    // Allocates a zero-initialized per-stream cache row.
    CacheAwareEncoder::State make_cache_state();
    void reset_cache_state(CacheAwareEncoder::State& state);

    // Runs one encoder step and updates state in place. Mel input is frame-major;
    // the mask spans cache_left_ctx + 1 + R, and prompt_index=-1 disables fusion.
    void encode_cache_aware(
        CacheAwareEncoder::State& state, const float* mel, int n_mel_frames, const float* attn_mask,
        int attn_mask_len, std::vector<float>& enc_out, int& T_enc, int prompt_index = -1,
        ggml_runtime::DeviceTensor* device_output = nullptr);

    std::vector<DiagSession> diagnostic_sessions() const override;
    BatchMetrics encoder_batch_metrics() const;
    BatchMetrics predictor_batch_metrics() const;
    BatchMetrics joint_batch_metrics() const;

    bool has_prompt() const override { return num_prompts_ > 0; }
    int prompt_index_for_lang(const std::string& lang) const override;
    std::vector<std::string> prompt_languages() const override {
        std::vector<std::string> out;
        out.reserve(prompt_dictionary_.size());
        for (const auto& kv : prompt_dictionary_) out.push_back(kv.first);
        return out;
    }

   private:
    // Predictor + joint-tail stages share the decoder Session. Encoder-side
    // joint weights and optional prompt fusion live in the cache encoder Session.
    class RnntDecoderStages;
    class DecoderBatchers;
    class RnntEncoderTail;
    class RnntDecoderState;
    class OfflineEncoderRoot;
    class OfflineEncoderBatcher;
    class S2SProjectionTail;
    RnntConfig rnnt_cfg_;
    std::unique_ptr<RnntPredictorModule> rnnt_predictor_;
    std::unique_ptr<RnntJointModule> rnnt_joint_;

    // Prompt metadata + fusion module for prompt-conditioned (multilingual)
    // RNNT models. It runs in rnnt_encoder_tail_; num_prompts_ == 0 disables it.
    int num_prompts_ = 0;
    std::unordered_map<std::string, int> prompt_dictionary_;
    std::unique_ptr<PromptFusionModule> prompt_fusion_;
    std::unique_ptr<RnntEncoderTail> rnnt_encoder_tail_;
    std::unique_ptr<RnntDecoderStages> rnnt_decoder_stages_;
    std::unique_ptr<ggml_runtime::Session> decoder_session_;
    std::unique_ptr<DecoderBatchers> decoder_batchers_;
    int decoder_arena_slots_ = 0;
    std::mutex decoder_slots_mu_;
    std::condition_variable decoder_slots_cv_;
    std::vector<bool> decoder_slots_used_;
    std::vector<bool> decoder_slots_need_reset_;
    // Steps released by the admission barrier. This excludes the next ingress
    // wave while it is still collecting, so the decoder scheduler knows how
    // many streams can actually declare ready work.
    std::atomic<int> admitted_decode_steps_{0};
    // One admission barrier per host decode step. It spends the queue budget
    // once to align a newly-arrived wave; predictor/joint iterations can then
    // batch immediately instead of fragmenting into permanently offset cohorts.
    std::mutex decode_admission_mu_;
    std::condition_variable decode_admission_cv_;
    uint64_t decode_admission_generation_ = 0;
    int decode_admission_waiting_ = 0;
    int decode_admission_target_ = 0;
    std::chrono::steady_clock::time_point decode_admission_deadline_;
    int acquire_decoder_slot(bool& needs_reset);
    void release_decoder_slot(int slot);
    void zero_decoder_slot(int slot);
    void zero_decoder_slots(std::vector<int> slots);

    // Cache-aware streaming encoder: owns the one shared device-resident encoder
    // Session. Each stream owns an indexed row in its persistent cache arena;
    // make_cache_state / encode_cache_aware / set_cache_right_ctx delegate to it.
    std::unique_ptr<CacheAwareEncoder> cache_encoder_;

    // Lazily initialized so streaming-only deployments do not duplicate all
    // encoder weights. Once used, full-utterance calls share one batched
    // frontend and one offline encoder Session.
    std::unique_ptr<MelSpectrogramExtractor> offline_fe_;
    std::unique_ptr<FastConformerEncoder> offline_encoder_;
    std::unique_ptr<S2SProjectionTail> s2s_projection_;
    int s2s_projection_dim_ = 0;
    std::unique_ptr<OfflineEncoderRoot> offline_encoder_root_;
    std::unique_ptr<ggml_runtime::Session> offline_encoder_session_;
    std::unique_ptr<OfflineEncoderBatcher> offline_encoder_batcher_;
    BatchingConfig batching_cfg_;
    mutable std::mutex offline_init_mu_;
    void ensure_offline_path();
};

}  // namespace nemo_speech::asr
