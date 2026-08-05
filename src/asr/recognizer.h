// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Shared ASR entry point for in-process and transport adapters. Recognizer is
// thread-safe; each RecognitionStream is driven by one caller thread.
#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "audio_resampler.h"
#include "batching.h"
#include "diar/diar_pipeline.h"
#include "parameter_parser.h"
#include "postproc/pipeline.h"
#include "runner.h"
#include "runtime.h"
#include "types.h"

namespace nemo_speech::asr {

class AsrModel;
class SileroVadModel;
class Recognizer;

struct BackendConfig {
    int gpu = 0;  // -1 = CPU
    void Register(common::ParameterParser& p) {
        p.Register("gpu", &gpu, "GPU device index (-1 = CPU)", {"--gpu", "-g"});
    }
};

// Startup configuration; per-request options live in AsrRequestOptions.
struct RecognizerConfig {
    BackendConfig backend;
    BatchingConfig batching;
    ModelConfig model;
    StreamingConfig streaming;  // chunk/padding geometry + cache-aware R
    DecoderConfig decoder;
    VadConfig vad;
    VadEndpointerCfg endpointing;
    DiarConfig diar;  // speaker diarization sidecar (Sortformer)
    postproc::PostprocConfig postproc;

    void Register(common::ParameterParser& p) {
        p.Register("backend", backend);
        p.Register("batching", batching);
        p.Register("model", model);
        p.Register("streaming", streaming);
        p.Register("decoder", decoder);
        p.Register("vad", vad);
        p.Register("endpointing", endpointing);
        p.Register("diar", diar);
        p.Register("postproc", postproc);
    }
};

// Single-threaded per-stream recognition with postprocessed Results.
class RecognitionStream {
   public:
    RecognitionStream(
        Recognizer* recognizer, std::unique_ptr<AsrRunner> runner, AsrRequestOptions options,
        bool coordinate_ingress);
    ~RecognitionStream();

    // Buffer mono float32 audio (no decode); next() drives decoding. A zero
    // sample rate means the model rate. The first non-empty push fixes the
    // stream's input rate; subsequent chunks must use the same rate.
    void push(const float* samples, size_t n, int sample_rate = 0);
    void force_endpoint();
    // Advance one decode step. Returns a final Result (is_final=true, possibly
    // empty transcript - the caller decides whether to emit) when an EOU fired
    // this step; an interim Result when the step advanced over new audio; or
    // nullopt when the buffered audio is exhausted (no progress this step). A
    // pull loop drains finals then a trailing interim and terminates on nullopt:
    // `while (auto r = next()) { ...; if (!r->is_final) break; }`.
    std::optional<Result> next();
    // No more audio: flush the tail and return the end-of-stream final Result.
    Result finish();

    const AsrRequestOptions& options() const { return opts_; }

   private:
    Result build_result_(const StreamingUpdate& u, bool is_final) const;
    // Extends diarization to cover all words in a final result.
    void flush_diar_deficit_(const StreamingUpdate& u);

    Recognizer* recognizer_;
    std::unique_ptr<AsrRunner> runner_;
    // Optional sidecar over the same model-rate audio as ASR.
    std::unique_ptr<DiarStream> diar_;
    AsrRequestOptions opts_;
    int input_sample_rate_ = 0;
    std::unique_ptr<audio::AudioResampler> resampler_;
    std::vector<float> resampled_audio_;
    bool resampler_flushed_ = false;
    bool coordinate_ingress_ = true;
    int pending_cohort_target_ = 0;
    // High-water mark of audio_processed_sec returned by next(). A step that
    // does not advance it processed no new window (audio exhausted), so next()
    // reports nullopt and a drain loop terminates. Monotonic per stream.
    float last_processed_sec_ = -1.0f;
};

class Recognizer {
   public:
    explicit Recognizer(RecognizerConfig config);
    ~Recognizer();

    Recognizer(const Recognizer&) = delete;
    Recognizer& operator=(const Recognizer&) = delete;

    // An empty language code selects the model default prompt. Network
    // transports can opt into ingress coordination; direct/library callers
    // avoid its queue-delay cost by default.
    std::unique_ptr<RecognitionStream> streaming_recognize(
        AsrRequestOptions opts, const std::string& language_code, bool coordinate_ingress = false);
    Result recognize(
        const float* samples, size_t n, AsrRequestOptions opts, const std::string& language_code,
        int sample_rate = 0);

    // Construct a buffered CTC or cache-aware transducer streaming runner.
    std::unique_ptr<AsrRunner> make_runner() const;

    // Build lazy graph shapes and cache-aware state before first use.
    void warmup();

    AsrModel* model() const { return model_.get(); }
    DiarModel* diar_model() const { return diar_model_.get(); }
    BatchMetrics vad_batch_metrics() const;
    int sample_rate() const;
    const std::string& model_name() const { return model_name_; }
    std::vector<std::string> supported_languages() const;
    double ms_per_enc_frame() const;
    postproc::Postprocessor& postproc() { return *postproc_; }
    const RecognizerConfig& config() const { return cfg_; }

   private:
    friend class RecognitionStream;
    void register_streaming_ingress() {
        active_streaming_ingress_.fetch_add(1, std::memory_order_relaxed);
    }
    void unregister_streaming_ingress() {
        active_streaming_ingress_.fetch_sub(1, std::memory_order_relaxed);
    }
    int arrive_streaming_ingress() {
        return streaming_ingress_batches_.arrive(
            active_streaming_ingress_.load(std::memory_order_relaxed));
    }

    // BackendManager first (model + postproc load against it; reverse-order
    // destruction). model_ is built in the ctor body after bm_.
    std::unique_ptr<ggml_runtime::BackendManager> bm_;
    std::shared_ptr<AsrModel> model_;
    std::unique_ptr<DiarModel> diar_model_;  // loaded iff cfg_.diar.model_path set
    std::string model_name_;
    RecognizerConfig cfg_;
    IngressBatchCoordinator streaming_ingress_batches_;
    IngressBatchCoordinator offline_ingress_batches_;
    std::atomic<int> active_streaming_ingress_{0};
    // Shared by CTC flashlight streams; null for other decoders.
    std::shared_ptr<const FlashlightResources> flashlight_resources_;
    std::unique_ptr<postproc::Postprocessor> postproc_;
    // Shared immutable VAD model; each stream owns recurrent state.
    std::shared_ptr<SileroVadModel> vad_model_;
};

}  // namespace nemo_speech::asr
