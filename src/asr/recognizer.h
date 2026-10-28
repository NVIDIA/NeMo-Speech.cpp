// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Shared ASR entry point for in-process and transport adapters. Recognizer is
// thread-safe; each RecognitionStream is driven by one caller thread.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
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
    int gpu = 0;      // -1 = CPU
    int threads = 0;  // CPU backend threads; 0 = min(8, hardware threads)
    void Register(common::ParameterParser& p) {
        p.Register("gpu", &gpu, "GPU device index (-1 = CPU)", {"--gpu", "-g"});
        p.Register("threads", &threads, "CPU backend threads (0 = min(8, hardware threads))");
    }
};

// SALM is selected by an encoder path; model.path supplies its LLM GGUF.
struct SalmBackendConfig {
    std::string encoder_path;
    // The CTC timestamp head: the encoder file when it carries one (resolve_salm_encoder); not a
    // setting.
    std::string ctc_path;
    // Defaults sized for a 32 GB GPU; raise max_sequences and the batch sizes on larger ones.
    int max_sequences = 16;
    int n_ctx_per_sequence = 8192;
    int max_context = 73728;
    int n_batch = 2048;
    int encoder_max_batch = 16;
    int encoder_ahead_seconds = 20000;
    int max_new_tokens = 1024;
    int mtp_draft = 7;
    std::string dflash_path;
    void Register(common::ParameterParser& p) {
        p.Register(
            "encoder_path", &encoder_path,
            "SALM encoder GGUF (selects SALM; default: the encoder file the decoder names)",
            {"--salm-encoder"});
        p.Register("max_sequences", &max_sequences, "SALM concurrent decode slots");
        p.Register(
            "n_ctx_per_sequence", &n_ctx_per_sequence,
            "SALM KV cells per decode slot (one pool for all slots)");
        p.Register(
            "max_context", &max_context,
            "SALM longest sequence in tokens (prompt + audio + output)");
        p.Register("n_batch", &n_batch, "SALM prefill token budget");
        p.Register("encoder_max_batch", &encoder_max_batch, "SALM encoder windows per batch");
        p.Register(
            "encoder_ahead_seconds", &encoder_ahead_seconds,
            "SALM audio seconds encoded ahead of admission, held on the device (0 = unbounded)");
        p.Register("max_new_tokens", &max_new_tokens, "SALM generation token limit");
        p.Register("mtp_draft", &mtp_draft, "SALM speculative draft tokens (0 disables MTP)");
        p.Register(
            "dflash_path", &dflash_path,
            "SALM DFlash2 drafter GGUF: drafts mtp_draft tokens per step instead of MTP");
    }
};

// Complete-audio backend extension. Implementations own scheduling and must be
// thread-safe. This keeps ASR independent of the optional speechlm library.
class OfflineRecognitionBackend {
   public:
    virtual ~OfflineRecognitionBackend() = default;
    virtual Result recognize(const float*, size_t, AsrRequestOptions, const std::string&, int) = 0;
    virtual void warmup() = 0;
    virtual int sample_rate() const = 0;
    virtual bool supports_word_timestamps() const = 0;
    virtual bool supports_speaker_diarization() const = 0;
    // Speaker turns alone, without a transcript (only with supports_offline_diarization()):
    // seconds, speaker = the diarizer's column.
    virtual bool supports_offline_diarization() const { return false; }
    virtual std::vector<SpeechLmSegment> diarize(const float*, size_t, int) { return {}; }
    // Speech translation (AsrRequestOptions::speechlm.task = "translate") and its target languages.
    virtual bool supports_translation() const { return false; }
    virtual std::vector<std::string> translation_languages() const { return {}; }
    // Client instructions and sampled decoding (AsrRequestOptions::speechlm instructions,
    // temperature and seed).
    virtual bool supports_instructions() const { return false; }
    virtual std::vector<std::string> supported_languages() const = 0;
    virtual BatchMetrics batch_metrics() const = 0;
};

// Startup configuration; per-request options live in AsrRequestOptions.
struct RecognizerConfig {
    BackendConfig backend;
    SalmBackendConfig salm;
    BatchingConfig batching;
    ModelConfig model;
    StreamingConfig streaming;  // chunk/padding geometry + cache-aware R
    DecoderConfig decoder;
    VadConfig vad;
    VadEndpointerCfg endpointing;
    DiarConfig diar;  // speaker diarization sidecar (Sortformer)
    postproc::PostprocConfig postproc;
    // Human-readable model and execution summaries. The CLI disables these
    // for --quiet and --json; verbose implementation logs are separate.
    bool log_status = true;

    void Register(common::ParameterParser& p) {
        p.Register("backend", backend);
        p.Register("salm", salm);
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
    // Interim results also carry word timings (when requested) and speaker tags.
    void set_interim_words(bool on) { interim_words_ = on; }
    // Retag buffered words from the diarizer's revised timeline.
    // Speaker labels strictly before this audio time are immutable.
    // Excludes provisional chunks and retrospective relabeling.
    double stable_speaker_time() const;
    void refresh_speaker_tags(Result& result) const;

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
    bool interim_words_ = false;
    // Late punctuation from steps that returned no Result; sent with the next.
    std::string late_punctuation_;
};

// A speech-LLM decoder GGUF names its encoder file (speech encoder and CTC timestamp head) in
// nemo_speech.speech_model, a file in the same directory: a SALM encoder path left empty names it,
// which selects SALM, and a CTC path left empty names the encoder file when it holds the head.
// Throws std::invalid_argument when the named encoder file is missing, or when the decoder and the
// encoder file carry different nemo_speech.checkpoint_id. Reads only GGUF headers; a file that is
// not a GGUF is left to the model loader.
void resolve_salm_encoder(RecognizerConfig& cfg);

class Recognizer {
   public:
    explicit Recognizer(RecognizerConfig config);
    Recognizer(RecognizerConfig config, std::unique_ptr<OfflineRecognitionBackend> backend);
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
    BatchMetrics pnc_batch_metrics() const;
    std::vector<std::pair<std::string, BatchMetrics>> batch_metrics() const;
    int sample_rate() const;
    bool supports_streaming() const;
    bool supports_word_timestamps() const;
    bool supports_speaker_diarization() const;
    bool supports_translation() const;
    // A complete-audio backend that takes client instructions and sampled decoding (a speech-LLM).
    bool supports_instructions() const;
    // A complete-audio backend that diarizes without transcribing (the speech-LLM's encoder);
    // diarize() throws without one.
    bool supports_offline_diarization() const;
    std::vector<SpeechLmSegment> diarize(const float* samples, size_t n, int sample_rate) const;
    std::vector<std::string> translation_languages() const;
    const std::string& model_name() const { return model_name_; }
    std::vector<std::string> supported_languages() const;
    double ms_per_enc_frame() const;
    const RecognizerConfig& config() const { return cfg_; }

   private:
    friend class RecognitionStream;
    postproc::Postprocessor& postproc() { return *postproc_; }
    void log_model_status() const;
    void log_execution_status(bool streaming) const;
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
    std::unique_ptr<OfflineRecognitionBackend> offline_backend_;
    std::unique_ptr<ggml_runtime::BackendManager> bm_;
    std::shared_ptr<AsrModel> model_;
    std::unique_ptr<DiarModel> diar_model_;  // loaded iff cfg_.diar.model_path set
    std::string model_name_;
    RecognizerConfig cfg_;
    IngressBatchCoordinator streaming_ingress_batches_;
    IngressBatchCoordinator offline_ingress_batches_;
    std::atomic<int> active_offline_requests_{0};
    std::atomic<int> active_streaming_ingress_{0};
    // Shared by CTC flashlight streams; null for other decoders.
    std::shared_ptr<const FlashlightResources> flashlight_resources_;
    std::unique_ptr<postproc::Postprocessor> postproc_;
    // Shared immutable VAD model; each stream owns recurrent state.
    std::shared_ptr<SileroVadModel> vad_model_;
    mutable std::once_flag offline_status_once_;
    mutable std::once_flag streaming_status_once_;
};

}  // namespace nemo_speech::asr
