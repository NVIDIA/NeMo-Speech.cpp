// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// NeMo-compatible log-mel extraction. GPU execution uses a cached ggml graph;
// CPU execution uses a radix-2 FFT. Both return the same frame-major layout.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "batching.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace ggml_runtime {
class Session;
}
class FeModule;

struct MelSpecConfig {
    int sample_rate = 16000;
    int n_fft = 512;
    int n_mels = 80;
    float window_size = 0.025f;   // seconds
    float window_stride = 0.01f;  // seconds
    float preemph = 0.97f;
    // NeMo uses the string value "per_feature" for utterance-level z-score
    // normalization and "NA" for frame-local features (notably the
    // cache-aware Nemotron models). Keep this in the shared frontend config so
    // both CPU and GPU paths honor the model contract.
    bool normalize_per_feature = true;
    // NeMo AudioToMelSpectrogramPreprocessor default log_zero_guard_value = 2^-24
    // (log_zero_guard_type="add"): log(mel + 2^-24). Using 1e-5 here floored
    // silence ~5 nats too high, shifting low-energy mel bins enough to perturb
    // the cache-aware encoder and flip marginal first-word language decisions.
    float log_zero_guard = 1.0f / 16777216.0f;  // 2^-24
    // mel filterbank range: 0..sr/2 by default
    float fmin = 0.0f;
    float fmax = 0.0f;  // 0 -> sr/2
    // Center the Hann window within the n_fft frame. false preserves the
    // right-aligned placement expected by legacy ASR GGUFs.
    bool stft_center_window = false;
    // Hann flavor: true = periodic (cos(2*pi*i/N), legacy ASR behavior),
    // false = symmetric (cos(2*pi*i/(N-1)) - torch.hann_window(periodic=False),
    // what NeMo's FilterbankFeatures actually uses).
    bool hann_periodic = true;
};

namespace ggml_runtime {
class BackendManager;
}

class MelSpectrogramExtractor {
   public:
    // A nullable manager enables GPU execution through the shared backend;
    // null selects the CPU implementation.
    explicit MelSpectrogramExtractor(
        const MelSpecConfig& cfg, ggml_runtime::BackendManager* bm = nullptr,
        const nemo_speech::asr::BatchingConfig& batching = {});
    ~MelSpectrogramExtractor();
    MelSpectrogramExtractor(const MelSpectrogramExtractor&) = delete;
    MelSpectrogramExtractor& operator=(const MelSpectrogramExtractor&) = delete;

    // Compute mel-spectrogram (n_mels, n_frames) row-major (outer=frames).
    // 'features' output is flattened as n_mels * n_frames (frame-major => feat_in=n_mels is the
    // inner dim to match ggml convention ne[0]=n_mels, ne[1]=T).
    //
    // Flags:
    //   reflect_left - true: pad the start with reflected samples (matches NeMo
    //     center=True). false: caller has prepended real left-context samples
    //     so frame 0's center sits at slice sample n_fft/2 (window covers
    //     slice[0..n_fft]). Used for streaming mid-stream calls.
    //   normalize - true: apply NeMo per_feature z-score over the frames in
    //     this call. false: leave raw log-mel values for the caller.
    void compute(
        const float* audio, size_t n_samples, std::vector<float>& features, int& n_frames,
        bool reflect_left = true, bool normalize = true) const;

    // Override the generated filterbank with the model's training-time basis.
    // `fb` is row-major (n_mels, n_fft / 2 + 1).
    void set_mel_basis(const float* fb, int n_mels, int n_bins);

    int win_length() const { return static_cast<int>(cfg_.window_size * cfg_.sample_rate + 0.5f); }
    int hop_length() const {
        return static_cast<int>(cfg_.window_stride * cfg_.sample_rate + 0.5f);
    }
    int n_fft() const { return cfg_.n_fft; }
    int n_mels() const { return cfg_.n_mels; }
    bool uses_gpu() const { return session_ != nullptr; }
    nemo_speech::asr::BatchMetrics batch_metrics() const;
    ggml_runtime::Session* diagnostic_session() const { return session_.get(); }

   private:
    struct BatchRequest {
        std::vector<float> audio;
        bool reflect_left = true;
        bool normalize = true;
    };
    struct BatchResult {
        std::vector<float> features;
        int n_frames = 0;
    };
    class GpuBatcher;

    MelSpecConfig cfg_;
    std::vector<float> window_;     // Hann window of length win_length
    std::vector<float> mel_basis_;  // (n_mels, n_fft/2 + 1) row-major

    void init_window();
    void init_mel_basis();

    // GPU log-mel for one window - runs through session_->run(). Output
    // layout matches the CPU path: `out` is resized to (n_mels * n_frames),
    // inner=n_mels (frame-major). `normalize=true` runs the host-side
    // per-feature z-score after download.
    void compute_gpu_via_session(
        const float* audio, size_t n_samples, std::vector<float>& out, int& n_frames,
        bool reflect_left, bool normalize) const;
    std::vector<BatchResult> compute_gpu_batch_via_session(
        std::vector<BatchRequest>&& requests) const;

    // Null on CPU-only execution.
    ggml_runtime::BackendManager* bm_ = nullptr;
    std::unique_ptr<FeModule> module_;
    std::unique_ptr<ggml_runtime::Session> session_;
    std::unique_ptr<GpuBatcher> batcher_;
    mutable std::mutex compute_mu_;  // serialises session_->run() across streams
};

// Read PCM16 or float32 WAV audio and downmix it to mono. Does not resample.
bool read_wav_mono_16k(const std::string& path, std::vector<float>& out_samples, int& out_sr);

// Streaming FE shared by the ASR runners and the diarizer pipeline: compute
// log-mel for frames newly covered by real audio since i_start. Only frames
// whose right edge is real samples are released (i*hop + n_fft/2 <=
// audio_end), so trailing frames near the frontier wait for more audio. NeMo
// center=true: reflect-pad the left only at the true stream start (frame
// centers within n_fft/2 of sample 0); mid-stream the real buffered left
// context is used (the caller's buffer trim must keep the FE's left edge).
// Returns the frames for [i_start, i_start+count) packed frame-major (n_mels
// each) in `out`, and count (0 if none ready). Callers append (and optionally
// VAD-mask) the returned raw frames themselves. Mel geometry (n_mels, hop,
// n_fft) is read from `fe` itself so it can never desync from the extractor
// that computes the frames.
int produce_new_mel_frames(
    MelSpectrogramExtractor& fe, const std::vector<float>& audio, size_t audio_base,
    int64_t i_start, std::vector<float>& out);
