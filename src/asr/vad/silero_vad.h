// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Silero VAD inference primitive (ggml runtime Module).
//
// Ported from whisper.cpp (src/whisper.cpp:4519-4653): STFT front-end
// (reflect-pad → Fourier-basis conv → magnitude), 4-layer Conv1D encoder,
// LSTMCell, pointwise conv → sigmoid. One speech probability per 512-sample
// (32 ms @ 16 kHz) window. Runs on the shared BackendManager (GPU or CPU).
//
// Per-stream: LSTM h/c state occupies an indexed row in the shared device
// arena; a <512-sample audio carry-over lives in each SileroVad instance.
#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "batching.h"
#include "runtime.h"

namespace nemo_speech::asr {

// Hyperparameters of the silero-v6 16 kHz model. Scalars are read from the
// sidecar GGUF's vad.* metadata; the per-layer encoder channel/stride lists
// are fixed for this architecture (and must match convert_model.py).
struct SileroVadConfig {
    int sample_rate = 16000;
    int window_size = 512;         // samples per inference window (one probability)
    int context_size = 64;         // reflect-pad each side before the STFT conv
    int lstm_hidden = 128;         // LSTM cell + hidden width
    int stft_filter_length = 256;  // STFT basis kernel length
    int stft_n_basis = 258;        // 2 * n_freqs (real ++ imag output channels)
    int n_freqs = 129;             // stft_n_basis / 2

    int n_encoder_layers = 4;
    // Conv1D encoder geometry. Layers 1 and 2 stride-2 (downsample the 4-frame
    // STFT output to a single frame); all kernels are size 3 with pad 1.
    std::vector<int> enc_in_channels{129, 128, 64, 64};
    std::vector<int> enc_out_channels{128, 64, 64, 128};
    std::vector<int> enc_strides{1, 2, 2, 1};
    int enc_kernel = 3;
};

class SileroVadModule;  // ggml Module impl, defined in silero_vad.cpp

// Immutable VAD weights + graph cache shared by every recognizer stream.  The
// recurrent h/c buffers are indexed persistent rows allocated per SileroVad,
// so sharing this object never shares speech state between streams.
class SileroVadModel {
   public:
    class State {
       public:
        State() = default;
        ~State();
        State(State&& other) noexcept;
        State& operator=(State&& other) noexcept;
        State(const State&) = delete;
        State& operator=(const State&) = delete;

       private:
        friend class SileroVadModel;
        State(SileroVadModel* owner, int slot, bool needs_reset)
            : owner_(owner), slot_(slot), needs_reset_(needs_reset) {}
        SileroVadModel* owner_ = nullptr;
        int slot_ = -1;
        bool needs_reset_ = false;
    };

    SileroVadModel(
        ggml_runtime::BackendManager& bm, const std::string& gguf_path,
        const BatchingConfig& batching = {});
    ~SileroVadModel();

    SileroVadModel(const SileroVadModel&) = delete;
    SileroVadModel& operator=(const SileroVadModel&) = delete;

    const SileroVadConfig& config() const { return cfg_; }
    State make_state();
    void reset_state(State& state);
    void infer(State& state, const float* frames, int n_windows, std::vector<float>& probabilities);
    BatchMetrics batch_metrics() const;

   private:
    SileroVadConfig cfg_;
    ggml_runtime::BackendManager* bm_ = nullptr;
    int arena_slots_ = 0;
    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    std::unique_ptr<SileroVadModule> module_;
    std::unique_ptr<ggml_runtime::Session> session_;
    class VadBatcher;
    std::unique_ptr<VadBatcher> batcher_;
    mutable std::mutex slots_mu_;
    std::vector<bool> slots_used_;
    std::vector<bool> slots_need_reset_;
    void release_slot(int slot);
    void zero_slot(int slot);
    void zero_slots(std::vector<int> slots);
};

class SileroVad {
   public:
    // Opens the sidecar VAD GGUF (general.architecture="vad") via its own
    // GGUFLoader and builds a Session against the shared `bm`. Throws if the
    // file isn't a VAD GGUF or a tensor is missing.
    SileroVad(ggml_runtime::BackendManager& bm, const std::string& gguf_path);
    explicit SileroVad(std::shared_ptr<SileroVadModel> model);
    ~SileroVad();

    SileroVad(const SileroVad&) = delete;
    SileroVad& operator=(const SileroVad&) = delete;

    // Window `samples` into 512-sample frames, append one probability per
    // completed window to `out_probs`, and carry any trailing (<512) samples
    // over to the next call. Returns the number of windows consumed.
    int feed_audio(const float* samples, size_t n_samples, std::vector<float>& out_probs);

    // Flush a trailing partial window (zero-padded to window_size) at
    // end-of-stream. Appends one probability and returns 1 if audio was
    // pending, else 0. Leaves the carry-over empty.
    int flush(std::vector<float>& out_probs);

    // Clear LSTM h/c state + audio carry-over for a fresh utterance. Also
    // clears the speech/silence timeline.
    void reset();

    int window_size() const { return cfg_.window_size; }
    const SileroVadConfig& config() const { return cfg_; }

    // Speech/silence timeline shared by feature masking and endpointing.
    //
    // The same Silero inference feeds both: observe_audio() runs the model on
    // completed windows AND binarizes each probability (onset/offset hysteresis)
    // into a per-global-mel-frame speech bit. VadMasker reads this for masking;
    // VadEndpointer reads it for VAD-driven EOU. Nothing is computed twice.
    //
    // Configure once before the first observe_audio(): `hop_length` is the FE
    // mel hop (maps 512-sample windows onto mel frames); onset/offset are the
    // binarization thresholds.
    void set_binarizer(float onset, float offset, int hop_length);
    // Feed `n` freshly-available samples (monotonic, in stream order): run
    // inference on completed windows and record their speech bits.
    void observe_audio(const float* samples, size_t n);
    // Binarize the trailing partial window at end-of-stream.
    void flush_timeline();
    // Raw speech bit for global mel frame g (recorded value, or the provisional
    // current state for not-yet-decided frames; false for g < 0 or below a
    // discarded base).
    bool frame_speech(int64_t g) const;
    // Number of mel frames whose covering window has been binarized.
    int64_t committed_frames() const { return speech_base_ + static_cast<int64_t>(speech_.size()); }
    // Most recent committed speech mel frame (inclusive), or -1 if none yet.
    int64_t last_speech_frame() const { return last_speech_frame_; }
    int hop_length() const { return hop_; }
    // Drop recorded speech bits for frames < g (the runner trims once its
    // decode cursor has safely passed them, bounding timeline memory on
    // endless streams). Frame indices stay global; only the storage base moves.
    void discard_timeline_before(int64_t g);

   private:
    // Run consecutive windows in one graph while preserving recurrent order.
    void run_windows(const float* frames, int n_windows, std::vector<float>& probabilities);
    // Binarize one window's probability (global window index k): update the
    // onset/offset hysteresis and record it across the mel frames the window
    // covers (floor(k*win/hop)..floor((k+1)*win/hop)).
    void binarize_window(float p, int64_t k);

    std::shared_ptr<SileroVadModel> model_;
    SileroVadConfig cfg_;
    // Per-stream recurrent state remains on the selected backend between
    // windows. A fresh handle is zero-initialized by Session::make_session_state.
    SileroVadModel::State recurrent_state_;
    std::vector<float> pending_audio_;  // <window_size carry-over
    // Reusable scratch for the zero-padded frame fed to run_window.
    std::vector<float> frame_;

    // Speech/silence timeline (binarizer). The defaults are placeholders: the
    // runners always call set_binarizer() (with VadMaskerCfg's canonical
    // onset/offset and the model FE's hop) before the first observe_audio().
    float onset_ = 0.5f;
    float offset_ = 0.3f;
    int hop_ = 160;  // FE mel hop in samples (window->mel-frame mapping)
    bool in_speech_ = false;
    int64_t windows_binarized_ = 0;
    int64_t last_speech_frame_ = -1;
    // Per-global-mel-frame speech bit for frames [speech_base_, speech_base_ +
    // speech_.size()); older frames are discarded by discard_timeline_before().
    std::vector<char> speech_;
    int64_t speech_base_ = 0;
    std::vector<float> prob_scratch_;  // reused by observe_audio/flush_timeline
};

}  // namespace nemo_speech::asr
