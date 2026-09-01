// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// S2S codec facade: streaming codes->wav decode with per-stream causal-conv
// and spec-overlap state, plus the one-shot silence-code computation used by
// the TTS EOS substitution.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "config.h"
#include "decoder.h"
#include "encoder.h"
#include "runtime.h"

namespace nemo_speech::s2s {

class S2SCodec {
   public:
    S2SCodec(ggml_runtime::BackendManager& bm, const std::string& gguf_path);
    ~S2SCodec();

    S2SCodec(const S2SCodec&) = delete;
    S2SCodec& operator=(const S2SCodec&) = delete;

    const CodecConfig& config() const { return cfg_; }
    int sample_rate() const { return cfg_.sample_rate; }
    int samples_per_frame() const { return cfg_.samples_per_frame(); }

    // Streaming decode of one stream's new frames. codes: row-major
    // (n_frames, Q) int32. Appends wav samples (22050 Hz) to out_wav.
    // Per-stream conv caches + spec tails thread across calls.
    void decode_wav(
        int64_t stream_id, const int32_t* codes, int n_frames, std::vector<float>& out_wav);

    void drop_stream(int64_t stream_id);

    // TTS silence codes: encode a zero waveform of 10 frames, take the
    // most common 31-tuple (mirrors model.py's _compute_tts_silence_codes).
    // For zero audio the STFT is identically zero, so the encoder input is
    // a zeros spec of the matching length.
    std::vector<int32_t> compute_silence_codes();

   private:
    struct StreamState {
        std::vector<std::vector<float>> per_block_cache;  // [n_blocks][(k-1)*C]
        std::vector<float> tail_re, tail_im;              // [n_bins * tail_len]
    };
    StreamState& get_or_create_stream(int64_t id);

    CodecConfig cfg_;
    int tail_len_ = 0;
    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    std::unique_ptr<CodecDecodeWavModule> decode_module_;
    std::unique_ptr<ggml_runtime::Session> decode_session_;
    std::unique_ptr<CodecEncodeModule> encode_module_;
    std::unique_ptr<ggml_runtime::Session> encode_session_;
    ggml_runtime::BackendManager* bm_ = nullptr;

    std::unordered_map<int64_t, StreamState> streams_;
    std::mutex streams_mutex_;

    // Per-n_frames envelope reciprocal (host), keyed by n_frames.
    std::unordered_map<int, std::vector<float>> envelope_cache_;
    std::vector<float> compute_envelope_recip(int n_frames, int* out_trim) const;
};

}  // namespace nemo_speech::s2s
