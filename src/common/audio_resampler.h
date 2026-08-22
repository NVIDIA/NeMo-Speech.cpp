// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Stateful mono float32 sample-rate conversion shared by speech pipelines. The
// precomputed windowed-sinc filter prevents aliases while retained history makes
// output independent of streaming chunk sizes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nemo_speech::audio {

// Matches the input-rate range accepted by Riva speech services.
bool supported_input_sample_rate(int sample_rate);

class AudioResampler {
   public:
    AudioResampler(int input_rate, int output_rate);

    int input_rate() const { return input_rate_; }
    int output_rate() const { return output_rate_; }

    // Append converted samples to `output`. Calls may use arbitrary chunk sizes.
    void process(const float* input, size_t count, std::vector<float>* output);

    // Emit the filter tail. No process() calls are allowed after finish().
    void finish(std::vector<float>* output);

   private:
    void emit_available(bool final, std::vector<float>* output);
    float filtered_sample(uint64_t center, size_t phase) const;
    float input_sample(int64_t index) const;
    void compact_history();
    void advance_source_position();
    size_t phase_index() const;

    int input_rate_;
    int output_rate_;
    uint64_t rate_gcd_ = 1;
    size_t phase_count_ = 1;
    bool exact_phase_table_ = true;
    std::vector<float> coefficients_;
    std::vector<float> input_;
    uint64_t input_start_ = 0;
    uint64_t total_input_ = 0;
    uint64_t next_output_ = 0;
    uint64_t source_index_ = 0;
    uint64_t source_remainder_ = 0;
    bool finished_ = false;
};

// Stateful little-endian signed-16 PCM wrapper around AudioResampler.
class Pcm16Resampler {
   public:
    Pcm16Resampler(int input_rate, int output_rate);

    // Each input chunk must contain complete 16-bit samples. Converted bytes
    // are appended to `output` and may be empty while filter lookahead builds.
    void process(const uint8_t* input, size_t bytes, std::vector<uint8_t>* output);
    void finish(std::vector<uint8_t>* output);

    uint64_t output_samples() const { return output_samples_; }

   private:
    void append_pcm(const std::vector<float>& samples, std::vector<uint8_t>* output);

    AudioResampler resampler_;
    std::vector<float> decoded_;
    std::vector<float> converted_;
    uint64_t output_samples_ = 0;
};

std::vector<float> resample_audio(
    const float* input, size_t count, int input_rate, int output_rate);

}  // namespace nemo_speech::audio
