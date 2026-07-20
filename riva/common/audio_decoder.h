// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Streaming mono signed-16 PCM decoder shared by every speech interface.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace nemo_speech::audio {

// Decodes raw little-endian signed-16 PCM and the PCM payload of a RIFF/WAVE
// stream. Input may be split at arbitrary byte boundaries, including inside
// the WAV header or a 16-bit sample.
class Pcm16StreamDecoder {
   public:
    // configured_sample_rate is 0 when the rate should come from a WAV header
    // or be selected by the consuming model for headerless PCM.
    explicit Pcm16StreamDecoder(int configured_sample_rate = 0);

    void process(const uint8_t* bytes, size_t size, std::vector<float>* output);
    void process(std::string_view bytes, std::vector<float>* output);

    // Completes header detection and rejects a truncated WAV header or sample.
    void finish(std::vector<float>* output);

    // The WAV rate when present, otherwise the configured rate (possibly 0).
    int sample_rate() const { return sample_rate_; }
    bool is_wav() const { return mode_ == Mode::Wav; }

   private:
    enum class Mode { Detect, Raw, Wav };

    bool parse_header(std::vector<float>* output);
    void decode_pcm(const uint8_t* bytes, size_t size, std::vector<float>* output);
    void decode_wav_payload(const uint8_t* bytes, size_t size, std::vector<float>* output);

    Mode mode_ = Mode::Detect;
    int configured_sample_rate_ = 0;
    int sample_rate_ = 0;
    std::vector<uint8_t> header_;
    size_t header_offset_ = 12;
    size_t wav_data_remaining_ = 0;
    bool saw_format_ = false;
    bool have_carry_ = false;
    uint8_t carry_ = 0;
    bool finished_ = false;
};

}  // namespace nemo_speech::audio
