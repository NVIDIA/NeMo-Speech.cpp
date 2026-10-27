// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Streaming PCM decoder shared by every speech interface: signed-16, mu-law and A-law, any channel
// count (downmixed by the integer mean, truncated toward zero), raw or in a RIFF/WAVE stream.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace nemo_speech::audio {

enum class StreamEncoding { Pcm16, Mulaw, Alaw };

// Decodes raw PCM (little-endian signed-16, or 8-bit G.711) and the payload of a
// RIFF/WAVE stream. Input may be split at arbitrary byte boundaries, including
// inside the WAV header or a sample frame. A WAV header overrides the encoding
// and channel count given for headerless input. Multi-channel frames become one
// mono sample: the integer mean of the channels' 16-bit values, truncated toward
// zero, divided by 32768.
class Pcm16StreamDecoder {
   public:
    // configured_sample_rate is 0 when the rate should come from a WAV header
    // or be selected by the consuming model for headerless PCM.
    explicit Pcm16StreamDecoder(
        int configured_sample_rate = 0, StreamEncoding encoding = StreamEncoding::Pcm16,
        int channels = 1);

    void process(const uint8_t* bytes, size_t size, std::vector<float>* output);
    void process(std::string_view bytes, std::vector<float>* output);

    // Completes header detection and rejects a truncated WAV header or sample.
    void finish(std::vector<float>* output);

    // The WAV rate when present, otherwise the configured rate (possibly 0).
    int sample_rate() const { return sample_rate_; }
    int channels() const { return channels_; }
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
    StreamEncoding encoding_ = StreamEncoding::Pcm16;
    int channels_ = 1;
    std::vector<uint8_t> pending_;  // bytes of an incomplete sample frame
    bool finished_ = false;
};

}  // namespace nemo_speech::audio
