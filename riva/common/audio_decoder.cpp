// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "audio_decoder.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "audio_resampler.h"

namespace nemo_speech::audio {
namespace {

constexpr size_t kMaxWavHeaderBytes = 1024 * 1024;
constexpr float kPcm16Scale = 1.0f / 32768.0f;

uint16_t
read_u16(const uint8_t* bytes) {
    return static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8);
}

uint32_t
read_u32(const uint8_t* bytes) {
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

bool
tag_is(const uint8_t* bytes, const char* tag) {
    return std::memcmp(bytes, tag, 4) == 0;
}

}  // namespace

Pcm16StreamDecoder::Pcm16StreamDecoder(int configured_sample_rate)
    : configured_sample_rate_(configured_sample_rate), sample_rate_(configured_sample_rate) {
    if (configured_sample_rate_ != 0 && !supported_input_sample_rate(configured_sample_rate_)) {
        throw std::invalid_argument(
            "sample rate must be between 8000 and 96000 Hz, or 0 (WAV/model rate)");
    }
}

void
Pcm16StreamDecoder::process(const uint8_t* bytes, size_t size, std::vector<float>* output) {
    if (output == nullptr)
        throw std::invalid_argument("PCM decoder output must not be null");
    if (finished_)
        throw std::logic_error("cannot append audio after PCM decoder finish");
    if (size == 0)
        return;
    if (bytes == nullptr)
        throw std::invalid_argument("PCM decoder input must not be null");

    if (mode_ == Mode::Raw) {
        decode_pcm(bytes, size, output);
        return;
    }
    if (mode_ == Mode::Wav) {
        decode_wav_payload(bytes, size, output);
        return;
    }

    size_t input_offset = 0;
    while (mode_ == Mode::Detect && input_offset < size) {
        const size_t limit =
            header_.size() < 12 ? 12 - header_.size() : kMaxWavHeaderBytes - header_.size();
        if (limit == 0)
            throw std::invalid_argument("WAV header exceeds 1 MiB");
        const size_t count = std::min(limit, size - input_offset);
        header_.insert(header_.end(), bytes + input_offset, bytes + input_offset + count);
        input_offset += count;

        if (header_.size() >= 12 &&
            (!tag_is(header_.data(), "RIFF") || !tag_is(header_.data() + 8, "WAVE"))) {
            mode_ = Mode::Raw;
            decode_pcm(header_.data(), header_.size(), output);
            header_.clear();
            break;
        }
        parse_header(output);
    }
    if (mode_ == Mode::Raw && input_offset < size)
        decode_pcm(bytes + input_offset, size - input_offset, output);
    else if (mode_ == Mode::Wav && input_offset < size)
        decode_wav_payload(bytes + input_offset, size - input_offset, output);
}

void
Pcm16StreamDecoder::process(std::string_view bytes, std::vector<float>* output) {
    process(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), output);
}

bool
Pcm16StreamDecoder::parse_header(std::vector<float>* output) {
    if (header_.size() < 12)
        return false;

    while (true) {
        if (header_offset_ + 8 > header_.size())
            return false;
        const uint8_t* chunk = header_.data() + header_offset_;
        const uint32_t chunk_size = read_u32(chunk + 4);
        const size_t payload_offset = header_offset_ + 8;

        if (tag_is(chunk, "data")) {
            if (!saw_format_)
                throw std::invalid_argument("WAV data chunk appears before its format chunk");
            mode_ = Mode::Wav;
            wav_data_remaining_ = chunk_size;
            if (payload_offset < header_.size()) {
                decode_wav_payload(
                    header_.data() + payload_offset, header_.size() - payload_offset, output);
            }
            header_.clear();
            return true;
        }

        const size_t padded_size =
            static_cast<size_t>(chunk_size) + static_cast<size_t>(chunk_size & 1u);
        if (padded_size > std::numeric_limits<size_t>::max() - payload_offset)
            throw std::invalid_argument("invalid WAV chunk size");
        const size_t next_offset = payload_offset + padded_size;
        if (next_offset > kMaxWavHeaderBytes)
            throw std::invalid_argument("WAV header exceeds 1 MiB");
        if (next_offset > header_.size())
            return false;

        if (tag_is(chunk, "fmt ")) {
            if (chunk_size < 16)
                throw std::invalid_argument("WAV format chunk is shorter than 16 bytes");
            const uint8_t* format = header_.data() + payload_offset;
            const uint16_t codec = read_u16(format);
            const uint16_t channels = read_u16(format + 2);
            const uint32_t rate = read_u32(format + 4);
            const uint16_t bits = read_u16(format + 14);
            if (codec != 1)
                throw std::invalid_argument("WAV codec is not PCM");
            if (channels != 1)
                throw std::invalid_argument("WAV audio must be mono");
            if (bits != 16)
                throw std::invalid_argument("WAV audio must use signed 16-bit PCM");
            if (rate > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
                !supported_input_sample_rate(static_cast<int>(rate))) {
                throw std::invalid_argument("WAV sample rate must be between 8000 and 96000 Hz");
            }
            if (configured_sample_rate_ != 0 &&
                rate != static_cast<uint32_t>(configured_sample_rate_)) {
                throw std::invalid_argument(
                    "WAV sample rate does not match the configured sample rate");
            }
            sample_rate_ = static_cast<int>(rate);
            saw_format_ = true;
        }
        header_offset_ = next_offset;
    }
}

void
Pcm16StreamDecoder::decode_wav_payload(
    const uint8_t* bytes, size_t size, std::vector<float>* output) {
    const size_t payload_size = std::min(size, wav_data_remaining_);
    if (payload_size != 0)
        decode_pcm(bytes, payload_size, output);
    wav_data_remaining_ -= payload_size;
}

void
Pcm16StreamDecoder::decode_pcm(const uint8_t* bytes, size_t size, std::vector<float>* output) {
    output->reserve(output->size() + (size + (have_carry_ ? 1u : 0u)) / 2);
    size_t offset = 0;
    if (have_carry_ && size != 0) {
        const uint16_t bits =
            static_cast<uint16_t>(carry_) | (static_cast<uint16_t>(bytes[0]) << 8);
        output->push_back(static_cast<float>(static_cast<int16_t>(bits)) * kPcm16Scale);
        have_carry_ = false;
        offset = 1;
    }
    while (offset + 1 < size) {
        const uint16_t bits = read_u16(bytes + offset);
        output->push_back(static_cast<float>(static_cast<int16_t>(bits)) * kPcm16Scale);
        offset += 2;
    }
    if (offset < size) {
        carry_ = bytes[offset];
        have_carry_ = true;
    }
}

void
Pcm16StreamDecoder::finish(std::vector<float>* output) {
    if (output == nullptr)
        throw std::invalid_argument("PCM decoder output must not be null");
    if (finished_)
        return;
    if (mode_ == Mode::Detect) {
        if (header_.size() >= 4 && tag_is(header_.data(), "RIFF")) {
            throw std::invalid_argument("truncated WAV header");
        }
        mode_ = Mode::Raw;
        decode_pcm(header_.data(), header_.size(), output);
        header_.clear();
    }
    if (mode_ == Mode::Wav && wav_data_remaining_ != 0)
        throw std::invalid_argument("truncated WAV data chunk");
    if (have_carry_)
        throw std::invalid_argument("signed 16-bit PCM stream has an odd byte count");
    finished_ = true;
}

}  // namespace nemo_speech::audio
