// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "audio_file.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace nemo_speech::audio {
namespace {

template <typename T>
T
read_le(const uint8_t* p) {
    T value{};
    std::memcpy(&value, p, sizeof(value));
    return value;
}

std::runtime_error
file_error(const std::string& path, const std::string& message) {
    return std::runtime_error(path + ": " + message);
}

constexpr const char* kConversionCommand =
    "convert with: ffmpeg -i INPUT -ac 1 -ar 16000 -c:a pcm_s16le output.wav";

AudioFile
parse_wav(const uint8_t* bytes, size_t byte_count, const std::string& source_name) {
    if (!bytes || byte_count < 12)
        throw file_error(source_name, "file is too small to be a RIFF/WAVE file");
    if (std::memcmp(bytes, "RIFF", 4) != 0 || std::memcmp(bytes + 8, "WAVE", 4) != 0)
        throw file_error(source_name, std::string("unsupported container; ") + kConversionCommand);

    uint16_t format = 0;
    uint16_t channels = 0;
    uint16_t bits = 0;
    uint16_t block_align = 0;
    uint32_t sample_rate = 0;
    const uint8_t* data = nullptr;
    size_t data_size = 0;
    bool have_format = false;
    for (size_t pos = 12; pos + 8 <= byte_count;) {
        const uint32_t declared = read_le<uint32_t>(bytes + pos + 4);
        const size_t body = pos + 8;
        const size_t available = byte_count - body;
        if (declared > available)
            throw file_error(source_name, "truncated WAV chunk");
        if (std::memcmp(bytes + pos, "fmt ", 4) == 0) {
            if (declared < 16)
                throw file_error(source_name, "invalid WAV format chunk");
            const auto* p = bytes + body;
            format = read_le<uint16_t>(p);
            channels = read_le<uint16_t>(p + 2);
            sample_rate = read_le<uint32_t>(p + 4);
            block_align = read_le<uint16_t>(p + 12);
            bits = read_le<uint16_t>(p + 14);
            have_format = true;
        } else if (std::memcmp(bytes + pos, "data", 4) == 0 && !data) {
            data = bytes + body;
            data_size = declared;
        }
        const uint64_t next = static_cast<uint64_t>(body) + declared + (declared & 1U);
        if (next > byte_count)
            break;
        pos = static_cast<size_t>(next);
    }

    if (!have_format || !data)
        throw file_error(source_name, "WAV file is missing a format or data chunk");
    if (channels != 1 && channels != 2)
        throw file_error(source_name, "only mono and stereo WAV files are supported");
    if (sample_rate < 8000 || sample_rate > 96000)
        throw file_error(source_name, "sample rate must be between 8 and 96 kHz");

    size_t bytes_per_sample = 0;
    if (format == 1 && bits == 16)
        bytes_per_sample = 2;
    else if (format == 3 && bits == 32)
        bytes_per_sample = 4;
    else
        throw file_error(
            source_name, std::string("unsupported WAV encoding; ") + kConversionCommand);
    const size_t expected_align = bytes_per_sample * channels;
    if (block_align != expected_align || data_size % expected_align != 0)
        throw file_error(source_name, "invalid or truncated WAV sample data");

    AudioFile output;
    output.sample_rate = static_cast<int>(sample_rate);
    output.source_channels = channels;
    const size_t frame_count = data_size / expected_align;
    output.samples.resize(frame_count);
    for (size_t frame = 0; frame < frame_count; ++frame) {
        float sum = 0.0f;
        for (size_t channel = 0; channel < channels; ++channel) {
            const auto* p = data + (frame * channels + channel) * bytes_per_sample;
            float sample = format == 1 ? read_le<int16_t>(p) / 32768.0f : read_le<float>(p);
            if (!std::isfinite(sample))
                sample = 0.0f;
            sum += std::clamp(sample, -1.0f, 1.0f);
        }
        output.samples[frame] = sum / channels;
    }
    if (output.samples.empty())
        throw file_error(source_name, "audio data is empty");
    return output;
}

}  // namespace

bool
is_wav_path(const std::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos)
        return false;
    std::string extension = path.substr(dot);
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension == ".wav" || extension == ".wave";
}

AudioFile
load_wav_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw file_error(path, "cannot open audio file");
    const auto end = input.tellg();
    if (end < 12)
        throw file_error(path, "file is too small to be a RIFF/WAVE file");
    if (static_cast<uint64_t>(end) > std::numeric_limits<size_t>::max())
        throw file_error(path, "file is too large");
    std::vector<uint8_t> bytes(static_cast<size_t>(end));
    input.seekg(0);
    if (!input.read(
            reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        throw file_error(path, "could not read the complete file");

    return parse_wav(bytes.data(), bytes.size(), path);
}

AudioFile
load_wav_memory(const void* data, size_t size, const std::string& source_name) {
    return parse_wav(static_cast<const uint8_t*>(data), size, source_name);
}

std::string
pcm16_wav(const std::string& pcm, int sample_rate) {
    if (sample_rate <= 0 || sample_rate > 192000)
        throw std::invalid_argument("PCM sample rate must be between 1 and 192000 Hz");
    if (pcm.size() > std::numeric_limits<uint32_t>::max() - 36 || pcm.size() % 2 != 0)
        throw std::runtime_error("PCM16 audio is too large or malformed");
    auto append_u16 = [](std::string& output, uint16_t value) {
        output.push_back(static_cast<char>(value));
        output.push_back(static_cast<char>(value >> 8));
    };
    auto append_u32 = [](std::string& output, uint32_t value) {
        for (int shift = 0; shift < 32; shift += 8)
            output.push_back(static_cast<char>(value >> shift));
    };
    std::string output;
    output.reserve(pcm.size() + 44);
    output += "RIFF";
    append_u32(output, static_cast<uint32_t>(pcm.size() + 36));
    output += "WAVEfmt ";
    append_u32(output, 16);
    append_u16(output, 1);
    append_u16(output, 1);
    append_u32(output, static_cast<uint32_t>(sample_rate));
    append_u32(output, static_cast<uint32_t>(sample_rate * 2));
    append_u16(output, 2);
    append_u16(output, 16);
    output += "data";
    append_u32(output, static_cast<uint32_t>(pcm.size()));
    output += pcm;
    return output;
}

}  // namespace nemo_speech::audio
