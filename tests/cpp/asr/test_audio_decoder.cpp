// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "audio_decoder.h"

using nemo_speech::audio::Pcm16StreamDecoder;

namespace {

int failures = 0;

void
check(bool condition, const char* label) {
    std::fprintf(stdout, "[%s] %s\n", condition ? "PASS" : "FAIL", label);
    if (!condition)
        ++failures;
}

void
append_u16(std::string& bytes, uint16_t value) {
    bytes.push_back(static_cast<char>(value & 0xff));
    bytes.push_back(static_cast<char>((value >> 8) & 0xff));
}

void
append_u32(std::string& bytes, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<char>((value >> shift) & 0xff));
}

void
write_u32(std::string& bytes, size_t offset, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8)
        bytes[offset++] = static_cast<char>((value >> shift) & 0xff);
}

void
append_pcm(std::string& bytes, const std::vector<int16_t>& samples) {
    for (int16_t sample : samples) append_u16(bytes, static_cast<uint16_t>(sample));
}

std::string
make_wav(const std::vector<int16_t>& samples, int sample_rate, int channels = 1, int bits = 16) {
    std::string body;
    body.append("fmt ", 4);
    append_u32(body, 16);
    append_u16(body, 1);
    append_u16(body, static_cast<uint16_t>(channels));
    append_u32(body, static_cast<uint32_t>(sample_rate));
    append_u32(body, static_cast<uint32_t>(sample_rate * channels * bits / 8));
    append_u16(body, static_cast<uint16_t>(channels * bits / 8));
    append_u16(body, static_cast<uint16_t>(bits));
    // Exercise an unknown odd-sized, word-padded chunk before data.
    body.append("JUNK", 4);
    append_u32(body, 3);
    body.append("abc\0", 4);
    body.append("data", 4);
    append_u32(body, static_cast<uint32_t>(samples.size() * 2));
    append_pcm(body, samples);

    std::string wav("RIFF", 4);
    append_u32(wav, static_cast<uint32_t>(body.size() + 4));
    wav.append("WAVE", 4);
    wav += body;
    return wav;
}

bool
matches(const std::vector<float>& decoded, const std::vector<int16_t>& expected) {
    if (decoded.size() != expected.size())
        return false;
    for (size_t i = 0; i < decoded.size(); ++i) {
        const float value = static_cast<float>(expected[i]) / 32768.0f;
        if (std::abs(decoded[i] - value) > 1.0e-7f)
            return false;
    }
    return true;
}

}  // namespace

int
main() {
    const std::vector<int16_t> expected = {-32768, -1234, 0, 1234, 32767};
    const std::string wav = make_wav(expected, 44100);
    Pcm16StreamDecoder wav_decoder;
    std::vector<float> decoded;
    // Split inside RIFF tags, chunk sizes, the format payload, and samples.
    for (size_t offset = 0; offset < wav.size();) {
        const size_t count = std::min<size_t>((offset % 7) + 1, wav.size() - offset);
        wav_decoder.process(std::string_view(wav).substr(offset, count), &decoded);
        offset += count;
    }
    wav_decoder.finish(&decoded);
    check(wav_decoder.is_wav(), "chunked RIFF/WAVE is detected");
    check(wav_decoder.sample_rate() == 44100, "WAV sample rate is surfaced");
    check(matches(decoded, expected), "chunked WAV PCM samples decode exactly");

    std::string wav_with_trailing_chunk = wav;
    wav_with_trailing_chunk.append("LIST", 4);
    append_u32(wav_with_trailing_chunk, 4);
    wav_with_trailing_chunk.append("meta", 4);
    write_u32(
        wav_with_trailing_chunk, 4, static_cast<uint32_t>(wav_with_trailing_chunk.size() - 8));
    Pcm16StreamDecoder trailing_chunk_decoder;
    decoded.clear();
    trailing_chunk_decoder.process(wav_with_trailing_chunk, &decoded);
    trailing_chunk_decoder.finish(&decoded);
    check(matches(decoded, expected), "WAV decoding stops at the declared data chunk boundary");

    bool truncated_data_rejected = false;
    try {
        std::string truncated_data = wav;
        const size_t data_offset = truncated_data.find("data");
        write_u32(
            truncated_data, data_offset + 4,
            static_cast<uint32_t>(expected.size() * sizeof(int16_t) + 2));
        Pcm16StreamDecoder decoder;
        decoded.clear();
        decoder.process(truncated_data, &decoded);
        decoder.finish(&decoded);
    }
    catch (const std::invalid_argument&) {
        truncated_data_rejected = true;
    }
    check(truncated_data_rejected, "truncated WAV data chunk is rejected");

    std::string raw;
    append_pcm(raw, expected);
    Pcm16StreamDecoder raw_decoder(16000);
    decoded.clear();
    for (char byte : raw) raw_decoder.process(reinterpret_cast<const uint8_t*>(&byte), 1, &decoded);
    raw_decoder.finish(&decoded);
    check(!raw_decoder.is_wav(), "headerless PCM remains raw");
    check(raw_decoder.sample_rate() == 16000, "raw PCM keeps its configured sample rate");
    check(matches(decoded, expected), "raw PCM supports sample-splitting chunk boundaries");

    bool mismatch_rejected = false;
    try {
        Pcm16StreamDecoder mismatch(16000);
        decoded.clear();
        mismatch.process(wav, &decoded);
    }
    catch (const std::invalid_argument&) {
        mismatch_rejected = true;
    }
    check(mismatch_rejected, "WAV/configured sample-rate mismatch is rejected");

    bool stereo_rejected = false;
    try {
        const std::string stereo = make_wav(expected, 16000, 2);
        Pcm16StreamDecoder decoder;
        decoded.clear();
        decoder.process(stereo, &decoded);
    }
    catch (const std::invalid_argument&) {
        stereo_rejected = true;
    }
    check(stereo_rejected, "non-mono WAV is rejected");

    bool odd_rejected = false;
    try {
        Pcm16StreamDecoder odd(16000);
        decoded.clear();
        const std::string bytes(13, '\1');
        odd.process(bytes, &decoded);
        odd.finish(&decoded);
    }
    catch (const std::invalid_argument&) {
        odd_rejected = true;
    }
    check(odd_rejected, "trailing half sample is rejected at finish");

    bool truncated_wav_rejected = false;
    try {
        Pcm16StreamDecoder truncated;
        decoded.clear();
        truncated.process(std::string_view("RIFF\0\0\0\0WAVE", 12), &decoded);
        truncated.finish(&decoded);
    }
    catch (const std::invalid_argument&) {
        truncated_wav_rejected = true;
    }
    check(truncated_wav_rejected, "truncated WAV header is rejected");

    // Large offline raw payloads must not be mistaken for oversized headers.
    std::string large_raw(1024 * 1024 + 2, '\0');
    Pcm16StreamDecoder large_decoder(16000);
    decoded.clear();
    large_decoder.process(large_raw, &decoded);
    large_decoder.finish(&decoded);
    check(decoded.size() == large_raw.size() / 2, "large raw PCM bypasses WAV header limit");

    std::fprintf(stdout, failures ? "FAILED (%d)\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
