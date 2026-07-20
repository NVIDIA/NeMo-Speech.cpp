// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Minimal WAV loader shared by the file-based examples: mono, 16 kHz, PCM16
// or float32. Walks the RIFF chunk list (tolerating unknown chunks),
// validates the format, and returns float samples in [-1, 1]. Intentionally
// minimal (the canonical case the models expect) so the examples stay focused
// on the recognition/diarization APIs rather than audio decoding.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace examples {

constexpr int kSampleRate = 16000;

// Read a little-endian scalar from a byte cursor.
template <typename T>
inline T
rd(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

inline bool
read_wav_mono_16k(const std::string& path, std::vector<float>& out, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open file";
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size < 44) {
        std::fclose(f);
        err = "file too small to be a WAV";
        return false;
    }
    std::vector<uint8_t> b(static_cast<size_t>(size));
    size_t got = std::fread(b.data(), 1, b.size(), f);
    std::fclose(f);
    if (got != b.size()) {
        err = "short read";
        return false;
    }

    if (std::memcmp(b.data(), "RIFF", 4) != 0 || std::memcmp(b.data() + 8, "WAVE", 4) != 0) {
        err = "not a RIFF/WAVE file";
        return false;
    }

    uint16_t fmt = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    const uint8_t* data = nullptr;
    size_t data_len = 0;

    // Chunk walk starts after the 12-byte RIFF header.
    size_t pos = 12;
    while (pos + 8 <= b.size()) {
        const uint8_t* id = b.data() + pos;
        uint32_t clen = rd<uint32_t>(b.data() + pos + 4);
        const uint8_t* body = b.data() + pos + 8;
        if (pos + 8 + clen > b.size())
            clen = static_cast<uint32_t>(b.size() - pos - 8);  // tolerate truncation
        if (std::memcmp(id, "fmt ", 4) == 0 && clen >= 16) {
            fmt = rd<uint16_t>(body + 0);
            channels = rd<uint16_t>(body + 2);
            rate = rd<uint32_t>(body + 4);
            bits = rd<uint16_t>(body + 14);
        } else if (std::memcmp(id, "data", 4) == 0) {
            data = body;
            data_len = clen;
        }
        pos += 8 + clen + (clen & 1);  // chunks are word-aligned
    }

    if (!data) {
        err = "no data chunk";
        return false;
    }
    if (channels != 1) {
        err = "expected mono (" + std::to_string(channels) + " channels); convert with e.g. sox";
        return false;
    }
    if (rate != kSampleRate) {
        err = "expected " + std::to_string(kSampleRate) + " Hz, got " + std::to_string(rate);
        return false;
    }

    if (fmt == 1 && bits == 16) {  // PCM16
        const size_t n = data_len / 2;
        out.resize(n);
        for (size_t i = 0; i < n; i++) out[i] = rd<int16_t>(data + i * 2) / 32768.0f;
        return true;
    }
    if (fmt == 3 && bits == 32) {  // IEEE float32
        const size_t n = data_len / 4;
        out.resize(n);
        for (size_t i = 0; i < n; i++) out[i] = rd<float>(data + i * 4);
        return true;
    }
    err = "unsupported sample format (need PCM16 or float32 mono)";
    return false;
}

}  // namespace examples
