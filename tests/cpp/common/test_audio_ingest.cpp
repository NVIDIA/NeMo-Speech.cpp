// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Audio ingest (audio_ingest.h).
//
//   test_audio_ingest                         unit tests (WAV parsing and downmix, G.711, PCM16
//   quantization,
//                                             resampling, unsupported formats, error statuses)
//   test_audio_ingest --dump <out_dir> <files...>
//                                             decode each file and write <out_dir>/<file name>.f32
//                                             (raw float32 samples) plus one JSON line per file on
//                                             stdout, for comparing against another decoder.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "audio_ingest.h"

namespace audio = nemo_speech::audio;

namespace {

int failures = 0;

void
check(bool condition, const std::string& label) {
    std::fprintf(stdout, "[%s] %s\n", condition ? "PASS" : "FAIL", label.c_str());
    if (!condition)
        ++failures;
}

void
put16(std::string& b, uint16_t v) {
    b.push_back(static_cast<char>(v & 0xff));
    b.push_back(static_cast<char>(v >> 8));
}

void
put32(std::string& b, uint32_t v) {
    for (int s = 0; s < 32; s += 8) b.push_back(static_cast<char>((v >> s) & 0xff));
}

std::string
wav_with_format(
    uint16_t format, uint16_t bits, uint16_t channels, uint32_t rate, const std::string& payload) {
    const uint16_t block_align = static_cast<uint16_t>(channels * ((bits + 7) / 8));
    std::string b = "RIFF";
    put32(b, 36 + static_cast<uint32_t>(payload.size()));
    b += "WAVEfmt ";
    put32(b, 16);
    put16(b, format);
    put16(b, channels);
    put32(b, rate);
    put32(b, rate * block_align);
    put16(b, block_align);
    put16(b, bits);
    b += "data";
    put32(b, static_cast<uint32_t>(payload.size()));
    return b + payload;
}

// WAVE_FORMAT_EXTENSIBLE with a 40-byte fmt chunk; `sub_format` is the first two bytes of the
// SubFormat GUID.
std::string
wav_extensible(
    uint16_t sub_format, uint16_t bits, uint16_t channels, uint32_t rate,
    const std::string& payload) {
    const uint16_t block_align = static_cast<uint16_t>(channels * bits / 8);
    std::string b = "RIFF";
    put32(b, 60 + static_cast<uint32_t>(payload.size()));
    b += "WAVEfmt ";
    put32(b, 40);
    put16(b, 0xFFFE);
    put16(b, channels);
    put32(b, rate);
    put32(b, rate * block_align);
    put16(b, block_align);
    put16(b, bits);
    put16(b, 22);
    put16(b, bits);
    put32(b, 0);
    put16(b, sub_format);
    b += std::string("\x00\x00\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71", 14);
    b += "data";
    put32(b, static_cast<uint32_t>(payload.size()));
    return b + payload;
}

std::string
wav16(uint16_t channels, uint32_t rate, const std::vector<int16_t>& samples) {
    std::string payload;
    for (int16_t s : samples) put16(payload, static_cast<uint16_t>(s));
    return wav_with_format(1, 16, channels, rate, payload);
}

std::string
float_payload(const std::vector<float>& samples) {
    std::string payload;
    for (float f : samples) {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        put32(payload, u);
    }
    return payload;
}

struct Outcome {
    bool ok = false;
    int status = 0;
    bool limit = false;  // AudioLimitError
    std::string message;
    audio::DecodedAudio audio;
};

Outcome
decode(const std::string& bytes, const audio::IngestOptions& options = {}) {
    Outcome o;
    try {
        o.audio = audio::decode_any(bytes.data(), bytes.size(), "test", options);
        o.ok = true;
    }
    catch (const audio::AudioLimitError& e) {
        o.status = e.http_status();
        o.limit = true;
        o.message = e.what();
    }
    catch (const audio::AudioError& e) {
        o.status = e.http_status();
        o.message = e.what();
    }
    return o;
}

bool
starts_with(const std::string& s, const std::string& prefix) {
    return s.compare(0, prefix.size(), prefix) == 0;
}

// A 16 kHz 16-bit FLAC of `frames` frames of 4096 samples, each channel a CONSTANT subframe of its
// value (about 10 bytes per frame); STREAMINFO declares `declared` samples (0 = unknown).
std::string
constant_flac(const std::vector<int16_t>& values, int frames, uint64_t declared) {
    const auto crc = [](const std::string& b, size_t from, int bits, uint32_t poly) {
        uint32_t c = 0;
        const uint32_t top = 1u << (bits - 1), mask = (bits == 32 ? 0u : 1u << bits) - 1;
        for (size_t i = from; i < b.size(); ++i) {
            c ^= static_cast<uint32_t>(static_cast<uint8_t>(b[i])) << (bits - 8);
            for (int k = 0; k < 8; ++k) c = (c & top) ? ((c << 1) ^ poly) & mask : (c << 1) & mask;
        }
        return c;
    };
    const auto channels = static_cast<uint64_t>(values.size());
    std::string b = "fLaC";
    b += std::string("\x80\0\0\x22", 4);  // last metadata block: STREAMINFO, 34 bytes
    b += std::string("\x10\0\x10\0", 4);  // block sizes 4096..4096
    b += std::string(6, '\0');            // frame sizes unknown
    const uint64_t info = (uint64_t{16000} << 44) | ((channels - 1) << 41) | (uint64_t{15} << 36) |
                          (declared & 0xFFFFFFFFFull);
    for (int s = 56; s >= 0; s -= 8) b.push_back(static_cast<char>((info >> s) & 0xff));
    b += std::string(16, '\0');  // MD5 unknown
    for (int n = 0; n < frames; ++n) {
        const size_t start = b.size();
        b += "\xFF\xF8\xC5";  // sync, fixed blocks; 4096 samples, 16 kHz
        b.push_back(static_cast<char>((channels == 1 ? 0x00 : 0x10) | 0x08));  // 16-bit
        if (n < 0x80) {  // frame number, UTF-8 coded
            b.push_back(static_cast<char>(n));
        } else if (n < 0x800) {
            b.push_back(static_cast<char>(0xC0 | (n >> 6)));
            b.push_back(static_cast<char>(0x80 | (n & 0x3F)));
        } else {
            b.push_back(static_cast<char>(0xE0 | (n >> 12)));
            b.push_back(static_cast<char>(0x80 | ((n >> 6) & 0x3F)));
            b.push_back(static_cast<char>(0x80 | (n & 0x3F)));
        }
        b.push_back(static_cast<char>(crc(b, start, 8, 0x07)));
        for (int16_t v : values) {
            b.push_back('\0');  // CONSTANT subframe
            b.push_back(static_cast<char>(static_cast<uint16_t>(v) >> 8));
            b.push_back(static_cast<char>(static_cast<uint16_t>(v) & 0xff));
        }
        const uint32_t footer = crc(b, start, 16, 0x8005);
        b.push_back(static_cast<char>(footer >> 8));
        b.push_back(static_cast<char>(footer & 0xff));
    }
    return b;
}

// Every sample is a multiple of 1/32768 in [-1, 1).
bool
on_pcm16_grid(const std::vector<float>& samples) {
    for (float v : samples) {
        const float scaled = v * 32768.0f;
        if (scaled != std::floor(scaled) || scaled < -32768.0f || scaled > 32767.0f)
            return false;
    }
    return true;
}

bool
count_near(size_t actual, long long expected) {
    return std::llabs(static_cast<long long>(actual) - expected) <= 2;
}

void
integer_wav_tests() {
    const size_t kMin = 320;
    {
        const auto o = decode(wav16(1, 16000, std::vector<int16_t>(kMin, 7)));
        check(
            o.ok && o.audio.decoder == "wav-passthrough" && o.audio.samples.size() == kMin &&
                o.audio.samples[0] == 7.0f / 32768,
            "16 kHz mono PCM16 passes through");
    }
    {
        std::vector<int16_t> frames = {100, 300, -100, 100};
        frames.resize(frames.size() + 2 * kMin, 0);
        const auto o = decode(wav16(2, 16000, frames));
        check(
            o.ok && o.audio.decoder == "wav-decoded" && o.audio.channels == 2 &&
                o.audio.samples[0] == 200.0f / 32768 && o.audio.samples[1] == 0.0f,
            "stereo is downmixed by the integer mean");
    }
    {
        std::vector<int16_t> frames = {600, 0, 0, 0, 0, 0};
        frames.resize(frames.size() + 6 * kMin, 0);
        const auto o = decode(wav16(6, 16000, frames));
        check(
            o.ok && o.audio.samples.size() == kMin + 1 && o.audio.samples[0] == 100.0f / 32768,
            "six-channel downmix is the integer mean");
    }
    {
        // truncation toward zero, not floor: (-1 + -2) / 2 = -1
        std::vector<int16_t> frames = {-1, -2};
        frames.resize(frames.size() + 2 * kMin, 0);
        const auto o = decode(wav16(2, 16000, frames));
        check(o.ok && o.audio.samples[0] == -1.0f / 32768, "integer mean truncates toward zero");
    }
    check(
        audio::decode_mulaw(0xff) == 0 && audio::decode_mulaw(0x7f) == 0 &&
            audio::decode_alaw(0xd5) == 8 && audio::decode_alaw(0x55) == -8 &&
            audio::decode_mulaw(0x00) == -32124,
        "G.711 reference points");
    for (uint32_t rate : {8000u, 22050u, 44100u, 48000u, 96000u}) {
        const size_t seconds = rate == 48000 ? 2 : 1;
        std::vector<int16_t> s(rate * seconds);
        for (size_t i = 0; i < s.size(); ++i)
            s[i] = static_cast<int16_t>((static_cast<int>(i % 200) - 100) * 50);
        const auto o = decode(wav16(1, rate, s));
        check(
            o.ok && o.audio.decoder == "wav-resampled" &&
                o.audio.source_rate == static_cast<int>(rate) &&
                count_near(o.audio.samples.size(), 16000LL * static_cast<long long>(seconds)) &&
                on_pcm16_grid(o.audio.samples),
            "resampling " + std::to_string(rate) +
                " Hz lands on 16 kHz PCM16 with the expected count");
    }
    {
        // a full-scale square wave overshoots in the resampler and is clipped to the PCM16 range
        std::vector<int16_t> s(8000);
        for (size_t i = 0; i < s.size(); ++i) s[i] = (i / 40) % 2 ? 32767 : -32768;
        const auto o = decode(wav16(1, 8000, s));
        check(
            o.ok && on_pcm16_grid(o.audio.samples),
            "resampled full-scale audio is clipped to PCM16");
    }
    for (uint16_t format : {uint16_t(7), uint16_t(6)}) {
        const auto o = decode(wav_with_format(format, 8, 1, 8000, std::string(8000, '\xff')));
        check(
            o.ok && o.audio.decoder == "wav-resampled" && count_near(o.audio.samples.size(), 16000),
            format == 7 ? "mu-law 8 kHz decodes to 16 kHz" : "A-law 8 kHz decodes to 16 kHz");
    }
    {
        std::string b = "RIFF";
        put32(b, 4 + 24 + 12 + 12 + 8 + static_cast<uint32_t>(kMin * 2));
        b += "WAVEfmt ";
        put32(b, 16);
        put16(b, 1);
        put16(b, 1);
        put32(b, 16000);
        put32(b, 32000);
        put16(b, 2);
        put16(b, 16);
        b += std::string("LIST\x04\0\0\0INFO", 12);
        b += std::string("junk\x03\0\0\0abc\0", 12);
        b += "data";
        put32(b, static_cast<uint32_t>(kMin * 2));
        b += std::string("\x01\0\x02\0", 4);
        b += std::string(kMin * 2 - 4, '\0');
        const auto o = decode(b);
        check(
            o.ok && o.audio.decoder == "wav-decoded" && o.audio.samples.size() == kMin &&
                o.audio.samples[0] == 1.0f / 32768 && o.audio.samples[1] == 2.0f / 32768,
            "extra chunks are skipped but disable the passthrough path");
    }
    {
        const std::string honest = wav16(1, 16000, {1, 2, 3});
        std::string riff_lie = honest;
        riff_lie.replace(4, 4, std::string("\xf0\xff\xff\xff", 4));
        auto o = decode(riff_lie);
        check(
            !o.ok && o.status == 400 && o.message == "malformed WAV: truncated RIFF payload",
            "RIFF size lie is malformed");
        std::string data_lie = honest;
        data_lie.replace(40, 4, std::string("\x40\0\0\0", 4));
        o = decode(data_lie);
        check(
            !o.ok && o.status == 400 && o.message == "malformed WAV: truncated chunk",
            "data size lie is malformed");
        // streamed WAV (ffmpeg pipe): RIFF and data sizes 0xFFFFFFFF, data to the end in whole
        // frames
        std::vector<int16_t> ramp(kMin);
        for (size_t i = 0; i < ramp.size(); ++i) ramp[i] = static_cast<int16_t>(i % 1000);
        std::string streamed = wav16(1, 16000, ramp);
        streamed.replace(4, 4, std::string("\xff\xff\xff\xff", 4));
        streamed.replace(40, 4, std::string("\xff\xff\xff\xff", 4));
        o = decode(streamed + std::string(1, '\x07'));  // a trailing partial frame is dropped
        check(
            o.ok && o.audio.samples.size() == kMin && o.audio.samples[300] == 300.0f / 32768,
            "streamed WAV with unknown sizes reads to the end");
        std::string riff_unknown = honest;
        riff_unknown.replace(4, 4, std::string("\xff\xff\xff\xff", 4));
        riff_unknown.replace(40, 4, std::string("\x40\0\0\0", 4));
        o = decode(riff_unknown);
        check(
            !o.ok && o.status == 400 && o.message == "malformed WAV: truncated chunk",
            "an unknown RIFF size does not excuse a data size lie");
        for (size_t cut : {12u, 20u, 30u, 36u, 43u}) {
            o = decode(honest.substr(0, cut));
            check(
                !o.ok && o.status == 400 && starts_with(o.message, "malformed WAV:"),
                "header cut at " + std::to_string(cut) + " is malformed");
        }
        std::string partial = honest.substr(0, honest.size() - 1);
        partial.replace(40, 4, std::string("\x05\0\0\0", 4));
        partial.replace(4, 4, std::string("\x29\0\0\0", 4));
        o = decode(partial);
        check(
            !o.ok && o.status == 400 && o.message == "malformed WAV: partial audio frame",
            "partial frame is malformed");
    }
    {
        std::string zc = wav16(1, 16000, {1});
        zc.replace(22, 2, std::string("\0\0", 2));
        auto o = decode(zc);
        check(
            !o.ok && o.status == 400 && o.message == "invalid audio: zero channels or sample rate",
            "zero channels is invalid");
        std::string zr = wav16(1, 16000, {1});
        zr.replace(24, 4, std::string("\0\0\0\0", 4));
        o = decode(zr);
        check(!o.ok && o.status == 400, "zero rate is invalid");
        o = decode(wav16(1, 4000, std::vector<int16_t>(4000)));
        check(
            !o.ok && o.status == 400 &&
                o.message == "invalid audio: sample rate must be between 8000 and 96000 Hz",
            "a PCM16 rate below 8 kHz is a 400");
        o = decode(
            wav_with_format(3, 32, 1, 192000, float_payload(std::vector<float>(19200, 0.1f))));
        check(
            !o.ok && o.status == 400 &&
                o.message == "invalid audio: sample rate must be between 8000 and 96000 Hz",
            "a float rate above 96 kHz is a 400");
    }
    {
        auto o = decode(wav16(1, 16000, {}));
        check(
            !o.ok && o.status == 400 &&
                o.message == "audio is shorter than one feature frame (320 samples at 16 kHz)",
            "zero-length data chunk is too short");
        for (const auto& input :
             {wav16(1, 16000, std::vector<int16_t>(160)),
              wav16(2, 16000, std::vector<int16_t>(320)), wav16(1, 8000, std::vector<int16_t>(100)),
              wav16(1, 16000, std::vector<int16_t>(319))}) {
            o = decode(input);
            check(!o.ok && o.status == 400, "shorter than one feature frame is rejected");
        }
        o = decode(wav16(1, 16000, std::vector<int16_t>(320)));
        check(o.ok && o.audio.samples.size() == 320, "exactly 320 samples is accepted");
    }
    {
        audio::IngestOptions small;
        small.max_samples = 1000;
        small.max_duration_seconds = 0;
        auto o = decode(wav16(1, 16000, std::vector<int16_t>(1002)), small);
        check(
            !o.ok && o.status == 400 && o.message == "audio exceeds configured limit",
            "too long before decoding");
        o = decode(wav16(1, 8000, std::vector<int16_t>(501)), small);
        check(
            !o.ok && o.status == 400 && o.message == "audio exceeds configured limit",
            "too long after resampling");
        o = decode(wav16(1, 16000, std::vector<int16_t>(1000)), small);
        check(o.ok && o.audio.samples.size() == 1000, "exactly at the cap passes");
        // the passthrough path only pre-checks (max + 1); the duration limit then applies
        audio::IngestOptions tight;
        tight.max_samples = 1000;
        tight.max_duration_seconds = 1000.0 / 16000.0;
        o = decode(wav16(1, 16000, std::vector<int16_t>(1001)), tight);
        check(
            !o.ok && o.status == 400 && starts_with(o.message, "audio is 0.1 s, the limit is 0 s"),
            "passthrough max+1 is rejected by the duration limit");
    }
}

// FLAC and MP3 decoding stops at the limits, so a small upload cannot expand past them.
void
compressed_limit_tests() {
    auto o = decode(constant_flac({1000, 3000}, 8, 8 * 4096));
    check(
        o.ok && o.audio.decoder == "flac" && o.audio.channels == 2 &&
            o.audio.samples.size() == 8 * 4096 && o.audio.samples[4100] == 2000.0f / 32768,
        "stereo FLAC decodes, mixed down per block");
    audio::IngestOptions second;
    second.max_samples = 16000;
    second.max_duration_seconds = 1.0;
    o = decode(constant_flac({0}, 8, 8 * 4096), second);
    check(
        !o.ok && o.limit && o.status == 400 && o.message == "audio is 2.0 s, the limit is 1 s",
        "FLAC over the limit is rejected from its header");
    o = decode(constant_flac({0}, 8, 0), second);
    check(
        !o.ok && o.limit && o.status == 400 && o.message == "audio exceeds configured limit",
        "FLAC without a declared length stops at the limit");
    // 61 minutes of silence in under 200 KB: the upload limits reject it from its header, and
    // without a declared length decoding stops at the limit (a minute here)
    const int frames = 61 * 60 * 16000 / 4096 + 1;
    o = decode(constant_flac({0}, frames, static_cast<uint64_t>(frames) * 4096));
    check(
        !o.ok && o.limit && o.message == "audio is 3660.0 s, the limit is 3600 s",
        "the FLAC bomb is rejected from its header");
    const std::string bomb = constant_flac({0}, frames, 0);
    check(bomb.size() < 200000, "the constant FLAC is small");
    audio::IngestOptions minute;
    minute.max_samples = 60 * 16000;
    minute.max_duration_seconds = 60;
    o = decode(bomb, minute);
    check(!o.ok && o.limit && o.message == "audio exceeds configured limit", "the FLAC bomb stops");
    check(
        audio::IngestOptions::local_file().max_duration_seconds == 0 &&
            audio::IngestOptions::local_file().max_samples == static_cast<size_t>(-1),
        "local files have no length limit");
    // The WAV family (here 8-bit PCM at 8 kHz) is rejected from its header, before decoding;
    // local files decode it whole.
    const std::string u8wav = wav_with_format(1, 8, 1, 8000, std::string(2 * 8000, '\x80'));
    o = decode(u8wav, second);
    check(
        !o.ok && o.limit && o.status == 400 && o.message == "audio is 2.0 s, the limit is 1 s",
        "an 8-bit WAV over the limit is rejected before decoding");
    o = decode(u8wav, audio::IngestOptions::local_file());
    check(o.ok && o.audio.samples.size() == 2 * 16000, "a local 8-bit WAV decodes whole");
}

void
float_path_tests() {
    {
        // clipping, then a floor to the PCM16 grid
        std::vector<float> v = {
            1.5f / 32768,
            2.5f / 32768,
            -1.5f / 32768,
            0.5f / 32768,
            32766.5f / 32768,
            -32767.5f / 32768,
            -32768.4f / 32768,
            32767.4f / 32768,
            32767.6f / 32768,
            12345.5f / 32768,
            1.5f,
            -1.5f};
        const int expect[] = {1,      2,     -2,    0,     32766, -32768,
                              -32768, 32767, 32767, 12345, 32767, -32768};
        audio::pcm16_round_trip(v);
        bool ok = true;
        for (size_t i = 0; i < v.size(); ++i)
            ok = ok && v[i] * 32768.0f == static_cast<float>(expect[i]);
        check(ok, "PCM16 round trip clips and floors");
    }
    {
        // IEEE float 16 kHz mono: no resampling; values off the PCM16 grid are floored
        const auto o = decode(wav_with_format(
            3, 32, 1, 16000, float_payload(std::vector<float>(400, 0.25f + 0.3f / 32768))));
        check(
            o.ok && o.audio.decoder == "wav" && o.audio.samples.size() == 400 &&
                o.audio.samples[0] == 8192.0f / 32768,
            "float WAV is floored to PCM16");
    }
    {
        // float stereo: float mean, then the floor
        std::vector<float> s;
        for (int i = 0; i < 400; ++i) {
            s.push_back(0.5f);
            s.push_back(-0.25f + 1.0f / 65536);
        }
        const auto o = decode(wav_with_format(3, 32, 2, 16000, float_payload(s)));
        check(
            o.ok && o.audio.channels == 2 && o.audio.samples.size() == 400 &&
                o.audio.samples[0] == 4096.0f / 32768,
            "float stereo is averaged then floored");
    }
    {
        // 24-bit PCM: left-justified int32 * 2^-31, then the floor
        std::string payload;
        for (int i = 0; i < 400; ++i) {
            const int32_t v = (i % 2) ? -0x123457 : 0x123457;
            payload.push_back(static_cast<char>(v & 0xff));
            payload.push_back(static_cast<char>((v >> 8) & 0xff));
            payload.push_back(static_cast<char>((v >> 16) & 0xff));
        }
        const auto o = decode(wav_with_format(1, 24, 1, 16000, payload));
        check(
            o.ok && o.audio.samples.size() == 400 &&
                o.audio.samples[0] * 32768 == static_cast<float>(0x1234) &&
                o.audio.samples[1] * 32768 == static_cast<float>(-0x1235),
            "24-bit PCM floors to PCM16");
    }
    {
        std::string u8(400, '\x80');
        u8[0] = '\xff';
        u8[1] = '\x00';
        const auto o = decode(wav_with_format(1, 8, 1, 16000, u8));
        check(
            o.ok && o.audio.samples[0] == 127.0f / 128 && o.audio.samples[1] == -1.0f &&
                o.audio.samples[2] == 0.0f,
            "unsigned 8-bit PCM is (u - 128) / 128");
    }
    {
        std::string payload;
        for (int i = 0; i < 400; ++i) put32(payload, static_cast<uint32_t>(0x7FFFFFFF));
        auto o = decode(wav_with_format(1, 32, 1, 16000, payload));
        check(
            o.ok && o.audio.samples[0] == 32767.0f / 32768, "32-bit PCM full scale clips to 32767");
        std::string ext;
        for (int i = 0; i < 400; ++i) put16(ext, static_cast<uint16_t>(-1234));
        o = decode(wav_extensible(1, 16, 1, 16000, ext));
        check(
            o.ok && o.audio.decoder == "wav" && o.audio.samples[0] == -1234.0f / 32768,
            "WAVE_FORMAT_EXTENSIBLE PCM16 decodes");
        std::string dbl;
        for (int i = 0; i < 400; ++i) {
            const double d = -0.5;
            uint64_t u;
            std::memcpy(&u, &d, 8);
            put32(dbl, static_cast<uint32_t>(u));
            put32(dbl, static_cast<uint32_t>(u >> 32));
        }
        o = decode(wav_with_format(3, 64, 1, 16000, dbl));
        check(o.ok && o.audio.samples[0] == -0.5f, "64-bit float WAV decodes");
    }
    {
        std::vector<float> s(44100);
        for (size_t i = 0; i < s.size(); ++i)
            s[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.05) * 0.9);
        const auto o = decode(wav_with_format(3, 32, 1, 44100, float_payload(s)));
        check(
            o.ok && o.audio.decoder == "wav" && o.audio.source_rate == 44100 &&
                count_near(o.audio.samples.size(), 16000) && on_pcm16_grid(o.audio.samples),
            "float 44.1 kHz resamples to 16 kHz PCM16");
    }
}

void
format_tests() {
    auto o = decode(std::string("this is not audio at all, just some text bytes"));
    check(
        !o.ok && o.status == 422 &&
            o.message == "unsupported audio format: this build decodes WAV, FLAC and MP3",
        "garbage is a 422 client error");
    o = decode(std::string());
    check(!o.ok && o.status == 422, "empty upload is a 422 client error");
    o = decode(std::string("fLaC") + std::string(60, '\0'));
    check(
        !o.ok && o.status == 422 && o.message == "invalid or unsupported audio file",
        "broken FLAC is a 422 client error");
    o = decode(std::string("ID3\x04\0\0\0\0\0\0", 10) + std::string(64, '\0'));
    check(!o.ok && o.status == 422, "an MP3 tag without frames is a 422 client error");
    const std::pair<std::string, std::string> others[] = {
        {"MP4/M4A", std::string(
                        "\0\0\0\x20"
                        "ftypisom",
                        12) +
                        std::string(64, '\0')},
        {"Ogg", std::string("OggS\0\x02", 6) + std::string(64, '\0')},
        {"AIFF", std::string(
                     "FORM\0\0\0\x40"
                     "AIFFCOMM",
                     16) +
                     std::string(64, '\0')},
        {"WebM/MKV", std::string("\x1a\x45\xdf\xa3", 4) + std::string(64, '\0')},
    };
    for (const auto& [label, bytes] : others) {
        o = decode(bytes);
        check(
            !o.ok && o.status == 422 && starts_with(o.message, "unsupported audio format:"),
            label + " is a 422 unsupported format");
    }
    o = decode(wav_with_format(2, 4, 1, 16000, std::string(512, '\0')));
    check(
        !o.ok && o.status == 422 &&
            starts_with(o.message, "unsupported audio format: WAV format tag 0x0002"),
        "ADPCM WAV is a 422 naming the format tag");
    o = decode(wav_with_format(0x55, 0, 1, 16000, std::string(512, '\0')));
    check(
        !o.ok && o.status == 422 &&
            starts_with(o.message, "unsupported audio format: WAV format tag 0x0055"),
        "MP3-in-WAV is a 422 naming the format tag");
    check(
        audio::is_audio_path("a/b.WAV") && audio::is_audio_path("x.flac") &&
            audio::is_audio_path("x.mp3") && !audio::is_audio_path("x.ogg") &&
            !audio::is_audio_path("x.m4a") && !audio::is_audio_path("dir.wav/x"),
        "is_audio_path lists the decoded extensions");
}

void
raw_tests() {
    // Headerless PCM / G.711 (Riva LINEAR_PCM, MULAW, ALAW) decode like the same samples in a WAV.
    std::vector<int16_t> s(16000);
    for (size_t i = 0; i < s.size(); ++i)
        s[i] = static_cast<int16_t>((static_cast<int>(i % 97) - 48) * 300);
    std::string pcm;
    for (int16_t v : s) put16(pcm, static_cast<uint16_t>(v));
    const auto from_wav = decode(wav16(1, 48000, s));
    audio::DecodedAudio raw;
    try {
        raw = audio::decode_raw(pcm.data(), pcm.size(), audio::RawEncoding::Pcm16, 48000, 1);
    }
    catch (const std::exception&) {
    }
    check(
        from_wav.ok && raw.samples == from_wav.audio.samples,
        "raw PCM16 decodes like a WAV upload");
    const std::string ulaw(8000, '\x55');
    const auto ulaw_wav = decode(wav_with_format(7, 8, 2, 8000, ulaw));
    audio::DecodedAudio ulaw_raw;
    try {
        ulaw_raw = audio::decode_raw(ulaw.data(), ulaw.size(), audio::RawEncoding::Mulaw, 8000, 2);
    }
    catch (const std::exception&) {
    }
    check(
        ulaw_wav.ok && ulaw_raw.samples == ulaw_wav.audio.samples && ulaw_raw.channels == 2,
        "raw stereo mu-law decodes like a WAV upload");
    int status = 0;
    try {
        audio::decode_raw(pcm.data(), 3, audio::RawEncoding::Pcm16, 16000, 1);
    }
    catch (const audio::AudioError& e) {
        status = e.http_status();
    }
    check(status == 400, "a partial raw sample frame is a 400");
}

int
dump(int argc, char** argv) {
    const std::string out_dir = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string path = argv[i];
        const std::string name = path.substr(path.find_last_of('/') + 1);
        std::ifstream in(path, std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const auto o = decode(bytes);
        std::string message = o.message;
        for (auto& c : message)
            if (c == '"' || c == '\\')
                c = '\'';
        if (o.ok) {
            std::ofstream out(out_dir + "/" + name + ".f32", std::ios::binary);
            out.write(
                reinterpret_cast<const char*>(o.audio.samples.data()),
                static_cast<std::streamsize>(o.audio.samples.size() * sizeof(float)));
            std::printf(
                "{\"file\":\"%s\",\"ok\":true,\"decoder\":\"%s\",\"source_rate\":%d,\"channels\":%"
                "d,\"n\":%zu}\n",
                name.c_str(), o.audio.decoder.c_str(), o.audio.source_rate, o.audio.channels,
                o.audio.samples.size());
        } else {
            std::printf(
                "{\"file\":\"%s\",\"ok\":false,\"status\":%d,\"message\":\"%s\"}\n", name.c_str(),
                o.status, message.c_str());
        }
    }
    return 0;
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--dump")
        return dump(argc, argv);
    integer_wav_tests();
    compressed_limit_tests();
    float_path_tests();
    format_tests();
    raw_tests();
    std::fprintf(stdout, "%s: %d failure(s)\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
