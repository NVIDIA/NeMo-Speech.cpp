// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Content probing and the WAV family reader of the audio ingest.
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "audio_ingest_detail.h"

namespace nemo_speech::audio::detail {
namespace {

uint16_t
rd16(const uint8_t* p, bool be) {
    return be ? static_cast<uint16_t>((p[0] << 8) | p[1])
              : static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t
rd32(const uint8_t* p, bool be) {
    return be ? (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
                    (static_cast<uint32_t>(p[2]) << 8) | p[3]
              : static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                    (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t
rd64le(const uint8_t* p) {
    return static_cast<uint64_t>(rd32(p, false)) |
           (static_cast<uint64_t>(rd32(p + 4, false)) << 32);
}

bool
is_mpeg_sync(const uint8_t* p, size_t n) {
    // MPEG-1/2/2.5 audio frame header: 11-bit sync, layer != 00 (00 is AAC ADTS), bitrate != 1111
    return n >= 4 && p[0] == 0xFF && (p[1] & 0xE0) == 0xE0 && (p[1] & 0x06) != 0 &&
           (p[2] & 0xF0) != 0xF0 && (p[1] & 0x18) != 0x08;
}

// WAV fmt fields the readers need.
struct WavFmt {
    uint16_t tag = 0;  // format tag, resolved through WAVE_FORMAT_EXTENSIBLE
    bool extensible = false;
    uint16_t channels = 0;
    uint32_t rate = 0;
    uint16_t block_align = 0;
    uint16_t bits = 0;
};

struct WavLayout {
    bool be = false;
    bool rf64 = false;
    WavFmt fmt;
    bool have_fmt = false;
    const uint8_t* data = nullptr;
    uint64_t data_len = 0;
};

// Chunk walk over RIFF/RIFX/RF64 (a data chunk that runs past the end of the file is cut to the
// bytes present).
bool
parse_wav_layout(const uint8_t* d, size_t n, WavLayout& w) {
    if (n < 12 || std::memcmp(d + 8, "WAVE", 4) != 0)
        return false;
    if (std::memcmp(d, "RIFF", 4) == 0) {
        w.be = false;
    } else if (std::memcmp(d, "RIFX", 4) == 0) {
        w.be = true;
    } else if (std::memcmp(d, "RF64", 4) == 0 || std::memcmp(d, "BW64", 4) == 0) {
        w.rf64 = true;
    } else {
        return false;
    }
    uint64_t ds64_data = 0;
    size_t pos = 12;
    while (pos + 8 <= n) {
        const uint8_t* id = d + pos;
        uint64_t len = rd32(d + pos + 4, w.be);
        const size_t body = pos + 8;
        if (std::memcmp(id, "ds64", 4) == 0 && len >= 24 && body + 24 <= n) {
            ds64_data = rd64le(d + body + 8);
        } else if (std::memcmp(id, "fmt ", 4) == 0 && !w.have_fmt) {
            if (len < 16 || body + 16 > n)
                return false;
            const uint8_t* f = d + body;
            w.fmt.tag = rd16(f, w.be);
            w.fmt.channels = rd16(f + 2, w.be);
            w.fmt.rate = rd32(f + 4, w.be);
            w.fmt.block_align = rd16(f + 12, w.be);
            w.fmt.bits = rd16(f + 14, w.be);
            if (w.fmt.tag == 0xFFFE) {
                w.fmt.extensible = true;
                if (len < 40 || body + 40 > n)
                    return false;
                w.fmt.tag = rd16(f + 24, w.be);  // first two bytes of the SubFormat GUID
            }
            w.have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0 && w.data == nullptr) {
            if (w.rf64 && len == 0xFFFFFFFFu)
                len = ds64_data;
            w.data = d + body;
            w.data_len = std::min<uint64_t>(len, n - body);
            if (body + len > n)
                break;
        }
        if (body + len > n)
            break;
        pos = static_cast<size_t>(body + len + (len & 1));
    }
    return w.have_fmt && w.data != nullptr;
}

}  // namespace

Kind
probe_kind(const uint8_t* d, size_t n) {
    if (n >= 12 && std::memcmp(d + 8, "WAVE", 4) == 0) {
        if (std::memcmp(d, "RIFF", 4) == 0)
            return Kind::RiffWave;
        if (std::memcmp(d, "RIFX", 4) == 0)
            return Kind::Rifx;
        if (std::memcmp(d, "RF64", 4) == 0 || std::memcmp(d, "BW64", 4) == 0)
            return Kind::Rf64;
    }
    if (n >= 4 && std::memcmp(d, "fLaC", 4) == 0)
        return Kind::Flac;
    if (n >= 10 && std::memcmp(d, "ID3", 3) == 0) {
        const size_t tag = 10 +
                           ((static_cast<size_t>(d[6] & 0x7f) << 21) | ((d[7] & 0x7f) << 14) |
                            ((d[8] & 0x7f) << 7) | (d[9] & 0x7f)) +
                           ((d[5] & 0x10) ? 10 : 0);
        if (tag + 4 <= n && std::memcmp(d + tag, "fLaC", 4) == 0)
            return Kind::Flac;
        return Kind::Mp3;
    }
    if (is_mpeg_sync(d, n))
        return Kind::Mp3;
    return Kind::Other;
}

int
wav_format_tag(const uint8_t* d, size_t n) {
    WavLayout w;
    return parse_wav_layout(d, n, w) ? static_cast<int>(w.fmt.tag) : -1;
}

bool
decode_wav_family(const uint8_t* d, size_t n, const IngestOptions& options, PcmFloat& out) {
    WavLayout w;
    if (!parse_wav_layout(d, n, w))
        return false;
    const WavFmt& f = w.fmt;
    if (f.channels == 0 || f.rate == 0 || f.block_align == 0)
        return false;
    const size_t width = (static_cast<size_t>(f.bits) + 7) / 8;
    const bool pcm = f.tag == 0x0001 && width >= 1 && width <= 4;
    const bool flt = f.tag == 0x0003 && (f.bits == 32 || f.bits == 64);
    const bool g711 = (f.tag == 0x0006 || f.tag == 0x0007) && f.bits == 8;
    if (!pcm && !flt && !g711)
        return false;
    const size_t sample_bytes = flt ? f.bits / 8u : (g711 ? 1u : width);
    if (static_cast<size_t>(f.block_align) < sample_bytes * f.channels)
        return false;
    const size_t frames = static_cast<size_t>(w.data_len / f.block_align);
    out.frames = frames;
    out.channels = f.channels;
    out.sample_rate = static_cast<int>(f.rate);
    out.wav_pcm16 = !w.rf64 && !f.extensible && pcm && width == 2;
    out.decoder = "wav";
    if (frames >= frame_cap(options, f.rate)) {  // over the limits: nothing is decoded
        out.over_limit = true;
        out.total_frames = frames;
        return true;
    }
    out.interleaved.resize(frames * f.channels);
    float* dst = out.interleaved.data();
    for (size_t fr = 0; fr < frames; ++fr) {
        const uint8_t* frame = w.data + fr * f.block_align;
        for (size_t c = 0; c < f.channels; ++c) {
            const uint8_t* s = frame + c * sample_bytes;
            float v = 0.0f;
            if (pcm) {
                switch (width) {
                    case 1:  // unsigned 8-bit: (u - 128) / 0x80
                        v = static_cast<float>(static_cast<int>(s[0]) - 128) *
                            (1.0f / static_cast<float>(0x80));
                        break;
                    case 2:
                        v = static_cast<float>(static_cast<int16_t>(rd16(s, w.be))) *
                            (1.0f / static_cast<float>(0x8000));
                        break;
                    case 3: {  // left-justified into 32 bits, then * 2^-31
                        const uint32_t u = w.be ? (static_cast<uint32_t>(s[0]) << 24) |
                                                      (static_cast<uint32_t>(s[1]) << 16) |
                                                      (static_cast<uint32_t>(s[2]) << 8)
                                                : (static_cast<uint32_t>(s[0]) << 8) |
                                                      (static_cast<uint32_t>(s[1]) << 16) |
                                                      (static_cast<uint32_t>(s[2]) << 24);
                        v = static_cast<float>(static_cast<int32_t>(u)) *
                            (1.0f / static_cast<float>(0x80000000u));
                        break;
                    }
                    default:
                        v = static_cast<float>(static_cast<int32_t>(rd32(s, w.be))) *
                            (1.0f / static_cast<float>(0x80000000u));
                        break;
                }
            } else if (flt) {
                if (f.bits == 32) {
                    uint32_t u = rd32(s, w.be);
                    std::memcpy(&v, &u, 4);
                } else {
                    uint64_t u =
                        w.be ? (static_cast<uint64_t>(rd32(s, true)) << 32) | rd32(s + 4, true)
                             : rd64le(s);
                    double dv;
                    std::memcpy(&dv, &u, 8);
                    v = static_cast<float>(dv);
                }
            } else {
                const int16_t q = f.tag == 0x0006 ? decode_alaw(s[0]) : decode_mulaw(s[0]);
                v = static_cast<float>(q) * (1.0f / static_cast<float>(0x8000));
            }
            *dst++ = v;
        }
    }
    return true;
}

}  // namespace nemo_speech::audio::detail
