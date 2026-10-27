// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Audio ingest (see audio_ingest.h).
#include "audio_ingest.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>

#include "audio_ingest_detail.h"
#include "audio_resampler.h"

namespace nemo_speech::audio {
namespace {

constexpr uint32_t kTargetRate = 16000;
constexpr uint16_t kPcm = 1;
constexpr uint16_t kAlaw = 6;
constexpr uint16_t kMulaw = 7;

[[noreturn]] void
fail(int status, const std::string& message) {
    throw AudioError(status, message);
}

[[noreturn]] void
over_limit(const std::string& message = "audio exceeds configured limit") {
    throw AudioLimitError(message);
}

[[noreturn]] void
malformed(const char* what) {
    fail(400, std::string("malformed WAV: ") + what);
}

[[noreturn]] void
too_short() {
    fail(400, "audio is shorter than one feature frame (320 samples at 16 kHz)");
}

uint16_t
le16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t
le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

struct StrictWav {
    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t sample_rate = 0;
    const uint8_t* data = nullptr;
    size_t data_len = 0;
    bool canonical_header = false;  // 44-byte header, nothing after the data chunk
    bool open_ended = false;        // streamed WAV: data size 0xFFFFFFFF, read to the end
};

// Strict RIFF/WAVE parsing for every RIFF upload: a declared size past the end of the file, a
// truncated chunk (even after the data chunk) or a missing fmt/data chunk is a 400. Only the first
// fmt and data chunks count; block_align and byte_rate are ignored. A streamed WAV (written where
// the writer could not seek back, e.g. an ffmpeg pipe) declares 0xFFFFFFFF for the RIFF and data
// sizes: its data runs to the end of the file, cut to whole frames.
StrictWav
parse_strict_wav(const uint8_t* in, size_t len) {
    if (len < 12 || std::memcmp(in, "RIFF", 4) != 0 || std::memcmp(in + 8, "WAVE", 4) != 0)
        malformed("missing RIFF/WAVE header");
    constexpr uint32_t kUnknownSize = 0xFFFFFFFFu;
    const uint64_t declared = le32(in + 4);
    if (declared != kUnknownSize && declared + 8 > len)
        malformed("truncated RIFF payload");
    size_t pos = 12;
    bool have_fmt = false, have_data = false;
    size_t data_offset = 0;
    StrictWav w;
    while (pos + 8 <= len) {
        const uint8_t* id = in + pos;
        uint64_t chunk = le32(in + pos + 4);
        const uint64_t start = pos + 8;
        const bool open_ended = chunk == kUnknownSize && std::memcmp(id, "data", 4) == 0;
        if (open_ended)
            chunk = len - start;
        const uint64_t end = start + chunk;
        if (end > len)
            malformed("truncated chunk");
        if (std::memcmp(id, "fmt ", 4) == 0 && !have_fmt) {
            if (chunk < 16)
                malformed("short fmt chunk");
            w.format = le16(in + start);
            w.channels = le16(in + start + 2);
            w.sample_rate = le32(in + start + 4);
            w.bits = le16(in + start + 14);
            have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0 && !have_data) {
            w.data = in + start;
            w.data_len = static_cast<size_t>(chunk);
            data_offset = static_cast<size_t>(start);
            have_data = true;
            if (open_ended) {
                w.open_ended = true;
                break;
            }
        }
        pos = static_cast<size_t>(end + (chunk & 1));
    }
    if (!have_fmt)
        malformed("missing fmt chunk");
    if (!have_data)
        malformed("missing data chunk");
    if (w.channels == 0 || w.sample_rate == 0)
        fail(400, "invalid audio: zero channels or sample rate");
    if (w.open_ended && w.bits >= 8) {
        const size_t frame = static_cast<size_t>(w.bits / 8) * w.channels;
        w.data_len -= w.data_len % frame;
    }
    w.canonical_header = data_offset == 44 && len == 44 + w.data_len;
    return w;
}

// Bytes per frame for the encodings decoded on the integer path (PCM16, A-law 8, mu-law 8); 0 for
// any other.
size_t
integer_frame_bytes(const StrictWav& w) {
    size_t bytes = 0;
    if (w.format == kPcm && w.bits == 16)
        bytes = 2;
    else if ((w.format == kAlaw || w.format == kMulaw) && w.bits == 8)
        bytes = 1;
    else
        return 0;
    const size_t frame = bytes * w.channels;
    if (w.data_len % frame != 0)
        malformed("partial audio frame");
    return frame;
}

// The most source frames at `rate` within max_samples, plus one (saturating).
uint64_t
max_sample_frames(const IngestOptions& options, uint32_t rate) {
    const uint64_t rate64 = rate;
    if (rate64 == 0 ||
        static_cast<uint64_t>(options.max_samples) > std::numeric_limits<uint64_t>::max() / rate64)
        return std::numeric_limits<uint64_t>::max();
    return static_cast<uint64_t>(options.max_samples) * rate64 / kTargetRate + 1;
}

void
check_duration(double seconds, const IngestOptions& options) {
    if (options.max_duration_seconds > 0 && seconds > options.max_duration_seconds + 1e-6) {
        char buf[128];
        std::snprintf(
            buf, sizeof(buf), "audio is %.1f s, the limit is %.0f s", seconds,
            options.max_duration_seconds);
        over_limit(buf);
    }
}

void
check_rate(uint32_t rate) {
    if (rate != kTargetRate && (rate > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
                                !supported_input_sample_rate(static_cast<int>(rate))))
        fail(400, "invalid audio: sample rate must be between 8000 and 96000 Hz");
}

std::vector<float>
to_16k(std::vector<float> mono, uint32_t rate) {
    if (rate == kTargetRate)
        return mono;
    std::vector<float> out = resample_audio(
        mono.data(), mono.size(), static_cast<int>(rate), static_cast<int>(kTargetRate));
    pcm16_round_trip(out);
    return out;
}

DecodedAudio
finish(
    std::vector<float> samples, int source_rate, int channels, std::string decoder,
    const IngestOptions& options) {
    DecodedAudio out;
    out.duration = static_cast<double>(samples.size()) / kTargetRate;
    check_duration(out.duration, options);
    if (samples.size() < options.min_samples)
        too_short();
    out.samples = std::move(samples);
    out.source_rate = source_rate;
    out.channels = channels;
    out.decoder = std::move(decoder);
    return out;
}

// PCM16, A-law and mu-law WAV: integer channel mean (truncated toward zero), then 16 kHz.
DecodedAudio
decode_integer_wav(const StrictWav& w, size_t frame_bytes, const IngestOptions& options) {
    const size_t frames = w.data_len / frame_bytes;
    // Bound the upload before decoding: max_samples * rate / 16000 + 1 source frames (saturating).
    if (static_cast<uint64_t>(frames) > max_sample_frames(options, w.sample_rate))
        over_limit();
    const bool passthrough = w.canonical_header && w.format == kPcm && w.channels == 1 &&
                             w.sample_rate == kTargetRate && w.bits == 16;
    if (passthrough && frames < options.min_samples)
        too_short();
    check_rate(w.sample_rate);
    std::vector<float> mono(frames);
    const size_t sample_bytes = frame_bytes / w.channels;
    for (size_t f = 0; f < frames; ++f) {
        const uint8_t* frame = w.data + f * frame_bytes;
        int64_t sum = 0;
        for (size_t c = 0; c < w.channels; ++c) {
            const uint8_t* s = frame + c * sample_bytes;
            sum += w.format == kPcm ? static_cast<int16_t>(le16(s))
                                    : (w.format == kAlaw ? decode_alaw(s[0]) : decode_mulaw(s[0]));
        }
        mono[f] = static_cast<float>(static_cast<int16_t>(sum / static_cast<int64_t>(w.channels))) *
                  (1.0f / 32768.0f);
    }
    const char* how = w.sample_rate != kTargetRate
                          ? "wav-resampled"
                          : (passthrough ? "wav-passthrough" : "wav-decoded");
    std::vector<float> samples = to_16k(std::move(mono), w.sample_rate);
    // The passthrough path has only the pre-check above (max + 1 frames); the duration limit covers
    // the rest.
    if (!passthrough && samples.size() > options.max_samples)
        over_limit();
    if (samples.size() < options.min_samples)
        too_short();
    return finish(std::move(samples), static_cast<int>(w.sample_rate), w.channels, how, options);
}

}  // namespace

namespace detail {

void
mix_down(const float* interleaved, size_t frames, int channels, float* mono) {
    for (size_t f = 0; f < frames; ++f) {
        const float* row = interleaved + f * static_cast<size_t>(channels);
        float sum = row[0];
        for (int c = 1; c < channels; ++c) sum += row[c];
        mono[f] = static_cast<float>(static_cast<double>(sum) / static_cast<double>(channels));
    }
}

uint64_t
frame_cap(const IngestOptions& options, uint32_t rate) {
    uint64_t cap = max_sample_frames(options, rate);
    if (options.max_duration_seconds > 0) {
        const double frames = std::floor(options.max_duration_seconds * rate) + 1;
        if (frames < static_cast<double>(cap))
            cap = static_cast<uint64_t>(frames);
    }
    return cap;
}

}  // namespace detail

int16_t
decode_mulaw(uint8_t value) {
    value = static_cast<uint8_t>(~value);
    const int sign = value & 0x80;
    const int exponent = (value >> 4) & 0x07;
    const int mantissa = value & 0x0f;
    const int magnitude = ((mantissa << 3) + 0x84) << exponent;
    const int sample = magnitude - 0x84;
    return static_cast<int16_t>(sign != 0 ? -sample : sample);
}

int16_t
decode_alaw(uint8_t value) {
    value = static_cast<uint8_t>(value ^ 0x55);
    int sample = (value & 0x0f) << 4;
    const int segment = (value & 0x70) >> 4;
    sample += 8;
    if (segment >= 1)
        sample += 0x100;
    if (segment > 1)
        sample <<= segment - 1;
    return static_cast<int16_t>((value & 0x80) != 0 ? sample : -sample);
}

void
pcm16_round_trip(std::vector<float>& samples) {
    for (float& v : samples) {
        const float scaled = v * static_cast<float>(8.0 * 0x10000000);
        int16_t q;
        if (static_cast<double>(scaled) >= 1.0 * 0x7FFFFFFF) {
            q = 32767;
        } else if (static_cast<double>(scaled) <= -8.0 * 0x10000000) {
            q = -32768;
        } else {
            const int32_t value = std::isnan(scaled) ? std::numeric_limits<int32_t>::min()
                                                     : static_cast<int32_t>(lrintf(scaled));
            q = static_cast<int16_t>(value >> 16);
        }
        v = static_cast<float>(q) * (1.0f / static_cast<float>(0x8000));
    }
}

DecodedAudio
decode_any(
    const void* data_ptr, size_t size, std::string_view /*name*/, const IngestOptions& options) {
    const auto* data = static_cast<const uint8_t*>(data_ptr);
    if (data == nullptr || size == 0)
        fail(422, "invalid or unsupported audio file");

    const detail::Kind kind = detail::probe_kind(data, size);
    if (kind == detail::Kind::RiffWave) {
        const StrictWav w = parse_strict_wav(data, size);
        if (const size_t frame_bytes = integer_frame_bytes(w))
            return decode_integer_wav(w, frame_bytes, options);
    }

    detail::PcmFloat pcm;
    bool decoded = false;
    switch (kind) {
        case detail::Kind::RiffWave:
        case detail::Kind::Rifx:
        case detail::Kind::Rf64:
            decoded = detail::decode_wav_family(data, size, options, pcm);
            if (!decoded) {
                const int tag = detail::wav_format_tag(data, size);
                if (tag >= 0) {
                    char buf[200];
                    std::snprintf(
                        buf, sizeof(buf),
                        "unsupported audio format: WAV format tag 0x%04X (this build decodes PCM, "
                        "IEEE float, "
                        "A-law and mu-law WAV)",
                        static_cast<unsigned>(tag));
                    fail(422, buf);
                }
            }
            break;
        case detail::Kind::Flac:
            decoded = detail::decode_flac(data, size, options, pcm);
            break;
        case detail::Kind::Mp3:
            decoded = detail::decode_mp3(data, size, options, pcm);
            break;
        case detail::Kind::Other:
            fail(422, "unsupported audio format: this build decodes WAV, FLAC and MP3");
    }
    if (!decoded)
        fail(422, "invalid or unsupported audio file");
    if (pcm.over_limit) {
        // the stream's own length when its header gives one, the plain limit otherwise
        if (pcm.total_frames > 0)
            check_duration(static_cast<double>(pcm.total_frames) / pcm.sample_rate, options);
        over_limit();
    }
    if (pcm.frames == 0)
        fail(422, "audio must be non-empty");
    if (pcm.channels < 1 || pcm.sample_rate <= 0)
        fail(400, "invalid audio: zero channels or sample rate");
    check_duration(static_cast<double>(pcm.frames) / pcm.sample_rate, options);
    check_rate(static_cast<uint32_t>(pcm.sample_rate));

    // float32 channel mean, divided in double
    std::vector<float> mono;
    if (pcm.mixed) {
        mono = std::move(pcm.interleaved);
    } else {
        mono.resize(pcm.frames);
        detail::mix_down(pcm.interleaved.data(), pcm.frames, pcm.channels, mono.data());
        std::vector<float>().swap(pcm.interleaved);
    }
    std::vector<float> samples;
    if (pcm.sample_rate != static_cast<int>(kTargetRate)) {
        samples = to_16k(std::move(mono), static_cast<uint32_t>(pcm.sample_rate));
    } else {
        samples = std::move(mono);
        if (!(pcm.wav_pcm16 && pcm.channels == 1))
            pcm16_round_trip(samples);
    }
    return finish(std::move(samples), pcm.sample_rate, pcm.channels, pcm.decoder, options);
}

DecodedAudio
decode_raw(
    const void* data, size_t size, RawEncoding encoding, int sample_rate, int channels,
    const IngestOptions& options) {
    if (size == 0)
        fail(422, "audio must be non-empty");
    if (sample_rate <= 0)
        fail(400, "invalid audio: zero channels or sample rate");
    if (channels < 1 || channels > 65535)
        fail(400, "invalid audio: channel count must be between 1 and 65535");
    const uint16_t bits = encoding == RawEncoding::Pcm16 ? 16 : 8;
    const uint16_t format =
        encoding == RawEncoding::Pcm16 ? kPcm : (encoding == RawEncoding::Mulaw ? kMulaw : kAlaw);
    const size_t block_align = static_cast<size_t>(channels) * bits / 8;
    if (size % block_align != 0)
        fail(400, "invalid audio: the byte count is not a whole number of sample frames");
    if (size > 0xFFFFFFFFull - 36)
        over_limit();
    std::string wav;
    wav.reserve(44 + size);
    auto u16 = [&](uint16_t v) {
        wav.push_back(static_cast<char>(v & 0xff));
        wav.push_back(static_cast<char>(v >> 8));
    };
    auto u32 = [&](uint32_t v) {
        for (int sh = 0; sh < 32; sh += 8) wav.push_back(static_cast<char>((v >> sh) & 0xff));
    };
    wav += "RIFF";
    u32(static_cast<uint32_t>(36 + size));
    wav += "WAVEfmt ";
    u32(16);
    u16(format);
    u16(static_cast<uint16_t>(channels));
    u32(static_cast<uint32_t>(sample_rate));
    u32(static_cast<uint32_t>(static_cast<uint64_t>(sample_rate) * block_align));
    u16(static_cast<uint16_t>(block_align));
    u16(bits);
    wav += "data";
    u32(static_cast<uint32_t>(size));
    wav.append(static_cast<const char*>(data), size);
    return decode_any(wav.data(), wav.size(), "audio", options);
}

bool
is_audio_path(const std::string& path) {
    const auto dot = path.find_last_of('.');
    const auto slash = path.find_last_of("/\\");
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
        return false;
    std::string ext = path.substr(dot + 1);
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext == "wav" || ext == "wave" || ext == "flac" || ext == "mp3";
}

DecodedAudio
decode_file(const std::string& path, const IngestOptions& options) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error(path + ": cannot open audio file");
    const std::streamoff end = input.tellg();
    std::string bytes(static_cast<size_t>(std::max<std::streamoff>(end, 0)), '\0');
    input.seekg(0);
    if (!bytes.empty() && !input.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error(path + ": could not read the complete file");
    try {
        return decode_any(bytes.data(), bytes.size(), path, options);
    }
    catch (const AudioLimitError& e) {
        throw AudioLimitError(path + ": " + e.what());
    }
    catch (const AudioError& e) {
        throw AudioError(e.http_status(), path + ": " + e.what());
    }
}

}  // namespace nemo_speech::audio
