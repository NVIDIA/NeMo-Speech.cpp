// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Audio ingest: turns an uploaded audio file into 16 kHz mono float samples on the PCM16 grid
// (int16 / 32768).
//
// Accepted input (decoded by content; file names and format hints are not used):
//   WAV: RIFF, RIFX and RF64/BW64; PCM 8/16/24/32-bit, IEEE float 32/64, A-law, mu-law,
//        WAVE_FORMAT_EXTENSIBLE with those encodings; any channel count and sample rate in 8-96
//        kHz.
//   FLAC and MP3 (miniaudio's dr_flac / dr_mp3).
// Every other container (Ogg, MP4/M4A, WebM, AAC, AIFF, ...) is rejected with 422.
//
// Processing:
//   PCM16, A-law and mu-law WAV: strict RIFF parsing (400 on malformed headers), integer channel
//   mean
//       (truncated toward zero), resampling to 16 kHz when needed.
//   Everything else: decoded to float32 (8-bit (u - 128) / 128, 16-bit / 32768, 24/32-bit
//   left-justified to
//       int32 * 2^-31, float as is), float32 channel mean, resampling to 16 kHz when needed.
//   Resampling uses audio::resample_audio. Results that are not already PCM16 values are quantized
//   to the PCM16 grid by clipping and flooring (pcm16_round_trip).
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace nemo_speech::audio {

// A client error with an HTTP status (400 or 422). Derives from std::invalid_argument, so
// transports that map invalid_argument to 400 / INVALID_ARGUMENT keep working.
class AudioError : public std::invalid_argument {
   public:
    AudioError(int http_status, const std::string& message)
        : std::invalid_argument(message), http_status_(http_status) {}
    int http_status() const { return http_status_; }

   private:
    int http_status_;
};

// Audio longer than the IngestOptions limits (400). Decoders stop reading at the limit.
class AudioLimitError : public AudioError {
   public:
    explicit AudioLimitError(const std::string& message) : AudioError(400, message) {}
};

// Longest upload the servers accept.
inline constexpr double kMaxUploadSeconds = 3600.0;

struct IngestOptions {
    // Cap on 16 kHz samples; SIZE_MAX disables.
    size_t max_samples = static_cast<size_t>(kMaxUploadSeconds) * 16000;
    // Duration limit in seconds; 0 disables.
    double max_duration_seconds = kMaxUploadSeconds;
    // Fewest 16 kHz samples (two 10 ms mel frames).
    size_t min_samples = 320;

    // Uploads: the defaults. Local files: no length limit (a model applies its own).
    static IngestOptions local_file() {
        IngestOptions options;
        options.max_samples = static_cast<size_t>(-1);
        options.max_duration_seconds = 0;
        return options;
    }
};

struct DecodedAudio {
    std::vector<float> samples;  // 16 kHz mono, int16 / 32768
    int source_rate = 0;         // sample rate of the upload
    int channels = 0;            // channel count of the upload
    double duration = 0.0;       // samples.size() / 16000
    // "wav-passthrough", "wav-decoded", "wav-resampled" (PCM16 / G.711 WAV); "wav", "flac", "mp3".
    std::string decoder;
};

// Decode an upload into 16 kHz mono samples. `name` (a filename or format) is used only in error
// messages. Throws AudioError:
//   400 "malformed WAV: <reason>", "invalid audio: <reason>",
//       "audio is shorter than one feature frame (320 samples at 16 kHz)",
//       AudioLimitError "audio exceeds configured limit", "audio is <d> s, the limit is <limit> s"
//       (FLAC and MP3 decoding stops at the limit, so a small upload cannot expand past it)
//   422 "invalid or unsupported audio file" (undecodable), "audio must be non-empty"
//       (no samples), "unsupported audio format: ..." (a container or WAV encoding this build does
//       not decode).
DecodedAudio decode_any(
    const void* data, size_t size, std::string_view name = {}, const IngestOptions& options = {});
DecodedAudio decode_file(const std::string& path, const IngestOptions& options = {});

// Headerless interleaved samples (little-endian signed-16, or 8-bit G.711) at `sample_rate` with
// `channels` channels, decoded exactly as the same samples in a WAV upload. Used for Riva
// LINEAR_PCM / MULAW / ALAW requests. Throws AudioError(400) for a partial sample frame.
enum class RawEncoding { Pcm16, Mulaw, Alaw };
DecodedAudio decode_raw(
    const void* data, size_t size, RawEncoding encoding, int sample_rate, int channels,
    const IngestOptions& options = {});

// Extensions directory scans treat as audio (decoding itself goes by content): wav, wave, flac,
// mp3.
bool is_audio_path(const std::string& path);

// ITU-T G.711 decoding to 16-bit PCM.
int16_t decode_mulaw(uint8_t value);
int16_t decode_alaw(uint8_t value);

// Quantize to the PCM16 grid: x * 2^31 clipped to int32, rounded to nearest, then the high 16 bits
// (a floor to multiples of 1/32768), returned as int16 / 32768. Values already on the grid are
// unchanged.
void pcm16_round_trip(std::vector<float>& samples);

}  // namespace nemo_speech::audio
