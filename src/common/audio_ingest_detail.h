// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Decoders of audio_ingest.cpp (internal).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "audio_ingest.h"

namespace nemo_speech::audio::detail {

// A decoded file before channel mixing and resampling.
struct PcmFloat {
    std::vector<float> interleaved;  // frames x channels, or frames when `mixed`
    size_t frames = 0;
    int channels = 0;
    int sample_rate = 0;
    // Plain (non-extensible) RIFF or RIFX WAV with 16-bit PCM: the samples are PCM16 values
    // already.
    bool wav_pcm16 = false;
    // FLAC and MP3: `interleaved` holds the channel mean (mix_down) of each frame.
    bool mixed = false;
    // Decoding stopped at frame_cap (FLAC, MP3) or did not start (WAV); total_frames is the
    // stream's length when its header gives one (0 otherwise).
    bool over_limit = false;
    uint64_t total_frames = 0;
    std::string decoder;
};

// The float32 channel mean of `frames` interleaved frames, divided in double.
void mix_down(const float* interleaved, size_t frames, int channels, float* mono);

// Source frames at `rate` that exceed the limits (the most they admit, plus one; saturating).
uint64_t frame_cap(const IngestOptions& options, uint32_t rate);

// Container kind from the leading bytes.
enum class Kind {
    RiffWave,  // "RIFF....WAVE"
    Rifx,      // big-endian WAV
    Rf64,      // "RF64"/"BW64"....WAVE
    Flac,      // "fLaC" (or ID3 + fLaC)
    Mp3,       // ID3v2 tag or MPEG audio frame sync
    Other,     // anything else (Ogg, MP4/M4A, WebM, AAC, AIFF, W64, ...): not decoded by this build
};
Kind probe_kind(const uint8_t* data, size_t size);

// WAV family reader (RIFF/RIFX/RF64) with the integer-to-float scaling of audio_ingest.h: PCM
// 8/16/24/32-bit, IEEE float 32/64, A-law, mu-law, WAVE_FORMAT_EXTENSIBLE with those. False when it
// does not handle the input. A stream of frame_cap(options, rate) frames or more is not decoded
// (over_limit, with total_frames).
bool decode_wav_family(
    const uint8_t* data, size_t size, const IngestOptions& options, PcmFloat& out);
// The WAV format tag (resolved through WAVE_FORMAT_EXTENSIBLE), or -1 when the header cannot be
// parsed.
int wav_format_tag(const uint8_t* data, size_t size);

// dr_flac / dr_mp3, mixed down per block; they stop (over_limit) at frame_cap(options, rate).
bool decode_flac(const uint8_t* data, size_t size, const IngestOptions& options, PcmFloat& out);
bool decode_mp3(const uint8_t* data, size_t size, const IngestOptions& options, PcmFloat& out);

}  // namespace nemo_speech::audio::detail
