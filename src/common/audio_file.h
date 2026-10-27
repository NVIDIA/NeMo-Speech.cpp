// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "audio_ingest.h"

namespace nemo_speech::audio {

struct AudioFile {
    std::vector<float> samples;
    int sample_rate = 0;
    int source_channels = 0;
};

// Load a WAV (PCM, IEEE float, A-law, mu-law), FLAC or MP3 file as 16 kHz mono samples through the
// audio ingest (audio_ingest.h); sample_rate is always 16000. A local file has no length limit (a
// model applies its own). Throws AudioError (std::invalid_argument; 400/422 status) for
// undecodable audio.
AudioFile load_wav_file(const std::string& path);

// The same for an in-memory upload, within the upload limits (IngestOptions). `source_name`
// prefixes error messages.
AudioFile load_wav_memory(
    const void* data, size_t size, const std::string& source_name = "audio upload",
    const IngestOptions& options = {});

// WAV at the file's native rate: PCM16 and IEEE float32, mono or stereo, 8-96 kHz, stereo averaged
// in float. Throws std::runtime_error.
AudioFile load_wav_native(const std::string& path);

// Wrap mono signed-16 little-endian PCM in a canonical RIFF/WAVE header.
std::string pcm16_wav(const std::string& pcm, int sample_rate);

bool is_wav_path(const std::string& path);

}  // namespace nemo_speech::audio
