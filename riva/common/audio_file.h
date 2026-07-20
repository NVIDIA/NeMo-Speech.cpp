// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace nemo_speech::audio {

struct AudioFile {
    std::vector<float> samples;
    int sample_rate = 0;
    int source_channels = 0;
};

// Load an uncompressed RIFF/WAVE file into mono float32 samples. PCM16 and
// IEEE float32, mono or stereo, and sample rates from 8 to 96 kHz are accepted.
// Stereo is downmixed by averaging channels. Throws std::runtime_error with an
// actionable message for malformed files and unsupported codecs.
AudioFile load_wav_file(const std::string& path);

// Parse an in-memory RIFF/WAVE body. `source_name` is included in validation
// errors.
AudioFile load_wav_memory(
    const void* data, size_t size, const std::string& source_name = "audio upload");

// Wrap mono signed-16 little-endian PCM in a canonical RIFF/WAVE header.
std::string pcm16_wav(const std::string& pcm, int sample_rate);

bool is_wav_path(const std::string& path);

}  // namespace nemo_speech::audio
