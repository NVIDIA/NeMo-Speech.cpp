// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// FLAC and MP3 decoding of the audio ingest: miniaudio's embedded dr_flac and dr_mp3
// (llama.cpp/vendor/miniaudio, Unlicense / MIT-0). Every miniaudio symbol is internal to this
// translation unit (MA_API static), so it cannot clash with the CLI's own miniaudio build for
// microphone capture.
#include <cstring>
#include <memory>
#include <vector>

#include "audio_ingest_detail.h"

#define MA_API static
// The one miniaudio global that is not declared through MA_API: rename it so it cannot collide with
// the CLI's microphone-capture build of miniaudio.
#define ma_atomic_global_lock nemo_speech_ingest_ma_atomic_global_lock
#define MA_NO_DEVICE_IO
#define MA_NO_THREADING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_WAV
#define MINIAUDIO_IMPLEMENTATION
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif
#include "miniaudio.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace nemo_speech::audio::detail {
namespace {

constexpr ma_uint64 kBlockFrames = 4096;

struct FlacCloser {
    void operator()(ma_dr_flac* flac) const { ma_dr_flac_close(flac); }
};

struct Mp3Closer {
    void operator()(ma_dr_mp3* mp3) const { ma_dr_mp3_uninit(mp3); }
};

void
set_format(PcmFloat& out, int channels, int rate, const char* decoder) {
    out.channels = channels;
    out.sample_rate = rate;
    out.wav_pcm16 = false;
    out.mixed = true;
    out.decoder = decoder;
}

// Appends the channel mean of `frames` interleaved frames; true at the cap.
bool
append_block(const float* block, size_t frames, uint64_t cap, PcmFloat& out) {
    const size_t at = out.interleaved.size();
    out.interleaved.resize(at + frames);
    mix_down(block, frames, out.channels, out.interleaved.data() + at);
    out.frames = out.interleaved.size();
    out.over_limit = out.frames >= cap;
    return out.over_limit;
}

}  // namespace

bool
decode_flac(const uint8_t* data, size_t size, const IngestOptions& options, PcmFloat& out) {
    const std::unique_ptr<ma_dr_flac, FlacCloser> flac(ma_dr_flac_open_memory(data, size, nullptr));
    if (flac == nullptr || flac->channels == 0 || flac->sampleRate == 0)
        return false;
    const int channels = flac->channels;
    set_format(out, channels, static_cast<int>(flac->sampleRate), "flac");
    const uint64_t cap = frame_cap(options, flac->sampleRate);
    out.total_frames = flac->totalPCMFrameCount;
    if (out.total_frames >= cap) {
        out.over_limit = true;
        return true;
    }
    std::vector<int32_t> block(static_cast<size_t>(kBlockFrames) * channels);
    std::vector<float> scaled(block.size());
    for (;;) {
        const ma_uint64 got =
            ma_dr_flac_read_pcm_frames_s32(flac.get(), kBlockFrames, block.data());
        if (got == 0)
            break;
        // Left-justified 32-bit samples scaled by 2^-31 (exact for up to 24 bits).
        const size_t n = static_cast<size_t>(got) * channels;
        for (size_t i = 0; i < n; ++i)
            scaled[i] = static_cast<float>(block[i]) * (1.0f / static_cast<float>(0x80000000u));
        if (append_block(scaled.data(), static_cast<size_t>(got), cap, out))
            break;
    }
    return true;
}

bool
decode_mp3(const uint8_t* data, size_t size, const IngestOptions& options, PcmFloat& out) {
    ma_dr_mp3 state;
    if (!ma_dr_mp3_init_memory(&state, data, size, nullptr))
        return false;
    const std::unique_ptr<ma_dr_mp3, Mp3Closer> mp3(&state);
    if (mp3->channels == 0 || mp3->sampleRate == 0)
        return false;
    const int channels = static_cast<int>(mp3->channels);
    set_format(out, channels, static_cast<int>(mp3->sampleRate), "mp3");
    const uint64_t cap = frame_cap(options, mp3->sampleRate);
    std::vector<float> block(static_cast<size_t>(kBlockFrames) * channels);
    for (;;) {
        const ma_uint64 got = ma_dr_mp3_read_pcm_frames_f32(mp3.get(), kBlockFrames, block.data());
        if (got == 0)
            break;
        if (append_block(block.data(), static_cast<size_t>(got), cap, out))
            break;
    }
    return true;
}

}  // namespace nemo_speech::audio::detail
