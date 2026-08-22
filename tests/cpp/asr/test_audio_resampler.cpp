// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "audio_resampler.h"

using nemo_speech::audio::AudioResampler;
using nemo_speech::audio::Pcm16Resampler;
using nemo_speech::audio::resample_audio;

namespace {

int failures = 0;

void
check(bool condition, const char* label) {
    std::fprintf(stdout, "[%s] %s\n", condition ? "PASS" : "FAIL", label);
    if (!condition)
        ++failures;
}

std::vector<float>
sine(int sample_rate, double frequency, double seconds) {
    constexpr double kPi = 3.14159265358979323846;
    const size_t count = static_cast<size_t>(sample_rate * seconds + 0.5);
    std::vector<float> output(count);
    for (size_t i = 0; i < count; ++i)
        output[i] = static_cast<float>(std::sin(2.0 * kPi * frequency * i / sample_rate));
    return output;
}

double
rms(const std::vector<float>& values, size_t trim = 0) {
    if (values.size() <= trim * 2)
        return 0.0;
    double sum = 0.0;
    for (size_t i = trim; i < values.size() - trim; ++i)
        sum += static_cast<double>(values[i]) * values[i];
    return std::sqrt(sum / static_cast<double>(values.size() - trim * 2));
}

double
rmse_against_sine(
    const std::vector<float>& values, int sample_rate, double frequency, size_t trim) {
    constexpr double kPi = 3.14159265358979323846;
    double sum = 0.0;
    size_t count = 0;
    for (size_t i = trim; i + trim < values.size(); ++i) {
        const double expected = std::sin(2.0 * kPi * frequency * i / sample_rate);
        const double error = values[i] - expected;
        sum += error * error;
        ++count;
    }
    return count ? std::sqrt(sum / count) : 0.0;
}

std::vector<uint8_t>
pcm16(const std::vector<float>& input) {
    std::vector<uint8_t> output;
    output.reserve(input.size() * 2);
    for (float sample : input) {
        sample = std::max(-1.0f, std::min(1.0f, sample));
        const int16_t value = static_cast<int16_t>(std::lrintf(sample * 32767.0f));
        const uint16_t bits = static_cast<uint16_t>(value);
        output.push_back(static_cast<uint8_t>(bits & 0xff));
        output.push_back(static_cast<uint8_t>((bits >> 8) & 0xff));
    }
    return output;
}

}  // namespace

int
main() {
    const auto source_48k = sine(48000, 1000.0, 1.0);
    const auto downsampled = resample_audio(source_48k.data(), source_48k.size(), 48000, 16000);
    check(downsampled.size() == 16000, "48 kHz -> 16 kHz output duration");
    check(
        rmse_against_sine(downsampled, 16000, 1000.0, 32) < 0.01,
        "downsampling preserves an in-band tone");

    const auto source_44k = sine(44100, 1000.0, 1.0);
    const auto fractional = resample_audio(source_44k.data(), source_44k.size(), 44100, 16000);
    check(fractional.size() == 16000, "44.1 kHz -> 16 kHz preserves duration without drift");
    check(
        rmse_against_sine(fractional, 16000, 1000.0, 32) < 0.01,
        "fractional-ratio conversion preserves an in-band tone");

    const auto source_coprime = sine(47999, 1000.0, 1.0);
    const auto coprime = resample_audio(source_coprime.data(), source_coprime.size(), 47999, 16000);
    check(coprime.size() == 16000, "co-prime rate conversion preserves duration");
    check(
        rmse_against_sine(coprime, 16000, 1000.0, 32) < 0.01,
        "co-prime rate conversion preserves an in-band tone");

    const auto source_8k = sine(8000, 1000.0, 1.0);
    const auto upsampled = resample_audio(source_8k.data(), source_8k.size(), 8000, 16000);
    check(upsampled.size() == 16000, "8 kHz -> 16 kHz output duration");
    check(
        rmse_against_sine(upsampled, 16000, 1000.0, 32) < 0.01,
        "upsampling preserves an in-band tone");

    const auto out_of_band = sine(48000, 12000.0, 1.0);
    const auto filtered = resample_audio(out_of_band.data(), out_of_band.size(), 48000, 16000);
    check(
        rms(filtered, 32) < 0.03, "downsampling suppresses frequencies above destination Nyquist");

    AudioResampler streaming(48000, 16000);
    std::vector<float> chunked;
    size_t offset = 0;
    static constexpr size_t kChunks[] = {1, 17, 511, 160, 997, 23};
    size_t chunk_index = 0;
    while (offset < source_48k.size()) {
        const size_t count = std::min(
            kChunks[chunk_index++ % (sizeof(kChunks) / sizeof(kChunks[0]))],
            source_48k.size() - offset);
        streaming.process(source_48k.data() + offset, count, &chunked);
        offset += count;
    }
    streaming.finish(&chunked);
    check(chunked == downsampled, "streaming output is independent of chunk boundaries");

    AudioResampler fractional_streaming(44100, 16000);
    chunked.clear();
    offset = 0;
    chunk_index = 0;
    while (offset < source_44k.size()) {
        const size_t count = std::min(
            kChunks[chunk_index++ % (sizeof(kChunks) / sizeof(kChunks[0]))],
            source_44k.size() - offset);
        fractional_streaming.process(source_44k.data() + offset, count, &chunked);
        offset += count;
    }
    fractional_streaming.finish(&chunked);
    check(
        chunked == fractional,
        "fractional-ratio streaming output is independent of chunk boundaries");

    const auto source_pcm = pcm16(source_48k);
    Pcm16Resampler pcm_resampler(48000, 11025);
    std::vector<uint8_t> resampled_pcm;
    size_t byte_offset = 0;
    static constexpr size_t kPcmChunkSamples[] = {1, 31, 512, 79, 1001};
    chunk_index = 0;
    while (byte_offset < source_pcm.size()) {
        const size_t samples = std::min(
            kPcmChunkSamples
                [chunk_index++ % (sizeof(kPcmChunkSamples) / sizeof(kPcmChunkSamples[0]))],
            (source_pcm.size() - byte_offset) / 2);
        pcm_resampler.process(source_pcm.data() + byte_offset, samples * 2, &resampled_pcm);
        byte_offset += samples * 2;
    }
    pcm_resampler.finish(&resampled_pcm);
    check(
        pcm_resampler.output_samples() == 11025 && resampled_pcm.size() == 11025 * 2,
        "signed-16 PCM streaming supports 11025 Hz output");

    Pcm16Resampler pcm_one_shot(48000, 11025);
    std::vector<uint8_t> one_shot_pcm;
    pcm_one_shot.process(source_pcm.data(), source_pcm.size(), &one_shot_pcm);
    pcm_one_shot.finish(&one_shot_pcm);
    check(
        resampled_pcm == one_shot_pcm,
        "signed-16 PCM output is independent of streaming chunk boundaries");

    bool odd_pcm_rejected = false;
    try {
        Pcm16Resampler odd_pcm(22050, 16000);
        std::vector<uint8_t> output;
        const uint8_t byte = 0;
        odd_pcm.process(&byte, 1, &output);
    }
    catch (const std::invalid_argument&) {
        odd_pcm_rejected = true;
    }
    check(odd_pcm_rejected, "signed-16 PCM rejects incomplete samples");

    const auto passthrough = resample_audio(source_8k.data(), source_8k.size(), 8000, 8000);
    check(passthrough == source_8k, "equal sample rates are an exact pass-through");

    bool rejected = false;
    try {
        AudioResampler invalid(4000, 16000);
        static_cast<void>(invalid);
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "unsupported input sample rate is rejected");

    std::fprintf(stdout, failures ? "FAILED (%d)\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
