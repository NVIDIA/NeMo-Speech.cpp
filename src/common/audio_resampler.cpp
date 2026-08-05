// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "audio_resampler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace nemo_speech::audio {
namespace {

constexpr int kMinInputRate = 8000;
constexpr int kMaxInputRate = 96000;
constexpr int kHalfTaps = 16;
constexpr size_t kTaps = kHalfTaps * 2;
constexpr size_t kMaxPhases = 2048;
constexpr long double kPi = 3.141592653589793238462643383279502884L;

long double
sinc(long double value) {
    if (std::abs(value) < 1.0e-12L)
        return 1.0L;
    const long double angle = kPi * value;
    return std::sin(angle) / angle;
}

uint64_t
rounded_output_count(uint64_t input_count, int input_rate, int output_rate) {
    const uint64_t input_rate_u64 = static_cast<uint64_t>(input_rate);
    const uint64_t output_rate_u64 = static_cast<uint64_t>(output_rate);
    const uint64_t whole = input_count / input_rate_u64;
    const uint64_t remainder = input_count % input_rate_u64;
    // Skip the __int128 path under the MSVC ABI: clang-cl defines
    // __SIZEOF_INT128__ but 128-bit division needs compiler-rt's __udivti3,
    // which isn't in the MSVC link environment. The #else branch is the
    // portable equivalent.
#if defined(__SIZEOF_INT128__) && !defined(_MSC_VER)
    using Wide = unsigned __int128;
    const Wide rounded_fraction =
        (static_cast<Wide>(remainder) * output_rate_u64 + input_rate_u64 / 2) / input_rate_u64;
    const Wide total = static_cast<Wide>(whole) * output_rate_u64 + rounded_fraction;
    if (total > std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("resampled audio is too large");
    return static_cast<uint64_t>(total);
#else
    const uint64_t half_input_rate = input_rate_u64 / 2;
    if (remainder > (std::numeric_limits<uint64_t>::max() - half_input_rate) / output_rate_u64)
        throw std::overflow_error("resampled audio is too large");
    const uint64_t rounded_fraction =
        (remainder * output_rate_u64 + half_input_rate) / input_rate_u64;
    if (whole > (std::numeric_limits<uint64_t>::max() - rounded_fraction) / output_rate_u64)
        throw std::overflow_error("resampled audio is too large");
    return whole * output_rate_u64 + rounded_fraction;
#endif
}

}  // namespace

bool
supported_input_sample_rate(int sample_rate) {
    return sample_rate >= kMinInputRate && sample_rate <= kMaxInputRate;
}

AudioResampler::AudioResampler(int input_rate, int output_rate)
    : input_rate_(input_rate), output_rate_(output_rate) {
    if (!supported_input_sample_rate(input_rate_))
        throw std::invalid_argument("input sample rate must be between 8000 and 96000 Hz");
    if (output_rate_ <= 0)
        throw std::invalid_argument("output sample rate must be positive");

    rate_gcd_ = std::gcd(static_cast<uint64_t>(input_rate_), static_cast<uint64_t>(output_rate_));
    const uint64_t exact_phase_count = static_cast<uint64_t>(output_rate_) / rate_gcd_;
    exact_phase_table_ = exact_phase_count <= kMaxPhases;
    phase_count_ = static_cast<size_t>(
        exact_phase_table_ ? exact_phase_count : static_cast<uint64_t>(kMaxPhases));
    coefficients_.resize(phase_count_ * kTaps);

    const long double rate_ratio =
        static_cast<long double>(output_rate_) / static_cast<long double>(input_rate_);
    const long double cutoff = rate_ratio < 1.0L ? rate_ratio * 0.94L : 1.0L;
    for (size_t phase = 0; phase < phase_count_; ++phase) {
        const long double fraction =
            static_cast<long double>(phase) / static_cast<long double>(phase_count_);
        std::array<long double, kTaps> weights{};
        long double weight_sum = 0.0L;
        for (size_t tap = 0; tap < kTaps; ++tap) {
            const int64_t offset = static_cast<int64_t>(tap) - static_cast<int64_t>(kHalfTaps) + 1;
            const long double distance = fraction - static_cast<long double>(offset);
            long double weight = 0.0L;
            if (std::abs(distance) < static_cast<long double>(kHalfTaps)) {
                const long double window =
                    0.5L + 0.5L * std::cos(kPi * distance / static_cast<long double>(kHalfTaps));
                weight = cutoff * sinc(cutoff * distance) * window;
            }
            weights[tap] = weight;
            weight_sum += weight;
        }
        if (std::abs(weight_sum) < 1.0e-12L)
            continue;
        for (size_t tap = 0; tap < kTaps; ++tap)
            coefficients_[phase * kTaps + tap] = static_cast<float>(weights[tap] / weight_sum);
    }
}

void
AudioResampler::process(const float* input, size_t count, std::vector<float>* output) {
    if (output == nullptr)
        throw std::invalid_argument("resampler output must not be null");
    if (finished_)
        throw std::logic_error("cannot append audio after resampler finish");
    if (count == 0)
        return;
    if (input == nullptr)
        throw std::invalid_argument("resampler input must not be null");

    if (input_rate_ == output_rate_) {
        output->insert(output->end(), input, input + count);
        total_input_ += count;
        next_output_ += count;
        return;
    }

    input_.insert(input_.end(), input, input + count);
    total_input_ += count;
    emit_available(false, output);
    compact_history();
}

void
AudioResampler::finish(std::vector<float>* output) {
    if (output == nullptr)
        throw std::invalid_argument("resampler output must not be null");
    if (finished_)
        return;
    if (input_rate_ != output_rate_)
        emit_available(true, output);
    input_.clear();
    input_start_ = total_input_;
    finished_ = true;
}

void
AudioResampler::emit_available(bool final, std::vector<float>* output) {
    if (total_input_ == 0)
        return;
    const uint64_t final_count =
        final ? rounded_output_count(total_input_, input_rate_, output_rate_) : 0;

    while (true) {
        if (final && next_output_ >= final_count)
            break;
        if (!final && (source_index_ >= total_input_ ||
                       total_input_ - source_index_ <= static_cast<uint64_t>(kHalfTaps)))
            break;
        output->push_back(filtered_sample(source_index_, phase_index()));
        ++next_output_;
        advance_source_position();
    }
}

float
AudioResampler::filtered_sample(uint64_t center, size_t phase) const {
    const int64_t first = static_cast<int64_t>(center) - static_cast<int64_t>(kHalfTaps) + 1;
    const float* coefficients = coefficients_.data() + phase * kTaps;
    double weighted = 0.0;
    for (size_t tap = 0; tap < kTaps; ++tap)
        weighted += static_cast<double>(input_sample(first + static_cast<int64_t>(tap))) *
                    coefficients[tap];
    return static_cast<float>(weighted);
}

void
AudioResampler::advance_source_position() {
    source_remainder_ += static_cast<uint64_t>(input_rate_);
    source_index_ += source_remainder_ / static_cast<uint64_t>(output_rate_);
    source_remainder_ %= static_cast<uint64_t>(output_rate_);
}

size_t
AudioResampler::phase_index() const {
    if (exact_phase_table_)
        return static_cast<size_t>(source_remainder_ / rate_gcd_);
    return static_cast<size_t>(
        source_remainder_ * static_cast<uint64_t>(phase_count_) /
        static_cast<uint64_t>(output_rate_));
}

float
AudioResampler::input_sample(int64_t index) const {
    if (index < 0)
        index = 0;
    if (index >= static_cast<int64_t>(total_input_))
        index = static_cast<int64_t>(total_input_) - 1;
    const uint64_t absolute = static_cast<uint64_t>(index);
    if (absolute < input_start_ || absolute - input_start_ >= input_.size())
        throw std::logic_error("resampler history underflow");
    return input_[static_cast<size_t>(absolute - input_start_)];
}

void
AudioResampler::compact_history() {
    if (input_.empty())
        return;
    if (source_index_ < static_cast<uint64_t>(kHalfTaps - 1))
        return;
    const uint64_t earliest_needed = source_index_ - static_cast<uint64_t>(kHalfTaps - 1);
    if (earliest_needed <= input_start_)
        return;
    const uint64_t discard64 = std::min<uint64_t>(earliest_needed - input_start_, input_.size());
    const size_t discard = static_cast<size_t>(discard64);
    input_.erase(input_.begin(), input_.begin() + discard);
    input_start_ += discard;
}

Pcm16Resampler::Pcm16Resampler(int input_rate, int output_rate)
    : resampler_(input_rate, output_rate) {}

void
Pcm16Resampler::process(const uint8_t* input, size_t bytes, std::vector<uint8_t>* output) {
    if (output == nullptr)
        throw std::invalid_argument("PCM resampler output must not be null");
    if ((bytes & 1u) != 0)
        throw std::invalid_argument("signed 16-bit PCM chunk has an odd byte count");
    if (bytes == 0)
        return;
    if (input == nullptr)
        throw std::invalid_argument("PCM resampler input must not be null");

    decoded_.resize(bytes / 2);
    constexpr float scale = 1.0f / 32768.0f;
    for (size_t i = 0; i < decoded_.size(); ++i) {
        const uint16_t bits =
            static_cast<uint16_t>(input[i * 2]) | (static_cast<uint16_t>(input[i * 2 + 1]) << 8);
        decoded_[i] = static_cast<float>(static_cast<int16_t>(bits)) * scale;
    }

    converted_.clear();
    resampler_.process(decoded_.data(), decoded_.size(), &converted_);
    append_pcm(converted_, output);
}

void
Pcm16Resampler::finish(std::vector<uint8_t>* output) {
    if (output == nullptr)
        throw std::invalid_argument("PCM resampler output must not be null");
    converted_.clear();
    resampler_.finish(&converted_);
    append_pcm(converted_, output);
}

void
Pcm16Resampler::append_pcm(const std::vector<float>& samples, std::vector<uint8_t>* output) {
    if (samples.size() > (output->max_size() - output->size()) / 2)
        throw std::overflow_error("resampled PCM audio is too large");
    const size_t offset = output->size();
    output->resize(offset + samples.size() * 2);
    for (size_t i = 0; i < samples.size(); ++i) {
        float sample = samples[i];
        sample = std::max(-1.0f, std::min(1.0f, sample));
        const int16_t value = static_cast<int16_t>(std::lrintf(sample * 32767.0f));
        const uint16_t bits = static_cast<uint16_t>(value);
        (*output)[offset + i * 2] = static_cast<uint8_t>(bits & 0xff);
        (*output)[offset + i * 2 + 1] = static_cast<uint8_t>((bits >> 8) & 0xff);
    }
    output_samples_ += samples.size();
}

std::vector<float>
resample_audio(const float* input, size_t count, int input_rate, int output_rate) {
    AudioResampler resampler(input_rate, output_rate);
    std::vector<float> output;
    if (count != 0) {
        const long double expected = static_cast<long double>(count) * output_rate / input_rate;
        if (expected < static_cast<long double>(std::numeric_limits<size_t>::max()))
            output.reserve(static_cast<size_t>(expected + 1));
    }
    resampler.process(input, count, &output);
    resampler.finish(&output);
    return output;
}

}  // namespace nemo_speech::audio
