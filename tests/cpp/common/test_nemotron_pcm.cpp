// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tts/nemotron/pcm_conversion.h"

using nemo_speech::tts::nemotron::pcm_f32_to_s16;
using nemo_speech::tts::nemotron::pcm_f32_to_s16_scalar;

namespace {

float
from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::string
encode(const std::vector<float>& values) {
    std::string encoded;
    encoded.reserve(values.size() * 4);
    for (float value : values) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        for (unsigned shift = 0; shift < 32; shift += 8)
            encoded.push_back(static_cast<char>((bits >> shift) & 255));
    }
    return encoded;
}

std::string
oracle(const std::vector<float>& values) {
    std::string output;
    output.reserve(values.size() * 2);
    for (float sample : values) {
        if (!std::isfinite(sample))
            throw std::runtime_error("nonfinite oracle input");
        const double scale = sample < 0 ? 32768.0 : 32767.0;
        const double clipped = std::max(-1.0, std::min(1.0, static_cast<double>(sample)));
        const volatile float product = static_cast<float>(clipped * scale);
        const double rounded = std::floor(std::abs(static_cast<double>(product)) + 0.5);
        const int signed_value = static_cast<int>(sample < 0 ? -rounded : rounded);
        const auto bits = static_cast<uint16_t>(signed_value);
        output.push_back(static_cast<char>(bits & 255));
        output.push_back(static_cast<char>(bits >> 8));
    }
    return output;
}

void
require_equal(const std::string& actual, const std::string& expected, const std::string& name) {
    if (actual != expected) {
        const auto difference =
            std::mismatch(actual.begin(), actual.end(), expected.begin(), expected.end());
        throw std::runtime_error(
            name + " mismatch at sample " +
            std::to_string((difference.first - actual.begin()) / 2));
    }
}

bool
has_avx2() {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

void
compare(const std::vector<float>& values, size_t alignment, const std::string& name) {
    const auto expected = oracle(values);
    const auto payload = encode(values);
    const std::string storage(alignment, 'x');
    const auto padded = storage + payload + std::string(32, 'x');
    const char* input = padded.data() + alignment;
    require_equal(pcm_f32_to_s16_scalar(input, payload.size()), expected, name + " scalar");
    require_equal(pcm_f32_to_s16(input, payload.size()), expected, name + " dispatch");
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if (has_avx2()) {
        std::string output(expected.size() + 32, 'x');
        nemo_speech::tts::nemotron::pcm_f32_avx2(input, output.data() + 1, values.size());
        require_equal(output.substr(1, expected.size()), expected, name + " AVX2");
        if (output.front() != 'x' || output.substr(expected.size() + 1) != std::string(31, 'x'))
            throw std::runtime_error("AVX2 output crossed its buffer extent");
    }
#endif
}

template <class Function>
void
require_rejected(Function function, const std::string& name) {
    try {
        function();
    }
    catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(name + " accepted invalid input");
}

void
test_nonfinite() {
    for (uint32_t bits :
         {0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xff800001u}) {
        for (size_t position = 0; position < 25; ++position) {
            std::vector<float> values(25, 0.25f);
            values[position] = from_bits(bits);
            const auto payload = encode(values);
            require_rejected(
                [&] { pcm_f32_to_s16_scalar(payload.data(), payload.size()); }, "scalar nonfinite");
            require_rejected(
                [&] { pcm_f32_to_s16(payload.data(), payload.size()); }, "dispatch nonfinite");
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
            if (has_avx2()) {
                std::string output(values.size() * 2, '\0');
                require_rejected(
                    [&] {
                        nemo_speech::tts::nemotron::pcm_f32_avx2(
                            payload.data(), output.data(), values.size());
                    },
                    "AVX2 nonfinite");
            }
#endif
        }
    }
}

}  // namespace

int
main() {
    try {
        const float infinity = std::numeric_limits<float>::infinity();
        std::vector<float> boundaries = {
            0.0f,
            -0.0f,
            1.0f,
            -1.0f,
            2.0f,
            -2.0f,
            std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max(),
            std::numeric_limits<float>::min(),
            -std::numeric_limits<float>::min(),
            std::numeric_limits<float>::denorm_min(),
            -std::numeric_limits<float>::denorm_min(),
            std::nextafter(1.0f, 0.0f),
            std::nextafter(-1.0f, 0.0f),
            std::nextafter(1.0f, infinity),
            std::nextafter(-1.0f, -infinity)};
        for (int sign : {-1, 1}) {
            const int scale = sign < 0 ? 32768 : 32767;
            for (int integer = 0; integer < scale; ++integer) {
                const float threshold = static_cast<float>(sign * (integer + 0.5) / scale);
                boundaries.push_back(std::nextafter(threshold, -infinity));
                boundaries.push_back(threshold);
                boundaries.push_back(std::nextafter(threshold, infinity));
            }
        }
        compare(boundaries, 1, "half-threshold neighbors");
        std::vector<float> random;
        random.reserve(100000);
        uint32_t state = 0x32a174f9;
        while (random.size() < 100000) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if ((state & 0x7f800000u) != 0x7f800000u)
                random.push_back(from_bits(state));
        }
        compare(random, 3, "deterministic random float bits");
        for (size_t alignment = 0; alignment < 32; ++alignment) {
            for (size_t count = 0; count <= 17; ++count) {
                compare(
                    std::vector<float>(boundaries.begin(), boundaries.begin() + count), alignment,
                    "alignment " + std::to_string(alignment) + " tail " + std::to_string(count));
            }
        }
        const std::string malformed(64, '\0');
        for (size_t size = 1; size < malformed.size(); ++size) {
            if (size % 4) {
                require_rejected(
                    [&] { pcm_f32_to_s16_scalar(malformed.data(), size); },
                    "scalar malformed extent");
                require_rejected(
                    [&] { pcm_f32_to_s16(malformed.data(), size); }, "dispatch malformed extent");
            }
        }
        test_nonfinite();
        std::cout
            << "PCM conversion passed " << boundaries.size()
            << " boundary samples, 100000 random samples, all tails and invalid samples; AVX2="
            << has_avx2() << '\n';
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
