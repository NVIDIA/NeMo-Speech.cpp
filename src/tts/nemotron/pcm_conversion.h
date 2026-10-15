// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define NEMO_NATIVE_PCM_AVX2 1
#endif

namespace nemo_speech::tts::nemotron {

inline void
pcm_f32_scalar(const char* input, char* output, size_t count) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    for (size_t index = 0; index < count; ++index) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(input + index * 4);
        const uint32_t bits = uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
                              (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
        float sample;
        std::memcpy(&sample, &bits, sizeof(sample));
        if (!std::isfinite(sample))
            throw std::runtime_error("native worker returned nonfinite PCM-F32LE audio");
        int value;
        if (sample <= -1.0f)
            value = -32768;
        else if (sample >= 1.0f)
            value = 32767;
        else {
            const float scaled = sample * (sample < 0.0f ? 32768.0f : 32767.0f);
            value = static_cast<int>(std::round(scaled));
        }
        const auto encoded = static_cast<uint16_t>(value);
        output[index * 2] = static_cast<char>(encoded & 255);
        output[index * 2 + 1] = static_cast<char>(encoded >> 8);
    }
}

inline std::string
pcm_f32_to_s16_scalar(const char* input, size_t bytes) {
    if (bytes % 4 != 0)
        throw std::runtime_error("native worker returned malformed PCM-F32LE audio");
    std::string output(bytes / 2, '\0');
    pcm_f32_scalar(input, output.data(), bytes / 4);
    return output;
}

#ifdef NEMO_NATIVE_PCM_AVX2
__attribute__((target("avx2"))) inline void
pcm_f32_avx2(const char* input, char* output, size_t count) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 negative_one = _mm256_set1_ps(-1.0f);
    const __m256i exponent_mask = _mm256_set1_epi32(0x7f800000);
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(-2147483647 - 1));
    size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        __m256 samples;
        std::memcpy(&samples, input + index * 4, sizeof(samples));
        const __m256i exponents = _mm256_and_si256(_mm256_castps_si256(samples), exponent_mask);
        if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(exponents, exponent_mask)) != 0)
            throw std::runtime_error("native worker returned nonfinite PCM-F32LE audio");
        samples = _mm256_max_ps(negative_one, _mm256_min_ps(one, samples));
        const __m256 negative = _mm256_cmp_ps(samples, zero, _CMP_LT_OQ);
        const __m256 scale =
            _mm256_blendv_ps(_mm256_set1_ps(32767.0f), _mm256_set1_ps(32768.0f), negative);
        const __m256 scaled = _mm256_mul_ps(samples, scale);
        const __m256 magnitude = _mm256_andnot_ps(sign_mask, scaled);
        const __m256 integral = _mm256_round_ps(magnitude, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        const __m256 increment = _mm256_and_ps(
            _mm256_cmp_ps(_mm256_sub_ps(magnitude, integral), _mm256_set1_ps(0.5f), _CMP_GE_OQ),
            one);
        const __m256 rounded =
            _mm256_or_ps(_mm256_add_ps(integral, increment), _mm256_and_ps(scaled, sign_mask));
        const __m256i integers = _mm256_cvttps_epi32(rounded);
        const __m128i packed = _mm_packs_epi32(
            _mm256_castsi256_si128(integers), _mm256_extracti128_si256(integers, 1));
        std::memcpy(output + index * 2, &packed, sizeof(packed));
    }
    pcm_f32_scalar(input + index * 4, output + index * 2, count - index);
}
#endif

inline std::string
pcm_f32_to_s16(const char* input, size_t bytes) {
    if (bytes % 4 != 0)
        throw std::runtime_error("native worker returned malformed PCM-F32LE audio");
    std::string output(bytes / 2, '\0');
#ifdef NEMO_NATIVE_PCM_AVX2
    if (__builtin_cpu_supports("avx2"))
        pcm_f32_avx2(input, output.data(), bytes / 4);
    else
#endif
        pcm_f32_scalar(input, output.data(), bytes / 4);
    return output;
}

}  // namespace nemo_speech::tts::nemotron

#ifdef NEMO_NATIVE_PCM_AVX2
#undef NEMO_NATIVE_PCM_AVX2
#endif
