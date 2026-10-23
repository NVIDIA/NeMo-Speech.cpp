// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// The CPU log-mel path must match a plain reference: an FFT with its twiddles
// computed inline and a dense filterbank product. Covers the generated
// filterbank and one set with set_mel_basis, including rows the sparse span has
// to handle at its edges. Builds that fuse multiply-adds (GCC by default, and
// clang on arm64) may fuse them differently here and in the library, so the
// comparison allows 1e-4, ten times below what the bugs this guards against
// produce.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "fe.h"
#include "numeric_parity.h"

namespace {

constexpr float kPi = 3.14159265358979323846f;

void
reference_fft(std::vector<float>& re, std::vector<float>& im) {
    const int n = static_cast<int>(re.size());
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * kPi / len;
        const float wre = std::cos(ang);
        const float wim = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            float cur_re = 1.0f, cur_im = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                const float ur = re[i + k];
                const float ui = im[i + k];
                const float vr = re[i + k + len / 2] * cur_re - im[i + k + len / 2] * cur_im;
                const float vi = re[i + k + len / 2] * cur_im + im[i + k + len / 2] * cur_re;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                const float nre = cur_re * wre - cur_im * wim;
                const float nim = cur_re * wim + cur_im * wre;
                cur_re = nre;
                cur_im = nim;
            }
        }
    }
}

// The filterbank MelSpectrogramExtractor generates when a model has none.
std::vector<float>
generated_basis(const MelSpecConfig& cfg) {
    const int n_bins = cfg.n_fft / 2 + 1;
    const float fmax = cfg.fmax > 0.0f ? cfg.fmax : 0.5f * cfg.sample_rate;
    auto hz_to_mel = [](float f) { return 1127.0f * std::log(1.0f + f / 700.0f); };
    auto mel_to_hz = [](float m) { return 700.0f * (std::exp(m / 1127.0f) - 1.0f); };
    const float mel_min = hz_to_mel(cfg.fmin);
    const float mel_max = hz_to_mel(fmax);
    std::vector<float> points(cfg.n_mels + 2);
    for (int i = 0; i < cfg.n_mels + 2; i++)
        points[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (cfg.n_mels + 1));
    std::vector<float> basis(static_cast<size_t>(cfg.n_mels) * n_bins, 0.0f);
    for (int m = 0; m < cfg.n_mels; m++) {
        for (int k = 0; k < n_bins; k++) {
            const float f = static_cast<float>(k) * cfg.sample_rate / cfg.n_fft;
            float w = 0.0f;
            if (f >= points[m] && f <= points[m + 1])
                w = (f - points[m]) / (points[m + 1] - points[m] + 1e-12f);
            else if (f >= points[m + 1] && f <= points[m + 2])
                w = (points[m + 2] - f) / (points[m + 2] - points[m + 1] + 1e-12f);
            basis[static_cast<size_t>(m) * n_bins + k] = std::max(0.0f, w);
        }
    }
    return basis;
}

// Unnormalized log-mel with reflect_left=true, as compute() produces it.
std::vector<float>
reference_log_mel(
    const MelSpecConfig& cfg, const std::vector<float>& basis, const std::vector<float>& audio) {
    const int win = static_cast<int>(cfg.window_size * cfg.sample_rate + 0.5f);
    const int hop = static_cast<int>(cfg.window_stride * cfg.sample_rate + 0.5f);
    const int n_fft = cfg.n_fft;
    const int n_bins = n_fft / 2 + 1;
    std::vector<float> window(win);
    const float denom = cfg.hann_periodic ? static_cast<float>(win) : static_cast<float>(win - 1);
    for (int i = 0; i < win; i++) window[i] = 0.5f * (1.0f - std::cos(2.0f * kPi * i / denom));

    std::vector<float> pre(audio);
    for (size_t i = pre.size() - 1; i > 0; --i) pre[i] = pre[i] - cfg.preemph * pre[i - 1];
    std::vector<float> padded(n_fft / 2, 0.0f);
    padded.insert(padded.end(), pre.begin(), pre.end());
    padded.insert(padded.end(), n_fft / 2, 0.0f);

    const int n_frames = static_cast<int>((padded.size() - n_fft) / hop + 1);
    std::vector<float> features(static_cast<size_t>(cfg.n_mels) * n_frames);
    std::vector<float> re(n_fft), im(n_fft);
    const int woff = cfg.stft_center_window ? (n_fft - win) / 2 : 0;
    for (int f = 0; f < n_frames; f++) {
        std::fill(re.begin(), re.end(), 0.0f);
        std::fill(im.begin(), im.end(), 0.0f);
        for (int i = 0; i < win; i++) re[woff + i] = padded[f * hop + woff + i] * window[i];
        reference_fft(re, im);
        for (int m = 0; m < cfg.n_mels; m++) {
            float acc = 0.0f;
            for (int k = 0; k < n_bins; k++) {
                const float power = re[k] * re[k] + im[k] * im[k];
                acc += basis[static_cast<size_t>(m) * n_bins + k] * power;
            }
            features[static_cast<size_t>(m) + static_cast<size_t>(f) * cfg.n_mels] =
                std::log(acc + cfg.log_zero_guard);
        }
    }
    return features;
}

constexpr float kTolerance = 1e-4f;

std::vector<float>
noise(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> audio(n);
    for (float& s : audio) s = dist(rng);
    return audio;
}

void
check_config(const MelSpecConfig& cfg, const char* name) {
    MelSpectrogramExtractor fe(cfg);
    const std::vector<float> audio = noise(20800, 7);
    std::vector<float> features;
    int n_frames = 0;

    fe.compute(audio.data(), audio.size(), features, n_frames, true, false);
    float diff = finite_max_abs_diff(features, reference_log_mel(cfg, generated_basis(cfg), audio));
    if (!(diff <= kTolerance)) {
        std::fprintf(stderr, "[FAIL] %s: generated filterbank off by %g\n", name, diff);
        throw std::runtime_error("mel features: generated filterbank");
    }

    // Replace the basis on the same extractor: the nonzero spans must follow.
    const int n_bins = cfg.n_fft / 2 + 1;
    std::vector<float> basis = generated_basis(cfg);
    std::reverse(basis.begin(), basis.end());
    std::fill_n(basis.begin(), n_bins, 0.0f);                    // an all-zero row
    std::fill_n(basis.begin() + n_bins, n_bins, 0.25f);          // a dense row
    basis[2 * static_cast<size_t>(n_bins) + n_bins - 1] = 1.0f;  // nonzero in the last bin
    basis[3 * static_cast<size_t>(n_bins)] = 1.0f;               // nonzero in the first bin
    basis[3 * static_cast<size_t>(n_bins) + n_bins / 2] = 0.5f;  // with a gap in between
    fe.set_mel_basis(basis.data(), cfg.n_mels, n_bins);
    fe.compute(audio.data(), audio.size(), features, n_frames, true, false);
    diff = finite_max_abs_diff(features, reference_log_mel(cfg, basis, audio));
    if (!(diff <= kTolerance)) {
        std::fprintf(stderr, "[FAIL] %s: set_mel_basis output off by %g\n", name, diff);
        throw std::runtime_error("mel features: set_mel_basis");
    }
}

}  // namespace

int
main() {
    MelSpecConfig asr;  // legacy ASR: periodic Hann, right-aligned window
    check_config(asr, "asr");

    MelSpecConfig nemo;  // NeMo parity: symmetric Hann, centered window
    nemo.n_mels = 128;
    nemo.hann_periodic = false;
    nemo.stft_center_window = true;
    check_config(nemo, "nemo");

    MelSpecConfig wide;
    wide.n_fft = 1024;
    wide.n_mels = 64;
    wide.fmin = 20.0f;
    wide.fmax = 7600.0f;
    check_config(wide, "n_fft 1024");

    std::printf("[PASS] CPU log-mel matches the reference\n");
    return 0;
}
