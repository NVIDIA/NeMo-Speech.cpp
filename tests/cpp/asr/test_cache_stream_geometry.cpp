// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// No arguments: model-independent causal first/steady-state shape checks.
// Optional: MODEL.gguf [--gpu N] exercises actual streaming, reset and EOF.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "recognizer.h"

using namespace nemo_speech::asr;

static void
require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

static void
check_shapes() {
    for (int right : {0, 1, 3, 6, 13}) {
        const auto cfg = make_cache_aware_config(EncoderConfig{}, right);
        const int first = 1 + 8 * right;
        const int steady = 9 + 8 * (1 + right);
        require(cfg.cache_first_chunk_mel_frames() == first, "wrong first mel length");
        require(cfg.subsample_time_length(first) == 1 + right, "first chunk needs no drop");
        require(
            cfg.subsample_time_length(steady) - cfg.cache_drop_extra == 1 + right,
            "steady chunk must drop only the overlap");
    }
}

static void
same_encoder(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size(), "encoder size differs from explicit reference chunk");
    float largest = 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite encoder projection");
        largest = std::max(largest, std::fabs(a[i] - b[i]));
        sum += static_cast<double>(a[i] - b[i]) * (a[i] - b[i]);
    }
    if (largest > 1e-5f)
        std::fprintf(
            stderr, "encoder max abs error=%g RMS=%g\n", largest, std::sqrt(sum / a.size()));
    require(largest <= 1e-5f, "encoder differs from explicit first/history chunk");
}

static void
check_model(const std::string& path, int gpu, int right) {
    RecognizerConfig cfg;
    cfg.model.path = path;
    cfg.backend.gpu = gpu;
    cfg.streaming.rnnt_right_context = right;
    cfg.log_status = false;
    Recognizer recognizer(cfg);
    auto* model = dynamic_cast<RnntModel*>(recognizer.model());
    require(model != nullptr, "expected cache-aware RNNT model");
    const auto enc = make_cache_aware_config(model->encoder_config(), right);
    const auto& fe = model->fe_config();
    const int first_mel = 1 + enc.subsampling_factor * right;
    const int new_mel = enc.subsampling_factor * (1 + right);
    const int overlap = 9;  // Existing runner policy, not inferred from subsampling.
    const int hop = model->fe().hop_length();
    const size_t first_samples = (first_mel - 1) * hop + fe.n_fft / 2;
    const size_t stride_samples = new_mel * hop;
    std::vector<float> audio(first_samples + 2 * stride_samples);
    for (size_t i = 0; i < audio.size(); ++i)
        audio[i] = 0.08f * std::sin(0.071f * i) + 0.03f * std::cos(0.019f * i);

    const int prompt = model->prompt_index_for_lang("auto");
    CacheStreamRunner runner(model, cfg);
    runner.set_prompt_index(prompt);
    runner.feed_audio(audio.data(), first_samples - 1);
    runner.step();
    require(runner.chunks_processed() == 0, "first chunk ran before its last stable mel frame");
    runner.feed_audio(audio.data() + first_samples - 1, 1);
    runner.step();
    require(
        runner.chunks_processed() == 1, "first chunk incorrectly waits for steady-state length");

    // Independent chunk assembly, including negative history for R=0. Use
    // another cache slot so runner buffer movement cannot affect the oracle.
    std::vector<float> all_mel;
    std::vector<float> initial_audio(audio.begin(), audio.begin() + first_samples - 1);
    produce_new_mel_frames(model->fe(), initial_audio, 0, 0, all_mel);
    auto state = model->make_cache_state();
    int consumed = 0;
    int cached = 0;
    std::vector<float> first_output;
    for (int step = 0; step < 3; ++step) {
        std::fprintf(stderr, "[check] R=%d chunk=%d\n", right, step);
        // Match the available audio frontier, not future audio: the first
        // reflected STFT frame of a minimal R=0 packet must use the same
        // boundary in the runner and in this chunk-assembly reference.
        const size_t available = first_samples + step * stride_samples;
        std::vector<float> prefix_audio(audio.begin(), audio.begin() + available);
        std::vector<float> new_features;
        produce_new_mel_frames(
            model->fe(), prefix_audio, 0, static_cast<int64_t>(all_mel.size() / fe.n_mels),
            new_features);
        all_mel.insert(all_mel.end(), new_features.begin(), new_features.end());
        const int prefix = step == 0 ? 0 : overlap;
        const int fresh = step == 0 ? first_mel : new_mel;
        const int frames = prefix + fresh;
        std::vector<float> chunk(static_cast<size_t>(frames) * fe.n_mels, 0.0f);
        for (int f = 0; f < frames; ++f) {
            const int source = consumed - prefix + f;
            if (source >= 0) {
                require(
                    static_cast<size_t>(source + 1) * fe.n_mels <= all_mel.size(),
                    "reference mel underrun");
                std::copy_n(
                    all_mel.data() + static_cast<size_t>(source) * fe.n_mels, fe.n_mels,
                    chunk.data() + static_cast<size_t>(f) * fe.n_mels);
            }
        }
        std::vector<float> mask(enc.cache_left_ctx + enc.cache_chunk_frames, 0.0f);
        std::fill_n(mask.begin(), std::max(0, enc.cache_left_ctx - cached), -1e9f);
        std::vector<float> expected;
        int expected_t = 0;
        model->encode_cache_aware(
            state, chunk.data(), frames, mask.data(), static_cast<int>(mask.size()), expected,
            expected_t, prompt);
        require(expected_t == right + 1, "wrong reference encoder frame count");
        if (step > 0) {
            runner.feed_audio(
                audio.data() + first_samples + (step - 1) * stride_samples, stride_samples);
            runner.step();
        }
        std::vector<float> actual;
        int actual_t = 0, dim = 0;
        runner.take_encoder_frames(actual, actual_t, dim);
        require(actual_t == expected_t, "wrong runner encoder frame count");
        require(runner.chunks_processed() == step + 1, "wrong chunk consumption");
        same_encoder(actual, expected);
        if (step == 0)
            first_output = actual;
        consumed += fresh;
        cached = std::min(enc.cache_left_ctx, cached + expected_t);
    }
    runner.reset();
    // Preserve FE packet shapes as well: Metal FFT rounding can differ across
    // packet shapes independently of the encoder's chunk geometry.
    runner.feed_audio(audio.data(), first_samples - 1);
    runner.step();
    runner.feed_audio(audio.data() + first_samples - 1, 1);
    runner.step();
    require(runner.chunks_processed() == 1, "reset did not restore first-chunk geometry");
    std::vector<float> reset_output;
    int reset_t = 0, reset_dim = 0;
    runner.take_encoder_frames(reset_output, reset_t, reset_dim);
    same_encoder(first_output, reset_output);

    auto decode = [&](size_t size, size_t packet, bool drain) {
        runner.reset();
        std::vector<int> ids;
        for (size_t off = 0; off < size;) {
            const size_t count = std::min(packet, size - off);
            runner.feed_audio(audio.data() + off, count);
            off += count;
            if (drain) {
                const auto update = runner.step();
                ids.insert(ids.end(), update.new_token_ids.begin(), update.new_token_ids.end());
            }
        }
        const auto final = runner.finalize();
        ids.insert(ids.end(), final.new_token_ids.begin(), final.new_token_ids.end());
        require(final.is_final, "EOF not final");
        require(runner.finalize().new_token_ids.empty(), "EOF duplicated tokens");
        return ids;
    };
    for (size_t size :
         {size_t(0), size_t(1), size_t(159), first_samples - 1, first_samples, first_samples + 1,
          audio.size()}) {
        const auto expected = decode(size, audio.size(), false);
        require(decode(size, 1279, true) == expected, "odd packet EOF differs");
        require(decode(size, 1600, true) == expected, "100ms packet EOF differs");
    }
    // A hard EOU starts a fresh encoder context but keeps the absolute frame
    // clock. The next chunk must nevertheless use startup geometry again.
    runner.reset();
    runner.feed_audio(audio.data(), first_samples);
    runner.force_eou();
    require(runner.step().is_final, "forced EOU did not fire");
    require(runner.cache_filled_frames() == 0, "forced EOU did not reset cache");
    const int before = runner.chunks_processed();
    runner.feed_audio(audio.data(), first_samples);
    runner.step();
    require(runner.chunks_processed() == before + 1, "post-EOU did not use first chunk");
    std::fprintf(stdout, "[PASS] R=%d first/history encoder, reset, EOF and EOU\n", right);
}

int
main(int argc, char** argv) try {
    check_shapes();
    if (argc > 1) {
        int gpu = -1;
        if (argc == 4 && std::string(argv[2]) == "--gpu")
            gpu = std::stoi(argv[3]);
        else if (argc != 2)
            throw std::runtime_error("Usage: test_cache_stream_geometry [MODEL.gguf [--gpu N]]");
        for (int right : {0, 3, 6, 13}) check_model(argv[1], gpu, right);
    }
    std::fprintf(stdout, "[PASS] cache-aware chunk geometry\n");
    return 0;
}
catch (const std::exception& error) {
    std::fprintf(stderr, "[FAIL] %s\n", error.what());
    return 1;
}
