// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// End-to-end RNNT streaming smoke test: load a Nemotron RNNT GGUF via
// AsrModel, feed WAV through CacheStreamRunner, print partials + final.
//
// Usage: ./test_rnnt_streaming <model.gguf> <audio.wav> [--gpu N]
//                            [--language CODE]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "fe.h"
#include "model.h"
#include "recognizer.h"  // RecognizerConfig
#include "runner.h"
#include "runtime.h"

using namespace nemo_speech::asr;

int
main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(
            stderr,
            "Usage: %s <model.gguf> <audio.wav> [--gpu N] [--chunk-ms N] "
            "[--right-ctx R] [--language CODE] [--dump-encoder PATH]\n",
            argv[0]);
        return 1;
    }
    std::string model_path = argv[1];
    std::string audio_path = argv[2];
    int gpu = 0;
    // Defaults match the Riva cache-aware low-latency streaming preset and how
    // a real client feeds audio: 160 ms packets, R=1 (one-chunk lookahead).
    int chunk_ms = 160;
    int right_ctx = 1;
    std::string language;
    std::string dump_encoder_path;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gpu" && i + 1 < argc)
            gpu = std::atoi(argv[++i]);
        else if (a == "--chunk-ms" && i + 1 < argc)
            chunk_ms = std::atoi(argv[++i]);
        else if (a == "--right-ctx" && i + 1 < argc)
            right_ctx = std::atoi(argv[++i]);
        else if (a == "--language" && i + 1 < argc)
            language = argv[++i];
        else if (a == "--dump-encoder" && i + 1 < argc)
            dump_encoder_path = argv[++i];
    }

    ggml_runtime::Params bm_params;
    bm_params.use_gpu = (gpu >= 0);
    bm_params.gpu_device_idx = std::max(gpu, 0);
    bm_params.pe_bin_path = const_cast<char*>("");
    ggml_runtime::BackendManager bm(bm_params);
    auto model = AsrModel::load(bm, model_path);
    auto* rnnt = dynamic_cast<RnntModel*>(model.get());
    if (!rnnt) {
        std::fprintf(stderr, "[test] expected an RNNT model\n");
        return 2;
    }

    std::vector<float> audio;
    int sr = 0;
    if (!read_wav_mono_16k(audio_path, audio, sr)) {
        std::fprintf(stderr, "failed to read WAV: %s\n", audio_path.c_str());
        return 2;
    }
    if (sr != model->sample_rate()) {
        std::fprintf(stderr, "WAV sr %d != model %d\n", sr, model->sample_rate());
        return 2;
    }

    // Default: Riva's `cache-aware-parakeet-rnnt` low-latency preset (R=1,
    // 160 ms streaming latency). Override with --right-ctx N (model is trained
    // on R ∈ {0, 1, 6, 13}; 13 is highest quality, ~1.12 s lookahead).
    RecognizerConfig cfg;
    cfg.streaming.rnnt_right_context = right_ctx;
    CacheStreamRunner runner(rnnt, cfg);
    runner.set_prompt_index(rnnt->prompt_index_for_lang(language));

    // Diagnostic: feed all audio in one shot, then inspect one encoder chunk
    // then run the staged predictor + joint argmax against its first frame.
    runner.feed_audio(audio.data(), audio.size());
    (void)runner.step();
    std::vector<float> enc_buf;
    int T_out = 0, d_model_out = 0;
    runner.take_encoder_frames(enc_buf, T_out, d_model_out);
    if (T_out > 0) {
        double sum = 0, sq = 0, mn = 1e30, mx = -1e30;
        int nan = 0;
        for (size_t i = 0; i < enc_buf.size(); i++) {
            float v = enc_buf[i];
            if (!std::isfinite(v)) {
                nan++;
                continue;
            }
            sum += v;
            sq += double(v) * v;
            if (v < mn)
                mn = v;
            if (v > mx)
                mx = v;
        }
        double mean = sum / enc_buf.size(), var = sq / enc_buf.size() - mean * mean;
        std::fprintf(
            stderr,
            "[diag] enc[chunk 0]: shape=(%d,%d) mean=%.3g std=%.3g min=%.3g max=%.3g nan=%d\n",
            d_model_out, T_out, mean, std::sqrt(std::max(0.0, var)), mn, mx, nan);

        // Take the first already-projected encoder frame and run the staged
        // predictor + joint argmax path against it (prev=blank, state=0).
        auto state = rnnt->make_rnnt_stream_state();
        rnnt->predict_rnnt(*state, rnnt->rnnt_config().blank_id, /*active_bank=*/0);
        int32_t best = -1;
        rnnt->joint_argmax(*state, enc_buf.data(), d_model_out, 1, &best);
        std::fprintf(
            stderr, "[diag] staged RNNT on enc_proj[0]: argmax=%d (blank=%d)\n", best,
            rnnt->rnnt_config().blank_id);
    }

    // Now reset and do the actual streaming run.
    runner.reset();

    const size_t chunk_samples = static_cast<size_t>(chunk_ms) * sr / 1000;
    std::string last_partial;
    std::vector<double> step_ms;
    std::vector<float> dumped_encoder;
    int dumped_encoder_dim = 0;
    int dumped_encoder_frames = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t off = 0; off < audio.size(); off += chunk_samples) {
        const size_t n = std::min(chunk_samples, audio.size() - off);
        runner.feed_audio(audio.data() + off, n);
        const int chunks_before = runner.chunks_processed();
        auto cs0 = std::chrono::high_resolution_clock::now();
        auto u = runner.step();
        auto cs1 = std::chrono::high_resolution_clock::now();
        if (!dump_encoder_path.empty() && runner.chunks_processed() == chunks_before + 1) {
            std::vector<float> chunk_encoder;
            int chunk_T = 0, chunk_D = 0;
            runner.take_encoder_frames(chunk_encoder, chunk_T, chunk_D);
            if (dumped_encoder_dim == 0)
                dumped_encoder_dim = chunk_D;
            if (chunk_D != dumped_encoder_dim) {
                std::fprintf(
                    stderr, "encoder dump dimension changed: %d -> %d\n", dumped_encoder_dim,
                    chunk_D);
                return 3;
            }
            dumped_encoder.insert(dumped_encoder.end(), chunk_encoder.begin(), chunk_encoder.end());
            dumped_encoder_frames += chunk_T;
        }
        step_ms.push_back(std::chrono::duration<double, std::milli>(cs1 - cs0).count());
        if (!u.new_token_ids.empty() && u.transcript_so_far != last_partial) {
            std::fprintf(
                stderr, "[partial @ %.2fs] %s\n", u.audio_processed_sec,
                u.transcript_so_far.c_str());
            last_partial = u.transcript_so_far;
        }
    }
    auto final_update = runner.finalize();
    auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const float audio_sec = static_cast<float>(audio.size()) / static_cast<float>(sr);
    std::fprintf(stderr, "[final] %s\n", final_update.transcript_so_far.c_str());
    std::fprintf(
        stderr,
        "[stats] %.2fs audio in %.0f ms (RTF %.4f) "
        "chunks=%d cache_filled_frames=%d\n",
        audio_sec, ms, ms / 1000.0 / audio_sec, runner.chunks_processed(),
        runner.cache_filled_frames());
    const auto decode_stats = runner.rnnt_decode_stats();
    std::fprintf(
        stderr,
        "[rnnt] encoder_frames=%llu emitted=%llu predictor_calls=%llu "
        "joint_calls=%llu joint_frames=%llu\n",
        static_cast<unsigned long long>(decode_stats.encoder_frames),
        static_cast<unsigned long long>(decode_stats.emitted_tokens),
        static_cast<unsigned long long>(decode_stats.predictor_calls),
        static_cast<unsigned long long>(decode_stats.joint_calls),
        static_cast<unsigned long long>(decode_stats.joint_frames));
    if (!step_ms.empty()) {
        std::sort(step_ms.begin(), step_ms.end());
        double sum = 0;
        for (double v : step_ms) sum += v;
        const double mean = sum / step_ms.size();
        auto pct = [&](double p) {
            size_t idx = (size_t)(p * (step_ms.size() - 1) + 0.5);
            return step_ms[std::min(idx, step_ms.size() - 1)];
        };
        char rtfx[16];
        if (mean > 0.0)
            std::snprintf(rtfx, sizeof(rtfx), "%.1f", (double)chunk_ms / mean);
        else
            std::snprintf(rtfx, sizeof(rtfx), "N/A");
        std::fprintf(
            stderr,
            "[per-chunk step() ms] n=%zu mean=%.3f p50=%.3f p95=%.3f max=%.3f | "
            "chunk=%d ms -> mean RTFx=%s, worst-chunk realtime budget %s (max %.3f vs %d ms)\n",
            step_ms.size(), mean, pct(0.5), pct(0.95), step_ms.back(), chunk_ms, rtfx,
            step_ms.back() <= chunk_ms ? "MET" : "MISSED", step_ms.back(), chunk_ms);
    }
    if (!dump_encoder_path.empty()) {
        std::ofstream out(dump_encoder_path, std::ios::binary);
        const int32_t d = dumped_encoder_dim;
        const int32_t t = dumped_encoder_frames;
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));
        out.write(reinterpret_cast<const char*>(&t), sizeof(t));
        out.write(
            reinterpret_cast<const char*>(dumped_encoder.data()),
            static_cast<std::streamsize>(dumped_encoder.size() * sizeof(float)));
        if (!out) {
            std::fprintf(stderr, "failed to write encoder dump: %s\n", dump_encoder_path.c_str());
            return 3;
        }
        std::fprintf(
            stderr, "[diag] wrote encoder projection (%d,%d) to %s\n", d, t,
            dump_encoder_path.c_str());
    }
    std::fprintf(stdout, "%s\n", final_update.transcript_so_far.c_str());
    return 0;
}
