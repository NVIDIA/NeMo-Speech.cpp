// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Streaming smoke test. Feeds a WAV in small chunks into BufferedStreamRunner
// and prints partial transcripts as they materialize, then the final.
//
// Usage: ./test_ctc_streaming <model.gguf> <audio.wav> [--chunk-ms N] [--gpu N]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
            "Usage: %s <model.gguf> <audio.wav> [--chunk-ms N] [--gpu N]\n"
            "         [--lm-path FILE --lexicon FILE]\n"
            "         [--beam-size N] [--beam-threshold F]\n"
            "         [--lm-weight F] [--word-score F]\n",
            argv[0]);
        return 1;
    }
    std::string model_path = argv[1];
    std::string audio_path = argv[2];
    int chunk_ms = 160;
    int gpu = 0;
    RecognizerConfig cfg;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--chunk-ms" && i + 1 < argc)
            chunk_ms = std::atoi(argv[++i]);
        else if (a == "--gpu" && i + 1 < argc)
            gpu = std::atoi(argv[++i]);
        else if (a == "--lm-path" && i + 1 < argc) {
            cfg.decoder.lm_path = argv[++i];
            cfg.decoder.kind = DecoderConfig::Kind::Flashlight;
        } else if (a == "--lexicon" && i + 1 < argc)
            cfg.decoder.lexicon_path = argv[++i];
        else if (a == "--beam-size" && i + 1 < argc) {
            cfg.decoder.beam_size = std::atoi(argv[++i]);
        } else if (a == "--beam-threshold" && i + 1 < argc)
            cfg.decoder.beam_threshold = std::atof(argv[++i]);
        else if (a == "--lm-weight" && i + 1 < argc)
            cfg.decoder.lm_weight = std::atof(argv[++i]);
        else if (a == "--word-score" && i + 1 < argc)
            cfg.decoder.word_insertion_score = std::atof(argv[++i]);
        else if (a == "--vad-model" && i + 1 < argc)
            cfg.vad.model_path = argv[++i];
        else if (a == "--vad-onset" && i + 1 < argc)
            cfg.vad.masker.onset = std::atof(argv[++i]);
        else if (a == "--vad-offset" && i + 1 < argc)
            cfg.vad.masker.offset = std::atof(argv[++i]);
        else if (a == "--vad-mask-value" && i + 1 < argc)
            cfg.vad.masker.mask_value = std::atof(argv[++i]);
    }

    ggml_runtime::Params bm_params;
    bm_params.use_gpu = (gpu >= 0);
    bm_params.gpu_device_idx = std::max(gpu, 0);
    bm_params.pe_bin_path = const_cast<char*>("");
    ggml_runtime::BackendManager bm(bm_params);
    auto model = AsrModel::load(bm, model_path);
    auto* ctc = dynamic_cast<CtcModel*>(model.get());
    if (!ctc) {
        std::fprintf(stderr, "[test] expected a CTC model\n");
        return 2;
    }
    std::vector<float> audio;
    int sr = 0;
    if (!read_wav_mono_16k(audio_path, audio, sr)) {
        std::fprintf(stderr, "failed to read WAV: %s\n", audio_path.c_str());
        return 2;
    }

    BufferedStreamRunner runner(ctc, cfg);
    if (!cfg.decoder.lm_path.empty()) {
        std::fprintf(
            stderr,
            "[ctc-stream] decoder: Flashlight LexiconDecoder + KenLM "
            "(beam=%d, lm_weight=%.2f)\n",
            cfg.decoder.beam_size, cfg.decoder.lm_weight);
    } else {
        std::fprintf(stderr, "[ctc-stream] decoder: greedy CTC\n");
    }

    const size_t chunk_samples = static_cast<size_t>(chunk_ms * sr / 1000);
    std::string last_partial;
    std::vector<double> step_ms;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t off = 0; off < audio.size(); off += chunk_samples) {
        const size_t n = std::min(chunk_samples, audio.size() - off);
        runner.feed_audio(audio.data() + off, n);
        auto cs0 = std::chrono::high_resolution_clock::now();
        auto u = runner.step();
        auto cs1 = std::chrono::high_resolution_clock::now();
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
        stderr, "[stats] %.2fs audio in %.0f ms (RTF %.4f)\n", audio_sec, ms,
        ms / 1000.0 / audio_sec);
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
    std::fprintf(stdout, "%s\n", final_update.transcript_so_far.c_str());
    return 0;
}
