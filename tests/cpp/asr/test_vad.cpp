// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Silero VAD test - two phases over one WAV:
//
//   1. primitive: feed the WAV through SileroVad and check the per-window
//      speech probabilities. Range-sane + speech detected; with --ref
//      <probs.txt> (one float per line, e.g. from whisper.cpp) compare
//      per-window probs to within --tol (the port's numerical-parity gate).
//
//   2. masker: feed the same WAV + a sentinel-filled mel buffer through
//      VadMasker, and check the rate-match + binarize/pad/merge FSM produced
//      a sane in-place mask (every frame sentinel-or-mask_value, both classes
//      present, speech-dominated). Prints a mask map.
//
// Usage: ./test_vad <vad.gguf> <audio.wav> [--gpu N] [--ref probs.txt] [--tol F]
//                   [--n-mels N] [--hop N] [--vad-pad-ms F] [--vad-min-duration-off-ms F]
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "fe.h"  // read_wav_mono_16k
#include "runtime.h"
#include "silero_vad.h"
#include "vad_masker.h"

using namespace nemo_speech::asr;

// Phase 1: SileroVad probabilities. Returns true on pass.
static bool
check_primitive(
    ggml_runtime::BackendManager& bm, const std::string& gguf, const std::vector<float>& audio,
    int sr, const std::string& ref_path, double tol) {
    SileroVad vad(bm, gguf);
    if (sr != vad.config().sample_rate) {
        std::fprintf(stderr, "FAIL: WAV sr %d != VAD %d\n", sr, vad.config().sample_rate);
        return false;
    }
    std::vector<float> probs;
    vad.feed_audio(audio.data(), audio.size(), probs);
    vad.flush(probs);

    float pmin = 1e9f, pmax = -1e9f;
    for (float p : probs) {
        pmin = std::min(pmin, p);
        pmax = std::max(pmax, p);
    }
    std::fprintf(
        stdout, "[primitive] windows=%zu prob range [%.6f, %.6f]\n", probs.size(), pmin, pmax);
    if (pmin < -1e-4f || pmax > 1.0f + 1e-4f) {
        std::fprintf(stderr, "FAIL: probabilities out of [0,1]\n");
        return false;
    }

    if (!ref_path.empty()) {
        std::ifstream f(ref_path);
        if (!f) {
            std::fprintf(stderr, "FAIL: cannot open ref %s\n", ref_path.c_str());
            return false;
        }
        std::vector<float> ref;
        float v;
        while (f >> v) ref.push_back(v);
        if (ref.size() != probs.size()) {
            std::fprintf(stderr, "FAIL: ref has %zu windows, got %zu\n", ref.size(), probs.size());
            return false;
        }
        double max_abs = 0.0;
        for (size_t i = 0; i < probs.size(); i++)
            max_abs = std::max(max_abs, std::abs(static_cast<double>(probs[i]) - ref[i]));
        std::fprintf(stdout, "[primitive] max abs diff vs ref: %.3e (tol %.3e)\n", max_abs, tol);
        if (max_abs > tol) {
            std::fprintf(stderr, "FAIL: exceeds parity tolerance\n");
            return false;
        }
        std::fprintf(stdout, "[primitive] PASS: within ref tolerance\n");
    } else if (pmax < 0.5f) {
        std::fprintf(stderr, "FAIL: no window exceeded 0.5 (expected speech)\n");
        return false;
    } else {
        std::fprintf(stdout, "[primitive] PASS: speech detected (max prob %.3f)\n", pmax);
    }
    return true;
}

// Shared immutable model/session with two interleaved recurrent states. This is
// the production ownership pattern and catches stale SessionState bindings.
static bool
check_shared_states(
    ggml_runtime::BackendManager& bm, const std::string& gguf, const std::vector<float>& audio) {
    BatchingConfig batching;
    batching.enabled = true;
    batching.max_batch_size = 4;
    batching.max_queue_delay_us = 20000;
    auto model = std::make_shared<SileroVadModel>(bm, gguf, batching);
    SileroVad a(model), b(model);
    std::vector<float> pa, pb;
    constexpr size_t kFeed = 733;  // deliberately not a 512-sample boundary
    for (size_t off = 0; off < audio.size(); off += kFeed) {
        const size_t n = std::min(kFeed, audio.size() - off);
        a.feed_audio(audio.data() + off, n, pa);
        b.feed_audio(audio.data() + off, n, pb);
    }
    a.flush(pa);
    b.flush(pb);
    if (pa.size() != pb.size()) {
        std::fprintf(stderr, "FAIL: shared VAD streams returned different lengths\n");
        return false;
    }
    double max_abs = 0.0;
    for (size_t i = 0; i < pa.size(); ++i)
        max_abs = std::max(max_abs, std::abs(static_cast<double>(pa[i]) - pb[i]));
    std::fprintf(
        stdout, "[shared-state] streams=%zu windows each, max abs diff %.3e\n", pa.size(), max_abs);
    if (max_abs > 1e-6) {
        std::fprintf(stderr, "FAIL: shared VAD SessionState crosstalk\n");
        return false;
    }
    a.reset();
    b.reset();
    pa.clear();
    pb.clear();
    const size_t prefix =
        std::min(audio.size(), static_cast<size_t>(model->config().window_size * 4));
    a.feed_audio(audio.data(), prefix, pa);
    b.feed_audio(audio.data(), prefix, pb);
    if (pa != pb) {
        std::fprintf(stderr, "FAIL: in-place VAD reset is not deterministic\n");
        return false;
    }
    std::fprintf(stdout, "[shared-state] PASS: one Session, isolated recurrent state\n");

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::future<std::vector<float>>> calls;
    const size_t batch_prefix =
        std::min(audio.size(), static_cast<size_t>(model->config().window_size * 8));
    for (int i = 0; i < 4; ++i) {
        calls.push_back(std::async(std::launch::async, [&, i] {
            (void)i;
            SileroVad vad(model);
            ready.fetch_add(1);
            while (!go.load()) std::this_thread::yield();
            std::vector<float> probs;
            vad.feed_audio(audio.data(), batch_prefix, probs);
            return probs;
        }));
    }
    while (ready.load() != 4) std::this_thread::yield();
    go.store(true);
    std::vector<float> batch_ref;
    for (auto& call : calls) {
        auto probs = call.get();
        if (batch_ref.empty())
            batch_ref = probs;
        if (probs != batch_ref) {
            std::fprintf(stderr, "FAIL: batched VAD recurrent outputs diverged\n");
            return false;
        }
    }
    if (model->batch_metrics().max_observed_batch < 4) {
        std::fprintf(stderr, "FAIL: VAD requests did not coalesce\n");
        return false;
    }
    std::fprintf(stdout, "[shared-state] B=4 indexed-arena parity PASS\n");
    return true;
}

// Phase 2: VadMasker in-place masking. Returns true on pass.
static bool
check_masker(
    ggml_runtime::BackendManager& bm, const std::string& gguf, const std::vector<float>& audio,
    int n_mels, int hop, const VadMaskerCfg& cfg) {
    SileroVad vad(bm, gguf);
    vad.set_binarizer(cfg.onset, cfg.offset, hop);
    VadMasker masker(&vad, n_mels, cfg);

    // Sentinel mel buffer: the masker overwrites silence frames with mask_value
    // and leaves speech frames as the sentinel, so we can tell them apart.
    const float kSentinel = 1.0f;
    const int n_frames = static_cast<int>(audio.size() / hop) + 1;
    std::vector<float> mel(static_cast<size_t>(n_frames) * n_mels, kSentinel);

    vad.observe_audio(audio.data(), audio.size());
    masker.apply(mel.data(), n_frames, /*first_global_frame=*/0);

    int masked = 0;
    std::string viz;
    for (int f = 0; f < n_frames; f++) {
        const float* row = mel.data() + static_cast<size_t>(f) * n_mels;
        const bool is_masked = (row[0] == cfg.mask_value);
        for (int m = 0; m < n_mels; m++) {
            const float want = is_masked ? cfg.mask_value : kSentinel;
            if (row[m] != want) {
                std::fprintf(stderr, "FAIL: frame %d bin %d = %f, partial mask\n", f, m, row[m]);
                return false;
            }
        }
        if (is_masked)
            masked++;
        if (f % 16 == 0)
            viz.push_back(is_masked ? '.' : '#');  // '.' = silence/masked, '#' = speech
    }

    const double frac = static_cast<double>(masked) / n_frames;
    std::fprintf(stdout, "[masker] frames=%d masked=%d (%.1f%%)\n", n_frames, masked, 100.0 * frac);
    std::fprintf(
        stdout, "[masker] mask map (1 char / 16 frames, '.'=silence '#'=speech):\n%s\n",
        viz.c_str());
    if (masked == 0 || masked == n_frames || frac > 0.6) {
        std::fprintf(
            stderr, "FAIL: implausible mask for a speech clip (%.1f%% masked)\n", 100.0 * frac);
        return false;
    }
    std::fprintf(stdout, "[masker] PASS: sane speech/silence mask\n");
    return true;
}

int
main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(
            stderr,
            "Usage: %s <vad.gguf> <audio.wav> [--gpu N] [--ref probs.txt] [--tol F]\n"
            "          [--n-mels N] [--hop N] [--vad-pad-ms F] [--vad-min-duration-off-ms F]\n",
            argv[0]);
        return 1;
    }
    std::string gguf_path = argv[1];
    std::string audio_path = argv[2];
    int gpu = 0, n_mels = 80, hop = 160;
    std::string ref_path;
    double tol = 1e-4;
    VadMaskerCfg cfg;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gpu" && i + 1 < argc)
            gpu = std::atoi(argv[++i]);
        else if (a == "--ref" && i + 1 < argc)
            ref_path = argv[++i];
        else if (a == "--tol" && i + 1 < argc)
            tol = std::atof(argv[++i]);
        else if (a == "--n-mels" && i + 1 < argc)
            n_mels = std::atoi(argv[++i]);
        else if (a == "--hop" && i + 1 < argc)
            hop = std::atoi(argv[++i]);
        else if (a == "--vad-pad-ms" && i + 1 < argc)
            cfg.pad_onset_ms = cfg.pad_offset_ms = std::atof(argv[++i]);
        else if (a == "--vad-min-duration-off-ms" && i + 1 < argc)
            cfg.min_duration_off_ms = std::atof(argv[++i]);
    }

    ggml_runtime::Params bm_params;
    bm_params.use_gpu = (gpu >= 0);
    bm_params.gpu_device_idx = std::max(gpu, 0);
    bm_params.pe_bin_path = const_cast<char*>("");
    ggml_runtime::BackendManager bm(bm_params);

    std::vector<float> audio;
    int sr = 0;
    if (!read_wav_mono_16k(audio_path, audio, sr)) {
        std::fprintf(stderr, "failed to read WAV: %s\n", audio_path.c_str());
        return 2;
    }

    const bool ok_primitive = check_primitive(bm, gguf_path, audio, sr, ref_path, tol);
    const bool ok_shared = check_shared_states(bm, gguf_path, audio);
    const bool ok_masker = check_masker(bm, gguf_path, audio, n_mels, hop, cfg);
    if (!ok_primitive || !ok_shared || !ok_masker)
        return 3;
    std::fprintf(stdout, "PASS\n");
    return 0;
}
