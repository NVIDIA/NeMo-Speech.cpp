// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Offline CTC smoke test. Loads a parakeet CTC GGUF, runs a WAV through
// the new AsrModel + BufferedStreamRunner pipeline, prints transcript.
//
// Usage: ./test_ctc_offline <model.gguf> <audio.wav> [--gpu N]
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "fe.h"  // read_wav_mono_16k
#include "model.h"
#include "recognizer.h"  // RecognizerConfig
#include "runner.h"
#include "runtime.h"

using namespace nemo_speech::asr;

static std::vector<int32_t>
frame_argmax(const std::vector<float>& log_probs, int T, int C) {
    std::vector<int32_t> ids(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t) {
        int best = 0;
        for (int c = 1; c < C; ++c)
            if (log_probs[static_cast<size_t>(t) * C + c] >
                log_probs[static_cast<size_t>(t) * C + best])
                best = c;
        ids[static_cast<size_t>(t)] = best;
    }
    return ids;
}

static std::vector<int32_t>
collapse_ids(const std::vector<int32_t>& ids, int blank) {
    std::vector<int32_t> out;
    int previous = -1;
    for (int id : ids) {
        if (id != blank && id != previous)
            out.push_back(id);
        previous = id;
    }
    return out;
}

static std::vector<int32_t>
collapsed_argmax(const std::vector<float>& log_probs, int T, int C, int blank) {
    return collapse_ids(frame_argmax(log_probs, T, C), blank);
}

int
main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <model.gguf> <audio.wav> [--gpu N]\n", argv[0]);
        return 1;
    }
    std::string model_path = argv[1];
    std::string audio_path = argv[2];
    int gpu = 0;
    bool verify_compact = false;
    bool verify_batch = false;
    bool verify_offline_fe = false;
    RecognizerConfig cfg;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gpu" && i + 1 < argc)
            gpu = std::atoi(argv[++i]);
        else if (a == "--verify-compact")
            verify_compact = true;
        else if (a == "--verify-batch")
            verify_batch = true;
        else if (a == "--verify-offline-fe")
            verify_offline_fe = true;
        else if (a == "--vad-model" && i + 1 < argc)
            cfg.vad.model_path = argv[++i];
        else if (a == "--vad-onset" && i + 1 < argc)
            cfg.vad.masker.onset = std::atof(argv[++i]);
        else if (a == "--vad-offset" && i + 1 < argc)
            cfg.vad.masker.offset = std::atof(argv[++i]);
        else if (a == "--vad-mask-value" && i + 1 < argc)
            cfg.vad.masker.mask_value = std::atof(argv[++i]);
        else if (a == "--vad-pad-ms" && i + 1 < argc)
            cfg.vad.masker.pad_onset_ms = cfg.vad.masker.pad_offset_ms = std::atof(argv[++i]);
        else if (a == "--vad-min-duration-off-ms" && i + 1 < argc)
            cfg.vad.masker.min_duration_off_ms = std::atof(argv[++i]);
        else if (a == "--vad-stddev-floor" && i + 1 < argc)
            cfg.vad.masker.stddev_floor = std::atof(argv[++i]);
    }

    ggml_runtime::Params bm_params;
    bm_params.use_gpu = (gpu >= 0);
    bm_params.gpu_device_idx = std::max(gpu, 0);
    bm_params.pe_bin_path = const_cast<char*>("");
    ggml_runtime::BackendManager bm(bm_params);
    if (verify_batch) {
        cfg.batching.enabled = true;
        cfg.batching.max_batch_size = 4;
        cfg.batching.max_queue_delay_us = 20000;
    }
    auto model = AsrModel::load(bm, model_path, cfg.batching);
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
    if (sr != model->sample_rate()) {
        std::fprintf(stderr, "WAV sr %d != model %d\n", sr, model->sample_rate());
        return 2;
    }

    if (verify_compact) {
        std::vector<float> mel, log_probs, best_probs;
        std::vector<int32_t> best_ids;
        int n_mel = 0, T_full = 0, C = 0, T_compact = 0;
        ctc->fe().compute(audio.data(), audio.size(), mel, n_mel);
        ctc->infer_ctc_from_mel(mel.data(), n_mel, log_probs, T_full, C);
        ctc->infer_ctc_greedy_from_mel(mel.data(), n_mel, best_ids, best_probs, T_compact);
        if (T_full != T_compact) {
            std::fprintf(
                stderr, "[compact] frame mismatch: full=%d compact=%d\n", T_full, T_compact);
            return 3;
        }
        float max_prob_err = 0.0f;
        for (int t = 0; t < T_full; ++t) {
            int expected = 0;
            for (int c = 1; c < C; ++c)
                if (log_probs[static_cast<size_t>(t) * C + c] >
                    log_probs[static_cast<size_t>(t) * C + expected])
                    expected = c;
            if (best_ids[static_cast<size_t>(t)] != expected) {
                std::fprintf(
                    stderr, "[compact] argmax mismatch at frame %d: full=%d compact=%d\n", t,
                    expected, best_ids[static_cast<size_t>(t)]);
                return 3;
            }
            max_prob_err = std::max(
                max_prob_err, std::fabs(
                                  std::exp(log_probs[static_cast<size_t>(t) * C + expected]) -
                                  best_probs[static_cast<size_t>(t)]));
        }
        if (max_prob_err > 1e-5f) {
            std::fprintf(stderr, "[compact] winning-prob max error %.8g\n", max_prob_err);
            return 3;
        }
        std::fprintf(
            stderr, "[compact] exact argmax parity over %d frames; max probability error %.3g\n",
            T_full, max_prob_err);
    }

    if (verify_offline_fe) {
        std::vector<float> cpu_mel, cpu_lp, gpu_lp;
        std::vector<float> compact_probs;
        std::vector<int32_t> compact_ids;
        int n_mel = 0, cpu_T = 0, cpu_C = 0, gpu_T = 0, gpu_C = 0;
        int compact_T = 0;
        // With batching disabled, the shared streaming frontend is the CPU
        // radix-2 reference while infer_ctc() selects the offline GPU frontend.
        ctc->fe().compute(audio.data(), audio.size(), cpu_mel, n_mel);
        ctc->infer_ctc_from_mel(cpu_mel.data(), n_mel, cpu_lp, cpu_T, cpu_C);
        ctc->infer_ctc(audio.data(), audio.size(), gpu_lp, gpu_T, gpu_C);
        ctc->infer_ctc_greedy(audio.data(), audio.size(), compact_ids, compact_probs, compact_T);
        const auto cpu_ids = collapsed_argmax(cpu_lp, cpu_T, cpu_C, ctc->ctc_config().blank_id);
        const auto gpu_ids = collapsed_argmax(gpu_lp, gpu_T, gpu_C, ctc->ctc_config().blank_id);
        if (cpu_ids != gpu_ids) {
            std::fprintf(
                stderr, "[offline-fe] CPU/GPU collapsed-token mismatch (%zu vs %zu)\n",
                cpu_ids.size(), gpu_ids.size());
            return 5;
        }
        if (compact_T != gpu_T || compact_ids != frame_argmax(gpu_lp, gpu_T, gpu_C)) {
            std::fprintf(stderr, "[offline-fe] compact/full token mismatch\n");
            return 5;
        }
        for (int t = 0; t < compact_T; ++t) {
            const int id = compact_ids[static_cast<size_t>(t)];
            const float expected = std::exp(gpu_lp[static_cast<size_t>(t) * gpu_C + id]);
            if (std::fabs(compact_probs[static_cast<size_t>(t)] - expected) > 1e-5f) {
                std::fprintf(stderr, "[offline-fe] compact/full probability mismatch\n");
                return 5;
            }
        }
        if (gpu >= 0 && !ctc->offline_frontend_uses_gpu()) {
            std::fprintf(stderr, "[offline-fe] CUDA run did not select GPU frontend\n");
            return 5;
        }
        std::fprintf(
            stderr, "[offline-fe] CPU/GPU exact collapsed-token parity (%zu tokens)\n",
            cpu_ids.size());
    }

    if (verify_batch) {
        std::vector<float> mel, ref_probs, ref_log_probs;
        std::vector<int32_t> ref_ids;
        int n_mel = 0, ref_T = 0, ref_full_T = 0, ref_C = 0;
        ctc->fe().compute(audio.data(), audio.size(), mel, n_mel);
        ctc->infer_ctc_from_mel(mel.data(), n_mel, ref_log_probs, ref_full_T, ref_C);
        ctc->infer_ctc_greedy_from_mel(mel.data(), n_mel, ref_ids, ref_probs, ref_T);
        const auto ref_full_ids =
            collapsed_argmax(ref_log_probs, ref_full_T, ref_C, ctc->ctc_config().blank_id);

        struct FullResult {
            std::vector<float> values;
            int T = 0;
            int C = 0;
        };
        std::atomic<int> full_ready{0};
        std::atomic<bool> full_go{false};
        std::vector<std::future<FullResult>> full_calls;
        for (int i = 0; i < 4; ++i) {
            full_calls.push_back(std::async(std::launch::async, [&] {
                full_ready.fetch_add(1);
                while (!full_go.load()) std::this_thread::yield();
                FullResult r;
                ctc->infer_ctc_from_mel(mel.data(), n_mel, r.values, r.T, r.C);
                return r;
            }));
        }
        while (full_ready.load() != 4) std::this_thread::yield();
        full_go.store(true);
        float full_max_error = 0.0f;
        std::vector<int32_t> batched_full_ids;
        std::vector<float> batched_full_probs;
        for (auto& call : full_calls) {
            auto r = call.get();
            if (r.T != ref_full_T || r.C != ref_C || r.values.size() != ref_log_probs.size()) {
                std::fprintf(stderr, "[batch] full CTC output shape mismatch\n");
                return 4;
            }
            for (size_t i = 0; i < r.values.size(); ++i)
                full_max_error =
                    std::max(full_max_error, std::fabs(r.values[i] - ref_log_probs[i]));
            if (!std::isfinite(full_max_error) ||
                collapsed_argmax(r.values, r.T, r.C, ctc->ctc_config().blank_id) != ref_full_ids) {
                std::fprintf(
                    stderr, "[batch] full CTC token mismatch (max logit error %.8g)\n",
                    full_max_error);
                return 4;
            }
            const auto ids = frame_argmax(r.values, r.T, r.C);
            if (batched_full_ids.empty()) {
                batched_full_ids = ids;
                batched_full_probs.resize(ids.size());
                for (size_t t = 0; t < ids.size(); ++t)
                    batched_full_probs[t] =
                        std::exp(r.values[t * static_cast<size_t>(r.C) + ids[t]]);
            } else if (ids != batched_full_ids) {
                std::fprintf(stderr, "[batch] B=4 full CTC peers are non-deterministic\n");
                return 4;
            }
        }
        std::fprintf(
            stderr, "[batch] full CTC exact collapsed-token parity (max logit error %.8g)\n",
            full_max_error);

        struct BatchResult {
            std::vector<int32_t> ids;
            std::vector<float> probs;
            int T = 0;
        };
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::vector<std::future<BatchResult>> calls;
        for (int i = 0; i < 4; ++i) {
            calls.push_back(std::async(std::launch::async, [&, i] {
                (void)i;
                ready.fetch_add(1);
                while (!go.load()) std::this_thread::yield();
                BatchResult r;
                ctc->infer_ctc_greedy_from_mel(mel.data(), n_mel, r.ids, r.probs, r.T);
                return r;
            }));
        }
        while (ready.load() != 4) std::this_thread::yield();
        go.store(true);
        for (auto& call : calls) {
            auto r = call.get();
            if (r.T != ref_T || r.probs.size() != batched_full_probs.size()) {
                size_t first = 0;
                while (first < r.ids.size() && first < ref_ids.size() &&
                       r.ids[first] == ref_ids[first])
                    ++first;
                std::fprintf(
                    stderr,
                    "[batch] batched CTC mismatch: T=%d/%d ids=%zu/%zu probs=%zu/%zu first=%zu "
                    "id=%d/%d\n",
                    r.T, ref_T, r.ids.size(), ref_ids.size(), r.probs.size(), ref_probs.size(),
                    first, first < r.ids.size() ? r.ids[first] : -1,
                    first < ref_ids.size() ? ref_ids[first] : -1);
                return 4;
            }
            float max_err = 0.0f;
            for (size_t i = 0; i < r.probs.size(); ++i) {
                max_err = std::max(max_err, std::fabs(r.probs[i] - batched_full_probs[i]));
            }
            if (!std::isfinite(max_err) || max_err > 1e-4f || r.ids != batched_full_ids) {
                std::fprintf(
                    stderr, "[batch] compact/full B=4 mismatch (probability max error %.8g)\n",
                    max_err);
                return 4;
            }
        }
        const auto metrics = ctc->batch_metrics();
        if (metrics.max_observed_batch < 4) {
            std::fprintf(
                stderr, "[batch] requests did not coalesce (max=%llu)\n",
                static_cast<unsigned long long>(metrics.max_observed_batch));
            return 4;
        }
        std::fprintf(stderr, "[batch] compact/full B=4 parity (%d frames)\n", ref_T);

        // Exercise the raw-audio path too: this must coalesce in the GPU
        // frontend before the already-tested encoder batcher.
        std::vector<float> raw_ref_lp;
        int raw_ref_T = 0, raw_ref_C = 0;
        ctc->infer_ctc(audio.data(), audio.size(), raw_ref_lp, raw_ref_T, raw_ref_C);
        const auto raw_ref_ids =
            collapsed_argmax(raw_ref_lp, raw_ref_T, raw_ref_C, ctc->ctc_config().blank_id);
        std::atomic<int> raw_ready{0};
        std::atomic<bool> raw_go{false};
        std::vector<std::future<FullResult>> raw_calls;
        for (int i = 0; i < 4; ++i) {
            raw_calls.push_back(std::async(std::launch::async, [&] {
                raw_ready.fetch_add(1);
                while (!raw_go.load()) std::this_thread::yield();
                FullResult r;
                ctc->infer_ctc(audio.data(), audio.size(), r.values, r.T, r.C);
                return r;
            }));
        }
        while (raw_ready.load() != 4) std::this_thread::yield();
        raw_go.store(true);
        for (auto& call : raw_calls) {
            const auto r = call.get();
            if (collapsed_argmax(r.values, r.T, r.C, ctc->ctc_config().blank_id) != raw_ref_ids) {
                std::fprintf(stderr, "[batch] raw-audio CTC token mismatch\n");
                return 4;
            }
        }
        const auto fe_metrics = ctc->offline_frontend_batch_metrics();
        if (fe_metrics.max_observed_batch < 4) {
            std::fprintf(
                stderr, "[batch] GPU frontend requests did not coalesce (max=%llu)\n",
                static_cast<unsigned long long>(fe_metrics.max_observed_batch));
            return 4;
        }
        std::fprintf(stderr, "[batch] GPU frontend exact token parity at B=4\n");

        // Exercise the shorter startup and tail shapes used by buffered
        // recognition, including their post-subsampling batch axis.
        const int startup_shapes[] = {209, 225, 241, 257, 273};
        for (int shape : startup_shapes) {
            if (shape > n_mel)
                continue;
            std::vector<int32_t> shape_ref_ids;
            std::vector<float> shape_ref_probs;
            int shape_ref_T = 0;
            ctc->infer_ctc_greedy_from_mel(
                mel.data(), shape, shape_ref_ids, shape_ref_probs, shape_ref_T);
            std::atomic<int> shape_ready{0};
            std::atomic<bool> shape_go{false};
            std::vector<std::future<BatchResult>> shape_calls;
            for (int i = 0; i < 4; ++i) {
                shape_calls.push_back(std::async(std::launch::async, [&] {
                    shape_ready.fetch_add(1);
                    while (!shape_go.load()) std::this_thread::yield();
                    BatchResult r;
                    ctc->infer_ctc_greedy_from_mel(mel.data(), shape, r.ids, r.probs, r.T);
                    return r;
                }));
            }
            while (shape_ready.load() != 4) std::this_thread::yield();
            shape_go.store(true);
            for (auto& call : shape_calls) {
                const auto r = call.get();
                if (r.T != shape_ref_T ||
                    collapse_ids(r.ids, ctc->ctc_config().blank_id) !=
                        collapse_ids(shape_ref_ids, ctc->ctc_config().blank_id)) {
                    size_t first = 0;
                    while (first < r.ids.size() && first < shape_ref_ids.size() &&
                           r.ids[first] == shape_ref_ids[first])
                        ++first;
                    std::fprintf(
                        stderr, "[batch] startup shape=%d mismatch T=%d/%d first=%zu id=%d/%d\n",
                        shape, r.T, shape_ref_T, first, first < r.ids.size() ? r.ids[first] : -1,
                        first < shape_ref_ids.size() ? shape_ref_ids[first] : -1);
                    return 4;
                }
            }
            std::fprintf(
                stderr, "[batch] startup shape=%d B=4 collapsed-token parity (%d frames)\n", shape,
                shape_ref_T);
        }
    }

    BufferedStreamRunner runner(ctc, cfg);
    runner.feed_audio(audio.data(), audio.size());
    auto final_update = runner.finalize();
    std::fprintf(stdout, "%s\n", final_update.transcript_so_far.c_str());
    return 0;
}
