// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Concurrency / crosstalk stress test for the cache-aware RNNT streaming path.
//
// One shared AsrModel + one shared cache-aware encoder Session is driven by many
// threads at once, each running its own CacheStreamRunner (its own indexed
// cache/predictor rows) over one of several DIFFERENT audios. Every transcript
// must match the single-stream reference for the audio it ran - any divergence
// means one stream's K/V/conv cache leaked into another (crosstalk). A watchdog
// thread flags a hang as a deadlock instead of letting the test wait forever.
//
// Usage: ./test_rnnt_concurrency <model.gguf> <a.wav> [b.wav ...]
//                                [--gpu N] [--threads N] [--iters N] [--chunk-ms N]
//                                [--language CODE]
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "fe.h"
#include "model.h"
#include "recognizer.h"  // RecognizerConfig
#include "runner.h"
#include "runtime.h"

using namespace nemo_speech::asr;

static bool
verify_frontend_batch(
    MelSpectrogramExtractor& fe, const std::vector<float>& audio, int sample_rate, int chunk_ms) {
    const size_t n = std::min(
        audio.size(), static_cast<size_t>(sample_rate) * static_cast<size_t>(chunk_ms) / 1000);
    std::vector<float> reference;
    int ref_frames = 0;
    fe.compute(audio.data(), n, reference, ref_frames, /*reflect_left=*/true, /*normalize=*/false);

    struct Result {
        std::vector<float> features;
        int frames = 0;
    };
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::future<Result>> calls;
    for (int i = 0; i < 4; ++i) {
        calls.push_back(std::async(std::launch::async, [&] {
            ready.fetch_add(1);
            while (!go.load()) std::this_thread::yield();
            Result r;
            fe.compute(
                audio.data(), n, r.features, r.frames, /*reflect_left=*/true,
                /*normalize=*/false);
            return r;
        }));
    }
    while (ready.load() != 4) std::this_thread::yield();
    go.store(true);
    float max_error = 0.0f;
    for (auto& call : calls) {
        const auto r = call.get();
        if (r.frames != ref_frames || r.features.size() != reference.size())
            return false;
        for (size_t i = 0; i < reference.size(); ++i)
            max_error = std::max(max_error, std::fabs(r.features[i] - reference[i]));
    }
    std::fprintf(stderr, "[frontend batch] B=4 max abs error=%.8g\n", max_error);
    return max_error == 0.0f;
}

namespace {

bool
verify_decoder_batch(RnntModel& model) {
    constexpr int B = 4;
    constexpr int T = 3;
    const int J = model.rnnt_config().joint_dim;
    std::vector<std::vector<float>> enc(B, std::vector<float>(static_cast<size_t>(J) * T));
    for (int b = 0; b < B; ++b)
        for (size_t i = 0; i < enc[b].size(); ++i)
            enc[b][i] = 0.35f * std::sin(0.013f * static_cast<float>(i + 17 * b));

    std::vector<std::vector<int32_t>> expected(B, std::vector<int32_t>(T));
    for (int b = 0; b < B; ++b) {
        auto state = model.make_rnnt_stream_state();
        for (int step = 0; step < 4; ++step) {
            const int token = (b + 13 * step) % (model.rnnt_config().vocab_size - 1);
            model.predict_rnnt(*state, token, step & 1);
        }
        model.joint_argmax(*state, enc[b].data(), J, T, expected[b].data());
    }

    std::vector<std::unique_ptr<RnntStreamState>> states;
    for (int b = 0; b < B; ++b) states.push_back(model.make_rnnt_stream_state());
    std::vector<std::vector<int32_t>> got(B, std::vector<int32_t>(T));
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> pool;
    for (int b = 0; b < B; ++b) {
        pool.emplace_back([&, b]() {
            ready.fetch_add(1);
            while (!go.load()) std::this_thread::yield();
            for (int step = 0; step < 4; ++step) {
                const int token = (b + 13 * step) % (model.rnnt_config().vocab_size - 1);
                model.predict_rnnt(*states[b], token, step & 1);
            }
            model.joint_argmax(*states[b], enc[b].data(), J, T, got[b].data());
        });
    }
    while (ready.load() != B) std::this_thread::yield();
    go.store(true);
    for (auto& thread : pool) thread.join();
    for (int b = 0; b < B; ++b) {
        if (got[b] != expected[b]) {
            std::fprintf(stderr, "[FAIL] batched RNNT decoder parity mismatch at item %d\n", b);
            return false;
        }
    }
    const auto pm = model.predictor_batch_metrics();
    const auto jm = model.joint_batch_metrics();
    std::fprintf(
        stderr, "[decoder batch] predictor max B=%llu joint max B=%llu\n",
        (unsigned long long)pm.max_observed_batch, (unsigned long long)jm.max_observed_batch);
    return pm.max_observed_batch >= B && jm.max_observed_batch >= B;
}

bool
verify_fresh_encoder_slots(RnntModel& model, int right_context) {
    constexpr int B = 4;
    const auto cfg = make_cache_aware_config(model.encoder_config(), right_context);
    const int M = model.fe_config().n_mels;
    const int mel_frames = 9 + cfg.subsampling_factor * cfg.cache_chunk_frames;
    const int mask_len = cfg.cache_left_ctx + cfg.cache_chunk_frames;
    std::vector<float> mel(static_cast<size_t>(M) * mel_frames);
    std::vector<float> mask(static_cast<size_t>(mask_len));
    for (size_t i = 0; i < mel.size(); ++i)
        mel[i] = 0.2f * std::sin(0.017f * static_cast<float>(i));
    for (int i = 0; i < cfg.cache_left_ctx; ++i) mask[static_cast<size_t>(i)] = -1e9f;

    std::vector<CacheAwareEncoder::State> states;
    for (int b = 0; b < B; ++b) states.push_back(model.make_cache_state());
    std::vector<std::vector<float>> output(B);
    std::vector<int> frames(B);
    for (int b = 0; b < B; ++b) {
        model.encode_cache_aware(
            states[b], mel.data(), mel_frames, mask.data(), mask_len, output[b], frames[b]);
    }

    float max_error = 0.0f;
    for (int b = 1; b < B; ++b) {
        if (frames[b] != frames[0] || output[b].size() != output[0].size())
            return false;
        for (size_t i = 0; i < output[0].size(); ++i)
            max_error = std::max(max_error, std::abs(output[b][i] - output[0][i]));
    }
    std::fprintf(stderr, "[encoder slots] first-use max abs error=%.6g\n", max_error);
    return max_error == 0.0f;
}

bool
verify_encoder_batch(RnntModel& model, int right_context) {
    constexpr int B = 4;
    constexpr float kMaxAbsError = 7.5e-2f;
    constexpr double kMaxRelativeRmse = 2.5e-2;
    const auto cfg = make_cache_aware_config(model.encoder_config(), right_context);
    const int M = model.fe_config().n_mels;
    const int mel_frames = 9 + cfg.subsampling_factor * cfg.cache_chunk_frames;
    const int mask_len = cfg.cache_left_ctx + cfg.cache_chunk_frames;
    constexpr int steps = 4;
    std::vector<std::vector<float>> mel(B, std::vector<float>(static_cast<size_t>(M) * mel_frames));
    std::vector<std::vector<float>> masks(steps, std::vector<float>(static_cast<size_t>(mask_len)));
    for (int step = 0; step < steps; ++step) {
        const int offset = std::max(0, cfg.cache_left_ctx - step * cfg.cache_chunk_frames);
        for (int i = 0; i < mask_len; ++i) masks[step][i] = i < offset ? -1e9f : 0.0f;
    }
    for (int b = 0; b < B; ++b)
        for (size_t i = 0; i < mel[b].size(); ++i)
            mel[b][i] = 0.2f * std::sin(0.017f * static_cast<float>(i + 31 * b));

    std::vector<std::vector<float>> expected(B);
    std::vector<int> expected_T(B);
    for (int b = 0; b < B; ++b) {
        auto state = model.make_cache_state();
        for (int step = 0; step < steps; ++step)
            model.encode_cache_aware(
                state, mel[b].data(), mel_frames, masks[step].data(), mask_len, expected[b],
                expected_T[b]);
    }

    struct BatchResult {
        std::vector<std::vector<float>> values = std::vector<std::vector<float>>(B);
        std::vector<int> frames = std::vector<int>(B);
    };
    auto run_batch = [&]() {
        BatchResult result;
        std::vector<CacheAwareEncoder::State> states;
        for (int b = 0; b < B; ++b) states.push_back(model.make_cache_state());
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::vector<std::thread> pool;
        for (int b = 0; b < B; ++b) {
            pool.emplace_back([&, b]() {
                ready.fetch_add(1);
                while (!go.load()) std::this_thread::yield();
                for (int step = 0; step < steps; ++step)
                    model.encode_cache_aware(
                        states[b], mel[b].data(), mel_frames, masks[step].data(), mask_len,
                        result.values[b], result.frames[b]);
            });
        }
        while (ready.load() != B) std::this_thread::yield();
        go.store(true);
        for (auto& thread : pool) thread.join();
        return result;
    };

    const auto got = run_batch();
    const auto repeated = run_batch();

    float batch_shape_max_error = 0.0f;
    float repeat_max_error = 0.0f;
    long double batch_shape_squared_error = 0.0L;
    long double repeat_squared_error = 0.0L;
    long double reference_squared = 0.0L;
    float reference_max = 0.0f;
    size_t compared_values = 0;
    for (int b = 0; b < B; ++b) {
        if (got.frames[b] != expected_T[b] || repeated.frames[b] != got.frames[b] ||
            got.values[b].size() != expected[b].size() ||
            repeated.values[b].size() != got.values[b].size())
            return false;
        for (size_t i = 0; i < got.values[b].size(); ++i) {
            if (!std::isfinite(expected[b][i]) || !std::isfinite(got.values[b][i]) ||
                !std::isfinite(repeated.values[b][i]))
                return false;
            batch_shape_max_error =
                std::max(batch_shape_max_error, std::abs(got.values[b][i] - expected[b][i]));
            repeat_max_error =
                std::max(repeat_max_error, std::abs(repeated.values[b][i] - got.values[b][i]));
            const long double batch_error =
                static_cast<long double>(got.values[b][i]) - expected[b][i];
            const long double repeat_error =
                static_cast<long double>(repeated.values[b][i]) - got.values[b][i];
            batch_shape_squared_error += batch_error * batch_error;
            repeat_squared_error += repeat_error * repeat_error;
            reference_squared += static_cast<long double>(expected[b][i]) * expected[b][i];
            reference_max = std::max(reference_max, std::abs(expected[b][i]));
            ++compared_values;
        }
    }
    const double batch_shape_rmse =
        std::sqrt(static_cast<double>(batch_shape_squared_error / compared_values));
    const double repeat_rmse =
        std::sqrt(static_cast<double>(repeat_squared_error / compared_values));
    const double reference_rms =
        std::sqrt(static_cast<double>(reference_squared / compared_values));
    const double batch_shape_relative_rmse = batch_shape_rmse / reference_rms;
    const double repeat_relative_rmse = repeat_rmse / reference_rms;
    const auto metrics = model.encoder_batch_metrics();
    std::fprintf(
        stderr,
        "[encoder batch] max B=%llu reference max=%.6g RMS=%.6g "
        "B=1/B=4 max abs error=%.6g relative RMSE=%.6g "
        "repeat B=4 max abs error=%.6g relative RMSE=%.6g\n",
        (unsigned long long)metrics.max_observed_batch, reference_max, reference_rms,
        batch_shape_max_error, batch_shape_relative_rmse, repeat_max_error, repeat_relative_rmse);
    // BF16 and quantized encoder graphs can change accumulation order with the
    // batch shape and with an item's row in that batch. Bound both effects; the
    // full streaming phase below separately requires exact transcript parity.
    return metrics.max_observed_batch >= B && batch_shape_max_error <= kMaxAbsError &&
           repeat_max_error <= kMaxAbsError && batch_shape_relative_rmse <= kMaxRelativeRmse &&
           repeat_relative_rmse <= kMaxRelativeRmse;
}

// Run one full streaming utterance on a fresh runner and return the final
// transcript. Mirrors test_rnnt_streaming's drive loop (no diagnostics).
std::string
run_stream(
    RnntModel& model, const RecognizerConfig& cfg, const std::vector<float>& audio, int sr,
    int chunk_ms, int prompt_index) {
    CacheStreamRunner runner(&model, cfg);
    runner.set_prompt_index(prompt_index);
    const size_t chunk_samples = static_cast<size_t>(chunk_ms) * sr / 1000;
    for (size_t off = 0; off < audio.size(); off += chunk_samples) {
        const size_t n = std::min(chunk_samples, audio.size() - off);
        runner.feed_audio(audio.data() + off, n);
        (void)runner.step();
    }
    return runner.finalize().transcript_so_far;
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(
            stderr,
            "Usage: %s <model.gguf> <a.wav> [b.wav ...] "
            "[--gpu N] [--threads N] [--iters N] [--chunk-ms N] "
            "[--right-context N] [--language CODE]\n",
            argv[0]);
        return 1;
    }
    std::string model_path = argv[1];
    std::vector<std::string> wavs;
    std::string language;
    int gpu = 0, threads = 8, iters = 4, chunk_ms = 160, right_context = 1, deadline_s = 120;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gpu" && i + 1 < argc)
            gpu = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc)
            threads = std::atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc)
            iters = std::atoi(argv[++i]);
        else if (a == "--chunk-ms" && i + 1 < argc)
            chunk_ms = std::atoi(argv[++i]);
        else if (a == "--right-context" && i + 1 < argc)
            right_context = std::atoi(argv[++i]);
        else if (a == "--deadline-s" && i + 1 < argc)
            deadline_s = std::atoi(argv[++i]);  // watchdog; raise under compute-sanitizer
        else if (a == "--language" && i + 1 < argc)
            language = argv[++i];
        else
            wavs.push_back(a);
    }
    if (wavs.empty()) {
        std::fprintf(stderr, "[test] need at least one wav\n");
        return 1;
    }

    ggml_runtime::Params bm_params;
    bm_params.use_gpu = (gpu >= 0);
    bm_params.gpu_device_idx = std::max(gpu, 0);
    bm_params.pe_bin_path = const_cast<char*>("");
    ggml_runtime::BackendManager bm(bm_params);
    BatchingConfig batching;
    batching.enabled = true;
    batching.max_batch_size = 4;
    batching.max_queue_delay_us = 20000;
    auto model = AsrModel::load(bm, model_path, batching);
    auto* rnnt = dynamic_cast<RnntModel*>(model.get());
    if (!rnnt) {
        std::fprintf(stderr, "[test] expected an RNNT model\n");
        return 2;
    }

    // Check the frontend before the independent decoder/encoder exactness
    // assertions. Optimized encoder kernels can deliberately change floating-
    // point accumulation order; that must not prevent this test from reporting
    // whether a frontend batching change is bit-identical to B=1.
    if (rnnt->fe().uses_gpu()) {
        std::vector<float> frontend_audio;
        int frontend_sr = 0;
        if (!read_wav_mono_16k(wavs.front(), frontend_audio, frontend_sr) ||
            frontend_sr != rnnt->sample_rate()) {
            std::fprintf(stderr, "[test] failed to read/resample %s\n", wavs.front().c_str());
            return 2;
        }
        if (!verify_frontend_batch(rnnt->fe(), frontend_audio, frontend_sr, chunk_ms)) {
            std::fprintf(stderr, "[FAIL] GPU frontend B=1/B=4 feature mismatch\n");
            return 5;
        }
    }
    if (!verify_decoder_batch(*rnnt))
        return 5;

    RecognizerConfig cfg;
    cfg.streaming.rnnt_right_context = right_context;
    rnnt->set_cache_right_ctx(cfg.streaming.rnnt_right_context);
    if (!verify_fresh_encoder_slots(*rnnt, right_context))
        return 5;
    if (!verify_encoder_batch(*rnnt, right_context))
        return 5;
    const int prompt_index = rnnt->prompt_index_for_lang(language);

    // Compute single-stream references and prime the shared encoder graph cache
    // before exercising concurrent cache-hit execution.
    std::vector<std::vector<float>> audios(wavs.size());
    std::vector<std::string> refs(wavs.size());
    for (size_t i = 0; i < wavs.size(); i++) {
        int sr = 0;
        if (!read_wav_mono_16k(wavs[i], audios[i], sr) || sr != rnnt->sample_rate()) {
            std::fprintf(stderr, "[test] failed to read/resample %s\n", wavs[i].c_str());
            return 2;
        }
        refs[i] = run_stream(*rnnt, cfg, audios[i], sr, chunk_ms, prompt_index);
        std::fprintf(stderr, "[ref %zu] %-28s -> \"%s\"\n", i, wavs[i].c_str(), refs[i].c_str());
    }
    const int sr = rnnt->sample_rate();

    // Watchdog: if the concurrent phase doesn't finish in `deadline`, declare a
    // deadlock and hard-exit (so the test fails loudly rather than hanging).
    std::atomic<bool> done{false};
    std::mutex wd_mu;
    std::condition_variable wd_cv;
    const auto deadline = std::chrono::seconds(deadline_s);
    std::thread watchdog([&]() {
        std::unique_lock<std::mutex> lk(wd_mu);
        if (!wd_cv.wait_for(lk, deadline, [&]() { return done.load(); })) {
            std::fprintf(
                stderr, "\n[FAIL] DEADLOCK: concurrent phase exceeded %llds\n",
                (long long)deadline.count());
            std::fflush(stderr);
            std::_Exit(3);
        }
    });

    std::atomic<int> mismatches{0};
    std::atomic<int> completed{0};
    std::mutex log_mu;

    auto worker = [&](int tid) {
        for (int k = 0; k < iters; k++) {
            // Mix audios across threads and iterations so different streams with
            // different state run truly concurrently.
            const size_t aidx = static_cast<size_t>(tid + k) % audios.size();
            const std::string got =
                run_stream(*rnnt, cfg, audios[aidx], sr, chunk_ms, prompt_index);
            completed.fetch_add(1);
            if (got != refs[aidx]) {
                mismatches.fetch_add(1);
                std::lock_guard<std::mutex> lg(log_mu);
                std::fprintf(
                    stderr,
                    "[CROSSTALK] tid=%d iter=%d audio=%zu\n  expected: \"%s\"\n  got:      "
                    "\"%s\"\n",
                    tid, k, aidx, refs[aidx].c_str(), got.c_str());
            }
        }
    };

    const auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int t = 0; t < threads; t++) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
    const auto t1 = std::chrono::high_resolution_clock::now();

    {
        std::lock_guard<std::mutex> lk(wd_mu);
        done.store(true);
    }
    wd_cv.notify_all();
    watchdog.join();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(
        stderr, "\n[stats] threads=%d iters=%d streams=%d audios=%zu in %.0f ms (%.1f ms/stream)\n",
        threads, iters, completed.load(), audios.size(), ms,
        completed.load() ? ms / completed.load() : 0.0);

    if (mismatches.load() != 0) {
        std::fprintf(
            stderr, "[FAIL] %d/%d streams diverged (crosstalk)\n", mismatches.load(),
            completed.load());
        return 4;
    }
    const auto fm = rnnt->fe().batch_metrics();
    const auto em = rnnt->encoder_batch_metrics();
    const auto pm = rnnt->predictor_batch_metrics();
    const auto jm = rnnt->joint_batch_metrics();
    std::fprintf(
        stderr,
        "[batch stats] frontend max B=%llu encoder max B=%llu predictor max B=%llu "
        "joint max B=%llu\n",
        (unsigned long long)fm.max_observed_batch, (unsigned long long)em.max_observed_batch,
        (unsigned long long)pm.max_observed_batch, (unsigned long long)jm.max_observed_batch);
    if (threads > 1 &&
        ((rnnt->fe().uses_gpu() && fm.max_observed_batch < 2) || em.max_observed_batch < 2 ||
         pm.max_observed_batch < 2 || jm.max_observed_batch < 2)) {
        std::fprintf(stderr, "[FAIL] concurrent run did not exercise every RNNT batch stage\n");
        return 5;
    }
    std::fprintf(
        stderr, "[PASS] %d concurrent streams, 0 crosstalk, no deadlock\n", completed.load());
    return 0;
}
