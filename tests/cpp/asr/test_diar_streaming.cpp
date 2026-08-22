// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Standalone streaming diarization CLI: stream a wav through the Sortformer
// pipeline and print speaker segments (optionally as RTTM for DER scoring).
//
// Usage:
//   test_diar_streaming <sortformer.gguf> <audio.wav> [--gpu] [--offline]
//       [--rttm NAME] [--dump-probs FILE] [--push-ms MS]
//       [--preset streaming|offline]
//       [--chunk N] [--rc N] [--lc N] [--fifo N] [--spkcache N] [--update N]
//       [--onset P] [--offset P] [--pad-onset S] [--pad-offset S]
//       [--min-on S] [--min-off S]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include "diar_pipeline.h"
#include "fe.h"

using namespace nemo_speech::asr;

static const char* kUsage =
    "usage: %s <sortformer.gguf> <audio.wav> [--gpu] [--offline] [--rttm NAME]\n"
    "    [--dump-probs FILE] [--push-ms MS] [--compact-frames N]\n"
    "    [--batching-check]\n"
    "    [--preset streaming|offline]\n"
    "    [--chunk N] [--rc N] [--lc N] [--fifo N] [--spkcache N] [--update N]\n"
    "    [--onset P] [--offset P] [--pad-onset S] [--pad-offset S] [--min-on S] [--min-off S]\n";

int
main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, kUsage, argv[0]);
        return 2;
    }
    const std::string gguf_path = argv[1];
    const std::string wav_path = argv[2];
    bool use_gpu = false;
    bool offline = false;    // full-attention single pass, no streaming state
    std::string probs_path;  // raw f32 (n_frames x n_spk) frame-probs dump
    std::string rttm_name;
    DiarGeometry geo;             // streaming preset
    DiarSegmentationCfg seg_cfg;  // NeMo v2 postprocessing defaults
    int push_ms = 160;
    int compact_frames = 0;  // 0 = library default compaction horizon
    bool batching_check = false;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        // Bounds-checked value fetch: a value-taking flag as the last token is
        // a usage error, not a read past argv.
        auto next_str = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        auto next = [&]() { return std::stoi(next_str()); };
        if (a == "--gpu")
            use_gpu = true;
        else if (a == "--offline")
            offline = true;
        else if (a == "--dump-probs")
            probs_path = next_str();
        else if (a == "--rttm")
            rttm_name = next_str();
        else if (a == "--preset")
            geo = DiarGeometry::preset(next_str());  // later flags override
        else if (a == "--chunk")
            geo.chunk_len = next();
        else if (a == "--rc")
            geo.chunk_right_context = next();
        else if (a == "--lc")
            geo.chunk_left_context = next();
        else if (a == "--fifo")
            geo.fifo_len = next();
        else if (a == "--spkcache")
            geo.spkcache_len = next();
        else if (a == "--update")
            geo.spkcache_update_period = next();
        else if (a == "--push-ms") {
            push_ms = next();
            // push_ms drives the feed loop's stride (push_ms * 16 samples);
            // <= 0 would make the offset never advance.
            if (push_ms <= 0) {
                std::fprintf(stderr, "--push-ms must be > 0 (got %d)\n", push_ms);
                return 2;
            }
        } else if (a == "--compact-frames")
            // Test hook: force aggressive timeline compaction (trigger after N
            // frames, retain N/2) to exercise the long-stream memory bound on
            // short clips. Output must match an uncompacted run exactly.
            compact_frames = next();
        else if (a == "--batching-check")
            batching_check = true;
        else if (a == "--onset")
            seg_cfg.onset = std::stof(next_str());
        else if (a == "--offset")
            seg_cfg.offset = std::stof(next_str());
        else if (a == "--pad-onset")
            seg_cfg.pad_onset = std::stod(next_str());
        else if (a == "--pad-offset")
            seg_cfg.pad_offset = std::stod(next_str());
        else if (a == "--min-on")
            seg_cfg.min_duration_on = std::stod(next_str());
        else if (a == "--min-off")
            seg_cfg.min_duration_off = std::stod(next_str());
        else {
            std::fprintf(stderr, "unknown arg: %s\n%s", a.c_str(), kUsage);
            return 2;
        }
    }

    std::vector<float> audio;
    int sr = 0;
    if (!read_wav_mono_16k(wav_path, audio, sr) || sr != 16000) {
        std::fprintf(stderr, "failed to read 16 kHz mono wav: %s\n", wav_path.c_str());
        return 1;
    }

    if (batching_check && offline) {
        std::fprintf(stderr, "--batching-check requires streaming mode\n");
        return 2;
    }
    ggml_runtime::Params backend_params;
    backend_params.use_gpu = use_gpu;
    ggml_runtime::BackendManager bm(backend_params);
    BatchingConfig batching;
    batching.enabled = batching_check;
    batching.max_batch_size = 8;
    batching.max_queue_delay_us = 500000;
    DiarModel model(bm, gguf_path, batching);
    const ScopedBatchCohort scalar_cohort(1);

    std::vector<float> probs;
    std::vector<DiarSegment> segs;
    int64_t n_frames = 0;
    double sec_per_frame = 0.0;
    if (offline) {
        probs = model.diarize_offline(audio.data(), audio.size(), &n_frames);
        sec_per_frame =
            model.cfg().encoder.subsampling_factor * static_cast<double>(model.cfg().window_stride);
        segs = diar_segments_from_probs(
            probs.data(), n_frames, model.cfg().num_speakers, sec_per_frame, seg_cfg);
    } else {
        DiarStream stream(model, geo);
        if (compact_frames > 0)
            stream.set_compaction(compact_frames, compact_frames / 2);
        const size_t push = static_cast<size_t>(push_ms) * 16;
        for (size_t off = 0; off < audio.size(); off += push) {
            stream.feed_audio(audio.data() + off, std::min(push, audio.size() - off));
        }
        stream.finish();
        // segments() folds any compaction-frozen prefix in; frame_probs() is
        // only the retained tail, so it must not be re-segmented with the
        // total frame count.
        segs = stream.segments(seg_cfg);
        probs = stream.frame_probs();
        n_frames = stream.n_frames();
        sec_per_frame = stream.seconds_per_frame();
    }
    if (batching_check) {
        struct ChunkInput {
            std::vector<float> mel;
            std::vector<float> spkcache;
            std::vector<float> fifo;
            int spkcache_frames = 0;
            int fifo_frames = 0;
        };
        constexpr int kChunkMelFrames = 64;
        const int d_model = model.cfg().encoder.d_model;
        std::vector<ChunkInput> chunk_inputs(4);
        const int state_lengths[4][2] = {{0, 0}, {4, 7}, {9, 3}, {2, 19}};
        for (size_t lane = 0; lane < chunk_inputs.size(); ++lane) {
            auto& input = chunk_inputs[lane];
            input.spkcache_frames = state_lengths[lane][0];
            input.fifo_frames = state_lengths[lane][1];
            input.mel.resize(static_cast<size_t>(model.cfg().n_mels) * kChunkMelFrames);
            input.spkcache.resize(static_cast<size_t>(d_model) * input.spkcache_frames);
            input.fifo.resize(static_cast<size_t>(d_model) * input.fifo_frames);
            for (size_t i = 0; i < input.mel.size(); ++i)
                input.mel[i] = 0.02f * std::sin(static_cast<float>(i + 17 * lane) * 0.013f);
            for (size_t i = 0; i < input.spkcache.size(); ++i)
                input.spkcache[i] = 0.01f * std::cos(static_cast<float>(i + 11 * lane) * 0.007f);
            for (size_t i = 0; i < input.fifo.size(); ++i)
                input.fifo[i] = 0.01f * std::sin(static_cast<float>(i + 23 * lane) * 0.009f);
        }
        auto run_chunk = [&](const ChunkInput& input) {
            return model.model().run_chunk(
                input.mel.data(), kChunkMelFrames,
                input.spkcache.empty() ? nullptr : input.spkcache.data(), input.spkcache_frames,
                input.fifo.empty() ? nullptr : input.fifo.data(), input.fifo_frames);
        };
        std::vector<SortformerModel::ChunkOutput> chunk_references;
        for (const auto& input : chunk_inputs) chunk_references.push_back(run_chunk(input));
        const auto chunk_metrics_before = model.batch_metrics();
        std::atomic<int> chunk_ready{0};
        std::atomic<bool> chunk_go{false};
        std::vector<std::future<SortformerModel::ChunkOutput>> chunk_calls;
        for (size_t lane = 0; lane < chunk_inputs.size(); ++lane) {
            chunk_calls.push_back(std::async(std::launch::async, [&, lane] {
                const ScopedBatchCohort cohort(4);
                chunk_ready.fetch_add(1);
                while (!chunk_go.load()) std::this_thread::yield();
                return run_chunk(chunk_inputs[lane]);
            }));
        }
        while (chunk_ready.load() != 4) std::this_thread::yield();
        chunk_go.store(true);
        double heterogeneous_max_abs = 0.0;
        double heterogeneous_max_rmse = 0.0;
        for (size_t lane = 0; lane < chunk_calls.size(); ++lane) {
            const auto result = chunk_calls[lane].get();
            const auto& reference = chunk_references[lane];
            if (result.total_frames != reference.total_frames ||
                result.chunk_frames != reference.chunk_frames ||
                result.preds.size() != reference.preds.size() ||
                result.chunk_embs.size() != reference.chunk_embs.size()) {
                std::fprintf(stderr, "heterogeneous state batch output shape changed\n");
                return 1;
            }
            double max_abs = 0.0;
            double square_error = 0.0;
            size_t count = 0;
            auto compare = [&](const std::vector<float>& got, const std::vector<float>& expected) {
                for (size_t i = 0; i < got.size(); ++i) {
                    const double delta = static_cast<double>(got[i]) - expected[i];
                    max_abs = std::max(max_abs, std::abs(delta));
                    square_error += delta * delta;
                    ++count;
                }
            };
            compare(result.preds, reference.preds);
            compare(result.chunk_embs, reference.chunk_embs);
            const double rmse = std::sqrt(square_error / std::max<size_t>(1, count));
            heterogeneous_max_abs = std::max(heterogeneous_max_abs, max_abs);
            heterogeneous_max_rmse = std::max(heterogeneous_max_rmse, rmse);
            // Allow bounded B=1/B>1 Q8 CUDA reduction drift; speaker segments
            // are still checked below for functional parity.
            // Single-chunk thresholds should be tighter than accumulated streaming thresholds
            if (max_abs > 1e-1 || rmse > 5e-3) {
                std::fprintf(
                    stderr, "heterogeneous state batch lane %zu parity delta max=%.3e rmse=%.3e\n",
                    lane, max_abs, rmse);
                return 1;
            }
        }
        const auto chunk_metrics_after = model.batch_metrics();
        if (chunk_metrics_after.target_reached_batches <=
                chunk_metrics_before.target_reached_batches ||
            chunk_metrics_after.max_observed_batch < 4) {
            std::fprintf(stderr, "heterogeneous state requests did not coalesce at B=4\n");
            return 1;
        }
        std::fprintf(
            stdout, "heterogeneous state B=4 parity PASS (max=%.3e rmse=%.3e)\n",
            heterogeneous_max_abs, heterogeneous_max_rmse);

        struct BatchedResult {
            std::vector<float> probabilities;
            std::vector<DiarSegment> segments;
        };
        auto run_stream = [&](const std::vector<float>& input) {
            DiarStream stream(model, geo);
            const size_t push = static_cast<size_t>(push_ms) * 16;
            for (size_t off = 0; off < input.size(); off += push) {
                stream.feed_audio(input.data() + off, std::min(push, input.size() - off));
            }
            stream.finish();
            return BatchedResult{stream.frame_probs(), stream.segments(seg_cfg)};
        };
        std::vector<std::vector<float>> inputs(4, audio);
        for (float& sample : inputs[1]) sample *= 0.5f;
        std::reverse(inputs[2].begin(), inputs[2].end());
        std::fill(inputs[3].begin(), inputs[3].end(), 0.0f);
        std::vector<BatchedResult> references;
        references.reserve(inputs.size());
        references.push_back({probs, segs});
        for (size_t i = 1; i < inputs.size(); ++i) references.push_back(run_stream(inputs[i]));
        const auto metrics_before = model.batch_metrics();

        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::vector<std::future<BatchedResult>> calls;
        for (int i = 0; i < 4; ++i) {
            calls.push_back(std::async(std::launch::async, [&, i] {
                const ScopedBatchCohort cohort(4);
                ready.fetch_add(1);
                while (!go.load()) std::this_thread::yield();
                return run_stream(inputs[static_cast<size_t>(i)]);
            }));
        }
        while (ready.load() != 4) std::this_thread::yield();
        const auto start = std::chrono::steady_clock::now();
        go.store(true);
        for (size_t result_index = 0; result_index < calls.size(); ++result_index) {
            auto& call = calls[result_index];
            const auto result = call.get();
            const auto& got = result.probabilities;
            const auto& reference = references[result_index];
            if (got.size() != reference.probabilities.size()) {
                std::fprintf(stderr, "batched diarization output length mismatch\n");
                return 1;
            }
            double max_abs = 0.0;
            double square_error = 0.0;
            for (size_t i = 0; i < got.size(); ++i) {
                const double delta = static_cast<double>(got[i]) - reference.probabilities[i];
                max_abs = std::max(max_abs, std::abs(delta));
                square_error += delta * delta;
            }
            const double rmse = std::sqrt(square_error / std::max<size_t>(1, got.size()));
            // Batched GEMMs can select a different deterministic reduction
            // geometry than scalar GEMMs. Through the stateful AOSC loop this
            // can amplify a few frame probabilities even for F32 weights, so
            // bound both the isolated and aggregate drift and verify the
            // resulting speaker segments exactly below.
            if (max_abs > 1.25e-1 || rmse > 7.5e-3) {
                std::fprintf(
                    stderr, "batched diarization parity delta max=%.3e rmse=%.3e\n", max_abs, rmse);
                return 1;
            }
            if (result.segments.size() != reference.segments.size()) {
                std::fprintf(stderr, "batched diarization segment count changed\n");
                return 1;
            }
            for (size_t i = 0; i < reference.segments.size(); ++i) {
                if (result.segments[i].speaker != reference.segments[i].speaker ||
                    std::abs(result.segments[i].t0 - reference.segments[i].t0) >
                        sec_per_frame + 1e-9 ||
                    std::abs(result.segments[i].t1 - reference.segments[i].t1) >
                        sec_per_frame + 1e-9) {
                    std::fprintf(
                        stderr,
                        "batched diarization segment %zu changed: expected spk=%d %.3f-%.3f, "
                        "got spk=%d %.3f-%.3f\n",
                        i, reference.segments[i].speaker, reference.segments[i].t0,
                        reference.segments[i].t1, result.segments[i].speaker, result.segments[i].t0,
                        result.segments[i].t1);
                    return 1;
                }
            }
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        const auto metrics_after = model.batch_metrics();
        if (metrics_after.deadline_batches != metrics_before.deadline_batches ||
            metrics_after.target_reached_batches <= metrics_before.target_reached_batches) {
            std::fprintf(stderr, "diarization cohort was not released by its batch target\n");
            return 1;
        }
        if (metrics_after.max_observed_batch < 4) {
            std::fprintf(stderr, "diarization requests did not coalesce\n");
            return 1;
        }
        std::fprintf(
            stdout, "B=4 diarization parity PASS (%lld ms)\n",
            static_cast<long long>(elapsed.count()));
    }
    if (!probs_path.empty()) {
        std::FILE* f = std::fopen(probs_path.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", probs_path.c_str());
            return 1;
        }
        std::fwrite(probs.data(), sizeof(float), probs.size(), f);
        std::fclose(f);
    }

    if (!rttm_name.empty()) {
        for (const auto& s : segs) {
            std::printf(
                "SPEAKER %s 1 %.3f %.3f <NA> <NA> speaker_%d <NA> <NA>\n", rttm_name.c_str(), s.t0,
                s.t1 - s.t0, s.speaker);
        }
        return 0;
    }

    std::printf(
        "%.1fs audio -> %ld diar frames (%.0f ms/frame)%s\n", audio.size() / 16000.0,
        (long)n_frames, sec_per_frame * 1000.0, offline ? " [offline]" : "");
    for (const auto& s : segs) {
        std::printf("  [%7.2fs - %7.2fs] speaker_%d\n", s.t0, s.t1, s.speaker);
    }
    return 0;
}
