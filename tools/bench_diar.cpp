// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Diarization neural-step benchmark with explicit geometry and batch size.

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "aosc_state.h"
#include "batching.h"
#include "diar_pipeline.h"
#include "sortformer_model.h"

using namespace nemo_speech::asr;

namespace {

struct Options {
    std::string model;
    bool gpu = false;
    int batch_size = 1;
    int warmups = 3;
    int reps = 5;
    int spkcache = -1;
    int fifo = -1;
    int chunk = -1;
    int left_context = -1;
    int right_context = -1;
};

[[noreturn]] void
usage(const char* program) {
    std::fprintf(
        stderr,
        "usage: %s MODEL [--gpu] [--batch-size N] [--warmups N] [--reps N]\n"
        "       [--spkcache N] [--fifo N] [--chunk N] [--lc N] [--rc N]\n",
        program);
    std::exit(2);
}

Options
parse_options(int argc, char** argv) {
    if (argc < 2)
        usage(argv[0]);
    Options options;
    options.model = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value = [&] {
            if (++i >= argc)
                usage(argv[0]);
            const std::string text = argv[i];
            int parsed = 0;
            const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
            if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || parsed < 0)
                throw std::invalid_argument("invalid value for " + argument + ": " + text);
            return parsed;
        };
        if (argument == "--gpu")
            options.gpu = true;
        else if (argument == "--batch-size")
            options.batch_size = value();
        else if (argument == "--warmups")
            options.warmups = value();
        else if (argument == "--reps")
            options.reps = value();
        else if (argument == "--spkcache")
            options.spkcache = value();
        else if (argument == "--fifo")
            options.fifo = value();
        else if (argument == "--chunk")
            options.chunk = value();
        else if (argument == "--lc")
            options.left_context = value();
        else if (argument == "--rc")
            options.right_context = value();
        else
            usage(argv[0]);
    }
    if (options.batch_size < 1 || options.warmups < 0 || options.reps < 1)
        usage(argv[0]);
    return options;
}

double
median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

}  // namespace

int
main(int argc, char** argv) try {
    const Options options = parse_options(argc, argv);

    ggml_runtime::Params backend_options;
    backend_options.use_gpu = options.gpu;
    ggml_runtime::BackendManager backend(backend_options);
    BatchingConfig batching;
    batching.enabled = options.batch_size > 1;
    batching.max_batch_size = options.batch_size;
    batching.max_queue_delay_us = 5000000;
    batching.max_queue_depth = std::max(2048, options.batch_size);
    SortformerModel model(backend, options.model, batching);

    DiarGeometry geometry =
        model.cfg().is_v3() ? DiarGeometry::v3_streaming() : DiarGeometry::riva_streaming();
    if (options.spkcache >= 0)
        geometry.spkcache_len = options.spkcache;
    if (options.fifo >= 0)
        geometry.fifo_len = options.fifo;
    if (options.chunk >= 0)
        geometry.chunk_len = options.chunk;
    if (options.left_context >= 0)
        geometry.chunk_left_context = options.left_context;
    if (options.right_context >= 0)
        geometry.chunk_right_context = options.right_context;
    geometry.validate(
        model.cfg().num_speakers, model.cfg().scoring.sil_frames_per_spk,
        model.cfg().encoder.pos_emb_max_len);

    const int state_frames = geometry.spkcache_len + geometry.fifo_len;
    const int window_frames =
        geometry.chunk_left_context + geometry.chunk_len + geometry.chunk_right_context;
    const int mel_frames = window_frames * model.cfg().encoder.subsampling_factor;
    const int dimension = model.cfg().encoder.d_model;
    std::vector<float> mel(static_cast<size_t>(model.cfg().n_mels) * mel_frames);
    std::vector<float> spkcache(static_cast<size_t>(dimension) * geometry.spkcache_len);
    std::vector<float> fifo(static_cast<size_t>(dimension) * geometry.fifo_len);
    for (size_t i = 0; i < mel.size(); ++i)
        mel[i] = 0.02f * std::sin(static_cast<float>(i) * 0.013f);
    for (size_t i = 0; i < spkcache.size(); ++i)
        spkcache[i] = 0.01f * std::cos(static_cast<float>(i) * 0.007f);
    for (size_t i = 0; i < fifo.size(); ++i)
        fifo[i] = 0.01f * std::sin(static_cast<float>(i) * 0.009f);

    volatile double checksum = 0.0;
    auto run_batch = [&] {
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::vector<std::future<SortformerModel::ChunkOutput>> calls;
        calls.reserve(options.batch_size);
        for (int lane = 0; lane < options.batch_size; ++lane) {
            calls.push_back(std::async(std::launch::async, [&] {
                const ScopedBatchCohort cohort(options.batch_size);
                ready.fetch_add(1, std::memory_order_release);
                while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
                return model.run_chunk(
                    mel.data(), mel_frames, spkcache.empty() ? nullptr : spkcache.data(),
                    geometry.spkcache_len, fifo.empty() ? nullptr : fifo.data(), geometry.fifo_len);
            }));
        }
        while (ready.load(std::memory_order_acquire) != options.batch_size)
            std::this_thread::yield();
        const auto start = std::chrono::steady_clock::now();
        go.store(true, std::memory_order_release);
        for (auto& call : calls) {
            const auto output = call.get();
            if (!output.preds.empty())
                checksum += output.preds.front();
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count();
    };

    for (int i = 0; i < options.warmups; ++i) run_batch();
    std::vector<double> elapsed;
    elapsed.reserve(options.reps);
    const BatchMetrics before = model.batch_metrics();
    for (int i = 0; i < options.reps; ++i) elapsed.push_back(run_batch());
    const BatchMetrics after = model.batch_metrics();
    const double median_ms = median(elapsed);
    const double audio_seconds = options.batch_size * geometry.chunk_len *
                                 model.cfg().encoder.subsampling_factor * model.cfg().window_stride;
    const double rtfx = audio_seconds / (median_ms / 1000.0);
    std::printf(
        "DIAR_BENCH version=%s batch=%d state=%d window=%d chunk=%d "
        "median_ms=%.3f per_item_ms=%.3f rtfx=%.3f observed_batch=%llu checksum=%.6f\n",
        model.cfg().is_v3() ? "v3" : "v2", options.batch_size, state_frames, window_frames,
        geometry.chunk_len, median_ms, median_ms / options.batch_size, rtfx,
        static_cast<unsigned long long>(after.max_observed_batch), static_cast<double>(checksum));
    if (options.batch_size > 1 &&
        (after.max_observed_batch < static_cast<uint64_t>(options.batch_size) ||
         after.target_reached_batches <= before.target_reached_batches)) {
        std::fprintf(stderr, "requested batch did not coalesce\n");
        return 1;
    }
    return 0;
}
catch (const std::exception& error) {
    std::fprintf(stderr, "bench_diar: %s\n", error.what());
    return 2;
}
