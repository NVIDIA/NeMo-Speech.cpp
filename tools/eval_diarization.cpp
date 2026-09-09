// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Batched standalone evaluation: retain native probabilities for rescoring and
// publish each completed file immediately. Input TSV: id, wav, offset, duration.
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "diar_pipeline.h"

using namespace nemo_speech::asr;
namespace fs = std::filesystem;

int
main(int argc, char** argv) try {
    if (argc < 10) {
        std::cerr << "usage: eval_diarization MODEL INPUT.tsv OUTDIR CHUNK RC CACHE FIFO UPDATE "
                     "CONCURRENCY [--cpu] [--lc N]\n";
        return 2;
    }
    auto integer = [](const std::string& text) {
        int value = 0;
        const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
        if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value < 0)
            throw std::invalid_argument("invalid non-negative integer: " + text);
        return value;
    };
    DiarGeometry geometry{integer(argv[6]), integer(argv[7]), integer(argv[4]), integer(argv[8]), 0,
                          integer(argv[5])};
    const int requested_concurrency = integer(argv[9]);
    if (requested_concurrency < 1)
        throw std::runtime_error("concurrency must be positive");
    ggml_runtime::Params params;
    params.use_gpu = true;
    for (int i = 10; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--cpu")
            params.use_gpu = false;
        else if (option == "--lc" && i + 1 < argc)
            geometry.chunk_left_context = integer(argv[++i]);
        else
            throw std::invalid_argument("unknown or incomplete option: " + option);
    }
    struct Input {
        std::string id, wav;
        double offset, duration;
    };
    std::vector<Input> inputs;
    std::set<std::string> ids;
    std::ifstream manifest(argv[2]);
    if (!manifest)
        throw std::runtime_error("cannot open manifest");
    for (std::string line; std::getline(manifest, line);) {
        if (line.empty())
            continue;
        std::istringstream row(line);
        Input input;
        std::getline(row, input.id, '\t');
        std::getline(row, input.wav, '\t');
        if (!(row >> input.offset >> input.duration) || !std::isfinite(input.offset) ||
            !std::isfinite(input.duration) || input.offset < 0 || input.duration <= 0 ||
            input.wav.empty() || input.id.empty() || input.id == "." || input.id == ".." ||
            input.id.find_first_of("/\\") != std::string::npos || !ids.insert(input.id).second)
            throw std::runtime_error("invalid/duplicate manifest entry");
        row >> std::ws;
        if (!row.eof())
            throw std::runtime_error("extra fields in manifest entry");
        inputs.push_back(input);
    }
    if (inputs.empty())
        throw std::runtime_error("empty manifest");
    const fs::path output(argv[3]);
    if (fs::exists(output))
        throw std::runtime_error("output already exists; use a new directory");
    const int concurrency =
        static_cast<int>(std::min<size_t>(requested_concurrency, inputs.size()));
    ggml_runtime::BackendManager backend(params);
    BatchingConfig batching;
    batching.enabled = true;
    batching.max_batch_size = concurrency;
    DiarModel model(backend, argv[1], batching);
    geometry.validate(
        model.cfg().num_speakers, model.cfg().scoring.sil_frames_per_spk,
        model.cfg().encoder.pos_emb_max_len);
    if (!fs::create_directories(output))
        throw std::runtime_error("cannot create fresh output directory");
    {
        std::ofstream metadata(output / "metadata.tsv");
        metadata << std::setprecision(17) << "format\tframe-major-f32\n"
                 << "version\t" << (model.cfg().is_v3() ? 3 : 2) << '\n'
                 << "speakers\t" << model.cfg().num_speakers << '\n'
                 << "seconds_per_frame\t" << model.cfg().seconds_per_output_frame() << '\n'
                 << "chunk\t" << geometry.chunk_len << '\n'
                 << "lc\t" << geometry.chunk_left_context << '\n'
                 << "rc\t" << geometry.chunk_right_context << '\n'
                 << "cache\t" << geometry.spkcache_len << '\n'
                 << "fifo\t" << geometry.fifo_len << '\n'
                 << "update\t" << geometry.spkcache_update_period << '\n'
                 << "concurrency\t" << concurrency << '\n';
        metadata.close();
        if (!metadata)
            throw std::runtime_error("metadata write failed");
    }
    std::atomic<size_t> next{0};
    size_t completed = 0, failures = 0;
    std::mutex progress;
    std::vector<std::thread> workers;
    for (int lane = 0; lane < concurrency; ++lane) {
        workers.emplace_back([&] {
            const ScopedBatchCohort cohort(concurrency);
            for (;;) {
                const size_t index = next.fetch_add(1);
                if (index >= inputs.size())
                    break;
                const auto& item = inputs[index];
                const auto started = std::chrono::steady_clock::now();
                std::string error;
                int64_t frames = 0;
                try {
                    std::vector<float> audio;
                    int sr = 0;
                    if (!read_wav_mono_16k(item.wav, audio, sr) || sr != 16000)
                        throw std::runtime_error("cannot read 16k mono WAV");
                    const double begin_sample = std::round(item.offset * sr);
                    const double region_samples = std::round(item.duration * sr);
                    if (!std::isfinite(begin_sample) || !std::isfinite(region_samples) ||
                        begin_sample >= audio.size() || region_samples < 1 ||
                        region_samples > audio.size() - begin_sample)
                        throw std::runtime_error(
                            "audio region is empty or exceeds file; refusing truncation");
                    const size_t begin = static_cast<size_t>(begin_sample);
                    const size_t end = begin + static_cast<size_t>(region_samples);
                    DiarStream stream(model, geometry);
                    stream.set_compaction(
                        std::numeric_limits<int64_t>::max(),
                        std::numeric_limits<int64_t>::max() / 2);
                    // Native streaming frontend/state, in 1s ingestion blocks.
                    for (size_t at = begin; at < end; at += sr)
                        stream.feed_audio(audio.data() + at, std::min<size_t>(sr, end - at));
                    stream.finish();
                    const auto& probs = stream.frame_probs();
                    frames = stream.n_frames();
                    if (stream.frame_probs_base() != 0 || probs.empty())
                        throw std::runtime_error("incomplete probability timeline");
                    for (float p : probs)
                        if (!std::isfinite(p) || p < 0 || p > 1)
                            throw std::runtime_error("invalid probability");
                    const auto path = output / (item.id + ".f32");
                    const auto temporary = path.string() + ".tmp";
                    {
                        std::ofstream file(temporary, std::ios::binary);
                        file.write(
                            reinterpret_cast<const char*>(probs.data()),
                            probs.size() * sizeof(float));
                        file.close();
                        if (!file)
                            throw std::runtime_error("probability write failed");
                    }
                    fs::rename(temporary, path);
                }
                catch (const std::exception& e) {
                    error = e.what();
                }
                const double seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                        .count();
                const std::lock_guard<std::mutex> lock(progress);
                ++completed;
                if (!error.empty())
                    ++failures;
                std::cout << '[' << completed << '/' << inputs.size() << "] " << item.id
                          << (error.empty() ? " DONE" : " FAILED: " + error) << " frames=" << frames
                          << " seconds=" << seconds << std::endl;
            }
        });
    }
    for (auto& worker : workers) worker.join();
    const auto metrics = model.batch_metrics();
    std::cout << "FINISHED files=" << completed << " failed=" << failures
              << " batches=" << metrics.batches << " items=" << metrics.items
              << " max_batch=" << metrics.max_observed_batch << std::endl;
    return failures ? 1 : 0;
}
catch (const std::exception& error) {
    std::cerr << "eval_diarization: " << error.what() << '\n';
    return 2;
}
