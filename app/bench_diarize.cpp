// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio_file.h"
#include "batching.h"
#include "bench.h"
#include "cli_util.h"
#include "engine_registry.h"
#include "model_utils.h"

namespace nemo_speech::bench {
namespace {
namespace fs = std::filesystem;

struct AudioInput {
    fs::path path;
    audio::AudioFile audio;
};

class DiarizeWorkload : public Workload {
   public:
    std::string task() const override { return "diarize"; }
    void register_parameters(common::ParameterParser& parser) override {
        parser.Register("diar", config_);
        parser.Register("batching", batching_);
    }
    bool parse_option(const std::string& arg, const std::function<std::string()>& value) override {
        if (arg == "--model" || arg == "-m")
            config_.model_path = value();
        else if (arg == "--offline")
            offline_ = true;
        else if (arg == "--preset")
            config_.preset = value();
        else if (arg == "--no-batching")
            enable_batching_ = false;
        else
            return false;
        return true;
    }
    void prepare(const CommonOptions& options) override {
        if (options.positional.size() != 1)
            throw std::invalid_argument(
                options.positional.empty() ? "bench diarize requires a WAV file or directory"
                                           : "unexpected argument: " + options.positional[1]);
        for (const auto& path : collect_files(
                 options.positional.front(), options.recursive,
                 [](const fs::path& file) { return audio::is_wav_path(file.string()); }, "WAV")) {
            auto audio = audio::load_wav_file(path.string());
            corpus_seconds_ += static_cast<double>(audio.samples.size()) / audio.sample_rate;
            inputs_.push_back({path, std::move(audio)});
        }
    }
    size_t input_count() const override { return inputs_.size(); }
    std::string input_name(size_t index) const override {
        return inputs_[index].path.filename().string();
    }
    void load(const CommonOptions& options, int max_concurrency) override {
        batching_.enabled = enable_batching_ && max_concurrency > 1;
        batching_.max_batch_size = std::min(batching_.max_batch_size, max_concurrency);
        batching_.max_queue_depth = std::max(batching_.max_queue_depth, max_concurrency * 4);
        batching_.state_arena_slots = std::max(batching_.state_arena_slots, max_concurrency);
        config_.model_path =
            resolve_model_file(config_.model_path, "diarization", "diarization model").string();
        engine_ = engines_.load_diarization(
            options.gpu, config_.model_path, config_.resolved_geometry(), batching_);
    }
    ItemResult run(size_t index) override {
        const auto& audio = inputs_[index].audio;
        const auto result = engine_->diarize(
            audio.samples.data(), audio.samples.size(), audio.sample_rate,
            offline_ ? asr::DiarizationMode::Offline : asr::DiarizationMode::Streaming);
        std::string signature;
        for (const auto& segment : result.segments) {
            char line[96];
            std::snprintf(
                line, sizeof(line), "%.3f %.3f %d;", segment.t0, segment.t1, segment.speaker);
            signature += line;
        }
        return {std::move(signature), {{"segments", static_cast<double>(result.segments.size())}}};
    }

    void describe(Value& output) const override {
        output["model"] = fs::path(config_.model_path).stem().string();
        output["mode"] = mode();
        output["batching"] = batching_.enabled;
        output["files"] = static_cast<double>(inputs_.size());
        output["corpus_audio_seconds"] = corpus_seconds_;
    }
    std::vector<std::string> header_lines() const override {
        return {"Model: " + fs::path(config_.model_path).filename().string(), "Mode: " + mode()};
    }
    std::string mismatch_key() const override { return "segment_mismatches"; }
    void summarize_run(
        Value& run, const std::vector<ItemResult>& items, double wall_seconds) const override {
        double audio_seconds = 0.0;
        for (const auto& item : items)
            audio_seconds += static_cast<double>(inputs_[item.input].audio.samples.size()) /
                             inputs_[item.input].audio.sample_rate;
        run["audio_seconds"] = audio_seconds;
        run["rtfx"] = audio_seconds / wall_seconds;
    }
    std::vector<Column> run_columns() const override {
        return {{"RTFx", [](const Value& run) { return run.number_or("rtfx"); }, 2}};
    }

   private:
    std::string mode() const { return offline_ ? "offline" : "streaming"; }

    asr::DiarConfig config_;
    asr::BatchingConfig batching_;
    bool offline_ = false;
    bool enable_batching_ = true;
    std::vector<AudioInput> inputs_;
    double corpus_seconds_ = 0.0;
    EngineRegistry engines_;
    std::shared_ptr<asr::Diarizer> engine_;
};

}  // namespace

std::unique_ptr<Workload>
make_diarize_workload() {
    return std::make_unique<DiarizeWorkload>();
}

}  // namespace nemo_speech::bench
