// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio_file.h"
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

std::string
first_transcript(const asr::Result& result) {
    if (result.alternatives.empty())
        return {};
    return result.alternatives.front().transcript;
}

void
append_text(std::string& transcript, const std::string& text) {
    if (text.empty())
        return;
    if (!transcript.empty())
        transcript += ' ';
    transcript += text;
}

class AsrWorkload : public Workload {
   public:
    AsrWorkload() { config_.backend.gpu = default_gpu_index(); }

    std::string task() const override { return "asr"; }
    void register_parameters(common::ParameterParser& parser) override {
        parser.Register("asr", config_);
    }
    bool parse_option(const std::string& arg, const std::function<std::string()>& value) override {
        if (arg == "--model" || arg == "-m") {
            model_ = value();
        } else if (arg == "--language" || arg == "-l") {
            language_ = value();
        } else if (arg == "--mode") {
            const auto mode = value();
            if (mode != "offline" && mode != "stream")
                throw std::invalid_argument("--mode must be offline or stream");
            stream_ = mode == "stream";
        } else {
            return false;
        }
        return true;
    }
    void prepare(const CommonOptions& options) override {
        if (options.positional.size() != 1)
            throw std::invalid_argument(
                options.positional.empty() ? "bench asr requires a WAV file or directory"
                                           : "unexpected argument: " + options.positional[1]);
        for (const auto& path : collect_files(
                 options.positional.front(), options.recursive,
                 [](const fs::path& file) { return audio::is_wav_path(file.string()); }, "WAV")) {
            auto audio = audio::load_wav_file(path.string());
            corpus_seconds_ += static_cast<double>(audio.samples.size()) / audio.sample_rate;
            inputs_.push_back({path, std::move(audio)});
        }
        if (options.device_set)
            config_.backend.gpu = options.gpu;
    }
    size_t input_count() const override { return inputs_.size(); }
    std::string input_name(size_t index) const override {
        return inputs_[index].path.filename().string();
    }
    void load(const CommonOptions&, int max_concurrency) override {
        config_.model.path =
            resolve_model_file(model_.empty() ? config_.model.path : model_, "asr", "ASR model")
                .string();
        config_.batching.enabled = max_concurrency > 1;
        config_.batching.max_batch_size =
            std::max(config_.batching.max_batch_size, max_concurrency);
        config_.batching.max_queue_depth =
            std::max(config_.batching.max_queue_depth, max_concurrency * 2);
        config_.batching.state_arena_slots =
            std::max(config_.batching.state_arena_slots, max_concurrency);
        config_.log_status = !cli_quiet() && !cli_json();
        recognizer_ = engines_.load_asr(config_);
    }
    void warmup_engine() override { engines_.warmup(); }
    ItemResult run(size_t index) override { return {recognize(inputs_[index]), {}}; }

    void describe(Value& output) const override {
        output["model"] = recognizer_->model_name();
        output["mode"] = stream_ ? "stream" : "offline";
        output["files"] = static_cast<double>(inputs_.size());
        output["corpus_audio_seconds"] = corpus_seconds_;
    }
    std::vector<std::string> header_lines() const override {
        return {
            "Model: " + recognizer_->model_name(),
            std::string("Mode: ") + (stream_ ? "stream" : "offline")};
    }
    std::string mismatch_key() const override { return "transcript_mismatches"; }
    void summarize_run(
        Value& run, const std::vector<ItemResult>& items, double wall_seconds) const override {
        double audio_seconds = 0.0;
        for (const auto& item : items)
            audio_seconds += static_cast<double>(inputs_[item.input].audio.samples.size()) /
                             inputs_[item.input].audio.sample_rate;
        run["utterances"] = static_cast<double>(items.size());
        run["audio_seconds"] = audio_seconds;
        run["rtfx"] = audio_seconds / wall_seconds;
        run["utterances_per_second"] = items.size() / wall_seconds;
    }
    std::vector<Column> run_columns() const override {
        return {{"RTFx", [](const Value& run) { return run.number_or("rtfx"); }, 2}};
    }

   private:
    std::string recognize(const AudioInput& input) {
        asr::AsrRequestOptions request;
        request.language_code = language_;
        if (!stream_) {
            return first_transcript(recognizer_->recognize(
                input.audio.samples.data(), input.audio.samples.size(), request, language_,
                input.audio.sample_rate));
        }
        auto stream = recognizer_->streaming_recognize(request, language_);
        const size_t chunk = std::max<size_t>(1, input.audio.sample_rate * 160 / 1000);
        std::string transcript;
        for (size_t offset = 0; offset < input.audio.samples.size(); offset += chunk) {
            const size_t count = std::min(chunk, input.audio.samples.size() - offset);
            stream->push(input.audio.samples.data() + offset, count, input.audio.sample_rate);
            while (auto result = stream->next()) {
                if (!result->is_final)
                    break;
                append_text(transcript, first_transcript(*result));
            }
        }
        append_text(transcript, first_transcript(stream->finish()));
        return transcript;
    }

    asr::RecognizerConfig config_;
    std::string model_;
    std::string language_;
    bool stream_ = false;
    std::vector<AudioInput> inputs_;
    double corpus_seconds_ = 0.0;
    EngineRegistry engines_;
    std::shared_ptr<asr::Recognizer> recognizer_;
};

}  // namespace

std::unique_ptr<Workload>
make_asr_workload() {
    return std::make_unique<AsrWorkload>();
}

}  // namespace nemo_speech::bench
