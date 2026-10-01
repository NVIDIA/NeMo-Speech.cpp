// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <fstream>
#include <mutex>
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
        } else if (arg == "--chunk-ms") {
            chunk_ms_ = parse_int(value(), arg, 10, 60000);
        } else if (arg == "--trace") {
            trace_path_ = value();
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
        if (!trace_path_.empty()) {
            if (!stream_)
                throw std::invalid_argument("--trace requires --mode stream");
            trace_.open(trace_path_, std::ios::trunc);
            if (!trace_)
                throw std::runtime_error("cannot write " + trace_path_);
        }
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
    ItemResult run(size_t index) override {
        std::vector<double> chunk_latency_ms;
        auto transcript = recognize(inputs_[index], chunk_latency_ms);
        if (!stream_)
            return {std::move(transcript), {}};
        return {std::move(transcript), {}, {{"chunk_latency_ms", std::move(chunk_latency_ms)}}};
    }

    void describe(Value& output) const override {
        output["model"] = recognizer_->model_name();
        output["mode"] = stream_ ? "stream" : "offline";
        if (stream_)
            output["chunk_ms"] = chunk_ms();
        output["files"] = static_cast<double>(inputs_.size());
        output["corpus_audio_seconds"] = corpus_seconds_;
    }
    std::vector<std::string> header_lines() const override {
        return {
            "Model: " + recognizer_->model_name(),
            std::string("Mode: ") +
                (stream_ ? "stream, " + std::to_string(chunk_ms()) + " ms chunks" : "offline")};
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
        std::vector<Column> columns = {
            {"RTFx", [](const Value& run) { return run.number_or("rtfx"); }, 2}};
        if (stream_) {
            // compute time per streamed chunk (push + drain), client-side
            columns.push_back(
                {"CHUNK avg (ms)",
                 [](const Value& run) { return stat(run, "metrics", "chunk_latency_ms", "mean"); },
                 2});
            columns.push_back(
                {"CHUNK p99 (ms)",
                 [](const Value& run) { return stat(run, "metrics", "chunk_latency_ms", "p99"); },
                 2});
        }
        return columns;
    }

   private:
    // Streamed chunk length: the model's cache-aware chunk ((right context + 1) x 80 ms) unless
    // --chunk-ms overrides it.
    int chunk_ms() const {
        if (chunk_ms_ > 0)
            return chunk_ms_;
        const int rc = config_.streaming.rnnt_right_context;
        return rc >= 0 ? (rc + 1) * 80 : 160;
    }
    std::string recognize(const AudioInput& input, std::vector<double>& chunk_latency_ms) {
        asr::AsrRequestOptions request;
        request.language_code = language_;
        if (!stream_) {
            return first_transcript(recognizer_->recognize(
                input.audio.samples.data(), input.audio.samples.size(), request, language_,
                input.audio.sample_rate));
        }
        auto stream = recognizer_->streaming_recognize(request, language_);
        const size_t chunk =
            std::max<size_t>(1, static_cast<size_t>(input.audio.sample_rate) * chunk_ms() / 1000);
        const auto stream_started = std::chrono::steady_clock::now();
        std::string transcript;
        for (size_t offset = 0; offset < input.audio.samples.size(); offset += chunk) {
            const size_t count = std::min(chunk, input.audio.samples.size() - offset);
            const auto started = std::chrono::steady_clock::now();
            stream->push(input.audio.samples.data() + offset, count, input.audio.sample_rate);
            std::string interim;
            while (auto result = stream->next()) {
                if (!result->is_final) {
                    interim = first_transcript(*result);
                    break;
                }
                append_text(transcript, first_transcript(*result));
            }
            const auto now = std::chrono::steady_clock::now();
            chunk_latency_ms.push_back(
                std::chrono::duration<double, std::milli>(now - started).count());
            if (trace_.is_open()) {
                std::string text = transcript;
                append_text(text, interim);
                write_trace(
                    input, static_cast<double>(offset + count) / input.audio.sample_rate,
                    std::chrono::duration<double, std::milli>(now - stream_started).count(), text);
            }
        }
        append_text(transcript, first_transcript(stream->finish()));
        if (trace_.is_open())
            write_trace(
                input, static_cast<double>(input.audio.samples.size()) / input.audio.sample_rate,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - stream_started)
                    .count(),
                transcript);
        return transcript;
    }

    // One JSON line per streamed chunk: audio position, elapsed time since the stream started, and
    // the transcript so far (finals plus the current interim). Used to render speed demos.
    void write_trace(
        const AudioInput& input, double audio_s, double elapsed_ms, const std::string& text) {
        Value line;
        line["input"] = input.path.filename().string();
        line["audio_s"] = audio_s;
        line["elapsed_ms"] = elapsed_ms;
        line["text"] = text;
        std::lock_guard<std::mutex> lock(trace_mutex_);
        trace_ << line.dump() << '\n';
    }

    asr::RecognizerConfig config_;
    std::string model_;
    std::string language_;
    bool stream_ = false;
    int chunk_ms_ = 0;
    std::string trace_path_;
    std::ofstream trace_;
    std::mutex trace_mutex_;
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
