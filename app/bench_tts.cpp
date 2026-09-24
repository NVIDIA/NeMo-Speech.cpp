// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench.h"
#include "cli_util.h"
#include "commands.h"
#include "config.h"
#include "engine_registry.h"

namespace nemo_speech::bench {
namespace {
namespace fs = std::filesystem;

constexpr int kDefaultSeed = 1;

struct TextInput {
    std::string name;
    std::string text;
};

std::string
trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
        return {};
    return value.substr(begin, value.find_last_not_of(" \t\r\n") - begin + 1);
}

size_t
utf8_length(const std::string& text) {
    size_t count = 0;
    for (const unsigned char c : text) count += (c & 0xC0) != 0x80;
    return count;
}

Column
metric_column(
    const std::string& header, const std::string& metric, const char* stat_name, int precision) {
    return {
        header, [=](const Value& row) { return stat(row, "metrics", metric, stat_name); },
        precision};
}

class TtsWorkload : public Workload {
   public:
    std::string task() const override { return "tts"; }
    std::vector<int> default_concurrency() const override { return {1}; }
    void register_parameters(common::ParameterParser& parser) override {
        parser.Register("tts", parsed_);
    }
    bool parse_option(const std::string& arg, const std::function<std::string()>& value) override {
        if (arg == "--text") {
            texts_.push_back(value());
        } else if (arg == "--text-file") {
            text_files_.push_back(value());
        } else if (arg == "--magpie-model" || arg == "--model" || arg == "-m") {
            parsed_.runtime.magpie_model = value();
        } else if (arg == "--codec-model") {
            parsed_.runtime.codec_model = value();
        } else if (arg == "--tokenizer-dir") {
            parsed_.tokenizer_model_dir = value();
        } else if (arg == "--tn-model-dir") {
            parsed_.tn_model_dir = value();
        } else if (arg == "--language") {
            language_ = value();
        } else if (arg == "--voice") {
            voice_ = value();
        } else if (arg == "--speaker") {
            options_.speaker = parse_int(value(), arg, 0, 100000);
        } else if (arg == "--sample-rate") {
            sample_rate_ = parse_int(value(), arg, 8000, 192000);
        } else if (arg == "--seed") {
            options_.seed = parse_int(value(), arg, -1, 2147483647);
            seed_set_ = true;
        } else if (arg == "--steps") {
            options_.steps = parse_int(value(), arg, 1, 1000000);
        } else if (arg == "--top-k") {
            options_.top_k = parse_int(value(), arg, 1, 1000000);
        } else if (arg == "--temperature") {
            options_.temperature = static_cast<float>(parse_double(value(), arg));
            options_.override_temperature = true;
        } else if (arg == "--cfg-scale") {
            options_.cfg_scale = static_cast<float>(parse_double(value(), arg));
            options_.override_cfg_scale = true;
        } else {
            return false;
        }
        return true;
    }
    void prepare(const CommonOptions& options) override {
        for (size_t i = 0; i < texts_.size(); ++i) {
            if (trim(texts_[i]).empty())
                throw std::invalid_argument("--text must not be empty");
            inputs_.push_back({"text" + std::to_string(i + 1), trim(texts_[i])});
        }
        // one utterance per line; `audio|text` filelist lines use the text
        for (const auto& file : text_files_) {
            std::istringstream lines(read_text_file(file));
            std::string line;
            int number = 0;
            while (std::getline(lines, line)) {
                ++number;
                std::string name = fs::path(file).stem().string() + ":" + std::to_string(number);
                const auto bar = line.rfind('|');
                if (bar != std::string::npos) {
                    const auto audio = fs::path(trim(line.substr(0, line.find('|'))));
                    if (!audio.empty())
                        name = audio.stem().string();
                    line = line.substr(bar + 1);
                }
                line = trim(line);
                if (!line.empty())
                    inputs_.push_back({name, line});
            }
        }
        for (const auto& positional : options.positional)
            for (const auto& path : collect_files(
                     positional, options.recursive,
                     [](const fs::path& file) { return file.extension() == ".txt"; }, ".txt")) {
                auto text = trim(read_text_file(path));
                if (text.empty())
                    throw std::invalid_argument(path.string() + " is empty");
                inputs_.push_back({path.stem().string(), std::move(text)});
            }
        if (inputs_.empty())
            throw std::invalid_argument(
                "bench tts requires --text TEXT, --text-file FILE, or a .txt file or directory");
        if (!seed_set_ && parsed_.runtime.seed < 0)
            options_.seed = kDefaultSeed;
        seed_ = options_.seed >= 0 ? options_.seed : parsed_.runtime.seed;
    }
    size_t input_count() const override { return inputs_.size(); }
    std::string input_name(size_t index) const override { return inputs_[index].name; }
    void load(const CommonOptions& options, int) override {
        auto config = make_synthesizer_config(parsed_, options.device, options.device_set);
        magpie_model_ = config.runtime.magpie_model;
        codec_model_ = config.runtime.codec_model;
        tokenizer_dir_ = config.tokenizer_model_dir;
        synthesizer_ = engines_.load_tts(std::move(config));
    }
    void warmup_engine() override { engines_.warmup(); }
    ItemResult run(size_t index) override {
        tts::SynthesisRequest request;
        request.text = inputs_[index].text;
        request.language_code = language_;
        request.voice_name = voice_;
        request.output_sample_rate = sample_rate_;
        request.options = options_;
        // client-side: submission to first audio chunk, and gaps between chunks
        using Clock = std::chrono::steady_clock;
        const auto submitted = Clock::now();
        Clock::time_point last{};
        double first_audio_ms = 0.0;
        std::vector<double> chunk_gaps_ms;
        const auto result =
            synthesizer_->synthesize(request, [&](const auto&, const std::string& chunk) {
                if (chunk.empty())
                    return true;
                const auto now = Clock::now();
                if (last == Clock::time_point{})
                    first_audio_ms =
                        std::chrono::duration<double, std::milli>(now - submitted).count();
                else
                    chunk_gaps_ms.push_back(
                        std::chrono::duration<double, std::milli>(now - last).count());
                last = now;
                return true;
            });
        if (result.output_samples == 0)
            throw std::runtime_error("synthesizer returned no audio for " + inputs_[index].name);
        const auto& stats = result.stats;
        return {
            std::to_string(result.output_samples),
            {{"first_audio_ms", first_audio_ms},
             {"audio_seconds", stats.audio_s},
             {"e2e_ttfa_ms", stats.e2e_ttfa_ms},
             {"e2e_rtfx", stats.e2e_rtfx},
             {"decoder_rtfx", stats.decoder_rtfx},
             {"decoder_itl_avg_ms", stats.decoder_itl_avg_ms},
             {"decoder_itl_p95_ms", stats.decoder_itl_p95_ms},
             {"codec_rtfx", stats.codec_rtfx},
             {"encoder_ms", stats.encoder_ms},
             {"decoder_frames", static_cast<double>(stats.generated_frames)}},
            {{"chunk_gap_ms", std::move(chunk_gaps_ms)}}};
    }

    void describe(Value& output) const override {
        output["model"] = synthesizer_->model_name();
        Value models(Value::Object{});
        models["magpie"] = magpie_model_;
        models["codec"] = codec_model_;
        models["tokenizer"] = tokenizer_dir_;
        output["models"] = std::move(models);
        output["language"] = language();
        output["voice"] = voice();
        output["sample_rate"] = sample_rate_ > 0 ? sample_rate_ : synthesizer_->sample_rate();
        output["seed"] = seed_;
        if (options_.steps > 0)
            output["steps"] = options_.steps;
        if (options_.top_k > 0)
            output["top_k"] = options_.top_k;
        if (options_.override_temperature)
            output["temperature"] = static_cast<double>(options_.temperature);
        if (options_.override_cfg_scale)
            output["cfg_scale"] = static_cast<double>(options_.cfg_scale);
        output["utterances"] = static_cast<double>(inputs_.size());
        size_t chars = 0;
        for (const auto& input : inputs_) chars += utf8_length(input.text);
        output["corpus_chars"] = static_cast<double>(chars);
    }
    std::vector<std::string> header_lines() const override {
        return {
            "Model: " + synthesizer_->model_name() + " (" +
                fs::path(magpie_model_).filename().string() + ", codec " +
                fs::path(codec_model_).filename().string() + ")",
            "Language: " + language() + "  Voice: " + voice() + "  Seed: " + std::to_string(seed_)};
    }
    std::string mismatch_key() const override { return "audio_length_mismatches"; }
    void summarize_run(
        Value& run, const std::vector<ItemResult>& items, double wall_seconds) const override {
        double audio_seconds = 0.0;
        for (const auto& item : items)
            for (const auto& [name, value] : item.metrics)
                if (name == "audio_seconds")
                    audio_seconds += value;
        run["audio_seconds"] = audio_seconds;
        run["rtfx"] = audio_seconds / wall_seconds;
    }
    std::vector<Column> run_columns() const override {
        // Riva/NIM TTS streaming metrics
        return {
            {"RTFx", [](const Value& run) { return run.number_or("rtfx"); }, 2},
            metric_column("TTFA avg", "first_audio_ms", "mean", 1),
            metric_column("TTFA p99", "first_audio_ms", "p99", 1),
            metric_column("ICL avg", "chunk_gap_ms", "mean", 2),
            metric_column("ICL p99", "chunk_gap_ms", "p99", 2)};
    }
    std::vector<Column> input_columns() const override {
        return {
            {"CHARS", [](const Value& row) { return row.number_or("chars"); }, 0},
            metric_column("AUDIO (s)", "audio_seconds", "mean", 2),
            metric_column("TTFA avg", "first_audio_ms", "mean", 1),
            metric_column("TTFA p99", "first_audio_ms", "p99", 1),
            metric_column("RTFx", "e2e_rtfx", "mean", 2),
            metric_column("DEC RTFx", "decoder_rtfx", "mean", 2),
            metric_column("ITL avg", "decoder_itl_avg_ms", "mean", 2),
            metric_column("ITL p95", "decoder_itl_p95_ms", "mean", 2),
            metric_column("CODEC RTFx", "codec_rtfx", "mean", 1),
            metric_column("ENC (ms)", "encoder_ms", "mean", 1),
            {"WALL (ms)", [](const Value& row) { return stat(row, "latency_ms", "", "mean"); }, 1}};
    }
    void describe_input(size_t index, Value& entry) const override {
        entry["chars"] = static_cast<double>(utf8_length(inputs_[index].text));
        entry["text"] = inputs_[index].text;
    }

   private:
    std::string language() const {
        return language_.empty() ? synthesizer_->default_language_code() : language_;
    }
    std::string voice() const {
        if (!voice_.empty())
            return voice_;
        const int speaker =
            options_.speaker >= 0 ? options_.speaker : synthesizer_->default_speaker();
        const auto& names = synthesizer_->speaker_names();
        return speaker < static_cast<int>(names.size()) ? names[speaker] : std::to_string(speaker);
    }

    tts::MagpieTtsServerConfig parsed_;
    tts::MagpieSynthesisOptions options_;
    std::vector<std::string> texts_;
    std::vector<std::string> text_files_;
    std::string language_;
    std::string voice_;
    int sample_rate_ = 0;
    bool seed_set_ = false;
    int seed_ = kDefaultSeed;
    std::vector<TextInput> inputs_;
    std::string magpie_model_;
    std::string codec_model_;
    std::string tokenizer_dir_;
    EngineRegistry engines_;
    std::shared_ptr<tts::Synthesizer> synthesizer_;
};

}  // namespace

std::unique_ptr<Workload>
make_tts_workload() {
    return std::make_unique<TtsWorkload>();
}

}  // namespace nemo_speech::bench
