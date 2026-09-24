// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench.h"
#include "cli_util.h"
#include "engine_registry.h"
#include "model_utils.h"
#include "translator.h"

namespace nemo_speech::bench {
namespace {

class TranslateWorkload : public Workload {
   public:
    TranslateWorkload() { config_.backend.gpu = default_gpu_index(); }

    std::string task() const override { return "translate"; }
    std::vector<int> default_concurrency() const override { return {1}; }
    void register_parameters(common::ParameterParser& parser) override {
        parser.Register("nmt", config_);
    }
    bool parse_option(const std::string& arg, const std::function<std::string()>& value) override {
        if (arg == "--model" || arg == "-m")
            model_ = value();
        else if (arg == "--from")
            source_ = value();
        else if (arg == "--to")
            target_ = value();
        else if (arg == "--text")
            texts_.push_back(value());
        else
            return false;
        return true;
    }
    void prepare(const CommonOptions& options) override {
        if (source_.empty() || target_.empty())
            throw std::invalid_argument("--from and --to are required");
        for (const auto& positional : options.positional) {
            std::istringstream lines(read_text_file(positional));
            for (std::string line; std::getline(lines, line);)
                if (!line.empty() && line.find_first_not_of(" \t\r") != std::string::npos)
                    texts_.push_back(std::move(line));
        }
        if (texts_.empty())
            throw std::invalid_argument("bench translate requires --text TEXT or a text file");
        for (const auto& text : texts_) corpus_bytes_ += text.size();
        if (options.device_set)
            config_.backend.gpu = options.gpu;
    }
    size_t input_count() const override { return texts_.size(); }
    std::string input_name(size_t index) const override {
        return "line" + std::to_string(index + 1);
    }
    void load(const CommonOptions&, int max_concurrency) override {
        config_.pool.contexts = std::max(config_.pool.contexts, max_concurrency);
        config_.verbose = cli_verbose();
        config_.model.path =
            require_model_file(model_.empty() ? config_.model.path : model_, "translation model")
                .string();
        translator_ = engines_.load_nmt(config_);
    }
    ItemResult run(size_t index) override {
        const auto result = translator_->translate({texts_[index]}, source_, target_);
        std::string text = result.empty() ? std::string() : result.front().text;
        const double bytes = static_cast<double>(text.size());
        return {std::move(text), {{"output_bytes", bytes}}};
    }

    void describe(Value& output) const override {
        output["model"] = translator_->model_name();
        output["source_language"] = source_;
        output["target_language"] = target_;
        output["contexts"] = config_.pool.contexts;
        output["lines"] = static_cast<double>(texts_.size());
        output["corpus_input_bytes"] = static_cast<double>(corpus_bytes_);
    }
    std::vector<std::string> header_lines() const override {
        return {"Model: " + translator_->model_name(), "Pair: " + source_ + "-" + target_};
    }
    std::string mismatch_key() const override { return "translation_mismatches"; }
    void summarize_run(
        Value& run, const std::vector<ItemResult>& items, double wall_seconds) const override {
        const double bytes = static_cast<double>(corpus_bytes_) * items.size() / texts_.size();
        run["input_bytes_per_second"] = bytes / wall_seconds;
    }
    std::vector<Column> run_columns() const override {
        return {
            {"IN B/s", [](const Value& run) { return run.number_or("input_bytes_per_second"); },
             1}};
    }

   private:
    nmt::TranslatorConfig config_;
    std::string model_;
    std::string source_;
    std::string target_;
    std::vector<std::string> texts_;
    size_t corpus_bytes_ = 0;
    EngineRegistry engines_;
    std::shared_ptr<nmt::Translator> translator_;
};

}  // namespace

std::unique_ptr<Workload>
make_translate_workload() {
    return std::make_unique<TranslateWorkload>();
}

}  // namespace nemo_speech::bench
