// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Shared `nemo-speech bench` harness. Each task implements a small Workload
// adapter; the harness owns option parsing, warmup, the worker pool, timing,
// percentile summaries, and the common JSON/table output.
#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "json.h"
#include "parameter_parser.h"

namespace nemo_speech::bench {

using json::Value;

struct CommonOptions {
    std::vector<int> concurrency;
    int repetitions = 3;
    int per_stream = 0;  // > 0: requests per concurrent stream (cycling inputs)
    int warmup = 1;
    int gpu = 0;
    std::string device = "auto";
    bool device_set = false;
    bool recursive = false;
    std::vector<std::string> positional;
};

// Per-item output of Workload::run. `signature` is compared against the first
// result for the same input to count mismatches across repetitions.
struct ItemResult {
    std::string signature;
    std::vector<std::pair<std::string, double>> metrics;
    // Per-item value lists (e.g. every chunk gap), pooled across items.
    std::vector<std::pair<std::string, std::vector<double>>> samples = {};
    size_t input = 0;  // index of the processed input (set by the harness)
};

// A table column; `value` reads a run object (summary table) or a per-input
// object (input table).
struct Column {
    std::string header;
    std::function<double(const Value&)> value;
    int precision = 2;
};

class Workload {
   public:
    virtual ~Workload() = default;

    virtual std::string task() const = 0;
    virtual std::vector<int> default_concurrency() const { return {1, 2, 4}; }
    virtual void register_parameters(common::ParameterParser& parser) = 0;
    // Consume a task-specific option; `value` reads its argument.
    virtual bool parse_option(const std::string& arg, const std::function<std::string()>& value) {
        (void)arg;
        (void)value;
        return false;
    }
    // Validate options and load inputs (not timed).
    virtual void prepare(const CommonOptions& options) = 0;
    virtual size_t input_count() const = 0;
    virtual std::string input_name(size_t index) const = 0;
    // Build the engine sized for `max_concurrency` (timed as load_ms).
    virtual void load(const CommonOptions& options, int max_concurrency) = 0;
    // Engine-internal warmup before the timed-input warmup iterations.
    virtual void warmup_engine() {}
    // Process one input; called concurrently from worker threads.
    virtual ItemResult run(size_t index) = 0;

    // Top-level report keys (model(s), mode, corpus facts).
    virtual void describe(Value& output) const = 0;
    virtual std::vector<std::string> header_lines() const = 0;
    virtual std::string mismatch_key() const = 0;
    // Task keys for one concurrency level (e.g. rtfx over the corpus).
    virtual void summarize_run(
        Value& run, const std::vector<ItemResult>& items, double wall_seconds) const {
        (void)run;
        (void)items;
        (void)wall_seconds;
    }
    virtual std::vector<Column> run_columns() const { return {}; }
    // Non-empty input columns enable the per-input breakdown.
    virtual std::vector<Column> input_columns() const { return {}; }
    virtual void describe_input(size_t index, Value& entry) const {
        (void)index;
        (void)entry;
    }
};

// Reads `object[group][name][stat]`, or 0 when absent.
double stat(
    const Value& object, const std::string& group, const std::string& name,
    const std::string& stat_name);
std::vector<std::filesystem::path> collect_files(
    const std::filesystem::path& input, bool recursive,
    const std::function<bool(const std::filesystem::path&)>& accept, const std::string& kind);

#if defined(NEMO_SPEECH_CLI_ASR)
std::unique_ptr<Workload> make_asr_workload();
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
std::unique_ptr<Workload> make_tts_workload();
#endif
#if defined(NEMO_SPEECH_CLI_DIAR)
std::unique_ptr<Workload> make_diarize_workload();
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
std::unique_ptr<Workload> make_translate_workload();
#endif

}  // namespace nemo_speech::bench
