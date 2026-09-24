// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "bench.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cli_util.h"
#include "commands.h"

namespace nemo_speech::bench {
namespace fs = std::filesystem;

double
stat(
    const Value& object, const std::string& group, const std::string& name,
    const std::string& stat_name) {
    const Value* node = object.find(group);
    if (node && !name.empty())
        node = node->find(name);
    if (node)
        node = node->find(stat_name);
    return node && node->is_number() ? node->number() : 0.0;
}

std::vector<fs::path>
collect_files(
    const fs::path& input, bool recursive, const std::function<bool(const fs::path&)>& accept,
    const std::string& kind) {
    std::error_code error;
    if (fs::is_regular_file(input, error))
        return {fs::absolute(input)};
    if (!fs::is_directory(input, error))
        throw std::invalid_argument(input.string() + " is not a file or directory");
    std::vector<fs::path> result;
    auto add = [&](const auto& entry) {
        if (entry.is_regular_file(error) && accept(entry.path()))
            result.push_back(fs::absolute(entry.path()));
    };
    if (recursive)
        for (const auto& entry : fs::recursive_directory_iterator(input)) add(entry);
    else
        for (const auto& entry : fs::directory_iterator(input)) add(entry);
    std::sort(result.begin(), result.end());
    if (result.empty())
        throw std::invalid_argument(input.string() + " contains no " + kind + " files");
    return result;
}

namespace {
using Clock = std::chrono::steady_clock;

const char* const kTasks[][2] = {
    {"asr", "ASR"}, {"tts", "TTS"}, {"diarize", "DIAR"}, {"translate", "NMT"}};

std::unique_ptr<Workload>
make_workload(const std::string& task) {
#if defined(NEMO_SPEECH_CLI_ASR)
    if (task == "asr")
        return make_asr_workload();
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
    if (task == "tts")
        return make_tts_workload();
#endif
#if defined(NEMO_SPEECH_CLI_DIAR)
    if (task == "diarize")
        return make_diarize_workload();
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
    if (task == "translate")
        return make_translate_workload();
#endif
    for (const auto& known : kTasks)
        if (task == known[0])
            throw UnsupportedFeatureError(
                "this build does not include the '" + task +
                "' benchmark workload; rebuild with -DNEMO_SPEECH_BUILD_" + known[1] + "=ON");
    throw std::invalid_argument(
        "unknown bench workload '" + task + "' (expected asr, tts, diarize, or translate)");
}

std::vector<int>
parse_concurrency(const std::string& value) {
    std::vector<int> result;
    size_t begin = 0;
    while (begin <= value.size()) {
        const size_t comma = value.find(',', begin);
        result.push_back(parse_int(
            value.substr(begin, comma == std::string::npos ? comma : comma - begin),
            "--concurrency", 1, 1024));
        if (comma == std::string::npos)
            break;
        begin = comma + 1;
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

double
percentile(std::vector<double> values, double fraction) {
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    return values[lower] + (values[upper] - values[lower]) * (position - lower);
}

Value
distribution(const std::vector<double>& values) {
    Value result(Value::Object{});
    double sum = 0.0;
    for (const double value : values) sum += value;
    result["mean"] = values.empty() ? 0.0 : sum / values.size();
    result["p50"] = percentile(values, 0.50);
    result["p90"] = percentile(values, 0.90);
    result["p95"] = percentile(values, 0.95);
    result["p99"] = percentile(values, 0.99);
    result["min"] = values.empty() ? 0.0 : *std::min_element(values.begin(), values.end());
    result["max"] = values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
    return result;
}

// Per-metric distributions in first-seen metric order; JSON objects sort keys.
Value
metric_distributions(const std::vector<const ItemResult*>& items) {
    std::vector<std::string> order;
    std::map<std::string, std::vector<double>> values;
    for (const auto* item : items) {
        for (const auto& [name, value] : item->metrics) {
            auto [it, inserted] = values.try_emplace(name);
            if (inserted)
                order.push_back(name);
            it->second.push_back(value);
        }
        for (const auto& [name, samples] : item->samples) {
            auto [it, inserted] = values.try_emplace(name);
            if (inserted)
                order.push_back(name);
            it->second.insert(it->second.end(), samples.begin(), samples.end());
        }
    }
    Value result(Value::Object{});
    for (const auto& name : order) result[name] = distribution(values[name]);
    return result;
}

void
print_table(
    const std::vector<Column>& columns, const Value::Array& rows,
    const std::vector<std::string>* names = nullptr) {
    size_t name_width = 6;
    if (names)
        for (const auto& name : *names) name_width = std::max(name_width, name.size() + 2);
    std::vector<int> widths;
    for (const auto& column : columns)
        widths.push_back(static_cast<int>(std::max<size_t>(10, column.header.size() + 2)));
    if (names)
        std::printf("%-*s", static_cast<int>(name_width), "INPUT");
    for (size_t c = 0; c < columns.size(); ++c)
        std::printf("%-*s", widths[c], columns[c].header.c_str());
    std::printf("\n");
    for (size_t r = 0; r < rows.size(); ++r) {
        if (names)
            std::printf("%-*s", static_cast<int>(name_width), (*names)[r].c_str());
        for (size_t c = 0; c < columns.size(); ++c)
            std::printf("%-*.*f", widths[c], columns[c].precision, columns[c].value(rows[r]));
        std::printf("\n");
    }
}

Column
key_column(const std::string& header, const std::string& key, int precision) {
    return {header, [key](const Value& row) { return row.number_or(key); }, precision};
}

Column
stat_column(
    const std::string& header, const std::string& group, const std::string& name,
    const std::string& stat_name, int precision) {
    return {header, [=](const Value& row) { return stat(row, group, name, stat_name); }, precision};
}

int
run_bench(int argc, char** argv) {
    if (argc == 0 || is_help_argument(argv[0])) {
        print_bench_help("nemo-speech");
        return 0;
    }
    auto workload = make_workload(argv[0]);
    CommonOptions options;
    options.gpu = default_gpu_index();
    options.concurrency = workload->default_concurrency();
    bool json = cli_json();

    common::ParameterParser parser;
    workload->register_parameters(parser);
    std::string config_file;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--config") {
            if (++i >= argc)
                throw std::invalid_argument("--config requires a value");
            config_file = argv[i];
        }
    if (!config_file.empty())
        parser.ApplyYaml(config_file);
    parser.ApplyEnv("NEMO_SPEECH");
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() {
            if (++i >= argc)
                throw std::invalid_argument(arg + " requires a value");
            return std::string(argv[i]);
        };
        if (is_help_argument(arg)) {
            print_bench_help("nemo-speech");
            return 0;
        } else if (arg == "--config") {
            ++i;
        } else if (arg == "--concurrency" || arg == "-c") {
            options.concurrency = parse_concurrency(value());
        } else if (arg == "--repetitions" || arg == "-n") {
            options.repetitions = parse_int(value(), arg, 1, 10000);
        } else if (arg == "--per-stream") {
            options.per_stream = parse_int(value(), arg, 1, 100000);
        } else if (arg == "--warmup") {
            options.warmup = parse_int(value(), arg, 0, 1000);
        } else if (arg == "--device" || arg == "--backend") {
            options.device = value();
            options.gpu = parse_device(options.device, arg);
            options.device_set = true;
        } else if (arg == "--gpu") {
            options.gpu = parse_int(value(), arg, -1, 1024);
            options.device = options.gpu < 0 ? "cpu" : "gpu:" + std::to_string(options.gpu);
            options.device_set = true;
        } else if (arg == "--recursive" || arg == "-r") {
            options.recursive = true;
        } else if (arg == "--json") {
            json = true;
        } else if (workload->parse_option(arg, value)) {
            continue;
        } else if (!arg.empty() && arg.front() == '-') {
            bool consumed = false;
            if (!parser.ParseCliArg(arg, i + 1 < argc ? argv[i + 1] : nullptr, &consumed))
                throw std::invalid_argument("unknown option: " + arg);
            if (consumed)
                ++i;
        } else {
            options.positional.push_back(arg);
        }
    }
    workload->prepare(options);
    const size_t inputs = workload->input_count();
    if (inputs == 0)
        throw std::invalid_argument("bench " + workload->task() + " has no inputs");
    const int max_concurrency =
        *std::max_element(options.concurrency.begin(), options.concurrency.end());

    const auto load_start = Clock::now();
    workload->load(options, max_concurrency);
    const double load_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - load_start).count();
    const auto warmup_start = Clock::now();
    workload->warmup_engine();
    for (int i = 0; i < options.warmup; ++i) (void)workload->run(static_cast<size_t>(i) % inputs);
    const double warmup_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - warmup_start).count();

    Value output(Value::Object{});
    output["command"] = "bench " + workload->task();
    output["task"] = workload->task();
    workload->describe(output);
    output["load_ms"] = load_ms;
    output["warmup_ms"] = warmup_ms;
    output["warmup"] = options.warmup;
    output["repetitions"] = options.repetitions;
    if (options.per_stream > 0)
        output["per_stream"] = options.per_stream;
    const auto input_columns = workload->input_columns();
    std::vector<std::string> names;
    for (size_t i = 0; i < inputs; ++i) names.push_back(workload->input_name(i));

    Value::Array runs;
    std::vector<std::string> reference(inputs);
    std::vector<bool> reference_set(inputs, false);
    for (const int concurrency : options.concurrency) {
        const size_t work_count =
            options.per_stream > 0
                ? static_cast<size_t>(options.per_stream) * static_cast<size_t>(concurrency)
                : inputs * static_cast<size_t>(options.repetitions);
        std::vector<ItemResult> results(work_count);
        std::vector<double> latency(work_count);
        std::atomic<size_t> next{0};
        std::mutex failure_mutex;
        std::exception_ptr failure;
        const auto started = Clock::now();
        std::vector<std::thread> workers;
        workers.reserve(concurrency);
        for (int thread = 0; thread < concurrency; ++thread) {
            workers.emplace_back([&] {
                try {
                    for (;;) {
                        const size_t work = next.fetch_add(1);
                        if (work >= work_count)
                            return;
                        const auto item_start = Clock::now();
                        results[work] = workload->run(work % inputs);
                        results[work].input = work % inputs;
                        latency[work] =
                            std::chrono::duration<double, std::milli>(Clock::now() - item_start)
                                .count();
                    }
                }
                catch (...) {
                    std::lock_guard<std::mutex> lock(failure_mutex);
                    if (!failure)
                        failure = std::current_exception();
                    next = work_count;
                }
            });
        }
        for (auto& worker : workers) worker.join();
        if (failure)
            std::rethrow_exception(failure);
        const double wall_seconds = std::chrono::duration<double>(Clock::now() - started).count();

        std::vector<int> mismatches(inputs, 0);
        int total_mismatches = 0;
        for (size_t work = 0; work < work_count; ++work) {
            const size_t index = work % inputs;
            if (!reference_set[index]) {
                reference[index] = results[work].signature;
                reference_set[index] = true;
            } else if (reference[index] != results[work].signature) {
                ++mismatches[index];
                ++total_mismatches;
            }
        }
        std::vector<const ItemResult*> all_items;
        for (const auto& item : results) all_items.push_back(&item);

        Value run(Value::Object{});
        run["concurrency"] = concurrency;
        run["items"] = static_cast<double>(work_count);
        run["wall_seconds"] = wall_seconds;
        run["items_per_second"] = work_count / wall_seconds;
        run["latency_ms"] = distribution(latency);
        Value metrics = metric_distributions(all_items);
        if (!metrics.object().empty())
            run["metrics"] = std::move(metrics);
        run[workload->mismatch_key()] = total_mismatches;
        workload->summarize_run(run, results, wall_seconds);
        if (!input_columns.empty()) {
            Value::Array per_input;
            for (size_t index = 0; index < inputs; ++index) {
                std::vector<const ItemResult*> items;
                std::vector<double> item_latency;
                for (size_t work = index; work < work_count; work += inputs) {
                    items.push_back(&results[work]);
                    item_latency.push_back(latency[work]);
                }
                Value entry(Value::Object{});
                entry["name"] = names[index];
                workload->describe_input(index, entry);
                entry["repetitions"] = static_cast<double>(items.size());
                entry["latency_ms"] = distribution(item_latency);
                entry["metrics"] = metric_distributions(items);
                entry[workload->mismatch_key()] = mismatches[index];
                per_input.emplace_back(std::move(entry));
            }
            run["inputs"] = std::move(per_input);
        }
        runs.emplace_back(std::move(run));
    }
    output["runs"] = std::move(runs);

    if (json) {
        std::printf("%s\n", output.dump(2).c_str());
        return 0;
    }
    for (const auto& line : workload->header_lines()) std::printf("%s\n", line.c_str());
    std::printf("Load: %.1f ms  Warmup: %.1f ms\n", load_ms, warmup_ms);
    std::vector<Column> columns{
        key_column("CONCURRENCY", "concurrency", 0),
        key_column("ITEMS", "items", 0),
        key_column("WALL (s)", "wall_seconds", 3),
        key_column("ITEMS/s", "items_per_second", 2),
        stat_column("P50 (ms)", "latency_ms", "", "p50", 1),
        stat_column("P95 (ms)", "latency_ms", "", "p95", 1),
    };
    for (auto& column : workload->run_columns()) columns.push_back(std::move(column));
    columns.push_back(key_column("MISMATCH", workload->mismatch_key(), 0));
    print_table(columns, output.at("runs").array());
    if (!input_columns.empty())
        for (const auto& run : output.at("runs").array()) {
            std::printf(
                "\nPer input, concurrency %d (%d requests):\n",
                static_cast<int>(run.at("concurrency").number()),
                static_cast<int>(run.at("items").number()));
            auto per_input = input_columns;
            per_input.push_back(key_column("MISMATCH", workload->mismatch_key(), 0));
            print_table(per_input, run.at("inputs").array(), &names);
        }
    return 0;
}

}  // namespace
}  // namespace nemo_speech::bench

void
print_bench_help(const char* program) {
    std::printf(
        "Usage: %s bench <task> [INPUT] [options]\n\n"
        "Benchmark an end-to-end workload with one shared engine and concurrent requests.\n"
        "Every task reports load/warmup time and, per concurrency level, wall time,\n"
        "items/s, per-item latency (mean/p50/p95), task metrics, and output mismatches\n"
        "against the first result seen for each input.\n\n"
        "Tasks:\n"
#if defined(NEMO_SPEECH_CLI_ASR)
        "  asr INPUT --model MODEL        WAV file or directory\n"
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
        "  tts [INPUT] [--text TEXT]...   .txt file or directory (one utterance per file)\n"
#endif
#if defined(NEMO_SPEECH_CLI_DIAR)
        "  diarize INPUT [--model MODEL]  WAV file or directory\n"
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
        "  translate INPUT --model MODEL --from SRC --to DST\n"
        "                                 Text file, one input per line\n"
#endif
        "\nCommon options:\n"
        "  -c, --concurrency LIST  Comma-separated levels (default: asr/diarize 1,2,4;\n"
        "                          tts/translate 1)\n"
        "  -n, --repetitions N     Corpus repetitions per level (default: 3)\n"
        "  --per-stream N          Instead: each concurrent stream sends N requests,\n"
        "                          cycling through the inputs\n"
        "  --warmup N              Timed-input warmup iterations (default: 1)\n"
        "  --device, --backend DEVICE\n"
        "                          auto, cpu, cuda[:N], metal, or vulkan[:N]\n"
        "  -r, --recursive         Recurse into input directories\n"
        "  --json                  Emit machine-readable results\n"
        "  --config FILE           Apply YAML configuration\n"
#if defined(NEMO_SPEECH_CLI_ASR)
        "\nasr options:\n"
        "  -m, --model MODEL       Local ASR GGUF path\n"
        "  --mode offline|stream   Recognition mode (default: offline)\n"
        "  -l, --language CODE     Prompt language code\n"
        "  --asr.* VALUE           Override any ASR engine setting\n"
#endif
#if defined(NEMO_SPEECH_CLI_TTS)
        "\ntts options:\n"
        "  --text TEXT             Utterance to synthesize (repeatable)\n"
        "  --text-file FILE        One utterance per line ('audio|text' filelists use the\n"
        "                          text; e.g. "
        "test_files/tts/ljs_audio_text_test_filelist_small.txt)\n"
        "  -m, --magpie-model MODEL\n"
        "                          MagpieTTS GGUF path or indexed HF repo\n"
        "  --codec-model MODEL     NanoCodec GGUF path or indexed HF repo\n"
        "  --tokenizer-dir MODEL   Tokenizer directory or indexed HF repo\n"
        "  --tn-model-dir DIR      Optional text-normalization grammars\n"
        "  --language CODE         Text language (default: en-US)\n"
        "  --voice NAME, --speaker N\n"
        "  --sample-rate HZ        Output rate (8 kHz through model rate)\n"
        "  --seed N                Sampling seed (default: 1; -1 = time-based)\n"
        "  --steps N --top-k N --temperature N --cfg-scale N\n"
        "  --tts.KEY VALUE         Override any C++ TTS setting\n"
        "  The MagpieTTS runtime serializes requests: concurrency > 1 measures\n"
        "  queueing, and client latency includes queue wait.\n"
#endif
#if defined(NEMO_SPEECH_CLI_DIAR)
        "\ndiarize options:\n"
        "  -m, --model MODEL       Sortformer GGUF path or indexed HF repo\n"
        "  --offline               Full-attention mode for short audio\n"
        "  --preset NAME           streaming or offline geometry\n"
        "  --no-batching           Disable dynamic batching\n"
        "  --diar.* --batching.* VALUE\n"
        "                          Override any diarization setting\n"
#endif
#if defined(NEMO_SPEECH_CLI_NMT)
        "\ntranslate options:\n"
        "  -m, --model MODEL       Local translation GGUF path\n"
        "  --from CODE --to CODE   Language pair (required)\n"
        "  --text TEXT             Input text (repeatable)\n"
        "  --nmt.* VALUE           Override any NMT engine setting\n"
#endif
        ,
        program);
}

int
command_bench(int argc, char** argv) {
    try {
        return nemo_speech::bench::run_bench(argc, argv);
    }
    catch (const std::exception& error) {
        return print_cli_exception("bench", error);
    }
}
