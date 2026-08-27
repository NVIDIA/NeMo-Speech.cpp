// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "audio_file.h"
#include "cli_util.h"
#include "commands.h"
#include "engine_registry.h"
#include "json.h"
#include "model_utils.h"
#include "parameter_parser.h"

namespace {
namespace fs = std::filesystem;
namespace asr = nemo_speech::asr;
using Clock = std::chrono::steady_clock;
using nemo_speech::json::Value;

struct AudioInput {
    fs::path path;
    nemo_speech::audio::AudioFile audio;
};

struct Options {
    fs::path input;
    std::string model;
    std::string language;
    std::string config_file;
    asr::RecognizerConfig config;
    std::vector<int> concurrency{1, 2, 4};
    int repetitions = 3;
    int warmup = 1;
    bool recursive = false;
    bool stream = false;
    bool json = cli_json();
};

std::string
required_value(int& index, int argc, char** argv, const std::string& option) {
    if (++index >= argc)
        throw std::invalid_argument(option + " requires a value");
    return argv[index];
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

Options
parse_options(int argc, char** argv) {
    Options options;
    options.config.backend.gpu = default_gpu_index();
    nemo_speech::common::ParameterParser parser;
    parser.Register("asr", options.config);
    for (int i = 0; i < argc; ++i)
        if (std::string(argv[i]) == "--config")
            options.config_file = required_value(i, argc, argv, "--config");
    if (!options.config_file.empty())
        parser.ApplyYaml(options.config_file);
    parser.ApplyEnv("NEMO_SPEECH");
    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "asr") {
            continue;
        } else if (arg == "--config") {
            ++i;
        } else if (arg == "--model" || arg == "-m") {
            options.model = required_value(i, argc, argv, arg);
        } else if (arg == "--language" || arg == "-l") {
            options.language = required_value(i, argc, argv, arg);
        } else if (arg == "--device" || arg == "--backend") {
            options.config.backend.gpu = parse_device(required_value(i, argc, argv, arg), arg);
        } else if (arg == "--gpu") {
            options.config.backend.gpu =
                parse_int(required_value(i, argc, argv, arg), arg, -1, 1024);
        } else if (arg == "--concurrency" || arg == "-c") {
            options.concurrency = parse_concurrency(required_value(i, argc, argv, arg));
        } else if (arg == "--repetitions" || arg == "-n") {
            options.repetitions = parse_int(required_value(i, argc, argv, arg), arg, 1, 10000);
        } else if (arg == "--warmup") {
            options.warmup = parse_int(required_value(i, argc, argv, arg), arg, 0, 1000);
        } else if (arg == "--mode") {
            const auto value = required_value(i, argc, argv, arg);
            if (value == "offline")
                options.stream = false;
            else if (value == "stream")
                options.stream = true;
            else
                throw std::invalid_argument("--mode must be offline or stream");
        } else if (arg == "--recursive" || arg == "-r") {
            options.recursive = true;
        } else if (arg == "--json") {
            options.json = true;
        } else if (!arg.empty() && arg.front() == '-') {
            bool consumed = false;
            const char* next = i + 1 < argc ? argv[i + 1] : nullptr;
            if (!parser.ParseCliArg(arg, next, &consumed))
                throw std::invalid_argument("unknown option: " + arg);
            if (consumed)
                ++i;
        } else if (options.input.empty()) {
            options.input = arg;
        } else {
            throw std::invalid_argument("unexpected argument: " + arg);
        }
    }
    if (options.input.empty())
        throw std::invalid_argument("bench asr requires a WAV file or directory");
    return options;
}

std::vector<fs::path>
collect_files(const Options& options) {
    std::error_code error;
    if (fs::is_regular_file(options.input, error))
        return {fs::absolute(options.input)};
    if (!fs::is_directory(options.input, error))
        throw std::invalid_argument(options.input.string() + " is not a file or directory");
    std::vector<fs::path> result;
    auto add = [&](const auto& entry) {
        if (entry.is_regular_file(error) && nemo_speech::audio::is_wav_path(entry.path().string()))
            result.push_back(fs::absolute(entry.path()));
    };
    if (options.recursive)
        for (const auto& entry : fs::recursive_directory_iterator(options.input)) add(entry);
    else
        for (const auto& entry : fs::directory_iterator(options.input)) add(entry);
    std::sort(result.begin(), result.end());
    if (result.empty())
        throw std::invalid_argument(options.input.string() + " contains no WAV files");
    return result;
}

std::string
append_result(const asr::Result& result) {
    if (result.alternatives.empty())
        return {};
    return result.alternatives.front().transcript;
}

std::string
recognize(asr::Recognizer& recognizer, const AudioInput& input, const Options& options) {
    asr::AsrRequestOptions request;
    request.language_code = options.language;
    if (!options.stream) {
        return append_result(recognizer.recognize(
            input.audio.samples.data(), input.audio.samples.size(), request, options.language,
            input.audio.sample_rate));
    }
    auto stream = recognizer.streaming_recognize(request, options.language);
    const size_t chunk = std::max<size_t>(1, input.audio.sample_rate * 160 / 1000);
    std::string transcript;
    for (size_t offset = 0; offset < input.audio.samples.size(); offset += chunk) {
        const size_t count = std::min(chunk, input.audio.samples.size() - offset);
        stream->push(input.audio.samples.data() + offset, count, input.audio.sample_rate);
        while (auto result = stream->next()) {
            if (result->is_final) {
                const auto text = append_result(*result);
                if (!text.empty()) {
                    if (!transcript.empty())
                        transcript += ' ';
                    transcript += text;
                }
            } else {
                break;
            }
        }
    }
    const auto final = append_result(stream->finish());
    if (!final.empty()) {
        if (!transcript.empty())
            transcript += ' ';
        transcript += final;
    }
    return transcript;
}

int
run_bench(int argc, char** argv) {
    if (argc == 0 || is_help_argument(argv[0])) {
        print_bench_help("nemo-speech");
        return 0;
    }
    if (std::string(argv[0]) != "asr")
        throw std::invalid_argument(
            "supported workload is 'asr' (use 'nemo-speech bench asr ...')");
    Options options = parse_options(argc, argv);
    std::vector<AudioInput> inputs;
    double corpus_seconds = 0.0;
    for (const auto& path : collect_files(options)) {
        auto audio = nemo_speech::audio::load_wav_file(path.string());
        corpus_seconds += static_cast<double>(audio.samples.size()) / audio.sample_rate;
        inputs.push_back({path, std::move(audio)});
    }
    const int max_concurrency =
        *std::max_element(options.concurrency.begin(), options.concurrency.end());
    options.config.model.path =
        resolve_model_file(
            options.model.empty() ? options.config.model.path : options.model, "asr", "ASR model")
            .string();
    options.config.batching.enabled = max_concurrency > 1;
    options.config.batching.max_batch_size =
        std::max(options.config.batching.max_batch_size, max_concurrency);
    options.config.batching.max_queue_depth =
        std::max(options.config.batching.max_queue_depth, max_concurrency * 2);
    options.config.batching.state_arena_slots =
        std::max(options.config.batching.state_arena_slots, max_concurrency);
    options.config.log_status = !cli_quiet() && !cli_json();

    nemo_speech::EngineRegistry engines;
    const auto load_start = Clock::now();
    auto recognizer = engines.load_asr(std::move(options.config));
    const double load_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - load_start).count();
    const auto warmup_start = Clock::now();
    engines.warmup();
    for (int i = 0; i < options.warmup; ++i)
        (void)recognize(*recognizer, inputs[static_cast<size_t>(i) % inputs.size()], options);
    const double warmup_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - warmup_start).count();

    Value output(Value::Object{});
    output["command"] = "bench asr";
    output["model"] = recognizer->model_name();
    output["mode"] = options.stream ? "stream" : "offline";
    output["load_ms"] = load_ms;
    output["warmup_ms"] = warmup_ms;
    output["files"] = static_cast<double>(inputs.size());
    output["corpus_audio_seconds"] = corpus_seconds;
    output["repetitions"] = options.repetitions;
    Value::Array runs;
    std::vector<std::string> reference(inputs.size());
    std::vector<bool> reference_set(inputs.size(), false);
    for (const int concurrency : options.concurrency) {
        const size_t work_count = inputs.size() * static_cast<size_t>(options.repetitions);
        std::atomic<size_t> next{0};
        std::atomic<int> mismatches{0};
        std::mutex error_mutex;
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
                        const size_t input_index = work % inputs.size();
                        const auto transcript =
                            recognize(*recognizer, inputs[input_index], options);
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!reference_set[input_index]) {
                            reference[input_index] = transcript;
                            reference_set[input_index] = true;
                        } else if (reference[input_index] != transcript) {
                            ++mismatches;
                        }
                    }
                }
                catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!failure)
                        failure = std::current_exception();
                }
            });
        }
        for (auto& worker : workers) worker.join();
        if (failure)
            std::rethrow_exception(failure);
        const double wall_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        const double audio_seconds = corpus_seconds * options.repetitions;
        Value run(Value::Object{});
        run["concurrency"] = concurrency;
        run["utterances"] = static_cast<double>(work_count);
        run["audio_seconds"] = audio_seconds;
        run["wall_seconds"] = wall_seconds;
        run["rtfx"] = audio_seconds / wall_seconds;
        run["utterances_per_second"] = work_count / wall_seconds;
        run["transcript_mismatches"] = mismatches.load();
        runs.emplace_back(std::move(run));
    }
    output["runs"] = std::move(runs);
    if (options.json) {
        std::printf("%s\n", output.dump(2).c_str());
    } else {
        std::printf(
            "Model: %s\nMode: %s\nLoad: %.1f ms  Warmup: %.1f ms\n",
            recognizer->model_name().c_str(), options.stream ? "stream" : "offline", load_ms,
            warmup_ms);
        std::printf(
            "%-12s %-12s %-12s %-12s %-10s\n", "CONCURRENCY", "WALL (s)", "RTFx", "UTT/s",
            "MISMATCH");
        for (const auto& run : output.at("runs").array())
            std::printf(
                "%-12d %-12.3f %-12.2f %-12.2f %-10d\n",
                static_cast<int>(run.at("concurrency").number()), run.at("wall_seconds").number(),
                run.at("rtfx").number(), run.at("utterances_per_second").number(),
                static_cast<int>(run.at("transcript_mismatches").number()));
    }
    return 0;
}

}  // namespace

void
print_bench_help(const char* program) {
    std::printf(
        "Usage: %s bench asr INPUT --model MODEL [options]\n\n"
        "Benchmark end-to-end ASR with one shared recognizer and concurrent utterances.\n"
        "Directory inputs are sorted and loaded once; GPU work uses the runtime's dynamic "
        "batching.\n\n"
        "Options:\n"
        "  -m, --model MODEL       Local ASR GGUF path\n"
        "  -c, --concurrency LIST  Comma-separated levels (default: 1,2,4)\n"
        "  -n, --repetitions N     Corpus repetitions per level (default: 3)\n"
        "  --warmup N              Timed-input warmup iterations (default: 1)\n"
        "  --mode offline|stream   Recognition mode (default: offline)\n"
        "  --device, --backend DEVICE\n"
        "                          auto, cpu, cuda[:N], metal, or vulkan[:N]\n"
        "  -l, --language CODE     Prompt language code\n"
        "  -r, --recursive         Recurse into input directories\n"
        "  --json                   Emit machine-readable results\n"
        "  --config FILE           Apply YAML configuration\n"
        "  --asr.* VALUE           Override any ASR engine setting\n",
        program);
}

int
command_bench(int argc, char** argv) {
#if defined(NEMO_SPEECH_CLI_ASR)
    try {
        return run_bench(argc, argv);
    }
    catch (const std::exception& error) {
        return print_cli_exception("bench", error);
    }
#else
    (void)argc;
    (void)argv;
    return print_cli_error(
        "bench", "this build does not include ASR", kCliExitUnsupportedFeature,
        "unsupported_feature");
#endif
}
