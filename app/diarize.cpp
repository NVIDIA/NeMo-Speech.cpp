// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "audio_file.h"
#include "batching.h"
#include "cli_util.h"
#include "commands.h"
#include "engine_registry.h"
#include "model_utils.h"
#include "parameter_parser.h"

namespace {
namespace fs = std::filesystem;

std::string
value_after(int& index, int argc, char** argv, const std::string& option) {
    if (++index >= argc)
        throw std::invalid_argument(option + " requires a value");
    return argv[index];
}

std::string
render_result(
    const nemo_speech::asr::DiarizationResult& result, const std::string& format,
    const fs::path& input, const std::string& recording_id) {
    std::string rendered;
    for (size_t i = 0; i < result.segments.size(); ++i) {
        const auto& segment = result.segments[i];
        char line[512];
        if (format == "rttm") {
            std::snprintf(
                line, sizeof(line), "SPEAKER %s 1 %.3f %.3f <NA> <NA> speaker_%d <NA> <NA>\n",
                recording_id.c_str(), segment.t0, segment.t1 - segment.t0, segment.speaker + 1);
        } else if (format == "json") {
            if (i == 0)
                rendered =
                    "{\n  \"file\": \"" + json_escape(input.string()) + "\",\n  \"segments\": [";
            std::snprintf(
                line, sizeof(line), "%s\n    {\"start\": %.3f, \"end\": %.3f, \"speaker\": %d}",
                i ? "," : "", segment.t0, segment.t1, segment.speaker + 1);
        } else {
            std::snprintf(
                line, sizeof(line), "%.3f\t%.3f\tspeaker %d\n", segment.t0, segment.t1,
                segment.speaker + 1);
        }
        rendered += line;
    }
    if (format == "json") {
        if (result.segments.empty())
            rendered = "{\n  \"file\": \"" + json_escape(input.string()) + "\",\n  \"segments\": [";
        rendered += !result.segments.empty() ? "\n  ]\n}\n" : "]\n}\n";
    }
    return rendered;
}

std::string
extension_for(const std::string& format) {
    if (format == "json")
        return ".json";
    if (format == "rttm")
        return ".rttm";
    return ".txt";
}

double
probability_after(int& index, int argc, char** argv, const std::string& option) {
    const double value = parse_double(value_after(index, argc, argv, option), option);
    if (value < 0.0 || value > 1.0)
        throw std::invalid_argument(option + " must be between 0 and 1");
    return value;
}

double
duration_after(int& index, int argc, char** argv, const std::string& option) {
    const double value = parse_double(value_after(index, argc, argv, option), option);
    if (value < 0.0)
        throw std::invalid_argument(option + " must not be negative");
    return value;
}

}  // namespace

void
print_diarize_help(const char* program) {
    std::printf(
        "Usage: %s diarize INPUT [--model MODEL] [options]\n\n"
        "Diarize one WAV file or every WAV file in a directory. Concurrent\n"
        "directory work shares one model and batches compatible GPU steps.\n\n"
        "Options:\n"
        "  -m, --model MODEL         Local Sortformer GGUF path\n"
        "  --device, --backend DEVICE\n"
        "                            auto, cpu, cuda[:N], metal, or vulkan[:N]\n"
        "  --offline                 Full-attention mode for short audio\n"
        "  --preset NAME             streaming or offline geometry\n"
        "  --config FILE             Load diarization YAML configuration\n"
        "  --format text|json|rttm   Output format (default: text)\n"
        "  --recording-id NAME       RTTM recording id\n"
        "  -o, --output PATH         Write output instead of stdout\n"
        "  --output-dir DIR          Preserve directory layout under DIR\n"
        "  -c, --concurrency N       Concurrent files; one shared model\n"
        "  -r, --recursive           Recurse into input directories\n"
        "  --onset VALUE             Speaker activation threshold\n"
        "  --offset VALUE            Speaker deactivation threshold\n"
        "  --pad-onset SECONDS       Extend segment starts\n"
        "  --pad-offset SECONDS      Extend segment ends\n"
        "  --min-duration-on SEC     Remove shorter speech segments\n"
        "  --min-duration-off SEC    Fill shorter silence gaps\n"
        "  --no-batching             Disable dynamic batching\n"
        "  --force                   Replace existing output files\n",
        program);
}

int
command_diarize(int argc, char** argv) {
    try {
        if (argc > 0 && is_help_argument(argv[0])) {
            print_diarize_help("nemo-speech");
            return 0;
        }
        fs::path input;
        fs::path output;
        fs::path output_dir;
        std::string config_file;
        nemo_speech::asr::DiarConfig config;
        nemo_speech::asr::DiarSegmentationCfg segmentation;
        nemo_speech::asr::BatchingConfig batching;
        nemo_speech::common::ParameterParser parser;
        parser.Register("diar", config);
        parser.Register("batching", batching);
        for (int i = 0; i < argc; ++i) {
            if (std::string(argv[i]) == "--config")
                config_file = value_after(i, argc, argv, "--config");
        }
        if (!config_file.empty())
            parser.ApplyYaml(config_file);
        parser.ApplyEnv("NEMO_SPEECH");
        std::string format = cli_json() ? "json" : "text";
        std::string recording_id;
        int gpu = default_gpu_index();
        int concurrency = 0;
        bool offline = false;
        bool force = false;
        bool recursive = false;
        bool enable_batching = true;
        for (int i = 0; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--config")
                ++i;
            else if (arg == "--model" || arg == "-m")
                config.model_path = value_after(i, argc, argv, arg);
            else if (arg == "--gpu")
                gpu = parse_int(value_after(i, argc, argv, arg), arg, -1, 1024);
            else if (arg == "--device" || arg == "--backend") {
                gpu = parse_device(value_after(i, argc, argv, arg), arg);
            } else if (arg == "--offline")
                offline = true;
            else if (arg == "--preset")
                config.preset = value_after(i, argc, argv, arg);
            else if (arg == "--format")
                format = value_after(i, argc, argv, arg);
            else if (arg == "--recording-id")
                recording_id = value_after(i, argc, argv, arg);
            else if (arg == "--output" || arg == "-o")
                output = value_after(i, argc, argv, arg);
            else if (arg == "--output-dir")
                output_dir = value_after(i, argc, argv, arg);
            else if (arg == "--concurrency" || arg == "-c")
                concurrency = parse_int(value_after(i, argc, argv, arg), arg, 1, 1024);
            else if (arg == "--recursive" || arg == "-r")
                recursive = true;
            else if (arg == "--onset")
                segmentation.onset = static_cast<float>(probability_after(i, argc, argv, arg));
            else if (arg == "--offset")
                segmentation.offset = static_cast<float>(probability_after(i, argc, argv, arg));
            else if (arg == "--pad-onset")
                segmentation.pad_onset = duration_after(i, argc, argv, arg);
            else if (arg == "--pad-offset")
                segmentation.pad_offset = duration_after(i, argc, argv, arg);
            else if (arg == "--min-duration-on")
                segmentation.min_duration_on = duration_after(i, argc, argv, arg);
            else if (arg == "--min-duration-off")
                segmentation.min_duration_off = duration_after(i, argc, argv, arg);
            else if (arg == "--no-batching")
                enable_batching = false;
            else if (arg == "--force")
                force = true;
            else if (!arg.empty() && arg[0] == '-') {
                bool consumed = false;
                if (!parser.ParseCliArg(arg, i + 1 < argc ? argv[i + 1] : nullptr, &consumed))
                    throw std::invalid_argument("unknown option: " + arg);
                if (consumed)
                    ++i;
            } else if (input.empty())
                input = arg;
            else
                throw std::invalid_argument("unexpected argument: " + arg);
        }
        if (input.empty())
            throw std::invalid_argument("a WAV file or directory is required");
        if (format != "text" && format != "json" && format != "rttm")
            throw std::invalid_argument("--format must be text, json, or rttm");
        const auto inputs = collect_wav_inputs(input, recursive);
        const bool directory = fs::is_directory(input);
        if (directory && !output.empty())
            throw std::invalid_argument("--output is only valid for one input; use --output-dir");
        if (!directory && !output_dir.empty())
            throw std::invalid_argument("--output-dir is only valid for a directory input");
        if (directory && !recording_id.empty())
            throw std::invalid_argument("--recording-id is only valid for one input");
        const int workers = std::min<int>(
            concurrency > 0 ? concurrency : (directory && gpu >= 0 ? 4 : 1), inputs.size());
        batching.enabled = enable_batching && workers > 1;
        batching.max_batch_size = std::min(batching.max_batch_size, workers);
        batching.max_queue_depth = std::max(batching.max_queue_depth, workers * 4);
        batching.state_arena_slots = std::max(batching.state_arena_slots, workers);
        const auto geometry = config.resolved_geometry();
        nemo_speech::EngineRegistry engines;
        config.model_path = require_model_file(config.model_path, "diarization model").string();
        if (cli_verbose())
            std::fprintf(
                stderr, "diarize: model=%s mode=%s inputs=%zu concurrency=%d device=%d\n",
                config.model_path.c_str(), offline ? "offline" : "streaming", inputs.size(),
                workers, gpu);
        auto engine = engines.load_diarization(gpu, config.model_path, geometry, batching);
        std::vector<nemo_speech::asr::DiarizationResult> results(inputs.size());
        std::vector<std::string> errors(inputs.size());
        std::atomic<size_t> next{0};
        std::vector<std::thread> worker_threads;
        worker_threads.reserve(workers);
        for (int worker = 0; worker < workers; ++worker) {
            worker_threads.emplace_back([&] {
                for (;;) {
                    const size_t index = next.fetch_add(1);
                    if (index >= inputs.size())
                        break;
                    try {
                        const auto source =
                            nemo_speech::audio::load_wav_file(inputs[index].string());
                        results[index] = engine->diarize(
                            source.samples.data(), source.samples.size(), source.sample_rate,
                            offline ? nemo_speech::asr::DiarizationMode::Offline
                                    : nemo_speech::asr::DiarizationMode::Streaming,
                            segmentation);
                    }
                    catch (const std::exception& error) {
                        errors[index] = error.what();
                    }
                }
            });
        }
        for (auto& worker : worker_threads) worker.join();
        if (cli_verbose() && batching.enabled) {
            const auto metrics = engine->batch_metrics();
            std::fprintf(
                stderr, "diarize: batches=%llu items=%llu max_batch=%llu\n",
                static_cast<unsigned long long>(metrics.batches),
                static_cast<unsigned long long>(metrics.items),
                static_cast<unsigned long long>(metrics.max_observed_batch));
        }

        if (directory && output_dir.empty())
            output_dir = fs::current_path() / "diarization";
        int failures = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!errors[i].empty()) {
                print_cli_error(
                    "diarize", inputs[i].string() + ": " + errors[i], 1, "runtime_error");
                ++failures;
                continue;
            }
            const std::string id = recording_id.empty() ? inputs[i].stem().string() : recording_id;
            const std::string rendered = render_result(results[i], format, inputs[i], id);
            if (!directory && output.empty()) {
                std::fwrite(rendered.data(), 1, rendered.size(), stdout);
                continue;
            }
            fs::path destination;
            if (!directory) {
                destination = output;
            } else {
                destination = relative_output_path(input, inputs[i]);
                destination.replace_extension(extension_for(format));
                destination = output_dir / destination;
            }
            try {
                write_text_file(destination, rendered, force);
                if (!cli_quiet())
                    std::fprintf(
                        stderr, "%s -> %s\n", inputs[i].string().c_str(),
                        destination.string().c_str());
            }
            catch (const std::exception& error) {
                print_cli_error("diarize", error.what(), 1, "runtime_error");
                ++failures;
            }
        }
        if (directory && !cli_quiet())
            std::fprintf(
                stderr, "diarized %zu file%s (%d failed)\n", inputs.size(),
                inputs.size() == 1 ? "" : "s", failures);
        return failures == 0 ? 0 : 1;
    }
    catch (const std::invalid_argument& error) {
        return print_cli_error(
            "diarize", error.what(), kCliExitInvalidArgument, "invalid_argument");
    }
    catch (const std::exception& error) {
        return print_cli_exception("diarize", error);
    }
}
