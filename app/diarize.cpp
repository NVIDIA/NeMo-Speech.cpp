// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio_file.h"
#include "cli_util.h"
#include "commands.h"
#include "engine_registry.h"
#include "model_utils.h"
#include "parameter_parser.h"

namespace {

std::string
value_after(int& index, int argc, char** argv, const std::string& option) {
    if (++index >= argc)
        throw std::invalid_argument(option + " requires a value");
    return argv[index];
}

}  // namespace

void
print_diarize_help(const char* program) {
    std::printf(
        "Usage: %s diarize AUDIO.wav [--model MODEL] [options]\n\n"
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
        "  --force                   Replace an existing output file\n",
        program);
}

int
command_diarize(int argc, char** argv) {
    try {
        if (argc > 0 && is_help_argument(argv[0])) {
            print_diarize_help("nemo-speech");
            return 0;
        }
        std::filesystem::path input;
        std::filesystem::path output;
        std::string config_file;
        nemo_speech::asr::DiarConfig config;
        nemo_speech::common::ParameterParser parser;
        parser.Register("diar", config);
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
        bool offline = false;
        bool force = false;
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
            throw std::invalid_argument("AUDIO.wav is required");
        if (format != "text" && format != "json" && format != "rttm")
            throw std::invalid_argument("--format must be text, json, or rttm");
        if (recording_id.empty())
            recording_id = input.stem().string();

        const auto source = nemo_speech::audio::load_wav_file(input.string());
        const auto geometry = config.resolved_geometry();
        nemo_speech::EngineRegistry engines;
        config.model_path = require_model_file(config.model_path, "diarization model").string();
        if (cli_verbose())
            std::fprintf(
                stderr, "diarize: model=%s mode=%s device=%d\n", config.model_path.c_str(),
                offline ? "offline" : "streaming", gpu);
        auto engine = engines.load_diarization(gpu, config.model_path, geometry);
        const auto result = engine->diarize(
            source.samples.data(), source.samples.size(), source.sample_rate,
            offline ? nemo_speech::asr::DiarizationMode::Offline
                    : nemo_speech::asr::DiarizationMode::Streaming);
        const auto& segments = result.segments;

        std::string rendered;
        for (size_t i = 0; i < segments.size(); ++i) {
            const auto& segment = segments[i];
            char line[512];
            if (format == "rttm") {
                std::snprintf(
                    line, sizeof(line), "SPEAKER %s 1 %.3f %.3f <NA> <NA> speaker_%d <NA> <NA>\n",
                    recording_id.c_str(), segment.t0, segment.t1 - segment.t0, segment.speaker + 1);
            } else if (format == "json") {
                if (i == 0)
                    rendered = "{\n  \"file\": \"" + json_escape(input.string()) +
                               "\",\n  \"segments\": [";
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
            if (segments.empty())
                rendered =
                    "{\n  \"file\": \"" + json_escape(input.string()) + "\",\n  \"segments\": [";
            rendered += !segments.empty() ? "\n  ]\n}\n" : "]\n}\n";
        }
        if (output.empty())
            std::fwrite(rendered.data(), 1, rendered.size(), stdout);
        else
            write_text_file(output, rendered, force);
        return 0;
    }
    catch (const std::invalid_argument& error) {
        return print_cli_error(
            "diarize", error.what(), kCliExitInvalidArgument, "invalid_argument");
    }
    catch (const std::exception& error) {
        return print_cli_exception("diarize", error);
    }
}
