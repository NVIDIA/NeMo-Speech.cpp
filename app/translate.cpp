// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli_util.h"
#include "commands.h"
#include "engine_registry.h"
#include "json.h"
#include "model_utils.h"
#include "parameter_parser.h"
#include "translator.h"

void
print_translate_help(const char* program) {
    std::printf(
        "Usage: %s translate TEXT [--model MODEL] --from SRC --to DST [options]\n\n"
        "If TEXT and --input are omitted, text is read from stdin.\n\n"
        "Options:\n"
        "  -m, --model MODEL      Local translation GGUF path\n"
        "  --from CODE            Source language (required)\n"
        "  --to CODE              Target language (required)\n"
        "  -i, --input PATH       Read one input text per line\n"
        "  -o, --output PATH      Write translations to a file\n"
        "  --device, --backend DEVICE\n"
        "                         auto, cpu, cuda[:N], metal, or vulkan[:N]\n"
        "  --json                  Emit source/target/text JSON\n"
        "  --config FILE           Apply YAML configuration\n"
        "  --nmt.* VALUE           Override any NMT engine setting\n"
        "  --force                Replace an existing output file\n",
        program);
}

int
command_translate(int argc, char** argv) {
    try {
        if (argc > 0 && is_help_argument(argv[0])) {
            print_translate_help("nemo-speech");
            return 0;
        }
        nemo_speech::nmt::TranslatorConfig config;
        config.backend.gpu = default_gpu_index();
        nemo_speech::common::ParameterParser parser;
        parser.Register("nmt", config);
        std::string config_file;
        for (int i = 0; i < argc; ++i) {
            if (std::string(argv[i]) == "--config") {
                if (++i >= argc)
                    throw std::invalid_argument("--config requires a value");
                config_file = argv[i];
            }
        }
        if (!config_file.empty())
            parser.ApplyYaml(config_file);
        parser.ApplyEnv("NEMO_SPEECH");
        std::string model_path, source, target, input_path, output_path;
        std::vector<std::string> texts;
        int gpu = default_gpu_index();
        bool device_set = false;
        bool force = false;
        bool json = cli_json();
        auto value = [&](int& i, const std::string& option) {
            if (++i >= argc)
                throw std::invalid_argument(option + " requires a value");
            return std::string(argv[i]);
        };
        for (int i = 0; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--model" || arg == "-m")
                model_path = value(i, arg);
            else if (arg == "--config")
                ++i;
            else if (arg == "--from")
                source = value(i, arg);
            else if (arg == "--to")
                target = value(i, arg);
            else if (arg == "--input" || arg == "-i")
                input_path = value(i, arg);
            else if (arg == "--output" || arg == "-o")
                output_path = value(i, arg);
            else if (arg == "--gpu") {
                gpu = parse_int(value(i, arg), arg, -1, 1024);
                device_set = true;
            } else if (arg == "--device" || arg == "--backend") {
                gpu = parse_device(value(i, arg), arg);
                device_set = true;
            } else if (arg == "--force")
                force = true;
            else if (arg == "--json")
                json = true;
            else if (!arg.empty() && arg[0] == '-') {
                bool consumed = false;
                if (!parser.ParseCliArg(arg, i + 1 < argc ? argv[i + 1] : nullptr, &consumed))
                    throw std::invalid_argument("unknown option: " + arg);
                if (consumed)
                    ++i;
            } else
                texts.push_back(arg);
        }
        if (source.empty() || target.empty())
            throw std::invalid_argument("--from and --to are required");
        if (!input_path.empty() && !texts.empty())
            throw std::invalid_argument("TEXT and --input cannot be used together");
        if (!input_path.empty()) {
            std::istringstream lines(read_text_file(input_path));
            for (std::string line; std::getline(lines, line);)
                if (!line.empty())
                    texts.push_back(std::move(line));
        } else if (texts.empty()) {
            for (std::string line; std::getline(std::cin, line);)
                if (!line.empty())
                    texts.push_back(std::move(line));
        }
        if (texts.empty())
            throw std::invalid_argument("no input text was provided");

        if (device_set)
            config.backend.gpu = gpu;
        config.verbose = cli_verbose();
        config.model.path =
            require_model_file(
                model_path.empty() ? config.model.path : model_path, "translation model")
                .string();
        if (cli_verbose())
            std::fprintf(
                stderr, "translate: model=%s pair=%s-%s inputs=%zu device=%d\n",
                config.model.path.c_str(), source.c_str(), target.c_str(), texts.size(),
                config.backend.gpu);
        nemo_speech::EngineRegistry engines;
        auto translator = engines.load_nmt(std::move(config));
        const auto result = translator->translate(texts, source, target);
        std::string rendered;
        if (json) {
            nemo_speech::json::Value::Array translations;
            for (size_t i = 0; i < result.size(); ++i) {
                nemo_speech::json::Value item(nemo_speech::json::Value::Object{});
                item["input"] = texts[i];
                item["text"] = result[i].text;
                item["source_language"] = source;
                item["target_language"] = target;
                translations.emplace_back(std::move(item));
            }
            rendered = nemo_speech::json::Value(std::move(translations)).dump(2) + "\n";
        } else {
            for (const auto& translation : result) {
                rendered += translation.text;
                rendered += '\n';
            }
        }
        if (output_path.empty())
            std::fwrite(rendered.data(), 1, rendered.size(), stdout);
        else
            write_text_file(output_path, rendered, force);
        return 0;
    }
    catch (const std::invalid_argument& error) {
        return print_cli_error(
            "translate", error.what(), kCliExitInvalidArgument, "invalid_argument");
    }
    catch (const std::exception& error) {
        return print_cli_exception("translate", error);
    }
}
