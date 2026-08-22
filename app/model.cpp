// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <filesystem>
#include <stdexcept>

#include "cli_util.h"
#include "commands.h"
#include "json.h"
#include "model_store.h"
#include "model_utils.h"

namespace {
namespace fs = std::filesystem;
using nemo_speech::json::Value;

int
run_pull(const std::string& repo) {
    const auto artifacts = pull_indexed_model(repo);
    if (cli_json()) {
        Value result(Value::Object{});
        result["repo"] = repo;
        Value::Array values;
        for (const auto& artifact : artifacts) {
            Value value(Value::Object{});
            value["repo"] = artifact.repo;
            value["role"] = artifact.role;
            value["path"] = artifact.path.u8string();
            value["cached"] = artifact.cached;
            values.emplace_back(std::move(value));
        }
        result["artifacts"] = std::move(values);
        std::printf("%s\n", result.dump(2).c_str());
    } else {
        for (const auto& artifact : artifacts)
            std::printf(
                "%s\t%s\t%s\n", artifact.repo.c_str(), artifact.role.c_str(),
                artifact.path.u8string().c_str());
    }
    return kCliExitSuccess;
}

int
run_model(int argc, char** argv) {
    if (argc == 0 || is_help_argument(argv[0])) {
        print_model_help("nemo-speech");
        return 0;
    }
    const std::string action = argv[0];
    if (action == "list") {
        if (argc != 1)
            throw std::invalid_argument("model list does not accept arguments");
        const std::string result = cli_json() ? indexed_models_json() : indexed_models_text();
        std::fwrite(result.data(), 1, result.size(), stdout);
        return kCliExitSuccess;
    }
    if (action == "pull") {
        if (argc != 2)
            throw std::invalid_argument("model pull requires one indexed Hugging Face repository");
        return run_pull(argv[1]);
    }
    if (action != "info")
        throw std::invalid_argument("unknown model action: " + action);
    if (argc != 2)
        throw std::invalid_argument("model info requires one local GGUF file");

    const fs::path path = require_model_file(argv[1], "model");
    const auto result = Value::parse(inspect_gguf_json(path));
    std::printf("%s\n", result.dump(2).c_str());
    return result.at("runtime_compatible").boolean() ? kCliExitSuccess : kCliExitUnsupportedFeature;
}

}  // namespace

void
print_model_help(const char* program) {
    std::printf(
        "Usage: %s model <action> [arguments]\n\n"
        "Actions:\n"
        "  list             List indexed repositories and command defaults\n"
        "  pull REPO        Download and verify an indexed Hugging Face repository\n"
        "  info FILE        Inspect a local GGUF file\n\n"
        "Models are cached under the platform user cache directory. Override it\n"
        "with NEMO_SPEECH_MODEL_DIR. Downloads require curl on PATH.\n",
        program);
}

int
command_model(int argc, char** argv) {
    try {
        return run_model(argc, argv);
    }
    catch (const std::exception& error) {
        return print_cli_exception("model", error);
    }
}

int
command_pull(int argc, char** argv) {
    try {
        if (argc > 0 && is_help_argument(argv[0])) {
            std::printf("Usage: nemo-speech pull REPO\n");
            return kCliExitSuccess;
        }
        if (argc != 1)
            throw std::invalid_argument("pull requires one indexed Hugging Face repository");
        return run_pull(argv[0]);
    }
    catch (const std::exception& error) {
        return print_cli_exception("pull", error);
    }
}
