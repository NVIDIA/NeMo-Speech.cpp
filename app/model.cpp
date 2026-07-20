// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <filesystem>
#include <stdexcept>

#include "cli_util.h"
#include "commands.h"
#include "json.h"
#include "model_utils.h"

namespace {
namespace fs = std::filesystem;
using nemo_speech::json::Value;

int
run_model(int argc, char** argv) {
    if (argc == 0 || is_help_argument(argv[0])) {
        print_model_help("nemo-speech");
        return 0;
    }
    if (std::string(argv[0]) != "info")
        throw std::invalid_argument("unknown model action: " + std::string(argv[0]));
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
        "Usage: %s model info FILE\n\n"
        "Inspect the metadata and runtime compatibility of a local GGUF file.\n",
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
