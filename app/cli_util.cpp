// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cli_util.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "audio_file.h"
#include "model_logging.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

struct CliOutputState {
    bool json = false;
    bool quiet = false;
    bool verbose = false;
};

CliOutputState output_state;

}  // namespace

#if defined(_WIN32)
#include <shellapi.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

int
parse_int(const std::string& value, const std::string& option, int minimum, int maximum) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < minimum || parsed > maximum)
        throw std::invalid_argument(
            option + " must be an integer from " + std::to_string(minimum) + " to " +
            std::to_string(maximum));
    return static_cast<int>(parsed);
}

double
parse_double(const std::string& value, const std::string& option) {
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value.c_str(), &end);
    if (errno != 0 || end == value.c_str() || *end != '\0' || !std::isfinite(parsed))
        throw std::invalid_argument(option + " must be a finite number");
    return parsed;
}

int
default_gpu_index() {
#if defined(NEMO_SPEECH_CLI_CUDA) || defined(NEMO_SPEECH_CLI_METAL) || \
    defined(NEMO_SPEECH_CLI_VULKAN)
    return 0;
#else
    return -1;
#endif
}

int
parse_device(const std::string& value, const std::string& option) {
    if (value == "auto")
        return default_gpu_index();
    if (value == "cpu")
        return -1;

    const auto parse_index = [&] {
        const auto colon = value.find(':');
        return colon == std::string::npos ? 0 : parse_int(value.substr(colon + 1), option, 0, 1024);
    };
    if (value == "cuda" || value.rfind("cuda:", 0) == 0) {
#if defined(NEMO_SPEECH_CLI_CUDA)
        return parse_index();
#else
        throw std::invalid_argument(option + " requested CUDA, but this build has no CUDA backend");
#endif
    }
    if (value == "metal") {
#if defined(NEMO_SPEECH_CLI_METAL)
        return 0;
#else
        throw std::invalid_argument(
            option + " requested Metal, but this build has no Metal backend");
#endif
    }
    if (value == "vulkan" || value.rfind("vulkan:", 0) == 0) {
#if defined(NEMO_SPEECH_CLI_VULKAN)
        return parse_index();
#else
        throw std::invalid_argument(
            option + " requested Vulkan, but this build has no Vulkan backend");
#endif
    }
    if (value == "gpu" || value.rfind("gpu:", 0) == 0) {
        if (default_gpu_index() < 0)
            throw std::invalid_argument(option + " requested a GPU, but this is a CPU-only build");
        return parse_index();
    }
    throw std::invalid_argument(
        option + " must be auto, cpu, cuda[:N], metal, vulkan[:N], or gpu[:N]");
}

std::string
json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char c : value) {
        switch (c) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (c < 0x20) {
                    static constexpr char hex[] = "0123456789abcdef";
                    output << "\\u00" << hex[c >> 4] << hex[c & 0xf];
                } else {
                    output << static_cast<char>(c);
                }
        }
    }
    return output.str();
}

std::string
read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot open " + path.string());
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

std::vector<std::filesystem::path>
collect_wav_inputs(const std::filesystem::path& input, bool recursive) {
    std::error_code error;
    if (std::filesystem::is_regular_file(input, error)) {
        if (!nemo_speech::audio::is_wav_path(input.string()))
            throw std::invalid_argument("input must be a .wav file");
        return {std::filesystem::absolute(input)};
    }
    if (!std::filesystem::is_directory(input, error))
        throw std::invalid_argument(input.string() + " is not a file or directory");
    std::vector<std::filesystem::path> files;
    auto add = [&](const auto& entry) {
        if (entry.is_regular_file(error) && nemo_speech::audio::is_wav_path(entry.path().string()))
            files.push_back(std::filesystem::absolute(entry.path()));
    };
    if (recursive) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(input)) add(entry);
    } else {
        for (const auto& entry : std::filesystem::directory_iterator(input)) add(entry);
    }
    std::sort(files.begin(), files.end());
    if (files.empty())
        throw std::invalid_argument(input.string() + " contains no WAV files");
    return files;
}

std::filesystem::path
relative_output_path(
    const std::filesystem::path& input_root, const std::filesystem::path& input_file) {
    const auto root = std::filesystem::absolute(input_root).lexically_normal();
    const auto file = std::filesystem::absolute(input_file).lexically_normal();
    auto relative = file.lexically_relative(root);
    if (relative.empty() || relative.is_absolute() ||
        (!relative.empty() && *relative.begin() == ".."))
        return file.filename();
    return relative;
}

void
write_text_file(const std::filesystem::path& path, const std::string& contents, bool force) {
    if (!force && std::filesystem::exists(path))
        throw std::runtime_error(path.string() + " already exists (use --force to replace it)");
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output || !(output << contents))
            throw std::runtime_error("cannot write " + path.string());
    }
    std::error_code error;
    if (force)
        std::filesystem::remove(path, error);
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot finish writing " + path.string() + ": " + error.message());
    }
}

bool
is_help_argument(const std::string& value) {
    return value == "-h" || value == "--help" || value == "help";
}

bool
open_url_in_browser(const std::string& url) {
#if defined(_WIN32)
    const auto result = reinterpret_cast<intptr_t>(
        ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
    return result > 32;
#else
#if defined(__APPLE__)
    const char* launcher = "open";
#else
    const char* launcher = "xdg-open";
#endif
    std::filesystem::path executable;
    std::istringstream paths(std::getenv("PATH") ? std::getenv("PATH") : "");
    std::string directory;
    while (std::getline(paths, directory, ':')) {
        const auto candidate =
            std::filesystem::path(directory.empty() ? "." : directory) / launcher;
        if (access(candidate.c_str(), X_OK) == 0) {
            executable = candidate;
            break;
        }
    }
    if (executable.empty())
        return false;
    const pid_t child = fork();
    if (child < 0)
        return false;
    if (child == 0) {
        const pid_t grandchild = fork();
        if (grandchild == 0) {
            execl(executable.c_str(), launcher, url.c_str(), static_cast<char*>(nullptr));
            _exit(127);
        }
        _exit(grandchild < 0 ? 1 : 0);
    }
    int status = 0;
    return waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

void
configure_cli_output(bool json, bool quiet, bool verbose) {
    output_state = {json, quiet, verbose};
    nemo_speech::common::configure_ggml_logging(verbose);
}

bool
cli_json() {
    return output_state.json;
}

bool
cli_quiet() {
    return output_state.quiet;
}

bool
cli_verbose() {
    return output_state.verbose;
}

int
print_cli_error(
    const std::string& command, const std::string& message, int exit_code,
    const std::string& type) {
    if (output_state.json) {
        std::fprintf(
            stderr,
            "{\"error\":{\"message\":\"%s\",\"type\":\"%s\",\"command\":\"%s\","
            "\"exit_code\":%d}}\n",
            json_escape(message).c_str(), json_escape(type).c_str(), json_escape(command).c_str(),
            exit_code);
    } else {
        std::fprintf(
            stderr, "nemo-speech%s%s: %s\n", command.empty() ? "" : " ", command.c_str(),
            message.c_str());
    }
    return exit_code;
}

int
print_cli_exception(const std::string& command, const std::exception& error) {
    if (dynamic_cast<const MissingModelError*>(&error))
        return print_cli_error(command, error.what(), kCliExitMissingModel, "missing_model");
    if (dynamic_cast<const UnsupportedFeatureError*>(&error))
        return print_cli_error(
            command, error.what(), kCliExitUnsupportedFeature, "unsupported_feature");
    if (dynamic_cast<const std::invalid_argument*>(&error))
        return print_cli_error(command, error.what(), kCliExitInvalidArgument, "invalid_argument");
    return print_cli_error(command, error.what(), kCliExitRuntime);
}
