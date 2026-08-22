// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

inline constexpr int kCliExitSuccess = 0;
inline constexpr int kCliExitRuntime = 1;
inline constexpr int kCliExitInvalidArgument = 2;
inline constexpr int kCliExitMissingModel = 3;
inline constexpr int kCliExitUnsupportedFeature = 4;

class MissingModelError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

class UnsupportedFeatureError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

int parse_int(const std::string& value, const std::string& option, int minimum, int maximum);
double parse_double(const std::string& value, const std::string& option);
int default_gpu_index();
int parse_device(const std::string& value, const std::string& option = "--device");
std::string json_escape(const std::string& value);
std::string read_text_file(const std::filesystem::path& path);
std::vector<std::filesystem::path> collect_wav_inputs(
    const std::filesystem::path& input, bool recursive);
std::filesystem::path relative_output_path(
    const std::filesystem::path& input_root, const std::filesystem::path& input_file);
void write_text_file(const std::filesystem::path& path, const std::string& contents, bool force);
bool is_help_argument(const std::string& value);
bool open_url_in_browser(const std::string& url);

void configure_cli_output(bool json, bool quiet, bool verbose);
bool cli_json();
bool cli_quiet();
bool cli_verbose();
int print_cli_error(
    const std::string& command, const std::string& message, int exit_code,
    const std::string& type = "runtime_error");
int print_cli_exception(const std::string& command, const std::exception& error);
