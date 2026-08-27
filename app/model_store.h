// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <string>
#include <vector>

struct PulledModelArtifact {
    std::string repo;
    std::string role;
    std::filesystem::path path;
    bool cached = false;
};

std::filesystem::path resolve_indexed_model_file(
    const std::string& reference, const std::string& role, const std::string& description);
std::filesystem::path resolve_indexed_model_directory(
    const std::string& reference, const std::string& role, const std::string& description);

std::vector<PulledModelArtifact> pull_indexed_model(const std::string& repo);
std::string indexed_models_json();
std::string indexed_models_text();
std::filesystem::path model_downloader_executable();
