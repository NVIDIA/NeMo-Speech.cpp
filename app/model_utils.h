// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <string>

std::filesystem::path require_model_file(const std::string& path, const std::string& description);
std::filesystem::path require_model_directory(
    const std::string& path, const std::string& description);

std::string inspect_gguf_json(const std::filesystem::path& path);
