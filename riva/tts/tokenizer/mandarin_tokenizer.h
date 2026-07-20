// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

class mandarin_tokenizer {
   public:
    explicit mandarin_tokenizer(const std::filesystem::path& model_dir);
    ~mandarin_tokenizer();

    mandarin_tokenizer(const mandarin_tokenizer&) = delete;
    mandarin_tokenizer& operator=(const mandarin_tokenizer&) = delete;

    std::vector<int> encode(const std::string& text) const;
    int pad_id() const;

   private:
    class impl;
    std::unique_ptr<impl> impl_;
};

bool mandarin_tokenizer_available(const std::filesystem::path& model_dir);
