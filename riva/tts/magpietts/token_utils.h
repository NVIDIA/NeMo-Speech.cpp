// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nemo_speech::tts {

inline std::vector<int32_t>
flatten_token_chunks(const std::vector<std::vector<int32_t>>& token_chunks) {
    std::vector<int32_t> tokens;
    size_t total = 0;
    for (const auto& chunk : token_chunks) {
        total += chunk.size();
    }
    tokens.reserve(total);
    for (const auto& chunk : token_chunks) {
        tokens.insert(tokens.end(), chunk.begin(), chunk.end());
    }
    return tokens;
}

}  // namespace nemo_speech::tts
