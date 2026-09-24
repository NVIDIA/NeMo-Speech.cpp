// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

inline float
finite_max_abs_diff(
    const std::vector<float>& actual, const std::vector<float>& expected, size_t count) {
    if (count > actual.size() || count > expected.size())
        throw std::runtime_error("comparison exceeds output size");
    float result = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
            return std::numeric_limits<float>::infinity();
        result = std::max(result, std::fabs(actual[i] - expected[i]));
    }
    return result;
}

inline float
finite_max_abs_diff(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size())
        throw std::runtime_error("output size mismatch");
    return finite_max_abs_diff(actual, expected, actual.size());
}
