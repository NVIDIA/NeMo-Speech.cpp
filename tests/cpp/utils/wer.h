// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <string>

namespace nemo_speech::tests {

struct ErrorRateResult {
    size_t edits = 0;
    size_t reference_units = 0;
    size_t hypothesis_units = 0;
    double rate = 0.0;
};

std::string lowercase_text(const std::string& text);
ErrorRateResult word_error_rate(const std::string& reference, const std::string& hypothesis);
ErrorRateResult char_error_rate(const std::string& reference, const std::string& hypothesis);

}  // namespace nemo_speech::tests
