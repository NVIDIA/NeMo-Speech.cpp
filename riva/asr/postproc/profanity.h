// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Profanity filter - whole-word masking (keep first char, mask the rest).
// Word list loaded from a file (one lowercased word per line); empty path =
// disabled.
#pragma once

#include <string>
#include <unordered_set>

namespace nemo_speech::asr::postproc {

class Profanity {
   public:
    // Empty path -> disabled (mask() is a no-op).
    explicit Profanity(const std::string& list_path = "");

    bool enabled() const { return !words_.empty(); }

    // Mask profane whole words in `text` (case-insensitive match; the original
    // casing of the first character is kept, the rest replaced with '*').
    // Trailing ASCII punctuation on a token is preserved.
    std::string mask(const std::string& text) const;

   private:
    std::unordered_set<std::string> words_;  // lowercased
};

}  // namespace nemo_speech::asr::postproc
