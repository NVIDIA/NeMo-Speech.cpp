// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Text normalization for TTS (written -> spoken, e.g. "2" -> "two").
// Uses the same optional Sparrowhawk/OpenFST stack as ASR ITN. Empty model dir,
// or a build without NEMO_SPEECH_WITH_NORM, is a pass-through.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace nemo_speech::tts::preproc {

class TextNormalizer {
   public:
    explicit TextNormalizer(const std::string& model_dir = "");
    ~TextNormalizer();

    bool enabled() const { return enabled_; }

    std::string normalize(const std::string& text) const;
    std::string normalize(const std::string& text, const std::string& language_code) const;
    std::vector<std::string> languages() const;

   private:
    bool enabled_ = false;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts::preproc
