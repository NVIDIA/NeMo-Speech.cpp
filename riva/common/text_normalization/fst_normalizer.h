// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Shared Sparrowhawk/OpenFST grammar runtime. The same two-FAR interface is
// used for ASR inverse text normalization and TTS text normalization.
#pragma once

#include <memory>
#include <string>

namespace nemo_speech::text_normalization {

class FstNormalizer {
   public:
    explicit FstNormalizer(const std::string& model_dir);
    ~FstNormalizer();

    // Returns the input unchanged on a per-utterance rewrite failure. When
    // requested, alignment_links describes the same successful rewrite.
    std::string normalize(const std::string& input, std::string* alignment_links = nullptr) const;
    std::string alignment(const std::string& input) const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::text_normalization
