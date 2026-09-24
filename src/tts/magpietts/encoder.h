// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "model.h"

namespace nemo_speech::tts {

class MagpieEncoder {
   public:
    explicit MagpieEncoder(const magpietts_model& model) : model_(model) {}
    MagpieEncoder(magpietts_model&&) = delete;

    bool eval(const std::vector<int32_t>& tokens, int threads, std::vector<float>& out) const;
    // backend/keep_allocr: optional side backend (own stream) and reusable graph allocator for
    // background evaluation (longform chunk prefetch).
    bool evalDevice(
        const std::vector<int32_t>& tokens, int threads, magpietts_backend_tensor& out,
        ggml_backend_t backend = nullptr, ggml_gallocr_t* keep_allocr = nullptr) const;

   private:
    const magpietts_model& model_;
};

}  // namespace nemo_speech::tts
