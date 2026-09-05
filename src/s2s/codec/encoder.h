// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Codec encode graph Module: spec -> codes (encoder + RVQ
// nearest-neighbour), non-streaming. Used once at startup to derive the TTS
// silence codes from a zeros spectrogram.
#pragma once

#include "config.h"
#include "runtime.h"

namespace nemo_speech::s2s {

class CodecEncodeModule : public ggml_runtime::Module {
   public:
    explicit CodecEncodeModule(const CodecConfig& cfg) : cfg_(cfg) {}

    // Inputs: spec (spec_t, spec_channels, B) f32.
    // Outputs: named code_<q> (BT,) i32 for q in [0, Q).
    void define_tensors(ggml_runtime::Session* s) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* s, ggml_runtime::TensorBag in,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* s) override;

   private:
    CodecConfig cfg_;
};

}  // namespace nemo_speech::s2s
