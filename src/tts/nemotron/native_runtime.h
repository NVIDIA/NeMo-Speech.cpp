// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tts/synthesizer.h"

namespace nemo_speech::tts::nemotron {

class NativeRuntime {
   public:
    NativeRuntime(const std::string& socket_path, int timeout_seconds);
    ~NativeRuntime();
    NativeRuntime(const NativeRuntime&) = delete;
    NativeRuntime& operator=(const NativeRuntime&) = delete;

    SynthesisResult synthesize(
        const PreparedSynthesis& request, const Synthesizer::PcmCallback& callback);
    const std::string& model_name() const;
    const std::vector<std::string>& speaker_names() const;
    bool text_normalization_enabled() const;
    int batch_size() const;
    const std::string& mode() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts::nemotron
