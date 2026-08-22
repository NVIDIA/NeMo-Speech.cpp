// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace nemo_speech::cli {

// Captures mono float32 samples from the default microphone. miniaudio's host
// backend stays behind the pimpl so transcribe.cpp remains independent of OS
// audio headers.
class MicrophoneCapture {
   public:
    MicrophoneCapture();
    ~MicrophoneCapture();

    MicrophoneCapture(const MicrophoneCapture&) = delete;
    MicrophoneCapture& operator=(const MicrophoneCapture&) = delete;

    void start();
    void stop();
    std::vector<float> drain();
    int sample_rate() const;
    const std::string& device_name() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::cli
