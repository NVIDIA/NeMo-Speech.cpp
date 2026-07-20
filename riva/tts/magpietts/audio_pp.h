// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace nemo_speech::tts {

class AudioPostProcessor {
   public:
    using WriteCallback = std::function<bool(const std::vector<float>&)>;

    AudioPostProcessor(int samples_per_frame, int future_frames, int window_samples);

    bool writeDecodedAudio(
        const std::vector<float>& audio, int history_frames, bool final_chunk,
        const WriteCallback& write_audio);
    bool flush(const WriteCallback& write_audio);

   private:
    static float hanningOverlapWeight(size_t i, size_t count);

    int samples_per_frame_ = 0;
    int future_frames_ = 0;
    int window_samples_ = 0;
    bool has_pending_audio_ = false;
    std::vector<float> pending_audio_;
};

}  // namespace nemo_speech::tts
