// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nemo_speech::tts::nanocodec {

struct NanoCodecHParams {
    int32_t sample_rate = 22050;
    int32_t samples_per_frame = 1024;
    int32_t num_codebooks = 8;
    int32_t codebook_size = 2016;
    int32_t latent_dim = 32;
    int32_t group_dim = 4;

    std::vector<int32_t> levels = {8, 7, 6, 6};
    std::vector<int32_t> base = {1, 8, 56, 336};
    std::vector<int32_t> scale = {4, 3, 3, 3};
    std::vector<int32_t> offset = {4, 3, 3, 3};
    std::vector<int32_t> up_rates = {8, 8, 4, 2, 2};
    std::vector<int32_t> res_kernels = {3, 7, 11};
    std::vector<int32_t> res_dilations = {1, 3, 5};
};

using NanoCodecFrame = std::array<int32_t, 8>;
using NanoCodecFrames = std::vector<NanoCodecFrame>;

class NanoCodecDecoder;

class NanoCodecModel {
   public:
    NanoCodecModel();
    ~NanoCodecModel();

    NanoCodecModel(NanoCodecModel&& other) noexcept;
    NanoCodecModel& operator=(NanoCodecModel&& other) noexcept;

    NanoCodecModel(const NanoCodecModel&) = delete;
    NanoCodecModel& operator=(const NanoCodecModel&) = delete;

    bool load(const std::string& path, bool force_cpu = false);
    void reset();
    bool loaded() const;

    const NanoCodecHParams& hparams() const;
    int sampleRate() const;
    int samplesPerFrame() const;
    int numCodebooks() const;
    int codebookSize() const;
    int decoderLeftContextFrames() const;
    int64_t decoderLeftContextSamples() const;

   private:
    friend class NanoCodecDecoder;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class NanoCodecStreamState {
   public:
    NanoCodecStreamState();
    ~NanoCodecStreamState();

    NanoCodecStreamState(NanoCodecStreamState&& other) noexcept;
    NanoCodecStreamState& operator=(NanoCodecStreamState&& other) noexcept;

    NanoCodecStreamState(const NanoCodecStreamState&) = delete;
    NanoCodecStreamState& operator=(const NanoCodecStreamState&) = delete;

    void clear();

   private:
    friend class NanoCodecDecoder;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class NanoCodecStreamGraph {
   public:
    NanoCodecStreamGraph();
    ~NanoCodecStreamGraph();

    NanoCodecStreamGraph(NanoCodecStreamGraph&& other) noexcept;
    NanoCodecStreamGraph& operator=(NanoCodecStreamGraph&& other) noexcept;

    NanoCodecStreamGraph(const NanoCodecStreamGraph&) = delete;
    NanoCodecStreamGraph& operator=(const NanoCodecStreamGraph&) = delete;

    void reset();
    bool initialized() const;
    int chunkFrames() const;

   private:
    friend class NanoCodecDecoder;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class NanoCodecDecoder {
   public:
    explicit NanoCodecDecoder(const NanoCodecModel& model);

    bool decode(const NanoCodecFrames& frames, int threads, std::vector<float>& audio) const;
    bool initStreamGraph(
        NanoCodecStreamState& state, int chunk_frames, NanoCodecStreamGraph& graph) const;
    bool decodeStream(
        NanoCodecStreamState& state, NanoCodecStreamGraph& graph, const NanoCodecFrames& frames,
        int threads, std::vector<float>& audio) const;
    bool decodeStreamAll(
        const NanoCodecFrames& frames, int chunk_frames, int threads,
        std::vector<float>& audio) const;

   private:
    const NanoCodecModel* model_;
};

}  // namespace nemo_speech::tts::nanocodec
