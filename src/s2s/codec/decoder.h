// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Codec decode graph Module: codes -> wav, fused RVQ depth-sum decode ->
// ConvNeXt decoder -> mag/phase -> IDFT -> overlap-add -> envelope
// normalise. Streaming-only: per-block causal-conv caches and the
// complex-spec overlap tail are per-call inputs / named outputs; the
// S2SCodec facade owns the per-stream host state.
#pragma once

#include "config.h"
#include "runtime.h"

namespace nemo_speech::s2s {

class CodecDecodeWavModule : public ggml_runtime::Module {
   public:
    explicit CodecDecodeWavModule(const CodecConfig& cfg) : cfg_(cfg) {}

    // Inputs (order): codes_qbt (BT, Q) i32 with per-q contiguous rows;
    //   cache_in_0..n_blocks-1 (k-1, C_b, B) f32; spec_re_tail_in /
    //   spec_im_tail_in (n_bins, tail_len, B) f32; envelope_recip (out_trim,)
    //   f32.
    // Outputs: [0] wav (out_trim, 1, B); named: cache_out_<b>,
    //   spec_re_tail_out, spec_im_tail_out.
    void define_tensors(ggml_runtime::Session* s) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* s, ggml_runtime::TensorBag in,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* s) override;

    int tail_len() const;

   private:
    CodecConfig cfg_;
    // Computed post-processing weights (host data uploaded in set_data).
    std::vector<float> idft_re_, idft_im_, window_, imag_mask_, id_kernel_;
};

}  // namespace nemo_speech::s2s
