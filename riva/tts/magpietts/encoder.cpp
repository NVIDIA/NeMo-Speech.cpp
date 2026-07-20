// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "encoder.h"

#include <cstdio>

#include "graph.h"
#include "nvtx_utils.h"

namespace nemo_speech::tts {

bool
MagpieEncoder::eval(
    const std::vector<int32_t>& tokens, int threads, std::vector<float>& out) const {
    const ggml_nvtx::range nvtx_range("magpietts_encoder_eval");
    const int n = (int)tokens.size();
    if (n <= 0 || n > model_.hparams.n_ctx) {
        fprintf(stderr, "invalid token length %d (max %d)\n", n, model_.hparams.n_ctx);
        return false;
    }

    ggml_context* ctx = new_graph_context();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAGPIETTS_MAX_NODES, false);

    ggml_tensor* tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(tok, "magpietts_encoder_tokens");
    ggml_set_input(tok);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(pos, "magpietts_encoder_positions");
    ggml_set_input(pos);

    ggml_tensor* x = ggml_get_rows(ctx, model_.text_embedding, tok);
    x = transformer_forward(ctx, model_.encoder, x, pos, nullptr);
    x = ggml_cont(ctx, ggml_cast(ctx, x, GGML_TYPE_F32));
    ggml_set_name(x, "magpietts_encoder_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    ggml_gallocr_t allocr = nullptr;
    const bool ok = compute_graph(
        model_, ctx, gf,
        {{"magpietts_encoder_tokens", tokens}, {"magpietts_encoder_positions", positions(n)}}, {},
        threads, &allocr);
    if (!ok) {
        ggml_free(ctx);
        return false;
    }

    out.resize((size_t)model_.hparams.n_embd * n);
    MagpiePinnedHostScratch output_staging;
    magpietts_backend_tensor_get_staged(
        model_, output_staging, x, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return true;
}

bool
MagpieEncoder::evalDevice(
    const std::vector<int32_t>& tokens, int threads, magpietts_backend_tensor& out) const {
    const ggml_nvtx::range nvtx_range("magpietts_encoder_eval_device");
    const int n = (int)tokens.size();
    if (n <= 0 || n > model_.hparams.n_ctx) {
        fprintf(stderr, "invalid token length %d (max %d)\n", n, model_.hparams.n_ctx);
        return false;
    }
    if (!out.alloc2d(model_, GGML_TYPE_F32, model_.hparams.n_embd, n, "text_cond_device")) {
        return false;
    }

    ggml_context* ctx = new_graph_context();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAGPIETTS_MAX_NODES, false);

    ggml_tensor* tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(tok, "magpietts_encoder_tokens");
    ggml_set_input(tok);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(pos, "magpietts_encoder_positions");
    ggml_set_input(pos);

    ggml_tensor* x = ggml_get_rows(ctx, model_.text_embedding, tok);
    x = transformer_forward(ctx, model_.encoder, x, pos, nullptr);
    x = ggml_cont(ctx, ggml_cast(ctx, x, GGML_TYPE_F32));
    ggml_set_name(x, "magpietts_encoder_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    ggml_gallocr_t allocr = nullptr;
    const bool ok = compute_graph(
        model_, ctx, gf,
        {{"magpietts_encoder_tokens", tokens}, {"magpietts_encoder_positions", positions(n)}}, {},
        threads, &allocr);
    if (!ok) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_copy(x, out.tensor);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return true;
}

}  // namespace nemo_speech::tts
