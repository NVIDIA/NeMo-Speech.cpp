// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Skinny-Q8 dispatch must not depend on call history (the skinny-Q8 ggml patch). The first
// 9..64-column mul_mat on an eligible Q8_0 weight repacks it in place; every column count must
// produce the same bits before and after that repack, including MMVQ's range (N <= 8, planar MMVQ
// after the repack) and widths above 64. Skips itself (exit 77) without a GPU backend.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace {

constexpr int64_t kK = 512;  // multiple of the skinny kernel's 128-byte k step
constexpr int64_t kM = 256;  // multiple of its 32-row block

std::vector<float>
run(ggml_backend_t backend, ggml_tensor* w, ggml_tensor* bias, int64_t n) {
    ggml_init_params params{ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kK, n);
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);
    if (bias) {
        y = ggml_add(ctx, y, bias);  // mul_mat + add is fused into the skinny-Q8 bias epilogue
    }
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    std::vector<float> xv((size_t)(kK * n));
    for (size_t i = 0; i < xv.size(); ++i) {
        xv[i] = std::sin(0.013f * (float)i + 0.5f * (float)n);
    }
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> out((size_t)(kM * n));
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return out;
}

}  // namespace

int
main() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        std::fprintf(stderr, "no GPU backend available; skipping\n");
        return 77;  // ctest SKIP_RETURN_CODE
    }

    // An eligible weight: Q8_0, contiguous, in a weights buffer, named like an encoder matrix.
    ggml_init_params params{ggml_tensor_overhead() * 4, nullptr, true};
    ggml_context* wctx = ggml_init(params);
    ggml_tensor* w = ggml_new_tensor_2d(wctx, GGML_TYPE_Q8_0, kK, kM);
    ggml_set_name(w, "encoder.test.weight");
    ggml_tensor* bias = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kM);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(wctx, backend);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> wf((size_t)(kK * kM)), bv((size_t)kM);
    for (size_t i = 0; i < wf.size(); ++i) {
        wf[i] = 0.05f * std::cos(0.007f * (float)i);
    }
    for (size_t i = 0; i < bv.size(); ++i) {
        bv[i] = 0.01f * (float)i;
    }
    std::vector<char> q(ggml_row_size(GGML_TYPE_Q8_0, kK) * kM);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), q.data(), 0, kM, kK, nullptr);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    ggml_backend_tensor_set(bias, bv.data(), 0, bv.size() * sizeof(float));

    // Widths outside the 9..64 skinny range, where the kernel used to depend on whether the weight
    // was already repacked: MMVQ's range and wider than the old 64-column cap. A skinny-range call
    // here would repack the weight before the reference outputs are recorded.
    const int64_t widths[] = {1, 2, 4, 8, 65, 69, 128};
    std::vector<std::vector<float>> before, before_bias;
    for (int64_t n : widths) {
        before.push_back(run(backend, w, nullptr, n));
        before_bias.push_back(run(backend, w, bias, n));
    }
    run(backend, w, nullptr, 16);  // a skinny-range call repacks the weight in place

    int failures = 0;
    for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); ++i) {
        const std::vector<float> after = run(backend, w, nullptr, widths[i]);
        const std::vector<float> after_bias = run(backend, w, bias, widths[i]);
        const bool same =
            std::memcmp(after.data(), before[i].data(), after.size() * sizeof(float)) == 0;
        const bool same_bias =
            std::memcmp(
                after_bias.data(), before_bias[i].data(), after_bias.size() * sizeof(float)) == 0;
        if (!same || !same_bias) {
            std::fprintf(
                stderr, "FAIL: N=%lld output changed after the repack (plain %s, bias %s)\n",
                (long long)widths[i], same ? "same" : "differs", same_bias ? "same" : "differs");
            ++failures;
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(wctx);
    ggml_backend_free(backend);
    if (failures) {
        return 1;
    }
    std::fprintf(stderr, "OK: every width gives the same bits before and after the repack\n");
    return 0;
}
