// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// ggml_argmax_bf16 (the ggml patch) against policy::greedy_pick: the batched path picks greedy
// tokens with the op on the device, the upstream path with greedy_pick on the host, and both must
// give the same token. Rows of vocabulary width and narrower, with equal values after the BF16
// rounding placed so that a plain argmax, or an argmax that does not take the lowest index, gives
// another column. Runs on the CPU backend and on the GPU backend when there is one that supports
// the op (CUDA; Vulkan and Metal reject it, so the scheduler runs it on the CPU there).
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "salm_policy.h"

namespace {

using nemo_speech::speechlm::policy::greedy_pick;

// rows x cols logits in [100, 230) (BF16 steps of 0.5 and 1), then per row a group of columns
// that round to the same largest value
std::vector<float>
make_rows(int64_t cols, int64_t rows) {
    std::vector<float> x((size_t)(cols * rows));
    uint32_t state = 12345u + (uint32_t)cols;
    for (float& v : x) {
        state = state * 1664525u + 1013904223u;
        v = 100.0f + 130.0f * (float)(state >> 8) / 16777216.0f;
    }
    const float inf = std::numeric_limits<float>::infinity();
    for (int64_t r = 0; r < rows; ++r) {
        float* row = x.data() + r * cols;
        const int64_t lo = (7 + 13 * r) % cols, mid = cols / 2 + r, hi = cols - 1 - r;
        switch (r % 4) {
            case 0:  // the largest logit is the highest column; all three round to 250
                row[lo] = 249.6f;
                row[mid] = 250.2f;
                row[hi] = 250.4f;
                break;
            case 1:  // ties to even: 250.5 rounds to 250 and loses to 251 further on
                row[lo] = 250.5f;
                row[hi] = 250.6f;
                break;
            case 2:  // masked columns, and equal values in neighbouring columns
                row[0] = -inf;
                row[mid] = 240.3f;
                row[mid + 1] = 240.4f;
                row[hi] = -inf;
                break;
            default:  // no tie at the top
                row[mid] = 245.0f;
                break;
        }
    }
    return x;
}

std::vector<int32_t>
run(ggml_backend_t backend, const std::vector<float>& x, int64_t cols, int64_t rows, bool bf16) {
    ggml_init_params params{ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor* ids = bf16 ? ggml_argmax_bf16(ctx, in) : ggml_argmax(ctx, in);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ids);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(in, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);
    std::vector<int32_t> out((size_t)rows);
    ggml_backend_tensor_get(ids, out.data(), 0, out.size() * sizeof(int32_t));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return out;
}

bool
supports_argmax_bf16(ggml_backend_t backend) {
    ggml_init_params params{ggml_tensor_overhead() * 2, nullptr, true};
    ggml_context* ctx = ggml_init(params);
    const bool ok = ggml_backend_supports_op(
        backend, ggml_argmax_bf16(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 1)));
    ggml_free(ctx);
    return ok;
}

}  // namespace

int
main() {
    ggml_backend_t backends[2] = {
        ggml_backend_cpu_init(), ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)};
    const char* names[2] = {"CPU", "GPU"};
    if (!backends[1]) {
        std::fprintf(stderr, "no GPU backend available; CPU only\n");
    } else if (!supports_argmax_bf16(backends[1])) {
        std::fprintf(
            stderr, "%s does not support ggml_argmax_bf16 (it runs on the CPU there); CPU only\n",
            ggml_backend_name(backends[1]));
        ggml_backend_free(backends[1]);
        backends[1] = nullptr;
    }
    if (!supports_argmax_bf16(backends[0])) {
        std::fprintf(stderr, "FAIL: the CPU backend does not support ggml_argmax_bf16\n");
        return 1;
    }

    int failures = 0;
    const int64_t widths[] = {131072, 4097, 1000, 70, 3};  // the vocabulary, and under one block
    for (int64_t cols : widths) {
        const int64_t rows = cols >= 70 ? 9 : 2;
        const std::vector<float> x = make_rows(cols, rows);
        std::vector<int32_t> want((size_t)rows);
        for (int64_t r = 0; r < rows; ++r)
            want[(size_t)r] = greedy_pick(x.data() + r * cols, (int)cols);
        for (int b = 0; b < 2; ++b) {
            if (!backends[b])
                continue;
            const std::vector<int32_t> got = run(backends[b], x, cols, rows, true);
            for (int64_t r = 0; r < rows; ++r) {
                if (got[(size_t)r] != want[(size_t)r]) {
                    std::fprintf(
                        stderr,
                        "FAIL: %s, %lld columns, row %lld: ggml_argmax_bf16 %d, greedy_pick %d\n",
                        names[b], (long long)cols, (long long)r, got[(size_t)r], want[(size_t)r]);
                    ++failures;
                }
            }
        }
        if (cols >= 70) {  // row 0 does tell the two rules apart
            const std::vector<int32_t> plain = run(backends[0], x, cols, rows, false);
            if (plain[0] == want[0]) {
                std::fprintf(
                    stderr, "FAIL: %lld columns: row 0 does not need the rounding\n",
                    (long long)cols);
                ++failures;
            }
        }
    }

    for (ggml_backend_t backend : backends)
        if (backend)
            ggml_backend_free(backend);
    if (failures)
        return 1;
    std::fprintf(stderr, "OK: ggml_argmax_bf16 picks the greedy_pick column on every backend\n");
    return 0;
}
