// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Regression test for the half_snake CUDA fusion's aliasing guard.
//
// The fusion collapses the seven nodes NanoCodec's activation emits -- mul, sin,
// sqr, mul, add, leaky_relu, concat -- into one kernel that reads the activation
// input through two views and writes the concat output. The graph allocator
// plans buffers for the *unfused* sequence, in which the input's last reader is
// the leaky_relu, so the input's memory is free to be recycled for the concat
// output. The fused kernel still reads the input while writing the concat, so
// where the allocator overlapped them at a shifted offset, threads clobbered
// input other threads had not read yet. It cost NanoCodec ~26 dB of SNR against
// a CPU reference, and made the same codes decode differently on every run.
//
// Two things make this awkward to test, and shape what is below.
//
// The arithmetic is correct at every shape in isolation, so a kernel-level unit
// test cannot see the bug -- it only appears once a real graph puts the output
// on top of the input. And the *safe* overlap, where the concat exactly covers
// its two input views end to end, is what a stock allocator produces for this
// shape: every thread then reads and writes one address, which the guard
// deliberately permits. A test that lets the allocator choose therefore passes
// with the guard removed, which is worse than no test.
//
// So this test places the buffers itself: the concat output is put at a
// deliberate byte offset inside the activation input, which is neither disjoint
// nor an exact alias, and is exactly the case the guard must reject. The
// unfused path is unaffected by that placement -- it materialises the snake and
// leaky-relu halves into their own buffers first -- so the reference stays
// valid.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace {

constexpr int64_t kLen = 512;
constexpr int64_t kChannels = 64;
constexpr int64_t kSnakeChannels = 32;  // half snake, half leaky relu
constexpr float kLeakySlope = 0.01f;
// Any non-zero shift smaller than the tensor gives a partial overlap. 1 KiB is
// comfortably past the alignment the backend would round to.
constexpr size_t kOverlapShift = 1024;

size_t
align_up(size_t v, size_t a) {
    return (v + a - 1) / a * a;
}

}  // namespace

int
main() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        std::fprintf(stderr, "no GPU backend available; skipping\n");
        return 77;  // ctest SKIP_RETURN_CODE
    }

    const size_t n = (size_t)kLen * (size_t)kChannels;
    const size_t tensor_bytes = n * sizeof(float);

    std::vector<float> host_in(n), host_alpha(kSnakeChannels), host_alpha_inv(kSnakeChannels);
    // Straddle zero so the leaky-relu half exercises both branches: a clobbered
    // input then shows up as a sign error, not a rounding one.
    for (size_t i = 0; i < n; ++i) {
        host_in[i] = std::sin((float)i * 0.037f) * 3.0f;
    }
    for (int64_t c = 0; c < kSnakeChannels; ++c) {
        host_alpha[(size_t)c] = 0.5f + 0.01f * (float)c;
        host_alpha_inv[(size_t)c] = 1.0f / host_alpha[(size_t)c];
    }

    ggml_init_params params{ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: could not create a graph context\n");
        return 1;
    }

    // One buffer, placed by hand. Layout: the activation input at 0, the concat
    // output shifted into it, then everything else well clear of both.
    const size_t align = 256;
    const size_t scratch_base = align_up(tensor_bytes + kOverlapShift, align);
    const size_t buf_bytes = scratch_base + 8 * align_up(tensor_bytes, align);
    ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(backend, buf_bytes);
    if (!buf) {
        std::fprintf(stderr, "FAIL: could not allocate a %zu byte backend buffer\n", buf_bytes);
        return 1;
    }
    char* base = (char*)ggml_backend_buffer_get_base(buf);

    size_t scratch = scratch_base;
    auto place = [&](ggml_tensor* t, char* at) {
        t->buffer = buf;
        t->data = at;
    };
    auto place_scratch = [&](ggml_tensor* t) {
        place(t, base + scratch);
        scratch += align_up(ggml_nbytes(t), align);
    };

    // The activation input. Its views resolve their pointers from it at
    // creation, so it has to be placed before they are built.
    ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kLen, kChannels, 1);
    ggml_set_name(x, "activation_input");
    place(x, base);

    ggml_tensor* alpha = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kSnakeChannels, 1);
    ggml_tensor* alpha_inv = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kSnakeChannels, 1);
    place_scratch(alpha);
    place_scratch(alpha_inv);

    ggml_tensor* x_snake = ggml_view_3d(ctx, x, kLen, kSnakeChannels, 1, x->nb[1], x->nb[2], 0);
    ggml_tensor* x_lrelu = ggml_view_3d(
        ctx, x, kLen, kChannels - kSnakeChannels, 1, x->nb[1], x->nb[2], kSnakeChannels * x->nb[1]);

    ggml_tensor* ax = ggml_mul(ctx, x_snake, alpha);
    ggml_tensor* periodic = ggml_mul(ctx, ggml_sqr(ctx, ggml_sin(ctx, ax)), alpha_inv);
    ggml_tensor* snake_out = ggml_add(ctx, x_snake, periodic);
    ggml_tensor* lrelu_out = ggml_leaky_relu(ctx, x_lrelu, kLeakySlope, false);
    ggml_tensor* out = ggml_concat(ctx, snake_out, lrelu_out, 1);
    ggml_set_name(out, "half_snake_out");
    ggml_set_output(out);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // Intermediates get their own space; only the concat is placed on top of the
    // input, which is the condition under test.
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor* node = ggml_graph_node(gf, i);
        if (node->data != nullptr) {
            continue;
        }
        if (node == out) {
            place(node, base + kOverlapShift);
        } else {
            place_scratch(node);
        }
    }

    const char* in_beg = (const char*)x->data;
    const char* in_end = in_beg + ggml_nbytes(x);
    const char* out_beg = (const char*)out->data;
    const char* out_end = out_beg + ggml_nbytes(out);
    const bool overlaps = in_beg < out_end && out_beg < in_end;
    const bool exact = out_beg == in_beg && out_end == in_end;
    if (!overlaps || exact) {
        std::fprintf(
            stderr,
            "FAIL: the test did not set up a shifted overlap (overlaps=%d exact=%d); "
            "it would not exercise the guard\n",
            (int)overlaps, (int)exact);
        return 1;
    }

    ggml_backend_tensor_set(x, host_in.data(), 0, tensor_bytes);
    ggml_backend_tensor_set(alpha, host_alpha.data(), 0, host_alpha.size() * sizeof(float));
    ggml_backend_tensor_set(
        alpha_inv, host_alpha_inv.data(), 0, host_alpha_inv.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: graph compute failed\n");
        return 1;
    }

    std::vector<float> got(n);
    ggml_backend_tensor_get(out, got.data(), 0, tensor_bytes);

    double worst = 0.0;
    size_t worst_at = 0;
    for (int64_t c = 0; c < kChannels; ++c) {
        for (int64_t i = 0; i < kLen; ++i) {
            const size_t idx = (size_t)c * kLen + (size_t)i;
            const double v = host_in[idx];
            double want;
            if (c < kSnakeChannels) {
                const double s = std::sin(v * host_alpha[(size_t)c]);
                want = v + s * s * host_alpha_inv[(size_t)c];
            } else {
                want = v >= 0.0 ? v : v * (double)kLeakySlope;
            }
            const double d = std::fabs((double)got[idx] - want);
            if (d > worst) {
                worst = d;
                worst_at = idx;
            }
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    // Fused and unfused differ only in float ordering; a clobbered input is off
    // by whole units, not ulps.
    if (worst >= 1.0e-4) {
        std::fprintf(
            stderr,
            "FAIL: half_snake output is wrong by %.6g at index %zu with the concat "
            "output shifted %zu bytes into its input. The fusion ran on a partial "
            "overlap, which it must refuse.\n",
            worst, worst_at, kOverlapShift);
        return 1;
    }
    std::fprintf(
        stderr,
        "OK: half_snake correct (max diff %.3g) with the output shifted %zu bytes "
        "into its input\n",
        worst, kOverlapShift);
    return 0;
}
