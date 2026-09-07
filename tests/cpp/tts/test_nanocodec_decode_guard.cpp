// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "tts/nanocodec/model.h"

namespace nc = nemo_speech::tts::nanocodec;

namespace {

int failures = 0;

void
expect(const char* name, const std::vector<float>& audio, bool want_finite) {
    if (nc::is_finite_audio(audio) != want_finite) {
        std::fprintf(stderr, "%s: expected %s\n", name, want_finite ? "acceptance" : "rejection");
        ++failures;
    }
}

// Runs one sample buffer through a graph tail shaped like the decoder's, with
// and without the clamp this change removes, and returns what the host reads
// back. Forces the CPU backend so the result is the same on every platform.
bool
graph_tail(const std::vector<float>& input, bool with_clamp, std::vector<float>& output) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "graph tail: CPU backend init failed\n");
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "graph tail: context allocation failed\n");
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor* inp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)input.size());
    ggml_set_input(inp);
    ggml_tensor* x = ggml_scale(ctx, inp, 1.0f);
    if (with_clamp) {
        x = ggml_clamp(ctx, x, -1.0f, 1.0f);
    }
    ggml_set_output(x);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(gf, x);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "graph tail: tensor allocation failed\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(inp, input.data(), 0, input.size() * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status == GGML_STATUS_SUCCESS) {
        output.resize((size_t)ggml_nelements(x));
        ggml_backend_tensor_get(x, output.data(), 0, output.size() * sizeof(float));
    } else {
        std::fprintf(stderr, "graph tail: compute failed (%d)\n", (int)status);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return status == GGML_STATUS_SUCCESS;
}

}  // namespace

int
main() {
    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    const float inf_value = std::numeric_limits<float>::infinity();

    expect("speech", {0.0f, 0.31f, -0.22f, 0.75f, -0.61f}, true);
    expect("digital silence", std::vector<float>(4096, 0.0f), true);
    expect("empty", {}, true);

    // A decode that legitimately reaches full scale is audio, not a fault. The
    // guard the writer would need if the clamp stayed could not tell them apart.
    expect("uniform negative full scale", std::vector<float>(1024, -1.0f), true);
    expect("uniform positive full scale", std::vector<float>(1024, 1.0f), true);
    expect("beyond full scale", {-1.7f, 2.4f, 0.1f}, true);

    expect("nan", {0.2f, nan_value, 0.4f}, false);
    expect("nan at end", {0.2f, 0.4f, nan_value}, false);
    expect("all nan", std::vector<float>(64, nan_value), false);
    expect("positive infinity", {0.2f, inf_value}, false);
    expect("negative infinity", {0.2f, -inf_value}, false);

    // Why the clamp had to go. Terminating the decoder graph in
    // ggml_clamp(-1, 1) folds a NaN onto a bound, so the buffer the host reads
    // back is finite and no downstream isfinite check can ever see the fault.
    // Without it the NaN survives the graph and this guard catches it.
    {
        std::vector<float> input = {0.2f, nan_value, -0.4f, 0.9f};
        std::vector<float> clamped;
        std::vector<float> unclamped;
        if (!graph_tail(input, true, clamped) || !graph_tail(input, false, unclamped)) {
            ++failures;
        } else {
            if (!nc::is_finite_audio(clamped)) {
                std::fprintf(stderr, "clamped graph tail: expected the NaN to be erased\n");
                ++failures;
            }
            if (nc::is_finite_audio(unclamped)) {
                std::fprintf(stderr, "unclamped graph tail: expected the NaN to survive\n");
                ++failures;
            }
        }
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d decode guard case(s) failed\n", failures);
        return 1;
    }
    return 0;
}
