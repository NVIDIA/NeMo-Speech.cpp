// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "runtime/ggml/runtime.h"
#include "tts/magpietts/model.h"

namespace {

constexpr int kHeadDim = 64;
constexpr int kHeads = 12;
constexpr int kBatch = 2;
constexpr int kCacheLength = 18;
constexpr int kFeatures = kHeadDim * kHeads;

size_t
qkv_index(int d, int head, int batch) {
    return static_cast<size_t>(d + kHeadDim * (head + kHeads * batch));
}

size_t
cache_index(int d, int head, int position, int slot, int plane) {
    return static_cast<size_t>(
        d + kHeadDim * head + kFeatures * position + kFeatures * kCacheLength * slot +
        kFeatures * kCacheLength * kBatch * plane);
}

std::vector<float>
reference_attention(
    const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
    const std::vector<float>& cache, const int32_t* ring_heads, const int32_t* active_lengths) {
    std::vector<float> output(q.size());
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    for (int batch = 0; batch < kBatch; ++batch) {
        const int key_begin = kCacheLength - active_lengths[batch];
        for (int head = 0; head < kHeads; ++head) {
            std::vector<float> scores;
            scores.reserve(static_cast<size_t>(active_lengths[batch] + 1));
            float maximum = -INFINITY;
            for (int logical = key_begin; logical <= kCacheLength; ++logical) {
                float score = 0.0f;
                for (int d = 0; d < kHeadDim; ++d) {
                    const float key =
                        logical == kCacheLength
                            ? k[qkv_index(d, head, batch)]
                            : cache[cache_index(
                                  d, head, (ring_heads[batch] + logical) % kCacheLength, batch, 0)];
                    score += q[qkv_index(d, head, batch)] * key;
                }
                score *= scale;
                scores.push_back(score);
                maximum = std::max(maximum, score);
            }

            float denominator = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (int d = 0; d < kHeadDim; ++d) {
                float context = 0.0f;
                for (int logical = key_begin; logical <= kCacheLength; ++logical) {
                    const size_t score_index = static_cast<size_t>(logical - key_begin);
                    const float value =
                        logical == kCacheLength
                            ? v[qkv_index(d, head, batch)]
                            : cache[cache_index(
                                  d, head, (ring_heads[batch] + logical) % kCacheLength, batch, 1)];
                    context += scores[score_index] * value;
                }
                output[qkv_index(d, head, batch)] = context / denominator;
            }
        }
    }
    return output;
}

}  // namespace

int
main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "SKIP: CUDA backend unavailable\n");
        return 0;
    }

    {
        ggml_runtime::Params params;
        params.use_gpu = false;
        ggml_runtime::BackendManager manager(params, backend);
        const auto backends = manager.get_backends();
        const auto buffer_types = manager.get_buft_list();
        if (manager.gpu_backend_handle() != backend ||
            std::find(backends.begin(), backends.end(), backend) == backends.end() ||
            std::none_of(
                buffer_types.begin(), buffer_types.end(),
                [backend](const auto& item) {
                    return item.first == ggml_backend_get_device(backend);
                })) {
            std::fprintf(stderr, "FAIL: borrowed CUDA backend was not registered\n");
            ggml_backend_free(backend);
            return 1;
        }
    }

    nemo_speech::tts::magpietts_model backend_model;
    backend_model.backend = backend;
    bool use_cuda_sampling = false;
    if (!nemo_speech::tts::magpietts_resolve_sampling_backend(
            backend_model, nemo_speech::tts::MAGPIETTS_BACKEND_AUTO, use_cuda_sampling) ||
        !use_cuda_sampling) {
        std::fprintf(stderr, "FAIL: automatic sampling did not select CUDA\n");
        backend_model.backend = nullptr;
        ggml_backend_free(backend);
        return 1;
    }
    backend_model.backend = nullptr;

    const size_t tensor_count = 7;
    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * tensor_count + ggml_graph_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: ggml context allocation\n");
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, 1, kHeads, kBatch);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, 1, kHeads, kBatch);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, 1, kHeads, kBatch);
    ggml_tensor* cache =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kFeatures * kCacheLength, kBatch, 2);
    ggml_tensor* slot_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kBatch);
    ggml_tensor* cache_state = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kBatch, 2);
    ggml_tensor* output = ggml_fused_attn_cached(
        ctx, q, k, v, nullptr, cache, slot_ids, cache_state, kCacheLength,
        1.0f / std::sqrt(static_cast<float>(kHeadDim)), true);
    ggml_set_output(output);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "FAIL: CUDA tensor allocation\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> q_data(ggml_nelements(q));
    std::vector<float> k_data(ggml_nelements(k));
    std::vector<float> v_data(ggml_nelements(v));
    std::vector<float> cache_data(ggml_nelements(cache));
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = 0.25f * std::sin(0.013f * static_cast<float>(i + 1));
        k_data[i] = 0.30f * std::cos(0.017f * static_cast<float>(i + 3));
        v_data[i] = 0.35f * std::sin(0.019f * static_cast<float>(i + 5));
    }
    for (size_t i = 0; i < cache_data.size(); ++i) {
        cache_data[i] = 0.40f * std::cos(0.007f * static_cast<float>(i + 7));
    }
    const int32_t slots[kBatch] = {0, 1};
    const int32_t state[kBatch * 2] = {0, 11, 0, 7};
    const std::vector<float> reference =
        reference_attention(q_data, k_data, v_data, cache_data, state, state + kBatch);

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));
    ggml_backend_tensor_set(cache, cache_data.data(), 0, cache_data.size() * sizeof(float));
    ggml_backend_tensor_set(slot_ids, slots, 0, sizeof(slots));
    ggml_backend_tensor_set(cache_state, state, 0, sizeof(state));

    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: CUDA graph compute (%d)\n", static_cast<int>(status));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> actual(reference.size());
    std::vector<float> updated_cache(cache_data.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    ggml_backend_tensor_get(cache, updated_cache.data(), 0, updated_cache.size() * sizeof(float));

    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_error = std::max(max_error, std::fabs(actual[i] - reference[i]));
    }
    bool cache_ok = true;
    for (int batch = 0; batch < kBatch; ++batch) {
        for (int head = 0; head < kHeads; ++head) {
            for (int d = 0; d < kHeadDim; ++d) {
                const size_t source = qkv_index(d, head, batch);
                cache_ok &=
                    updated_cache[cache_index(d, head, state[batch], batch, 0)] == k_data[source];
                cache_ok &=
                    updated_cache[cache_index(d, head, state[batch], batch, 1)] == v_data[source];
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (max_error > 2.0e-5f) {
        std::fprintf(stderr, "FAIL: cached attention max error %.8g\n", max_error);
        return 1;
    }
    if (!cache_ok) {
        std::fprintf(stderr, "FAIL: circular cache update mismatch\n");
        return 1;
    }
    std::printf("cached attention max error %.8g\n", max_error);
    return 0;
}
