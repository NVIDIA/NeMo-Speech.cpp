// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Persistent (single-launch) CUDA kernel for one MagpieTTS decoder step: audio-code embedding,
// all decoder layers (self-attention over the per-lane ring-buffer KV cache, text
// cross-attention with additive log-prior/padding bias, FFN), the final LayerNorm and the mean
// cross-attention alignment. Two CFG lanes (conditional, unconditional). Requires Q8_0
// projections; the caller keeps the ggml graph path as the fallback.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

struct magpietts_decoder_fused;

struct magpietts_decoder_fused_layer_weights {
    const void* qkv = nullptr;  // ggml Q8_0 [n_embd, 3*n_embd]
    int qkv_type = 0;
    const void* o = nullptr;  // [n_embd, n_embd]
    int o_type = 0;
    const void* cross_q = nullptr;  // [n_embd, cross_dim]
    int cross_q_type = 0;
    const void* cross_o = nullptr;  // [cross_dim, n_embd]
    int cross_o_type = 0;
    const void* ff1 = nullptr;  // [n_embd, n_ff]
    int ff1_type = 0;
    const void* ff2 = nullptr;  // [n_ff, n_embd]
    int ff2_type = 0;
    const float* norm_self = nullptr;  // [n_embd] F32
    const float* norm_xq = nullptr;    // [n_embd] F32 (cross-attention query norm)
    const float* norm_ff = nullptr;    // [n_embd] F32
    bool has_cross = true;
    bool collect_alignment = false;  // contributes to the mean alignment output
    bool apply_prior = false;        // softmax bias = log prior (+ pad mask) instead of pad mask
};

struct magpietts_decoder_fused_weights {
    int n_embd = 0;
    int n_head = 0;
    int n_ff = 0;
    int cross_dim = 0;  // n_cross_head * n_cross_dhead (single head)
    int n_layers = 0;
    int n_codebooks = 0;  // stacked audio codebooks summed into the input embedding
    float ln_eps = 1e-5f;
    const void* const* audio_emb = nullptr;  // n_codebooks tables [vocab][n_embd]
    int emb_type = 0;                        // ggml type id: F32 = 0, F16 = 1
    const void* pos_emb = nullptr;           // [n_ctx][n_embd]
    int pos_type = 0;
    const float* norm_out = nullptr;  // [n_embd] F32
    const magpietts_decoder_fused_layer_weights* layers = nullptr;
};

struct magpietts_decoder_fused_cache {
    int cache_len = 0;                 // self-attention rows per lane
    float* const* kv_arena = nullptr;  // per layer: [2 planes][2 lanes][cache_len][n_embd] F32
    int text_capacity = 0;             // cross-attention rows per layer (padded text length)
    const float* cross_k = nullptr;    // [n_layers][text_capacity][cross_dim] F32
    const float* cross_v = nullptr;
};

struct magpietts_decoder_fused_step_args {
    const int32_t* tokens = nullptr;  // device [n_codebooks]
    int position = 0;
    int ring_head = 0;  // oldest physical cache row before this step (the new row's slot)
    int valid_len = 0;  // valid cache rows before this step
    const float* prior = nullptr;  // device [text_capacity]: log prior on real rows, -1e30 on pads
    const float* mask = nullptr;   // device [text_capacity]: 0 on real rows, -1e30 on pads
    float* hidden_cond = nullptr;  // device [n_embd] outputs after the final norm
    float* hidden_uncond = nullptr;  // device [n_embd]
    float* alignment = nullptr;      // device [text_capacity] mean attention over collect layers
};

magpietts_decoder_fused* magpietts_decoder_fused_create(
    const magpietts_decoder_fused_weights& w, const magpietts_decoder_fused_cache& cache,
    char* error, size_t error_size);
void magpietts_decoder_fused_free(magpietts_decoder_fused* f);
bool magpietts_decoder_fused_supported(const magpietts_decoder_fused* f);
int magpietts_decoder_fused_grid(const magpietts_decoder_fused* f);
bool magpietts_decoder_fused_step(
    magpietts_decoder_fused* f, cudaStream_t stream, const magpietts_decoder_fused_step_args& step,
    char* error, size_t error_size);
