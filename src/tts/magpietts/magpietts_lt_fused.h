// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Fused CUDA implementation of one MagpieTTS local-transformer round (CFG pair lanes):
// input assembly + LayerNorm + GEMV kernels with fused epilogues, a tiny cached attention
// kernel, and the per-codebook output projection. Replaces the per-round ggml graph
// (~26 kernels plus views/casts) with 11 kernel launches per round.
#pragma once

#include <cstddef>
#include <cstdint>

struct magpietts_lt_fused;

struct magpietts_lt_fused_layer_weights {
    const void* qkv = nullptr;  // [n_embd, 3*n_embd] (ggml: ne0 = n_embd)
    int qkv_type = 0;
    const void* o = nullptr;  // [n_embd, n_embd]
    int o_type = 0;
    const void* ff1 = nullptr;  // [n_embd, n_ff]
    int ff1_type = 0;
    const void* ff2 = nullptr;  // [n_ff, n_embd]
    int ff2_type = 0;
    const float* norm_self = nullptr;  // [n_embd] F32
    const float* norm_ff = nullptr;    // [n_embd] F32
};

struct magpietts_lt_fused_weights {
    int n_embd = 0;
    int n_head = 0;
    int n_ff = 0;
    int n_layers = 0;
    int vocab = 0;     // audio vocab (logits width)
    int n_rounds = 0;  // stacked codebooks
    int pos_rows = 0;  // local-transformer context length (rows of pos_emb)
    float ln_eps = 1.0e-5f;
    const magpietts_lt_fused_layer_weights* layers = nullptr;
    const void* pos_emb = nullptr;  // [n_embd, pos_rows]
    int pos_type = 0;
    const void* const* audio_emb = nullptr;  // n_rounds tables [n_embd, vocab]
    int audio_emb_type = 0;
    const void* const* out_w = nullptr;  // n_rounds [n_embd, vocab]
    int out_type = 0;
    const float* const* out_b = nullptr;  // n_rounds [vocab] F32 (may contain nullptrs)
};

// Returns nullptr (with error text) when the weight layout is unsupported.
magpietts_lt_fused* magpietts_lt_fused_create(
    const magpietts_lt_fused_weights& weights, char* error, size_t error_size);
void magpietts_lt_fused_free(magpietts_lt_fused* fused);

// Device buffers receiving the round-0 inputs (decoder hidden states), n_embd floats each.
float* magpietts_lt_fused_input_cond(magpietts_lt_fused* fused);
float* magpietts_lt_fused_input_uncond(magpietts_lt_fused* fused);

// Optional in-kernel sampling: when set, the output projection's last-finishing block runs the
// exact top-k sampler on the logits (top_k <= 256) and writes codes[c] / argmax[c].
struct magpietts_lt_fused_sampler {
    const void* config = nullptr;  // device magpietts_cuda_sampling_config
    int32_t* codes = nullptr;      // [n_rounds]
    int32_t* argmax = nullptr;     // [n_rounds]
    int32_t* top_ids = nullptr;    // scratch [vocab]
    float* top_vals = nullptr;     // scratch [vocab]
    int audio_codebook_size = 0;
    int audio_eos_id = 0;
};

// Enqueue the kernels for round `c` on `stream`. `codes_device` holds the sampled codes of
// the current frame (index c-1 is read for c > 0). On return the logits pointers refer to
// device F32 buffers [vocab] for the conditional and unconditional lanes.
bool magpietts_lt_fused_round(
    magpietts_lt_fused* fused, int c, const int32_t* codes_device, void* stream,
    const float** logits_cond, const float** logits_uncond, char* error, size_t error_size,
    const magpietts_lt_fused_sampler* sampler = nullptr);

// Capture the kernels of round `c` into a CUDA graph (returned as an opaque cudaGraph_t)
// suitable for cudaGraphAddChildGraphNode. The caller owns the returned graph.
bool magpietts_lt_fused_capture_round(
    magpietts_lt_fused* fused, int c, const int32_t* codes_device, void** graph_out,
    const float** logits_cond, const float** logits_uncond, char* error, size_t error_size,
    const magpietts_lt_fused_sampler* sampler = nullptr);
void magpietts_lt_fused_destroy_graph(void* graph);

// True when the output projection kernel can run the exact top-k sampler itself (planar Q8
// output weights and a vocab within the fast-path limit). Otherwise the caller must sample.
bool magpietts_lt_fused_sampling_supported(const magpietts_lt_fused* fused);

// Persistent chain: all rounds of the frame (input assembly, both layers, output projection and
// exact top-k sampling per codebook) in a single kernel launch with software grid barriers.
// Requires planar Q8 weights for every projection and an in-kernel sampler (top_k <= 256).
bool magpietts_lt_fused_chain_supported(const magpietts_lt_fused* fused);
bool magpietts_lt_fused_chain(
    magpietts_lt_fused* fused, void* stream, const magpietts_lt_fused_sampler* sampler, char* error,
    size_t error_size);
bool magpietts_lt_fused_capture_chain(
    magpietts_lt_fused* fused, void** graph_out, const magpietts_lt_fused_sampler* sampler,
    char* error, size_t error_size);
