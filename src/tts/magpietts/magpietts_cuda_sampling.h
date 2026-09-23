// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

struct magpietts_cuda_sampler;

magpietts_cuda_sampler* magpietts_cuda_sampler_create(int codebooks);
// Device array of the current frame's sampled codes (one int32 per stacked codebook).
const int32_t* magpietts_cuda_sampler_codes_device(const magpietts_cuda_sampler* sampler);
// Raw device buffers of the sampler for kernels that sample in-place (fused local transformer).
struct magpietts_cuda_sampler_device_pointers_t {
    const void* config = nullptr;  // magpietts_cuda_sampling_config on the device
    int32_t* codes = nullptr;      // [codebooks]
    int32_t* argmax = nullptr;     // [codebooks]
    int32_t* top_ids = nullptr;    // [codebooks][MAX_VOCAB] scratch
    float* top_vals = nullptr;     // [codebooks][MAX_VOCAB] scratch
    int top_k = 0;                 // host mirror of the configured top-k
};
bool magpietts_cuda_sampler_device_pointers(
    const magpietts_cuda_sampler* sampler, magpietts_cuda_sampler_device_pointers_t* out);
void magpietts_cuda_sampler_free(magpietts_cuda_sampler* sampler);
bool magpietts_cuda_device_is_uma(void);

// Bind to a caller-owned stream.
bool magpietts_cuda_sampler_bind_stream(
    magpietts_cuda_sampler* sampler, void* stream, char* error, size_t error_size);

// Configure per-frame values stored in a stable device buffer.
bool magpietts_cuda_sampler_configure(
    magpietts_cuda_sampler* sampler, bool use_cfg, float cfg_scale, float temperature, int top_k,
    bool forbid_audio_eos, uint64_t seed, int frame_index, char* error, size_t error_size);
bool magpietts_cuda_sampler_upload_config(
    magpietts_cuda_sampler* sampler, char* error, size_t error_size);

// Compose and launch the local-transformer sequence as a CUDA graph.
bool magpietts_cuda_sampler_sequence_is_warm(const magpietts_cuda_sampler* sampler);
bool magpietts_cuda_sampler_sequence_is_ready(const magpietts_cuda_sampler* sampler);
bool magpietts_cuda_sampler_sequence_is_disabled(const magpietts_cuda_sampler* sampler);
bool magpietts_cuda_sampler_sequence_build_active(const magpietts_cuda_sampler* sampler);
void magpietts_cuda_sampler_sequence_mark_warm(magpietts_cuda_sampler* sampler);
bool magpietts_cuda_sampler_sequence_begin_build(
    magpietts_cuda_sampler* sampler, char* error, size_t error_size);
bool magpietts_cuda_sampler_sequence_finish_build_and_launch(
    magpietts_cuda_sampler* sampler, char* error, size_t error_size);
void magpietts_cuda_sampler_sequence_abort_build(magpietts_cuda_sampler* sampler);
void magpietts_cuda_sampler_sequence_disable(magpietts_cuda_sampler* sampler);
// The composed sequence bakes in which kernels the caller enqueued (fused vs ggml local
// transformer, in-kernel vs separate sampling, CFG pair vs single). Callers tag it with a key
// describing that structure; reset drops the warm/composed state so a different structure is
// composed afresh instead of replaying the old graph.
uint64_t magpietts_cuda_sampler_sequence_key(const magpietts_cuda_sampler* sampler);
void magpietts_cuda_sampler_sequence_set_key(magpietts_cuda_sampler* sampler, uint64_t key);
void magpietts_cuda_sampler_sequence_reset(magpietts_cuda_sampler* sampler);
bool magpietts_cuda_sampler_sequence_launch(
    magpietts_cuda_sampler* sampler, char* error, size_t error_size);
bool magpietts_cuda_sampler_sequence_add_ggml_graph(
    magpietts_cuda_sampler* sampler, void* graph_template, char* error, size_t error_size);
bool magpietts_cuda_sampler_sequence_add_device_copy(
    magpietts_cuda_sampler* sampler, const void* src_device, void* dst_device, size_t bytes,
    char* error, size_t error_size);

bool magpietts_cuda_sample_codebooks(
    magpietts_cuda_sampler* sampler, const float* logits_cond, const float* logits_uncond,
    int codebooks, int vocab_size, int audio_codebook_size, int audio_eos_id, bool use_cfg,
    float cfg_scale, float temperature, int top_k, bool forbid_audio_eos, uint64_t seed,
    int frame_index, int codebook_offset, int32_t* codes_out, int32_t* argmax_out, char* error,
    size_t error_size);

bool magpietts_cuda_sample_codebooks_device(
    magpietts_cuda_sampler* sampler, const float* logits_cond, const float* logits_uncond,
    int codebooks, int vocab_size, int audio_codebook_size, int audio_eos_id, bool use_cfg,
    float cfg_scale, float temperature, int top_k, bool forbid_audio_eos, uint64_t seed,
    int frame_index, int codebook_offset, int output_offset, char* error, size_t error_size);

// Launch using the most recently uploaded configuration.
bool magpietts_cuda_sample_codebooks_device_configured(
    magpietts_cuda_sampler* sampler, const float* logits_cond, const float* logits_uncond,
    int codebooks, int vocab_size, int audio_codebook_size, int audio_eos_id, int codebook_offset,
    int output_offset, char* error, size_t error_size);

bool magpietts_cuda_copy_sampled_code_to_device(
    magpietts_cuda_sampler* sampler, int codebook, void* dst_device, char* error,
    size_t error_size);

bool magpietts_cuda_copy_sampled_codebooks(
    magpietts_cuda_sampler* sampler, int codebooks, int32_t* codes_out, int32_t* argmax_out,
    char* error, size_t error_size);
