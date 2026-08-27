// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

struct magpietts_cuda_sampler;

magpietts_cuda_sampler* magpietts_cuda_sampler_create(int codebooks);
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
