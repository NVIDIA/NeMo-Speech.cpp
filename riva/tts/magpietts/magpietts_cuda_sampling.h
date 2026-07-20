// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

struct magpietts_cuda_sampler;

magpietts_cuda_sampler* magpietts_cuda_sampler_create(int codebooks);
void magpietts_cuda_sampler_free(magpietts_cuda_sampler* sampler);
bool magpietts_cuda_device_is_uma(void);

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

bool magpietts_cuda_copy_sampled_code_to_device(
    magpietts_cuda_sampler* sampler, int codebook, void* dst_device, char* error,
    size_t error_size);

bool magpietts_cuda_copy_sampled_codebooks(
    magpietts_cuda_sampler* sampler, int codebooks, int32_t* codes_out, int32_t* argmax_out,
    char* error, size_t error_size);
