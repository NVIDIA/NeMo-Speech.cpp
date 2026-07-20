// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>

#include "magpietts_cuda_sampling.h"

static constexpr int MAGPIETTS_CUDA_MAX_VOCAB = 4096;
static constexpr int MAGPIETTS_CUDA_BLOCK_SIZE = 256;

struct magpietts_cuda_sampler {
    int codebooks = 0;
    cudaStream_t stream = nullptr;
    int32_t* d_codes = nullptr;
    int32_t* d_argmax = nullptr;
    int32_t* d_top_ids = nullptr;
    float* d_top_vals = nullptr;
};

bool
magpietts_cuda_device_is_uma(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        (void)cudaGetLastError();
        return false;
    }

    int device = 0;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess || device < 0 || device >= count) {
        (void)cudaGetLastError();
        device = 0;
    }

    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return prop.integrated != 0 && prop.unifiedAddressing != 0;
}

static void
set_error(char* error, size_t error_size, const char* message, cudaError_t err = cudaSuccess) {
    if (!error || error_size == 0) {
        return;
    }
    if (err == cudaSuccess) {
        snprintf(error, error_size, "%s", message);
    } else {
        snprintf(error, error_size, "%s: %s", message, cudaGetErrorString(err));
    }
}

static __device__ __forceinline__ uint64_t
splitmix64_next(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static __device__ __forceinline__ double
uniform01(uint64_t seed, int frame_index, int codebook) {
    uint64_t state = seed ^ (0xd1b54a32d192ed03ULL * (uint64_t)(frame_index + 1)) ^
                     (0xabc98388fb8fac03ULL * (uint64_t)(codebook + 1));
    const uint64_t r = splitmix64_next(state);
    return (double)(r >> 11) * 0x1.0p-53;
}

static __device__ __forceinline__ bool
forbidden_token(int id, int audio_codebook_size, int audio_eos_id, bool forbid_audio_eos) {
    const int base = audio_codebook_size;
    if (id == base + 0 || id == base + 2 || id == base + 3 || id == base + 4 || id == base + 5 ||
        id == base + 6 || id == base + 7) {
        return true;
    }
    return forbid_audio_eos && id == audio_eos_id;
}

static __device__ __forceinline__ float
sampled_logit(
    const float* logits_cond, const float* logits_uncond, int off, int id, int audio_codebook_size,
    int audio_eos_id, bool use_cfg, float cfg_scale, bool forbid_audio_eos) {
    float logit = logits_cond[off + id];
    if (use_cfg && logits_uncond) {
        logit = cfg_scale * logit + (1.0f - cfg_scale) * logits_uncond[off + id];
    }
    if (forbidden_token(id, audio_codebook_size, audio_eos_id, forbid_audio_eos)) {
        logit = -INFINITY;
    }
    return logit;
}

static __device__ __forceinline__ bool
better_logit(int lhs_id, float lhs, int rhs_id, float rhs) {
    return lhs > rhs || (lhs == rhs && lhs_id < rhs_id);
}

__global__ void
magpietts_sample_codebooks_kernel(
    const float* logits_cond, const float* logits_uncond, int codebooks, int vocab_size,
    int audio_codebook_size, int audio_eos_id, bool use_cfg, float cfg_scale, float temperature,
    int top_k, bool forbid_audio_eos, uint64_t seed, int frame_index, int codebook_offset,
    int output_offset, int32_t* top_ids_scratch, float* top_vals_scratch, int32_t* codes_out,
    int32_t* argmax_out) {
    const int c = blockIdx.x;
    if (c >= codebooks) {
        return;
    }

    __shared__ float s_vals[MAGPIETTS_CUDA_BLOCK_SIZE];
    __shared__ int s_ids[MAGPIETTS_CUDA_BLOCK_SIZE];
    __shared__ double s_sums[MAGPIETTS_CUDA_BLOCK_SIZE];

    int k = top_k < vocab_size ? top_k : vocab_size;
    if (k < 1) {
        k = 1;
    }

    const int off = c * vocab_size;
    int32_t* top_ids = top_ids_scratch + (size_t)c * vocab_size;
    float* top_vals = top_vals_scratch + (size_t)c * vocab_size;

    for (int rank = 0; rank < k; ++rank) {
        float local_val = -INFINITY;
        int local_id = vocab_size;

        for (int id = threadIdx.x; id < vocab_size; id += blockDim.x) {
            bool selected = false;
            for (int prev = 0; prev < rank; ++prev) {
                selected = selected || id == top_ids[prev];
            }
            if (selected) {
                continue;
            }
            const float logit = sampled_logit(
                logits_cond, logits_uncond, off, id, audio_codebook_size, audio_eos_id, use_cfg,
                cfg_scale, forbid_audio_eos);
            if (better_logit(id, logit, local_id, local_val)) {
                local_val = logit;
                local_id = id;
            }
        }

        s_vals[threadIdx.x] = local_val;
        s_ids[threadIdx.x] = local_id;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride &&
                better_logit(
                    s_ids[threadIdx.x + stride], s_vals[threadIdx.x + stride], s_ids[threadIdx.x],
                    s_vals[threadIdx.x])) {
                s_vals[threadIdx.x] = s_vals[threadIdx.x + stride];
                s_ids[threadIdx.x] = s_ids[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            top_vals[rank] = s_vals[0];
            top_ids[rank] = s_ids[0] == vocab_size ? 0 : s_ids[0];
        }
        __syncthreads();
    }

    if (threadIdx.x != 0) {
        s_sums[threadIdx.x] = 0.0;
    }

    int sampled = top_ids[0];
    if (temperature <= 0.0f) {
        sampled = top_ids[0];
    } else {
        const float max_logit = top_vals[0];
        double local_sum = 0.0;
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            if (isfinite(top_vals[i])) {
                local_sum += exp((double)(top_vals[i] - max_logit) / (double)temperature);
            }
        }
        s_sums[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s_sums[threadIdx.x] += s_sums[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && s_sums[0] > 0.0) {
            const double target = uniform01(seed, frame_index, codebook_offset + c) * s_sums[0];
            double acc = 0.0;
            for (int i = 0; i < k; ++i) {
                if (isfinite(top_vals[i])) {
                    acc += exp((double)(top_vals[i] - max_logit) / (double)temperature);
                }
                if (target <= acc) {
                    sampled = top_ids[i];
                    break;
                }
            }
        }
    }

    if (threadIdx.x == 0) {
        codes_out[output_offset + c] = sampled;
        argmax_out[output_offset + c] = top_ids[0];
    }
}

magpietts_cuda_sampler*
magpietts_cuda_sampler_create(int codebooks) {
    if (codebooks <= 0) {
        return nullptr;
    }
    magpietts_cuda_sampler* sampler = new magpietts_cuda_sampler;
    sampler->codebooks = codebooks;
    cudaError_t err = cudaStreamCreateWithFlags(&sampler->stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        delete sampler;
        return nullptr;
    }
    err = cudaMalloc(&sampler->d_codes, (size_t)codebooks * sizeof(int32_t));
    if (err != cudaSuccess) {
        cudaStreamDestroy(sampler->stream);
        delete sampler;
        return nullptr;
    }
    err = cudaMalloc(&sampler->d_argmax, (size_t)codebooks * sizeof(int32_t));
    if (err != cudaSuccess) {
        cudaFree(sampler->d_codes);
        cudaStreamDestroy(sampler->stream);
        delete sampler;
        return nullptr;
    }
    err = cudaMalloc(
        &sampler->d_top_ids, (size_t)codebooks * MAGPIETTS_CUDA_MAX_VOCAB * sizeof(int32_t));
    if (err != cudaSuccess) {
        cudaFree(sampler->d_argmax);
        cudaFree(sampler->d_codes);
        cudaStreamDestroy(sampler->stream);
        delete sampler;
        return nullptr;
    }
    err = cudaMalloc(
        &sampler->d_top_vals, (size_t)codebooks * MAGPIETTS_CUDA_MAX_VOCAB * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(sampler->d_top_ids);
        cudaFree(sampler->d_argmax);
        cudaFree(sampler->d_codes);
        cudaStreamDestroy(sampler->stream);
        delete sampler;
        return nullptr;
    }
    return sampler;
}

void
magpietts_cuda_sampler_free(magpietts_cuda_sampler* sampler) {
    if (!sampler) {
        return;
    }
    cudaFree(sampler->d_codes);
    cudaFree(sampler->d_argmax);
    cudaFree(sampler->d_top_ids);
    cudaFree(sampler->d_top_vals);
    cudaStreamDestroy(sampler->stream);
    delete sampler;
}

bool
magpietts_cuda_sample_codebooks(
    magpietts_cuda_sampler* sampler, const float* logits_cond, const float* logits_uncond,
    int codebooks, int vocab_size, int audio_codebook_size, int audio_eos_id, bool use_cfg,
    float cfg_scale, float temperature, int top_k, bool forbid_audio_eos, uint64_t seed,
    int frame_index, int codebook_offset, int32_t* codes_out, int32_t* argmax_out, char* error,
    size_t error_size) {
    if (!sampler || !logits_cond || !codes_out || !argmax_out) {
        set_error(error, error_size, "invalid CUDA sampler arguments");
        return false;
    }
    if (!magpietts_cuda_sample_codebooks_device(
            sampler, logits_cond, logits_uncond, codebooks, vocab_size, audio_codebook_size,
            audio_eos_id, use_cfg, cfg_scale, temperature, top_k, forbid_audio_eos, seed,
            frame_index, codebook_offset, 0, error, error_size)) {
        return false;
    }
    return magpietts_cuda_copy_sampled_codebooks(
        sampler, codebooks, codes_out, argmax_out, error, error_size);
}

bool
magpietts_cuda_sample_codebooks_device(
    magpietts_cuda_sampler* sampler, const float* logits_cond, const float* logits_uncond,
    int codebooks, int vocab_size, int audio_codebook_size, int audio_eos_id, bool use_cfg,
    float cfg_scale, float temperature, int top_k, bool forbid_audio_eos, uint64_t seed,
    int frame_index, int codebook_offset, int output_offset, char* error, size_t error_size) {
    if (!sampler || !logits_cond) {
        set_error(error, error_size, "invalid CUDA sampler device arguments");
        return false;
    }
    if (codebooks <= 0 || codebooks > sampler->codebooks) {
        set_error(error, error_size, "invalid CUDA sampler codebook count");
        return false;
    }
    if (output_offset < 0 || output_offset + codebooks > sampler->codebooks) {
        set_error(error, error_size, "invalid CUDA sampler output offset");
        return false;
    }
    if (vocab_size <= 0 || vocab_size > MAGPIETTS_CUDA_MAX_VOCAB) {
        set_error(error, error_size, "CUDA sampler vocab size exceeds supported limit");
        return false;
    }

    magpietts_sample_codebooks_kernel<<<codebooks, MAGPIETTS_CUDA_BLOCK_SIZE, 0, sampler->stream>>>(
        logits_cond, logits_uncond, codebooks, vocab_size, audio_codebook_size, audio_eos_id,
        use_cfg, cfg_scale, temperature, top_k, forbid_audio_eos, seed, frame_index,
        codebook_offset, output_offset, sampler->d_top_ids, sampler->d_top_vals, sampler->d_codes,
        sampler->d_argmax);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_error(error, error_size, "failed to launch CUDA sampler", err);
        return false;
    }

    if (error && error_size > 0) {
        error[0] = '\0';
    }
    return true;
}

bool
magpietts_cuda_copy_sampled_code_to_device(
    magpietts_cuda_sampler* sampler, int codebook, void* dst_device, char* error,
    size_t error_size) {
    if (!sampler || !dst_device) {
        set_error(error, error_size, "invalid CUDA sampled-code copy arguments");
        return false;
    }
    if (codebook < 0 || codebook >= sampler->codebooks) {
        set_error(error, error_size, "invalid CUDA sampled-code index");
        return false;
    }

    cudaError_t err = cudaStreamSynchronize(sampler->stream);
    if (err != cudaSuccess) {
        set_error(error, error_size, "failed to synchronize CUDA sampled code", err);
        return false;
    }
    err = cudaMemcpyAsync(
        dst_device, sampler->d_codes + codebook, sizeof(int32_t), cudaMemcpyDeviceToDevice,
        sampler->stream);
    if (err != cudaSuccess) {
        set_error(error, error_size, "failed to copy CUDA sampled code to device", err);
        return false;
    }
    err = cudaStreamSynchronize(sampler->stream);
    if (err != cudaSuccess) {
        set_error(error, error_size, "failed to synchronize CUDA sampled-code copy", err);
        return false;
    }
    if (error && error_size > 0) {
        error[0] = '\0';
    }
    return true;
}

bool
magpietts_cuda_copy_sampled_codebooks(
    magpietts_cuda_sampler* sampler, int codebooks, int32_t* codes_out, int32_t* argmax_out,
    char* error, size_t error_size) {
    if (!sampler || !codes_out || !argmax_out) {
        set_error(error, error_size, "invalid CUDA sampled-code host copy arguments");
        return false;
    }
    if (codebooks <= 0 || codebooks > sampler->codebooks) {
        set_error(error, error_size, "invalid CUDA sampled-code host copy count");
        return false;
    }

    cudaError_t err = cudaMemcpyAsync(
        codes_out, sampler->d_codes, (size_t)codebooks * sizeof(int32_t), cudaMemcpyDeviceToHost,
        sampler->stream);
    if (err != cudaSuccess) {
        set_error(error, error_size, "failed to copy CUDA sampled codes", err);
        return false;
    }
    err = cudaMemcpyAsync(
        argmax_out, sampler->d_argmax, (size_t)codebooks * sizeof(int32_t), cudaMemcpyDeviceToHost,
        sampler->stream);
    if (err != cudaSuccess) {
        set_error(error, error_size, "failed to copy CUDA argmax codes", err);
        return false;
    }
    err = cudaStreamSynchronize(sampler->stream);
    if (err != cudaSuccess) {
        set_error(error, error_size, "failed to synchronize CUDA sampled-code host copy", err);
        return false;
    }
    if (error && error_size > 0) {
        error[0] = '\0';
    }
    return true;
}
