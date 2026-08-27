// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cub/block/block_radix_sort.cuh>

#include "magpietts_cuda_sampling.h"

static constexpr int MAGPIETTS_CUDA_MAX_VOCAB = 4096;
static constexpr int MAGPIETTS_CUDA_BLOCK_SIZE = 256;
static constexpr int MAGPIETTS_CUDA_SMALL_VOCAB = 2048;
static constexpr int MAGPIETTS_CUDA_SMALL_ITEMS_PER_THREAD =
    MAGPIETTS_CUDA_SMALL_VOCAB / MAGPIETTS_CUDA_BLOCK_SIZE;
static constexpr int MAGPIETTS_CUDA_MAX_ITEMS_PER_THREAD =
    MAGPIETTS_CUDA_MAX_VOCAB / MAGPIETTS_CUDA_BLOCK_SIZE;

struct alignas(16) magpietts_cuda_sampling_config {
    float cfg_scale = 1.0f;
    float temperature = 0.0f;
    int top_k = 1;
    int frame_index = 0;
    uint64_t seed = 0;
    int use_cfg = 0;
    int forbid_audio_eos = 0;
};

struct magpietts_cuda_sampler {
    int codebooks = 0;
    cudaStream_t stream = nullptr;
    cudaStream_t owned_stream = nullptr;
    bool uses_external_stream = false;
    int32_t* d_codes = nullptr;
    int32_t* d_argmax = nullptr;
    int32_t* d_top_ids = nullptr;
    float* d_top_vals = nullptr;
    magpietts_cuda_sampling_config* h_config = nullptr;
    magpietts_cuda_sampling_config* d_config = nullptr;
    cudaGraph_t sequence_graph = nullptr;
    cudaGraphExec_t sequence_exec = nullptr;
    cudaGraphNode_t sequence_tail = nullptr;
    bool sequence_warm = false;
    bool sequence_build_active = false;
    bool sequence_disabled = false;
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

template <int items_per_thread>
__global__ void
magpietts_sample_codebooks_kernel(
    const float* logits_cond, const float* logits_uncond, int codebooks, int vocab_size,
    int audio_codebook_size, int audio_eos_id, const magpietts_cuda_sampling_config* config_ptr,
    int codebook_offset, int output_offset, int32_t* top_ids_scratch, float* top_vals_scratch,
    int32_t* codes_out, int32_t* argmax_out) {
    const int c = blockIdx.x;
    if (c >= codebooks) {
        return;
    }

    const magpietts_cuda_sampling_config config = *config_ptr;

    using block_sort = cub::BlockRadixSort<float, MAGPIETTS_CUDA_BLOCK_SIZE, items_per_thread, int>;
    __shared__ typename block_sort::TempStorage sort_storage;
    __shared__ double s_sums[MAGPIETTS_CUDA_BLOCK_SIZE];

    int k = config.top_k < vocab_size ? config.top_k : vocab_size;
    if (k < 1) {
        k = 1;
    }

    const int off = c * vocab_size;
    int32_t* top_ids = top_ids_scratch + (size_t)c * vocab_size;
    float* top_vals = top_vals_scratch + (size_t)c * vocab_size;

    // Sort once per codebook with work independent of top-k.
    float thread_vals[items_per_thread];
    int thread_ids[items_per_thread];
#pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        const int id = (int)threadIdx.x * items_per_thread + item;
        thread_vals[item] =
            id < vocab_size
                ? sampled_logit(
                      logits_cond, logits_uncond, off, id, audio_codebook_size, audio_eos_id,
                      config.use_cfg != 0, config.cfg_scale, config.forbid_audio_eos != 0)
                : -INFINITY;
        thread_ids[item] = id;
    }
    block_sort(sort_storage).SortDescendingBlockedToStriped(thread_vals, thread_ids);
    __syncthreads();

#pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        const int rank = item * MAGPIETTS_CUDA_BLOCK_SIZE + (int)threadIdx.x;
        if (rank < k) {
            top_vals[rank] = thread_vals[item];
            top_ids[rank] = thread_ids[item];
        }
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        s_sums[threadIdx.x] = 0.0;
    }

    int sampled = top_ids[0];
    if (config.temperature <= 0.0f) {
        sampled = top_ids[0];
    } else {
        const float max_logit = top_vals[0];
        double local_sum = 0.0;
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            if (isfinite(top_vals[i])) {
                local_sum += exp((double)(top_vals[i] - max_logit) / (double)config.temperature);
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
            const double target =
                uniform01(config.seed, config.frame_index, codebook_offset + c) * s_sums[0];
            double acc = 0.0;
            for (int i = 0; i < k; ++i) {
                if (isfinite(top_vals[i])) {
                    acc += exp((double)(top_vals[i] - max_logit) / (double)config.temperature);
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
    cudaError_t err = cudaStreamCreateWithFlags(&sampler->owned_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        delete sampler;
        return nullptr;
    }
    sampler->stream = sampler->owned_stream;
    err = cudaMalloc(&sampler->d_codes, (size_t)codebooks * sizeof(int32_t));
    if (err != cudaSuccess) {
        cudaStreamDestroy(sampler->owned_stream);
        delete sampler;
        return nullptr;
    }
    err = cudaMalloc(&sampler->d_argmax, (size_t)codebooks * sizeof(int32_t));
    if (err != cudaSuccess) {
        cudaFree(sampler->d_codes);
        cudaStreamDestroy(sampler->owned_stream);
        delete sampler;
        return nullptr;
    }
    err = cudaMalloc(
        &sampler->d_top_ids, (size_t)codebooks * MAGPIETTS_CUDA_MAX_VOCAB * sizeof(int32_t));
    if (err != cudaSuccess) {
        cudaFree(sampler->d_argmax);
        cudaFree(sampler->d_codes);
        cudaStreamDestroy(sampler->owned_stream);
        delete sampler;
        return nullptr;
    }
    err = cudaMalloc(
        &sampler->d_top_vals, (size_t)codebooks * MAGPIETTS_CUDA_MAX_VOCAB * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(sampler->d_top_ids);
        cudaFree(sampler->d_argmax);
        cudaFree(sampler->d_codes);
        cudaStreamDestroy(sampler->owned_stream);
        delete sampler;
        return nullptr;
    }
    err = cudaHostAlloc(
        reinterpret_cast<void**>(&sampler->h_config), sizeof(*sampler->h_config),
        cudaHostAllocDefault);
    if (err != cudaSuccess) {
        cudaFree(sampler->d_top_vals);
        cudaFree(sampler->d_top_ids);
        cudaFree(sampler->d_argmax);
        cudaFree(sampler->d_codes);
        cudaStreamDestroy(sampler->owned_stream);
        delete sampler;
        return nullptr;
    }
    *sampler->h_config = magpietts_cuda_sampling_config{};
    err = cudaMalloc(&sampler->d_config, sizeof(*sampler->d_config));
    if (err != cudaSuccess) {
        cudaFreeHost(sampler->h_config);
        cudaFree(sampler->d_top_vals);
        cudaFree(sampler->d_top_ids);
        cudaFree(sampler->d_argmax);
        cudaFree(sampler->d_codes);
        cudaStreamDestroy(sampler->owned_stream);
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
    if (sampler->sequence_exec) {
        cudaGraphExecDestroy(sampler->sequence_exec);
    }
    if (sampler->sequence_graph) {
        cudaGraphDestroy(sampler->sequence_graph);
    }
    cudaFree(sampler->d_config);
    cudaFreeHost(sampler->h_config);
    cudaFree(sampler->d_codes);
    cudaFree(sampler->d_argmax);
    cudaFree(sampler->d_top_ids);
    cudaFree(sampler->d_top_vals);
    cudaStreamDestroy(sampler->owned_stream);
    delete sampler;
}

bool
magpietts_cuda_sampler_bind_stream(
    magpietts_cuda_sampler* sampler, void* stream, char* error, size_t error_size) {
    if (!sampler || !stream) {
        set_error(error, error_size, "invalid CUDA sampler stream binding");
        return false;
    }
    sampler->stream = (cudaStream_t)stream;
    sampler->uses_external_stream = true;
    if (error && error_size > 0) {
        error[0] = '\0';
    }
    return true;
}

bool
magpietts_cuda_sampler_configure(
    magpietts_cuda_sampler* sampler, bool use_cfg, float cfg_scale, float temperature, int top_k,
    bool forbid_audio_eos, uint64_t seed, int frame_index, char* error, size_t error_size) {
    if (!sampler || !sampler->h_config || !sampler->d_config) {
        set_error(error, error_size, "invalid CUDA sampler configuration");
        return false;
    }
    sampler->h_config->cfg_scale = cfg_scale;
    sampler->h_config->temperature = temperature;
    sampler->h_config->top_k = top_k;
    sampler->h_config->frame_index = frame_index;
    sampler->h_config->seed = seed;
    sampler->h_config->use_cfg = use_cfg ? 1 : 0;
    sampler->h_config->forbid_audio_eos = forbid_audio_eos ? 1 : 0;
    if (error && error_size > 0)
        error[0] = '\0';
    return true;
}

bool
magpietts_cuda_sampler_upload_config(
    magpietts_cuda_sampler* sampler, char* error, size_t error_size) {
    if (!sampler || !sampler->h_config || !sampler->d_config) {
        set_error(error, error_size, "invalid CUDA sampler config upload");
        return false;
    }
    cudaError_t err = cudaSuccess;
    if (sampler->sequence_build_active) {
        cudaGraphNode_t node = nullptr;
        const cudaGraphNode_t* deps = sampler->sequence_tail ? &sampler->sequence_tail : nullptr;
        const size_t dependency_count = sampler->sequence_tail ? 1 : 0;
        err = cudaGraphAddMemcpyNode1D(
            &node, sampler->sequence_graph, deps, dependency_count, sampler->d_config,
            sampler->h_config, sizeof(*sampler->d_config), cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            sampler->sequence_tail = node;
    } else {
        err = cudaMemcpyAsync(
            sampler->d_config, sampler->h_config, sizeof(*sampler->d_config),
            cudaMemcpyHostToDevice, sampler->stream);
    }
    if (err != cudaSuccess) {
        set_error(error, error_size, "failed to upload CUDA sampler config", err);
        return false;
    }
    if (error && error_size > 0)
        error[0] = '\0';
    return true;
}

bool
magpietts_cuda_sampler_sequence_is_warm(const magpietts_cuda_sampler* sampler) {
    return sampler && sampler->sequence_warm;
}

bool
magpietts_cuda_sampler_sequence_is_ready(const magpietts_cuda_sampler* sampler) {
    return sampler && sampler->sequence_exec;
}

bool
magpietts_cuda_sampler_sequence_is_disabled(const magpietts_cuda_sampler* sampler) {
    return !sampler || sampler->sequence_disabled;
}

bool
magpietts_cuda_sampler_sequence_build_active(const magpietts_cuda_sampler* sampler) {
    return sampler && sampler->sequence_build_active;
}

void
magpietts_cuda_sampler_sequence_mark_warm(magpietts_cuda_sampler* sampler) {
    if (sampler)
        sampler->sequence_warm = true;
}

bool
magpietts_cuda_sampler_sequence_begin_build(
    magpietts_cuda_sampler* sampler, char* error, size_t error_size) {
    if (!sampler || !sampler->stream || sampler->sequence_exec || sampler->sequence_disabled ||
        sampler->sequence_build_active) {
        set_error(error, error_size, "invalid CUDA local sequence graph-build state");
        return false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t err = cudaGraphCreate(&graph, 0);
    if (err != cudaSuccess) {
        set_error(error, error_size, "failed to create CUDA local sequence graph", err);
        return false;
    }
    sampler->sequence_graph = graph;
    sampler->sequence_tail = nullptr;
    sampler->sequence_build_active = true;
    if (error && error_size > 0)
        error[0] = '\0';
    return true;
}

bool
magpietts_cuda_sampler_sequence_finish_build_and_launch(
    magpietts_cuda_sampler* sampler, char* error, size_t error_size) {
    if (!sampler || !sampler->sequence_build_active) {
        set_error(error, error_size, "CUDA local sequence graph build is not active");
        return false;
    }
    cudaGraph_t graph = sampler->sequence_graph;
    cudaError_t err = cudaSuccess;
    sampler->sequence_build_active = false;
    sampler->sequence_tail = nullptr;
    if (!graph) {
        set_error(error, error_size, "CUDA local sequence graph is empty");
        return false;
    }
    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph);
        sampler->sequence_graph = nullptr;
        set_error(error, error_size, "failed to instantiate CUDA local sequence graph", err);
        return false;
    }
    err = cudaGraphLaunch(exec, sampler->stream);
    if (err != cudaSuccess) {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        sampler->sequence_graph = nullptr;
        set_error(error, error_size, "failed to launch composed CUDA local sequence", err);
        return false;
    }
    sampler->sequence_exec = exec;
    if (error && error_size > 0)
        error[0] = '\0';
    return true;
}

void
magpietts_cuda_sampler_sequence_abort_build(magpietts_cuda_sampler* sampler) {
    if (!sampler || !sampler->sequence_build_active)
        return;
    if (sampler->sequence_graph)
        cudaGraphDestroy(sampler->sequence_graph);
    sampler->sequence_graph = nullptr;
    sampler->sequence_tail = nullptr;
    sampler->sequence_build_active = false;
}

void
magpietts_cuda_sampler_sequence_disable(magpietts_cuda_sampler* sampler) {
    if (!sampler)
        return;
    magpietts_cuda_sampler_sequence_abort_build(sampler);
    if (sampler->sequence_exec) {
        cudaGraphExecDestroy(sampler->sequence_exec);
        sampler->sequence_exec = nullptr;
    }
    if (sampler->sequence_graph) {
        cudaGraphDestroy(sampler->sequence_graph);
        sampler->sequence_graph = nullptr;
    }
    sampler->sequence_tail = nullptr;
    sampler->sequence_disabled = true;
}

bool
magpietts_cuda_sampler_sequence_launch(
    magpietts_cuda_sampler* sampler, char* error, size_t error_size) {
    if (!sampler || !sampler->sequence_exec) {
        set_error(error, error_size, "CUDA local sequence graph is not ready");
        return false;
    }
    const cudaError_t err = cudaGraphLaunch(sampler->sequence_exec, sampler->stream);
    if (err != cudaSuccess) {
        set_error(error, error_size, "failed to launch CUDA local sequence graph", err);
        return false;
    }
    if (error && error_size > 0)
        error[0] = '\0';
    return true;
}

bool
magpietts_cuda_sampler_sequence_add_ggml_graph(
    magpietts_cuda_sampler* sampler, void* graph_template, char* error, size_t error_size) {
    if (!sampler || !sampler->sequence_build_active || !sampler->sequence_graph ||
        !graph_template) {
        set_error(error, error_size, "invalid GGML child graph for CUDA local sequence");
        return false;
    }
    cudaGraphNode_t node = nullptr;
    const cudaGraphNode_t* deps = sampler->sequence_tail ? &sampler->sequence_tail : nullptr;
    const size_t dependency_count = sampler->sequence_tail ? 1 : 0;
    const cudaError_t err = cudaGraphAddChildGraphNode(
        &node, sampler->sequence_graph, deps, dependency_count,
        reinterpret_cast<cudaGraph_t>(graph_template));
    if (err != cudaSuccess) {
        set_error(error, error_size, "failed to add GGML child graph to CUDA local sequence", err);
        return false;
    }
    sampler->sequence_tail = node;
    if (error && error_size > 0)
        error[0] = '\0';
    return true;
}

bool
magpietts_cuda_sampler_sequence_add_device_copy(
    magpietts_cuda_sampler* sampler, const void* src_device, void* dst_device, size_t bytes,
    char* error, size_t error_size) {
    if (!sampler || !sampler->sequence_build_active || !sampler->sequence_graph || !src_device ||
        !dst_device || bytes == 0) {
        set_error(error, error_size, "invalid device copy for CUDA local sequence");
        return false;
    }
    cudaGraphNode_t node = nullptr;
    const cudaGraphNode_t* deps = sampler->sequence_tail ? &sampler->sequence_tail : nullptr;
    const size_t dependency_count = sampler->sequence_tail ? 1 : 0;
    const cudaError_t err = cudaGraphAddMemcpyNode1D(
        &node, sampler->sequence_graph, deps, dependency_count, dst_device, src_device, bytes,
        cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        set_error(error, error_size, "failed to add device copy to CUDA local sequence", err);
        return false;
    }
    sampler->sequence_tail = node;
    if (error && error_size > 0)
        error[0] = '\0';
    return true;
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
    if (!magpietts_cuda_sampler_configure(
            sampler, use_cfg, cfg_scale, temperature, top_k, forbid_audio_eos, seed, frame_index,
            error, error_size) ||
        !magpietts_cuda_sampler_upload_config(sampler, error, error_size)) {
        return false;
    }
    return magpietts_cuda_sample_codebooks_device_configured(
        sampler, logits_cond, logits_uncond, codebooks, vocab_size, audio_codebook_size,
        audio_eos_id, codebook_offset, output_offset, error, error_size);
}

bool
magpietts_cuda_sample_codebooks_device_configured(
    magpietts_cuda_sampler* sampler, const float* logits_cond, const float* logits_uncond,
    int codebooks, int vocab_size, int audio_codebook_size, int audio_eos_id, int codebook_offset,
    int output_offset, char* error, size_t error_size) {
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

    if (sampler->sequence_build_active) {
        magpietts_cuda_sampling_config* config = sampler->d_config;
        int32_t* top_ids = sampler->d_top_ids;
        float* top_vals = sampler->d_top_vals;
        int32_t* codes = sampler->d_codes;
        int32_t* argmax = sampler->d_argmax;
        void* kernel_args[] = {
            &logits_cond,  &logits_uncond, &codebooks,       &vocab_size,    &audio_codebook_size,
            &audio_eos_id, &config,        &codebook_offset, &output_offset, &top_ids,
            &top_vals,     &codes,         &argmax};
        cudaKernelNodeParams params{};
        params.func =
            vocab_size <= MAGPIETTS_CUDA_SMALL_VOCAB
                ? reinterpret_cast<void*>(
                      magpietts_sample_codebooks_kernel<MAGPIETTS_CUDA_SMALL_ITEMS_PER_THREAD>)
                : reinterpret_cast<void*>(
                      magpietts_sample_codebooks_kernel<MAGPIETTS_CUDA_MAX_ITEMS_PER_THREAD>);
        params.gridDim = dim3((unsigned int)codebooks, 1, 1);
        params.blockDim = dim3(MAGPIETTS_CUDA_BLOCK_SIZE, 1, 1);
        params.sharedMemBytes = 0;
        params.kernelParams = kernel_args;
        params.extra = nullptr;
        cudaGraphNode_t node = nullptr;
        const cudaGraphNode_t* deps = sampler->sequence_tail ? &sampler->sequence_tail : nullptr;
        const size_t dependency_count = sampler->sequence_tail ? 1 : 0;
        const cudaError_t err =
            cudaGraphAddKernelNode(&node, sampler->sequence_graph, deps, dependency_count, &params);
        if (err != cudaSuccess) {
            set_error(error, error_size, "failed to add sampler to CUDA local sequence", err);
            return false;
        }
        sampler->sequence_tail = node;
        if (error && error_size > 0)
            error[0] = '\0';
        return true;
    }

    if (vocab_size <= MAGPIETTS_CUDA_SMALL_VOCAB) {
        magpietts_sample_codebooks_kernel<MAGPIETTS_CUDA_SMALL_ITEMS_PER_THREAD>
            <<<codebooks, MAGPIETTS_CUDA_BLOCK_SIZE, 0, sampler->stream>>>(
                logits_cond, logits_uncond, codebooks, vocab_size, audio_codebook_size,
                audio_eos_id, sampler->d_config, codebook_offset, output_offset, sampler->d_top_ids,
                sampler->d_top_vals, sampler->d_codes, sampler->d_argmax);
    } else {
        magpietts_sample_codebooks_kernel<MAGPIETTS_CUDA_MAX_ITEMS_PER_THREAD>
            <<<codebooks, MAGPIETTS_CUDA_BLOCK_SIZE, 0, sampler->stream>>>(
                logits_cond, logits_uncond, codebooks, vocab_size, audio_codebook_size,
                audio_eos_id, sampler->d_config, codebook_offset, output_offset, sampler->d_top_ids,
                sampler->d_top_vals, sampler->d_codes, sampler->d_argmax);
    }
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

    if (sampler->sequence_build_active) {
        return magpietts_cuda_sampler_sequence_add_device_copy(
            sampler, sampler->d_codes + codebook, dst_device, sizeof(int32_t), error, error_size);
    }

    cudaError_t err = cudaSuccess;
    if (!sampler->uses_external_stream) {
        err = cudaStreamSynchronize(sampler->stream);
        if (err != cudaSuccess) {
            set_error(error, error_size, "failed to synchronize CUDA sampled code", err);
            return false;
        }
    }
    err = cudaMemcpyAsync(
        dst_device, sampler->d_codes + codebook, sizeof(int32_t), cudaMemcpyDeviceToDevice,
        sampler->stream);
    if (err != cudaSuccess) {
        set_error(error, error_size, "failed to copy CUDA sampled code to device", err);
        return false;
    }
    if (!sampler->uses_external_stream) {
        err = cudaStreamSynchronize(sampler->stream);
        if (err != cudaSuccess) {
            set_error(error, error_size, "failed to synchronize CUDA sampled-code copy", err);
            return false;
        }
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
