// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Device-side MagpieTTS codebook sampling shared by the standalone sampling kernel and the
// fused local-transformer output projection (which samples in its last-finishing block).
#pragma once
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

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

// Largest top-k handled by the exact radix-select fast path; larger k falls back to the
// full block radix sort.
static constexpr int MAGPIETTS_CUDA_MAX_FAST_TOPK = 256;

// Monotone key: ascending key order == descending logit order (larger logit -> smaller
// key), bit-exact and invertible.
static __device__ __forceinline__ uint32_t
descending_key(float f) {
    const uint32_t bits = __float_as_uint(f);
    const uint32_t ascending = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    return ~ascending;
}

static __device__ __forceinline__ float
descending_key_to_float(uint32_t key) {
    const uint32_t ascending = ~key;
    const uint32_t bits = (ascending & 0x80000000u) ? (ascending & 0x7fffffffu) : ~ascending;
    return __uint_as_float(bits);
}

// Block-wide inclusive scan of one int per thread (blockDim.x == MAGPIETTS_CUDA_BLOCK_SIZE).
static __device__ __forceinline__ int
block_inclusive_scan(int value, int* warp_scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr int warps = MAGPIETTS_CUDA_BLOCK_SIZE / 32;
    int inclusive = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) {
            inclusive += other;
        }
    }
    if (lane == 31) {
        warp_scratch[warp] = inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        int total = lane < warps ? warp_scratch[lane] : 0;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int other = __shfl_up_sync(0xffffffffu, total, offset);
            if (lane >= offset) {
                total += other;
            }
        }
        if (lane < warps) {
            warp_scratch[lane] = total;
        }
    }
    __syncthreads();
    const int warp_offset = warp > 0 ? warp_scratch[warp - 1] : 0;
    return inclusive + warp_offset;
}

// Shared top-k temperature sampling given `top_vals`/`top_ids` sorted by descending logit
// (ties by ascending id). `exps` receives the k Boltzmann terms so the sum and the
// cumulative scan use identical values and identical (original) accumulation order.
static __device__ __forceinline__ int
sample_from_topk(
    const float* top_vals, const int32_t* top_ids, int k,
    const magpietts_cuda_sampling_config& config, int rng_codebook, double* exps, double* s_sums) {
    int sampled = top_ids[0];
    if (config.temperature <= 0.0f) {
        return sampled;
    }
    const float max_logit = top_vals[0];
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const float v = top_vals[i];
        exps[i] = isfinite(v) ? exp((double)(v - max_logit) / (double)config.temperature) : 0.0;
    }
    __syncthreads();
    double local_sum = 0.0;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        local_sum += exps[i];
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
        const double target = uniform01(config.seed, config.frame_index, rng_codebook) * s_sums[0];
        double acc = 0.0;
        for (int i = 0; i < k; ++i) {
            acc += exps[i];
            if (target <= acc) {
                sampled = top_ids[i];
                break;
            }
        }
    }
    return sampled;
}


// Exact top-k (k <= MAGPIETTS_CUDA_MAX_FAST_TOPK) temperature sampling of one codebook by a
// full 256-thread block. Shared buffers are provided by the caller (see
// magpietts_sample_fast_shared). Writes codes_out[out_index] / argmax_out[out_index].
//
// Latency-oriented: this runs on one block while the rest of a persistent kernel waits, so the
// number of block-wide synchronizations is what matters. Radix select with double-buffered
// histograms (2 syncs per pass), no sort of the selected set (the sampler is order-independent,
// argmax is a separate min-key reduction), and warp-shuffle reductions/scans for the Boltzmann
// weights instead of a shared-memory tree.
struct magpietts_sample_fast_shared {
    int s_hist[2][256];  // double-buffered so a pass can clear the next pass's histogram
    int s_scan[MAGPIETTS_CUDA_BLOCK_SIZE / 32];
    int s_ctrl[4];
    unsigned int s_minkey;
    int s_argmax;
    unsigned long long s_sel[MAGPIETTS_CUDA_MAX_FAST_TOPK];
    float s_fsum[MAGPIETTS_CUDA_BLOCK_SIZE / 32];
    int s_pick;
    unsigned long long s_best;  // Gumbel-max fast path: (key of the best score) << 32 | id
};

// Per-item uniform in (0, 1] for the Gumbel draw: counter-based on (seed, frame, codebook, id),
// so runs are reproducible and every item's draw is independent of the others.
static __device__ __forceinline__ float
gumbel_u01(uint64_t seed, int frame_index, int codebook, int id) {
    uint64_t state = seed ^ (0xd1b54a32d192ed03ULL * (uint64_t)(frame_index + 1)) ^
                     (0xabc98388fb8fac03ULL * (uint64_t)(codebook + 1)) ^
                     (0x9e3779b97f4a7c15ULL * (uint64_t)(id + 1));
    const uint64_t r = splitmix64_next(state);
    return (float)((r >> 40) + 1) * (1.0f / 16777217.0f);
}

#ifdef MAGPIETTS_SAMPLER_DBG
__device__ unsigned long long g_sampler_dbg[8];
#define SAMPLER_STAMP(i)                  \
    do {                                  \
        if (threadIdx.x == 0)             \
            g_sampler_dbg[i] = clock64(); \
    } while (0)
#else
#define SAMPLER_STAMP(i) \
    do {                 \
    } while (0)
#endif

template <int items_per_thread>
static __device__ __forceinline__ void
magpietts_sample_codebook_fast_block(
    const float* logits_cond, const float* logits_uncond, int vocab_size, int audio_codebook_size,
    int audio_eos_id, const magpietts_cuda_sampling_config& config, int rng_codebook, int k,
    int32_t* top_ids, float* top_vals, int32_t* code_out, int32_t* argmax_out,
    magpietts_sample_fast_shared& sh) {
    constexpr int block = MAGPIETTS_CUDA_BLOCK_SIZE;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    (void)top_ids;
    (void)top_vals;
    SAMPLER_STAMP(0);
    float thread_vals[items_per_thread];
    uint32_t keys[items_per_thread];
    uint32_t minkey = 0xffffffffu;
#pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        const int id = tid * items_per_thread + item;
        thread_vals[item] =
            id < vocab_size
                ? sampled_logit(
                      logits_cond, logits_uncond, 0, id, audio_codebook_size, audio_eos_id,
                      config.use_cfg != 0, config.cfg_scale, config.forbid_audio_eos != 0)
                : -INFINITY;
        keys[item] = descending_key(thread_vals[item]);
        minkey = min(minkey, keys[item]);
    }
    // argmax = smallest key (ties -> smallest id, which is the first item holding that key)
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        minkey = min(minkey, __shfl_xor_sync(0xffffffffu, minkey, offset));
    sh.s_hist[0][tid] = 0;
    sh.s_hist[1][tid] = 0;
    if (tid == 0) {
        sh.s_minkey = 0xffffffffu;
        sh.s_argmax = 0x7fffffff;
    }
    __syncthreads();
    if (lane == 0)
        atomicMin(&sh.s_minkey, minkey);
    // ---- Fast path: Gumbel-max over the whole distribution, accepted when the draw lies in the
    // top-k set. A softmax sample conditioned on falling inside the top-k set IS the top-k
    // truncated softmax, so this is exact; it costs one max and one count reduction instead of
    // four radix-select passes. Rejections (mass outside the top-k, rare at k = 80) fall through
    // to the exact radix path below.
    if (config.temperature > 0.0f && k < vocab_size) {
        const float inv_t = 1.0f / config.temperature;
        unsigned long long best = ~0ull;
#pragma unroll
        for (int item = 0; item < items_per_thread; ++item) {
            const int id = tid * items_per_thread + item;
            const float v = thread_vals[item];
            if (isfinite(v)) {
                const float u = gumbel_u01(config.seed, config.frame_index, rng_codebook, id);
                const float g = v * inv_t - __logf(-__logf(u));
                const unsigned long long packed =
                    ((unsigned long long)descending_key(g) << 32) | (uint32_t)id;
                best = min(best, packed);
            }
            if (keys[item] == minkey)  // candidates for the argmax (smallest id among maxima)
                atomicMin(&sh.s_argmax, id);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            best = min(best, __shfl_xor_sync(0xffffffffu, best, offset));
        if (tid == 0)
            sh.s_best = ~0ull;
        __syncthreads();  // s_minkey final, s_best initialized
        if (lane == 0)
            atomicMin(&sh.s_best, best);
        __syncthreads();
        const unsigned long long sel = sh.s_best;
        if (sel != ~0ull) {
            const int cand = (int)(uint32_t)sel;
            const uint32_t cand_key = descending_key(sampled_logit(
                logits_cond, logits_uncond, 0, cand, audio_codebook_size, audio_eos_id,
                config.use_cfg != 0, config.cfg_scale, config.forbid_audio_eos != 0));
            int n_better = 0;
#pragma unroll
            for (int item = 0; item < items_per_thread; ++item) {
                const int id = tid * items_per_thread + item;
                n_better +=
                    (keys[item] < cand_key || (keys[item] == cand_key && id < cand)) ? 1 : 0;
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                n_better += __shfl_xor_sync(0xffffffffu, n_better, offset);
            if (lane == 0)
                sh.s_scan[warp] = n_better;
            __syncthreads();
            int rank = 0;
#pragma unroll
            for (int w = 0; w < block / 32; ++w) rank += sh.s_scan[w];
            if (rank < k) {
                if (tid == 0) {
                    *code_out = cand;
                    *argmax_out = sh.s_argmax;  // final: all argmax atomics preceded the sync
                }
                SAMPLER_STAMP(5);
                return;
            }
            __syncthreads();  // s_scan is reused by the exact path's block scan
        }
    }
    uint32_t prefix = 0;
    uint32_t mask = 0;
    int remaining = k;
    SAMPLER_STAMP(1);
#pragma unroll 1
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        int* hist = sh.s_hist[pass & 1];
#pragma unroll
        for (int item = 0; item < items_per_thread; ++item) {
            // Warp-aggregated histogram update: logits share most high-order key bits, so plain
            // shared atomics would serialize thousands of updates on a few bins.
            const bool active = (keys[item] & mask) == prefix;
            const int bin = active ? (int)((keys[item] >> shift) & 0xffu) : -1;
            const unsigned peers = __match_any_sync(0xffffffffu, bin);
            if (active && (__ffs(peers) - 1) == lane) {
                atomicAdd(&hist[bin], __popc(peers));
            }
        }
        __syncthreads();
        if (tid < 32) {
            int bins[8];
            int local = 0;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                bins[b] = hist[tid * 8 + b];
                local += bins[b];
            }
            int inclusive = local;
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
                if (tid >= offset)
                    inclusive += other;
            }
            int running = inclusive - local;
            if (running < remaining && remaining <= inclusive) {
#pragma unroll
                for (int b = 0; b < 8; ++b) {
                    if (running < remaining && remaining <= running + bins[b]) {
                        sh.s_ctrl[0] = tid * 8 + b;
                        sh.s_ctrl[1] = remaining - running;
                    }
                    running += bins[b];
                }
            }
        }
        // The other histogram buffer is free now: clear it for the next pass while warp 0 scans.
        if (pass + 1 < 4)
            sh.s_hist[(pass + 1) & 1][tid] = 0;
        __syncthreads();
        prefix |= (uint32_t)sh.s_ctrl[0] << shift;
        mask |= 0xffu << shift;
        remaining = sh.s_ctrl[1];
    }
    SAMPLER_STAMP(2);
    // Compact the k selected (key, id) pairs: all keys below the threshold plus the first
    // `remaining` keys equal to it (ascending id).
    const uint32_t threshold = prefix;
    const int need_equal = remaining;
    int n_less = 0;
    int n_equal = 0;
#pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        n_less += keys[item] < threshold ? 1 : 0;
        n_equal += keys[item] == threshold ? 1 : 0;
    }
    const int packed_inclusive = block_inclusive_scan(n_less | (n_equal << 16), sh.s_scan);
    if (tid == block - 1)
        sh.s_ctrl[2] = packed_inclusive & 0xffff;
    __syncthreads();
    const int total_less = sh.s_ctrl[2];
    int less_pos = (packed_inclusive & 0xffff) - n_less;
    int equal_rank = (packed_inclusive >> 16) - n_equal;
#pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        const int id = tid * items_per_thread + item;
        const unsigned long long packed = ((unsigned long long)keys[item] << 32) | (uint32_t)id;
        if (keys[item] < threshold) {
            sh.s_sel[less_pos++] = packed;
        } else if (keys[item] == threshold) {
            if (equal_rank < need_equal)
                sh.s_sel[total_less + equal_rank] = packed;
            ++equal_rank;
        }
        if (keys[item] == sh.s_minkey)
            atomicMin(&sh.s_argmax, id);  // argmax: smallest id among the maximal logits
    }
    __syncthreads();
    SAMPLER_STAMP(3);
    // Temperature sampling over the k selected logits: one candidate per thread (k <= block).
    const float max_logit = descending_key_to_float(sh.s_minkey);
    float e = 0.0f;
    int my_id = -1;
    if (tid < k) {
        const unsigned long long mine = sh.s_sel[tid];
        const float v = descending_key_to_float((uint32_t)(mine >> 32));
        my_id = (int32_t)(uint32_t)mine;
        // Boltzmann weights and their scan in f32: FP64 runs at 1/64 rate on consumer Ampere and
        // dominated this tail. k <= 256 terms in (0, 1] keep the f32 prefix sums well conditioned.
        if (config.temperature > 0.0f)
            e = isfinite(v) ? __expf((v - max_logit) / config.temperature) : 0.0f;
    }
    SAMPLER_STAMP(6);
    // Inclusive scan of e over the block (warp shuffles + one cross-warp step).
    float incl = e;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float other = __shfl_up_sync(0xffffffffu, incl, offset);
        if (lane >= offset)
            incl += other;
    }
    // exclusive prefix = the previous lane's inclusive value (monotone by construction)
    const float prev_incl = __shfl_up_sync(0xffffffffu, incl, 1);
    if (lane == 31)
        sh.s_fsum[warp] = incl;
    __syncthreads();
    float warp_offset = 0.0f, total = 0.0f;
#pragma unroll
    for (int w = 0; w < block / 32; ++w) {
        const float t = sh.s_fsum[w];
        if (w < warp)
            warp_offset += t;
        total += t;
    }
    incl += warp_offset;
    const float excl = lane == 0 ? warp_offset : prev_incl + warp_offset;
    if (tid == 0)
        sh.s_pick = -1;
    __syncthreads();
    SAMPLER_STAMP(7);
    if (config.temperature > 0.0f && total > 0.0f) {
        const float target =
            (float)(uniform01(config.seed, config.frame_index, rng_codebook) * (double)total);
        // The inclusive prefix is monotone and excl is the previous candidate's inclusive value,
        // so exactly one candidate satisfies excl < target <= incl when target <= total; if
        // rounding leaves none, the last candidate takes it.
        if (tid < k && excl < target && target <= incl)
            sh.s_pick = my_id;
        __syncthreads();
        if (tid == k - 1 && sh.s_pick < 0)
            sh.s_pick = my_id;
        __syncthreads();
    }
    SAMPLER_STAMP(4);
    if (tid == 0) {
        const int argmax = sh.s_argmax;
        const int sampled = sh.s_pick >= 0 ? sh.s_pick : argmax;
        *code_out = sampled;
        *argmax_out = argmax;
    }
    SAMPLER_STAMP(5);
}
