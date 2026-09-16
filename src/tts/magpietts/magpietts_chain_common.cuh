// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Device building blocks shared by the persistent (single-launch) MagpieTTS kernels: the
// local-transformer chain and the decoder stack. Planar Q8_0 GEMV phases with producer-side
// activation quantization and LayerNorm folding, cp.async weight staging, grid barriers,
// arrival elections and the optional phase-timing instrumentation.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "magpietts_cuda_sampling_device.cuh"

// ggml type ids (ggml.h): F32 = 0, F16 = 1, Q8_0 = 8.
static constexpr int LTF_TYPE_F32 = 0;
static constexpr int LTF_TYPE_F16 = 1;
static constexpr int LTF_TYPE_Q8_0 = 8;
static constexpr int LTF_LANES = 2;
#ifndef LTF_THREADS_OVERRIDE
#define LTF_THREADS_OVERRIDE 256
#endif
// Resident blocks per SM. Fewer, fatter blocks mean fewer barrier participants and a smaller
// spread of per-block latency chains at each barrier; more blocks mean more warps per SM to
// hide latency within a phase.
#ifndef LTF_CHAIN_BLOCKS_PER_SM
#define LTF_CHAIN_BLOCKS_PER_SM 1
#endif
// Register cap for the chain kernel. The kernel is resident on every SM for the whole step;
// leaving registers (and shared memory, see the carveout in setup) free lets the concurrent
// NanoCodec kernels co-schedule on the same SMs instead of serializing behind the chain.
#ifndef LTF_CHAIN_MAX_REGS
#define LTF_CHAIN_MAX_REGS 168
#endif
#ifndef LTF_ROWS_PER_WARP_OVERRIDE
#define LTF_ROWS_PER_WARP_OVERRIDE 4
#endif
static constexpr int LTF_THREADS = LTF_THREADS_OVERRIDE;
static constexpr int LTF_WARPS = LTF_THREADS / 32;
static constexpr int LTF_MAX_EMBD = 1024;  // max LayerNorm width
static constexpr int LTF_MAX_K = 3072;     // max GEMV input width (FFN hidden)
static constexpr int LTF_MAX_POS = 32;

// Planar Q8 matrix: quants int8 [N][K] (16-byte aligned rows since K % 64 == 0) and f16
// scales [N][K/32]; repacked once from ggml Q8_0 (34-byte interleaved blocks) so the GEMV
// streams whole 32-byte sectors with two uint4 loads per block instead of ten 4-byte loads.
struct ltf_q8p {
    const int8_t* qs = nullptr;
    const __half* d = nullptr;
};


static inline void
ltf_set_error(char* error, size_t size, const char* msg, cudaError_t err = cudaSuccess) {
    if (!error || size == 0)
        return;
    if (err == cudaSuccess)
        snprintf(error, size, "%s", msg);
    else
        snprintf(error, size, "%s: %s", msg, cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------------------
// device helpers
// ---------------------------------------------------------------------------------------
static __device__ __forceinline__ float
ltf_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

static __device__ __forceinline__ float
ltf_gelu(float x) {
    const float GELU_COEF_A = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

static __device__ __forceinline__ float
ltf_load_scalar(const void* base, int type, size_t index) {
    if (type == LTF_TYPE_F16)
        return __half2float(reinterpret_cast<const __half*>(base)[index]);
    return reinterpret_cast<const float*>(base)[index];
}

// ---------------------------------------------------------------------------------------
// GEMV building blocks (M = 2 CFG lanes). One warp computes ROWS_PER_WARP output rows at a
// time; a block handles LTF_WARPS * ROWS_PER_WARP rows. Q8_0 weights are combined with
// activations quantized once per block to symmetric int8 (32-element blocks, like ggml's
// Q8_1 path) using dp4a; F16 weights use f32 activations.
// ---------------------------------------------------------------------------------------
static constexpr int LTF_ROWS_PER_WARP = LTF_ROWS_PER_WARP_OVERRIDE;
static constexpr int LTF_ROWS_PER_BLOCK = LTF_WARPS * LTF_ROWS_PER_WARP;

enum ltf_epilogue { LTF_EPI_STORE = 0, LTF_EPI_RESIDUAL = 1, LTF_EPI_GELU = 2, LTF_EPI_BIAS = 3 };

static __device__ __forceinline__ int
ltf_dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}

// Register-prefetched Q8 path: each lane owns up to MB blocks per row (blocks = K/32 <= 32*MB).
// Phase A loads every weight word before the activation prologue; phase B does the dp4a.
template <int R, int MB>
struct ltf_q8_prefetch {
    int q[R][MB][8];
    float d[R][MB];
};

template <int R, int MB>
static __device__ __forceinline__ void
ltf_prefetch_q8(
    const int8_t* Wq, const __half* Wd, int K, int N, int row0, int lane,
    ltf_q8_prefetch<R, MB>& pf) {
    const int blocks = K >> 5;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const bool valid = row0 + r < N;
        const int row = valid ? row0 + r : 0;
        const uint4* qrow = reinterpret_cast<const uint4*>(Wq + (size_t)row * K);
        const __half* drow = Wd + (size_t)row * blocks;
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            const int b = lane + m * 32;
            if (valid && b < blocks) {
                const uint4 lo = __ldg(qrow + b * 2);
                const uint4 hi = __ldg(qrow + b * 2 + 1);
                pf.q[r][m][0] = (int)lo.x;
                pf.q[r][m][1] = (int)lo.y;
                pf.q[r][m][2] = (int)lo.z;
                pf.q[r][m][3] = (int)lo.w;
                pf.q[r][m][4] = (int)hi.x;
                pf.q[r][m][5] = (int)hi.y;
                pf.q[r][m][6] = (int)hi.z;
                pf.q[r][m][7] = (int)hi.w;
                pf.d[r][m] = __half2float(drow[b]);
            } else {
                pf.d[r][m] = 0.f;
#pragma unroll
                for (int i = 0; i < 8; ++i) pf.q[r][m][i] = 0;
            }
        }
    }
}

template <int R, int MB>
static __device__ __forceinline__ void
ltf_dot_q8_prefetched(
    const ltf_q8_prefetch<R, MB>& pf, int K, const int* xq0, const int* xq1, const float* xd0,
    const float* xd1, int lane, float (&acc0)[R], float (&acc1)[R]) {
    const int blocks = K >> 5;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc0[r] = 0.f;
        acc1[r] = 0.f;
    }
#pragma unroll
    for (int m = 0; m < MB; ++m) {
        const int b = lane + m * 32;
        if (b < blocks) {
            const int* u0 = xq0 + b * 8;
            const int* u1 = xq1 + b * 8;
            int x0[8], x1[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                x0[i] = u0[i];
                x1[i] = u1[i];
            }
            const float e0 = xd0[b];
            const float e1 = xd1[b];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    s0 = ltf_dp4a(pf.q[r][m][i], x0[i], s0);
                    s1 = ltf_dp4a(pf.q[r][m][i], x1[i], s1);
                }
                acc0[r] += pf.d[r][m] * e0 * (float)s0;
                acc1[r] += pf.d[r][m] * e1 * (float)s1;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc0[r] = ltf_warp_sum(acc0[r]);
        acc1[r] = ltf_warp_sum(acc1[r]);
    }
}

// Same dot, but the int8 activations come from a producer-quantized global buffer with XG
// elements per scale (32: one scale per Q8 block; 16: two scales per block).
template <int R, int MB, int XG>
static __device__ __forceinline__ void
ltf_dot_q8_prefetched_xq(
    const ltf_q8_prefetch<R, MB>& pf, int K, const int8_t* xq, const float* xd, int lane,
    float (&acc0)[R], float (&acc1)[R]) {
    const int blocks = K >> 5;
    const int* xq0 = reinterpret_cast<const int*>(xq);
    const int* xq1 = reinterpret_cast<const int*>(xq + K);
    const float* xd0 = xd;
    const float* xd1 = xd + K / XG;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc0[r] = 0.f;
        acc1[r] = 0.f;
    }
#pragma unroll
    for (int m = 0; m < MB; ++m) {
        const int b = lane + m * 32;
        if (b < blocks) {
            int x0[8], x1[8];
            const int4* p0 = reinterpret_cast<const int4*>(xq0 + b * 8);
            const int4* p1 = reinterpret_cast<const int4*>(xq1 + b * 8);
            const int4 a0 = p0[0], b0 = p0[1], a1 = p1[0], b1 = p1[1];
            x0[0] = a0.x;
            x0[1] = a0.y;
            x0[2] = a0.z;
            x0[3] = a0.w;
            x0[4] = b0.x;
            x0[5] = b0.y;
            x0[6] = b0.z;
            x0[7] = b0.w;
            x1[0] = a1.x;
            x1[1] = a1.y;
            x1[2] = a1.z;
            x1[3] = a1.w;
            x1[4] = b1.x;
            x1[5] = b1.y;
            x1[6] = b1.z;
            x1[7] = b1.w;
            if (XG == 32) {
                const float e0 = xd0[b];
                const float e1 = xd1[b];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int s0 = 0, s1 = 0;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        s0 = ltf_dp4a(pf.q[r][m][i], x0[i], s0);
                        s1 = ltf_dp4a(pf.q[r][m][i], x1[i], s1);
                    }
                    acc0[r] += pf.d[r][m] * e0 * (float)s0;
                    acc1[r] += pf.d[r][m] * e1 * (float)s1;
                }
            } else if (XG == 16) {
                const float2 e0 = *reinterpret_cast<const float2*>(xd0 + 2 * b);
                const float2 e1 = *reinterpret_cast<const float2*>(xd1 + 2 * b);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int s0a = 0, s0b = 0, s1a = 0, s1b = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        s0a = ltf_dp4a(pf.q[r][m][i], x0[i], s0a);
                        s1a = ltf_dp4a(pf.q[r][m][i], x1[i], s1a);
                        s0b = ltf_dp4a(pf.q[r][m][4 + i], x0[4 + i], s0b);
                        s1b = ltf_dp4a(pf.q[r][m][4 + i], x1[4 + i], s1b);
                    }
                    acc0[r] += pf.d[r][m] * (e0.x * (float)s0a + e0.y * (float)s0b);
                    acc1[r] += pf.d[r][m] * (e1.x * (float)s1a + e1.y * (float)s1b);
                }
            } else {  // XG == 4: one scale per int (4 activations)
                const float4 ea0 = *reinterpret_cast<const float4*>(xd0 + 8 * b);
                const float4 eb0 = *reinterpret_cast<const float4*>(xd0 + 8 * b + 4);
                const float4 ea1 = *reinterpret_cast<const float4*>(xd1 + 8 * b);
                const float4 eb1 = *reinterpret_cast<const float4*>(xd1 + 8 * b + 4);
                const float e0[8] = {ea0.x, ea0.y, ea0.z, ea0.w, eb0.x, eb0.y, eb0.z, eb0.w};
                const float e1[8] = {ea1.x, ea1.y, ea1.z, ea1.w, eb1.x, eb1.y, eb1.z, eb1.w};
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float t0 = 0.f, t1 = 0.f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        t0 += e0[i] * (float)ltf_dp4a(pf.q[r][m][i], x0[i], 0);
                        t1 += e1[i] * (float)ltf_dp4a(pf.q[r][m][i], x1[i], 0);
                    }
                    acc0[r] += pf.d[r][m] * t0;
                    acc1[r] += pf.d[r][m] * t1;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc0[r] = ltf_warp_sum(acc0[r]);
        acc1[r] = ltf_warp_sum(acc1[r]);
    }
}

// F16 weights x f32 activations (xn0/xn1 in shared memory, 16-byte aligned).
static __device__ __forceinline__ void
ltf_rows_dot_f16(
    const __half* W, int K, int row0, int row1, const float* xn0, const float* xn1, int lane,
    float& acc00, float& acc01, float& acc10, float& acc11) {
    const int chunks = K >> 3;
    const uint4* r0 = reinterpret_cast<const uint4*>(W + (size_t)row0 * K);
    const uint4* r1 = row1 >= 0 ? reinterpret_cast<const uint4*>(W + (size_t)row1 * K) : nullptr;
    float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
    for (int c = lane; c < chunks; c += 32) {
        const uint4 p0 = __ldg(r0 + c);
        uint4 p1 = make_uint4(0, 0, 0, 0);
        if (r1)
            p1 = __ldg(r1 + c);
        const int k = c * 8;
        const float4 x0a = *reinterpret_cast<const float4*>(xn0 + k);
        const float4 x0b = *reinterpret_cast<const float4*>(xn0 + k + 4);
        const float4 x1a = *reinterpret_cast<const float4*>(xn1 + k);
        const float4 x1b = *reinterpret_cast<const float4*>(xn1 + k + 4);
        const __half2* h0 = reinterpret_cast<const __half2*>(&p0);
        const __half2* h1 = reinterpret_cast<const __half2*>(&p1);
        float w[8], v[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f0 = __half22float2(h0[i]);
            const float2 f1 = __half22float2(h1[i]);
            w[2 * i] = f0.x;
            w[2 * i + 1] = f0.y;
            v[2 * i] = f1.x;
            v[2 * i + 1] = f1.y;
        }
        const float xa[8] = {x0a.x, x0a.y, x0a.z, x0a.w, x0b.x, x0b.y, x0b.z, x0b.w};
        const float xb[8] = {x1a.x, x1a.y, x1a.z, x1a.w, x1b.x, x1b.y, x1b.z, x1b.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            a00 += w[i] * xa[i];
            a01 += w[i] * xb[i];
            a10 += v[i] * xa[i];
            a11 += v[i] * xb[i];
        }
    }
    acc00 = ltf_warp_sum(a00);
    acc01 = ltf_warp_sum(a01);
    acc10 = ltf_warp_sum(a10);
    acc11 = ltf_warp_sum(a11);
}

// Block-wide LayerNorm of two lanes of K values (x in shared, result xn in shared); one
// reduction pass over sum and sum of squares.
static __device__ __forceinline__ void
ltf_block_layernorm(const float* x, float* xn, const float* weight, int K, float eps, float* red) {
    const int tid = threadIdx.x;
    float s0 = 0.f, s1 = 0.f, q0 = 0.f, q1 = 0.f;
#pragma unroll 4
    for (int k = tid; k < K; k += LTF_THREADS) {
        const float a = x[k];
        const float b = x[K + k];
        s0 += a;
        q0 += a * a;
        s1 += b;
        q1 += b * b;
    }
    s0 = ltf_warp_sum(s0);
    s1 = ltf_warp_sum(s1);
    q0 = ltf_warp_sum(q0);
    q1 = ltf_warp_sum(q1);
    if ((tid & 31) == 0) {
        red[(tid >> 5) * 4 + 0] = s0;
        red[(tid >> 5) * 4 + 1] = s1;
        red[(tid >> 5) * 4 + 2] = q0;
        red[(tid >> 5) * 4 + 3] = q1;
    }
    __syncthreads();
    float S0 = 0.f, S1 = 0.f, Q0 = 0.f, Q1 = 0.f;
#pragma unroll
    for (int w = 0; w < LTF_WARPS; ++w) {
        S0 += red[w * 4 + 0];
        S1 += red[w * 4 + 1];
        Q0 += red[w * 4 + 2];
        Q1 += red[w * 4 + 3];
    }
    const float m0 = S0 / (float)K;
    const float m1 = S1 / (float)K;
    const float r0 = rsqrtf(fmaxf(Q0 / (float)K - m0 * m0, 0.f) + eps);
    const float r1 = rsqrtf(fmaxf(Q1 / (float)K - m1 * m1, 0.f) + eps);
#pragma unroll 4
    for (int k = tid; k < K; k += LTF_THREADS) {
        xn[k] = (x[k] - m0) * r0 * weight[k];
        xn[K + k] = (x[K + k] - m1) * r1 * weight[k];
    }
    __syncthreads();
}

static constexpr int LTF_MAX_LAYERS = 4;
static constexpr int LTF_MAX_ROUNDS = LTF_MAX_POS;
// Rows per warp of each GEMV phase. The grid is two blocks per SM (>= LTF_CHAIN_MIN_GRID blocks),
// so a phase of N rows gives each block ceil(N / grid) rows spread over LTF_WARPS warps; setup
// verifies these bounds for the actual shapes and grid.
static constexpr int LTF_CHAIN_MIN_GRID = 64 * LTF_CHAIN_BLOCKS_PER_SM;
static constexpr int LTF_CHAIN_MAX_GRID = 256;
constexpr int
ltf_rows_per_warp(int rows_at_128_blocks) {
    return (rows_at_128_blocks * 2 / LTF_CHAIN_BLOCKS_PER_SM + LTF_WARPS - 1) / LTF_WARPS;
}
static constexpr int LTF_R_QKV = ltf_rows_per_warp(18);  // 3K rows:  2304 / 128 = 18
static constexpr int LTF_R_O = ltf_rows_per_warp(6);     // K rows:    768 / 128 = 6
static constexpr int LTF_R_FF1 = ltf_rows_per_warp(24);  // NFF rows: 3072 / 128 = 24
static constexpr int LTF_R_OUT = ltf_rows_per_warp(16);  // vocab:    2024 / 128 = 16
static constexpr int LTF_STAGE_BYTES =
    LTF_CHAIN_BLOCKS_PER_SM == 1 ? 36864 : 24576;  // max staged quants per block
static constexpr int LTF_STAGE_SCALES =
    LTF_CHAIN_BLOCKS_PER_SM == 1 ? 1152 : 768;  // max staged scales per block
#ifdef LTF_CHAIN_TIMING
// Microbenchmark instrumentation (-DLTF_CHAIN_TIMING): block 0 stamps %globaltimer when it
// arrives at and leaves every grid barrier: [round][phase][arrive, release], plus start/end.
static constexpr int LTF_TIMING_PHASES = 16;
__device__ unsigned long long g_ltf_chain_timing[2 + LTF_MAX_ROUNDS * LTF_TIMING_PHASES * 2];
static __device__ __forceinline__ unsigned long long
ltf_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#define LTF_STAMP(idx)                                     \
    do {                                                   \
        if (tile == 0 && threadIdx.x == 0)                 \
            g_ltf_chain_timing[(idx)] = ltf_globaltimer(); \
    } while (0)
// Round-8 detail: [phase][6] sub-phase stamps of block 0 inside the GEMV, and every block's
// barrier arrival time [phase][block] to separate imbalance from barrier latency.
static constexpr int LTF_TIMING_DETAIL_ROUND = 8;
static constexpr int LTF_TIMING_MAX_GRID = 512;
__device__ unsigned long long g_ltf_chain_detail[LTF_TIMING_PHASES * 6];
__device__ unsigned long long g_ltf_chain_arrive[LTF_TIMING_PHASES * LTF_TIMING_MAX_GRID];
__device__ int g_ltf_chain_detail_phase;  // set by block 0 as it enters each phase of round 8
#define LTF_DETAIL(sub)                                                                   \
    do {                                                                                  \
        if (tile == 0 && threadIdx.x == 0 && g_ltf_chain_detail_phase >= 0)               \
            g_ltf_chain_detail[g_ltf_chain_detail_phase * 6 + (sub)] = ltf_globaltimer(); \
    } while (0)
#define LTF_ARRIVE(ph)                                                                 \
    do {                                                                               \
        if (threadIdx.x == 0 && (ph) >= 0 && tile < LTF_TIMING_MAX_GRID)               \
            g_ltf_chain_arrive[(ph) * LTF_TIMING_MAX_GRID + tile] = ltf_globaltimer(); \
    } while (0)
#else
#define LTF_STAMP(idx) \
    do {               \
    } while (0)
#define LTF_DETAIL(sub) \
    do {                \
    } while (0)
#define LTF_ARRIVE(ph) \
    do {               \
    } while (0)
#endif

// Repack a ggml Q8_0 device tensor [N rows][K] into planar quants/scales device buffers.
static inline bool
ltf_repack_q8(const void* w, int N, int K, std::vector<void*>& owned, ltf_q8p& out) {
    const int blocks = K / 32;
    const size_t src_bytes = (size_t)N * blocks * 34;
    std::vector<uint8_t> host(src_bytes);
    if (cudaMemcpy(host.data(), w, src_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    std::vector<int8_t> qs((size_t)N * K);
    std::vector<uint16_t> d((size_t)N * blocks);
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks; ++b) {
            const uint8_t* blk = host.data() + ((size_t)n * blocks + b) * 34;
            uint16_t scale;
            memcpy(&scale, blk, 2);
            d[(size_t)n * blocks + b] = scale;
            memcpy(qs.data() + (size_t)n * K + (size_t)b * 32, blk + 2, 32);
        }
    }
    int8_t* d_qs = nullptr;
    __half* d_d = nullptr;
    if (cudaMalloc(&d_qs, qs.size()) != cudaSuccess)
        return false;
    owned.push_back(d_qs);
    if (cudaMalloc(&d_d, d.size() * sizeof(uint16_t)) != cudaSuccess)
        return false;
    owned.push_back(d_d);
    if (cudaMemcpy(d_qs, qs.data(), qs.size(), cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    if (cudaMemcpy(d_d, d.data(), d.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) !=
        cudaSuccess)
        return false;
    out.qs = d_qs;
    out.d = d_d;
    return true;
}

// c[r] = sum_k gamma[k] * dequant(W[r][k]) for the LayerNorm fold: LN(x) W = r * ((x*gamma) W - mu
// * c).
static __global__ void
ltf_ln_fold_kernel(const int8_t* qs, const __half* d, const float* gamma, int N, int K, float* c) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= N)
        return;
    const int blocks = K >> 5;
    float acc = 0.f;
    for (int b = lane; b < blocks; b += 32) {
        float s = 0.f;
        const int8_t* q = qs + (size_t)row * K + (size_t)b * 32;
#pragma unroll
        for (int i = 0; i < 32; ++i) s += gamma[b * 32 + i] * (float)q[i];
        acc += s * __half2float(d[(size_t)row * blocks + b]);
    }
    acc = ltf_warp_sum(acc);
    if (lane == 0)
        c[row] = acc;
}

static inline bool
ltf_ln_fold_constants(
    const ltf_q8p& W, const float* gamma, int N, int K, std::vector<void*>& owned,
    const float*& out) {
    float* c = nullptr;
    if (cudaMalloc(&c, (size_t)N * sizeof(float)) != cudaSuccess)
        return false;
    owned.push_back(c);
    ltf_ln_fold_kernel<<<(N + 7) / 8, 256>>>(W.qs, W.d, gamma, N, K, c);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
        return false;
    out = c;
    return true;
}

// Shared memory of the chain kernel. The GEMV scratch and the sampler scratch are never live in
// the same phase and alias each other.
struct ltf_chain_gemv_smem {
    __align__(16) int s_q[LTF_LANES * (LTF_MAX_K / 4)];
    __align__(16) float s_d[LTF_LANES * (LTF_MAX_K / 16)];  // scales for groups >= 16
    float s_red[4 * LTF_WARPS];
    __align__(
        16) float s_x[LTF_LANES * LTF_MAX_EMBD];  // staged input (LN phases) / attention output
};
union ltf_chain_smem_union {
    ltf_chain_gemv_smem g;
    magpietts_sample_fast_shared samp;
};
struct ltf_chain_smem {
    __align__(16) int8_t s_w[LTF_STAGE_BYTES];
    __align__(16) __half s_wd[LTF_STAGE_SCALES];
    ltf_chain_smem_union u;
};

// Weights are streamed once per round (17 MB/round vs a 6 MB L2), so they are fetched with an
// evict-first L2 policy; the small activation buffers and the K/V cache then stay L2-resident
// across rounds instead of being flushed by the weight stream.
static __device__ __forceinline__ unsigned long long
ltf_evict_first_policy() {
    unsigned long long policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
    return policy;
}
static __device__ __forceinline__ void
ltf_cp_async16(void* smem, const void* gmem, unsigned long long policy) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem),
        "l"(policy));
}
static __device__ __forceinline__ void
ltf_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
static __device__ __forceinline__ void
ltf_cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

static __device__ __forceinline__ void
ltf_grid_barrier(unsigned int* counter, unsigned int& target, unsigned int nblocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        target += nblocks;
        // Release this block's writes with the arrival, acquire everyone else's with the poll
        // (no fences, no back-off: measured ~7% faster per step than fence + atomic + nanosleep).
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(counter) : "memory");
        unsigned int seen;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                         : "=r"(seen)
                         : "l"(counter)
                         : "memory");
        } while (seen < target);
    }
    __syncthreads();
}

// Arrival election for the current phase: every block increments a monotonic counter (reset per
// launch; a grid barrier separates uses). Returns this block's arrival rank in [0, nblocks); the
// blocks with rank >= nblocks - n_tail spin until all blocks have arrived and then own the tail
// work (e.g. attention over the just-completed K/V rows) in parallel.
static __device__ __forceinline__ int
ltf_chain_arrival_rank(unsigned int* counter, unsigned int nblocks, int n_tail, int* s_flag) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned int prev = atomicAdd(counter, 1u);
        const unsigned int rank = prev % nblocks;
        const unsigned int done_value = prev - rank + nblocks;
        if ((int)rank >= (int)nblocks - n_tail) {
            while (*reinterpret_cast<volatile unsigned int*>(counter) < done_value) {
            }
            __threadfence();
        }
        *s_flag = (int)rank;
    }
    __syncthreads();
    return *s_flag;
}

// Rows of an N-row projection owned by block `tile` of `nblocks`: an even split so every
// resident block streams the same number of weight bytes in every phase (the grid is a whole
// multiple of the SM count, so per-SM load is balanced too).
static __device__ __forceinline__ void
ltf_chain_rows(int N, int tile, int nblocks, int& row_begin, int& rows) {
    row_begin = (int)(((long long)N * tile) / nblocks);
    rows = (int)(((long long)N * (tile + 1)) / nblocks) - row_begin;
}

// Stage this block's rows of W (contiguous in the planar layout) into shared memory
// asynchronously. Callers must have finished reading the previous stage (__syncthreads)
// before issuing.
static __device__ __forceinline__ void
ltf_chain_stage(const ltf_q8p& W, int K, int N, int tile, ltf_chain_smem& sm) {
    int row_begin, rows;
    ltf_chain_rows(N, tile, gridDim.x, row_begin, rows);
    if (rows <= 0)
        return;
    const int blocks = K >> 5;
    const int qbytes = rows * K;
    const int dbytes = (rows * blocks * 2 + 15) & ~15;
    const int8_t* gq = W.qs + (size_t)row_begin * K;
    const int8_t* gd = reinterpret_cast<const int8_t*>(W.d + (size_t)row_begin * blocks);
    int8_t* sd = reinterpret_cast<int8_t*>(sm.s_wd);
    const unsigned long long policy = ltf_evict_first_policy();
    for (int i = threadIdx.x * 16; i < qbytes; i += LTF_THREADS * 16)
        ltf_cp_async16(sm.s_w + i, gq + i, policy);
    for (int i = threadIdx.x * 16; i < dbytes; i += LTF_THREADS * 16)
        ltf_cp_async16(sd + i, gd + i, policy);
    ltf_cp_async_commit();
}

// Load this warp's rows of the staged tile into the register prefetch struct.
template <int R, int MB>
static __device__ __forceinline__ void
ltf_chain_load_staged(
    const ltf_chain_smem& sm, int K, int rows_valid, int wrow0, int lane,
    ltf_q8_prefetch<R, MB>& pf) {
    const int blocks = K >> 5;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = wrow0 + r;
        const bool valid = row < rows_valid;
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            const int b = lane + m * 32;
            if (valid && b < blocks) {
                const uint4* q = reinterpret_cast<const uint4*>(sm.s_w + (size_t)row * K + b * 32);
                const uint4 lo = q[0];
                const uint4 hi = q[1];
                pf.q[r][m][0] = (int)lo.x;
                pf.q[r][m][1] = (int)lo.y;
                pf.q[r][m][2] = (int)lo.z;
                pf.q[r][m][3] = (int)lo.w;
                pf.q[r][m][4] = (int)hi.x;
                pf.q[r][m][5] = (int)hi.y;
                pf.q[r][m][6] = (int)hi.z;
                pf.q[r][m][7] = (int)hi.w;
                pf.d[r][m] = __half2float(sm.s_wd[row * blocks + b]);
            } else {
                pf.d[r][m] = 0.f;
#pragma unroll
                for (int i = 0; i < 8; ++i) pf.q[r][m][i] = 0;
            }
        }
    }
}

struct ltf_chain_phase_args {
    const float* x;       // [2][K] global input (ignored when x_smem or assemble)
    const float* x_smem;  // [2][K] shared-memory input (attention output), or nullptr
    // Producer-quantized input: int8 [2][K] with one f32 scale per xgroup (32 or 16) elements.
    // When set, the prologue is skipped and the dot reads these directly.
    const int8_t* xq;
    const float* xd;
    int xgroup;
    // Producer-side quantization of this phase's output: int8 [2][N] in groups of out_group
    // rows (this block's row range must be group-aligned); the f32 output is not written.
    int8_t* out_q;
    float* out_d;
    int out_group;
    int out_keep_f32;        // also write the f32 output (residual stream producers)
    const float* out_gamma;  // multiply by gamma[row] before quantizing (LN fold for the consumer)
    float* out_stats;        // [2][2][nblocks]: this block's partial sum / sumsq of the f32 output
    // LN fold on the consumer side: input is q(x*gamma); mu, r come from the producers' partial
    // stats and the epilogue applies v = r * (acc - mu * ln_c[row]).
    const float* stats;
    const float* ln_c;
    const float* norm_w;  // LayerNorm weight or nullptr
    int N, K;
    float* out;         // [2][N]
    const float* bias;  // [N] or nullptr
    // input assembly (in-block LayerNorm path): x = h_in + pos, or emb_row + pos, or
    // emb_scale * sum(emb_rows[i]) + pos (decoder: mean of the codebook embeddings)
    int assemble;
    const float* h_in;
    const void* emb_row;  // embedding row of the previous code (F16/F32), or nullptr for round 0
    const void* const* emb_rows;  // n_emb_rows embedding rows to sum, or nullptr
    int n_emb_rows;
    float emb_scale;
    const void* pos_row;  // positional row
    int emb_type, pos_type;
    float* x_store;  // residual stream initialized by tile 0 when assembling
    float ln_eps;
    // Residual lanes that receive this phase's result (bit 0: lane 0, bit 1: lane 1; 0 = both).
    // Disabled lanes keep their residual value (decoder cross-attention updates the cond lane
    // only).
    int lane_mask;
    // K/V cache write for rows >= K (qkv phase): element (lane, row, f) at
    // kv_*[lane * kv_lane_stride + kv_row * kv_row_stride + f]
    float* kcache;
    float* vcache;
    size_t kv_lane_stride, kv_row_stride;
    int kv_row;
};

static __device__ __forceinline__ float
ltf_chain_input(const ltf_chain_phase_args& a, int l, int k) {
    if (a.x_smem)
        return a.x_smem[(size_t)l * a.K + k];
    if (!a.assemble)
        return a.x[(size_t)l * a.K + k];
    const float p = ltf_load_scalar(a.pos_row, a.pos_type, (size_t)k);
    if (a.emb_rows) {
        float s = 0.f;
        for (int i = 0; i < a.n_emb_rows; ++i)
            s += ltf_load_scalar(a.emb_rows[i], a.emb_type, (size_t)k);
        return s * a.emb_scale + p;
    }
    if (!a.emb_row)
        return a.h_in[(size_t)l * a.K + k] + p;
    return ltf_load_scalar(a.emb_row, a.emb_type, (size_t)k) + p;
}

// One GEMV phase on this block's staged tile: prologue (optional assembly + LayerNorm stats +
// int8 quantization of both lanes), dp4a dot, epilogue. Leaves the block synchronized so the
// caller may immediately restage the shared weight buffer.
template <int EPI, int R, int MB>
static __device__ __forceinline__ void
ltf_chain_gemv(const ltf_chain_phase_args& a, int tile, ltf_chain_smem& sm) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int K = a.K;
    const int blocks = K >> 5;
    ltf_chain_gemv_smem& g = sm.u.g;
    int row_begin, rows_valid;
    ltf_chain_rows(a.N, tile, gridDim.x, row_begin, rows_valid);
    if (rows_valid <= 0)
        return;                  // block-uniform: no rows for this block
    const int wrow0 = warp * R;  // this warp's rows within the block's staged range
    const int row0 = row_begin + wrow0;
    const bool staged = a.norm_w != nullptr;  // LN phases stage the input in shared memory
    LTF_DETAIL(0);

    // Residual/bias operands for this warp's rows, loaded early to hide their latency.
    float res0[R], res1[R], bia[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
        res0[r] = res1[r] = bia[r] = 0.f;
        if (lane == 0 && row < a.N) {
            if (EPI == LTF_EPI_RESIDUAL) {
                res0[r] = a.out[row];
                res1[r] = a.out[a.N + row];
            }
            if (a.bias)
                bia[r] = a.bias[row];
        }
    }

    float m0 = 0.f, m1 = 0.f, r0 = 1.f, r1 = 1.f;
    float stats_partial = 0.f;
    if (staged) {
        float s0 = 0.f, s1 = 0.f, q0 = 0.f, q1 = 0.f;
#pragma unroll 3
        for (int k = tid; k < K; k += LTF_THREADS) {
            const float x0 = ltf_chain_input(a, 0, k);
            const float x1 = (a.assemble && a.emb_rows) ? x0 : ltf_chain_input(a, 1, k);
            g.s_x[k] = x0;
            g.s_x[K + k] = x1;
            if (a.assemble && tile == 0) {
                a.x_store[k] = x0;
                a.x_store[K + k] = x1;
            }
            s0 += x0;
            q0 += x0 * x0;
            s1 += x1;
            q1 += x1 * x1;
        }
        s0 = ltf_warp_sum(s0);
        s1 = ltf_warp_sum(s1);
        q0 = ltf_warp_sum(q0);
        q1 = ltf_warp_sum(q1);
        if (lane == 0) {
            g.s_red[warp * 4 + 0] = s0;
            g.s_red[warp * 4 + 1] = s1;
            g.s_red[warp * 4 + 2] = q0;
            g.s_red[warp * 4 + 3] = q1;
        }
        __syncthreads();
        float S0 = 0.f, S1 = 0.f, Q0 = 0.f, Q1 = 0.f;
#pragma unroll
        for (int w = 0; w < LTF_WARPS; ++w) {
            S0 += g.s_red[w * 4 + 0];
            S1 += g.s_red[w * 4 + 1];
            Q0 += g.s_red[w * 4 + 2];
            Q1 += g.s_red[w * 4 + 3];
        }
        m0 = S0 / (float)K;
        m1 = S1 / (float)K;
        r0 = rsqrtf(fmaxf(Q0 / (float)K - m0 * m0, 0.f) + a.ln_eps);
        r1 = rsqrtf(fmaxf(Q1 / (float)K - m1 * m1, 0.f) + a.ln_eps);
        LTF_DETAIL(1);
    } else if (a.stats) {
        // LN statistics from the producers' per-block partial sums (deterministic order). Only
        // the loads are issued here; the reduction runs after the dot, which does not need them.
        const int nb = (int)gridDim.x;
        const int grp = tid >> 6;  // 0: l0 sum, 1: l0 sumsq, 2: l1 sum, 3: l1 sumsq
        for (int b = tid & 63; b < nb; b += 64) stats_partial += a.stats[grp * nb + b];
    } else if (a.assemble && tile == 0) {
#pragma unroll 3
        for (int k = tid; k < K; k += LTF_THREADS) {
            a.x_store[k] = ltf_chain_input(a, 0, k);
            a.x_store[K + k] = ltf_chain_input(a, 1, k);
        }
    }
    if (!a.xq) {
        const int group = lane >> 3;
        const int sub = lane & 7;
        const int total = 2 * blocks;
        constexpr int PF = 4;
        for (int base = warp * 4 + group; base < total; base += LTF_WARPS * 4 * PF) {
            float4 v[PF];
#pragma unroll
            for (int j = 0; j < PF; ++j) {
                const int idx = base + j * LTF_WARPS * 4;
                if (idx < total) {
                    const int l = idx / blocks;
                    const int b = idx - l * blocks;
                    const int k0 = b * 32 + sub * 4;
                    if (staged) {
                        v[j] = *reinterpret_cast<const float4*>(g.s_x + (size_t)l * K + k0);
                    } else if (a.x_smem) {
                        v[j] = *reinterpret_cast<const float4*>(a.x_smem + (size_t)l * K + k0);
                    } else if (!a.assemble) {
                        v[j] = __ldg(reinterpret_cast<const float4*>(a.x + (size_t)l * K + k0));
                    } else {
                        v[j] = make_float4(
                            ltf_chain_input(a, l, k0), ltf_chain_input(a, l, k0 + 1),
                            ltf_chain_input(a, l, k0 + 2), ltf_chain_input(a, l, k0 + 3));
                    }
                } else {
                    v[j] = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
#pragma unroll
            for (int j = 0; j < PF; ++j) {
                const int idx = base + j * LTF_WARPS * 4;
                if (idx >= total)
                    break;
                const int l = idx / blocks;
                const int b = idx - l * blocks;
                const int k0 = b * 32 + sub * 4;
                float4 x = v[j];
                if (staged) {
                    const float m = l == 0 ? m0 : m1;
                    const float r = l == 0 ? r0 : r1;
                    const float4 w4 = *reinterpret_cast<const float4*>(a.norm_w + k0);
                    x.x = (x.x - m) * r * w4.x;
                    x.y = (x.y - m) * r * w4.y;
                    x.z = (x.z - m) * r * w4.z;
                    x.w = (x.w - m) * r * w4.w;
                }
                float amax = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
                const float inv = amax > 0.f ? 127.f / amax : 0.f;
                const int q0 = __float2int_rn(x.x * inv) & 0xff;
                const int q1 = __float2int_rn(x.y * inv) & 0xff;
                const int q2 = __float2int_rn(x.z * inv) & 0xff;
                const int q3 = __float2int_rn(x.w * inv) & 0xff;
                g.s_q[((size_t)l * (K >> 2)) + b * 8 + sub] =
                    q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
                if (sub == 0)
                    g.s_d[l * blocks + b] = amax / 127.f;
            }
        }
    }
    // Wide inputs (MB > 1, e.g. K = 3072) are read three times per warp with dependent global
    // round trips; stage them into shared memory once, overlapping the weight-tile wait.
    const bool x_in_smem = a.xq && MB > 1 && a.xgroup >= 16;
    if (x_in_smem) {
        const int n16 = (2 * K) / 16;  // int8 [2][K] as 16-byte chunks
        const int4* src = reinterpret_cast<const int4*>(a.xq);
        int4* dst = reinterpret_cast<int4*>(g.s_q);
        for (int i = tid; i < n16; i += LTF_THREADS) dst[i] = src[i];
        const int nd = 2 * (K / a.xgroup);
        for (int i = tid; i < nd; i += LTF_THREADS) g.s_d[i] = a.xd[i];
    }
    LTF_DETAIL(2);
    ltf_cp_async_wait_all();
    __syncthreads();
    LTF_DETAIL(3);

    float acc0[R], acc1[R];
    if (wrow0 < rows_valid) {
        ltf_q8_prefetch<R, MB> pf;
        ltf_chain_load_staged<R, MB>(sm, K, rows_valid, wrow0, lane, pf);
        if (!a.xq) {
            ltf_dot_q8_prefetched<R, MB>(
                pf, K, g.s_q, g.s_q + (K >> 2), g.s_d, g.s_d + blocks, lane, acc0, acc1);
        } else if (a.xgroup == 32) {
            ltf_dot_q8_prefetched_xq<R, MB, 32>(
                pf, K, x_in_smem ? reinterpret_cast<const int8_t*>(g.s_q) : a.xq,
                x_in_smem ? g.s_d : a.xd, lane, acc0, acc1);
        } else if (a.xgroup == 16) {
            ltf_dot_q8_prefetched_xq<R, MB, 16>(
                pf, K, x_in_smem ? reinterpret_cast<const int8_t*>(g.s_q) : a.xq,
                x_in_smem ? g.s_d : a.xd, lane, acc0, acc1);
        } else {
            ltf_dot_q8_prefetched_xq<R, MB, 4>(pf, K, a.xq, a.xd, lane, acc0, acc1);
        }
    }
    LTF_DETAIL(4);
    if (a.stats && !staged) {
        const float v = ltf_warp_sum(stats_partial);
        if (lane == 0)
            g.s_red[warp] = v;
        __syncthreads();
        const float S0 = g.s_red[0] + g.s_red[1], Q0 = g.s_red[2] + g.s_red[3];
        const float S1 = g.s_red[4] + g.s_red[5], Q1 = g.s_red[6] + g.s_red[7];
        m0 = S0 / (float)K;
        m1 = S1 / (float)K;
        r0 = rsqrtf(fmaxf(Q0 / (float)K - m0 * m0, 0.f) + a.ln_eps);
        r1 = rsqrtf(fmaxf(Q1 / (float)K - m1 * m1, 0.f) + a.ln_eps);
    }
    if (lane == 0 && wrow0 < rows_valid) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (wrow0 + r >= rows_valid)
                break;  // rows past this block's range belong to another block
            float v0 = acc0[r], v1 = acc1[r];
            if (a.ln_c) {  // LN fold: LN(x) W = r * ((x*gamma) W - mu * c)
                const float cr = a.ln_c[row];
                v0 = r0 * (v0 - m0 * cr);
                v1 = r1 * (v1 - m1 * cr);
            }
            v0 += bia[r];
            v1 += bia[r];
            if (EPI == LTF_EPI_GELU) {
                v0 = ltf_gelu(v0);
                v1 = ltf_gelu(v1);
            }
            if (EPI == LTF_EPI_RESIDUAL) {
                v0 = (a.lane_mask == 0 || (a.lane_mask & 1)) ? v0 + res0[r] : res0[r];
                v1 = (a.lane_mask == 0 || (a.lane_mask & 2)) ? v1 + res1[r] : res1[r];
            }
            if (a.out_q) {
                // stage for the producer-side quantization below (rows_valid <= 2 * LTF_MAX_EMBD)
                g.s_x[wrow0 + r] = v0;
                g.s_x[rows_valid + wrow0 + r] = v1;
            }
            if (!a.out_q || a.out_keep_f32) {
                a.out[row] = v0;
                a.out[a.N + row] = v1;
            }
            if (a.kcache && row >= K) {
                float* cache = row < 2 * K ? a.kcache : a.vcache;
                const int f = row < 2 * K ? row - K : row - 2 * K;
                cache[(size_t)a.kv_row * a.kv_row_stride + f] = v0;
                cache[a.kv_lane_stride + (size_t)a.kv_row * a.kv_row_stride + f] = v1;
            }
        }
    }
    __syncthreads();  // all warps done with the staged weights and the scratch
    if (a.out_q) {
        // Quantize this block's rows in groups of out_group (16 or 32) per CFG lane: one group
        // per out_group threads, amax over the group, symmetric int8, one f32 scale.
        const int G = a.out_group;
        const int n_groups = rows_valid / G;
        const int task = tid / G;  // (lane, group)
        const int j = tid - task * G;
        if (a.out_stats && warp < 2) {
            // partial LN statistics of this block's f32 output rows (lane = warp)
            float sv = 0.f, sq = 0.f;
            for (int i = lane; i < rows_valid; i += 32) {
                const float v = g.s_x[warp * rows_valid + i];
                sv += v;
                sq += v * v;
            }
            sv = ltf_warp_sum(sv);
            sq = ltf_warp_sum(sq);
            if (lane == 0) {
                a.out_stats[(warp * 2 + 0) * gridDim.x + tile] = sv;
                a.out_stats[(warp * 2 + 1) * gridDim.x + tile] = sq;
            }
        }
        // Every thread runs the group shuffles (full mask); only threads with a valid task store.
        const bool valid = task < 2 * n_groups;
        const int l = valid ? task / n_groups : 0;
        const int grp = valid ? task - l * n_groups : 0;
        const int grow = row_begin + grp * G;  // group-aligned by construction
        float v = valid ? g.s_x[l * rows_valid + grp * G + j] : 0.f;
        if (valid && a.out_gamma)
            v *= a.out_gamma[grow + j];
        float amax = fabsf(v);
        for (int o = G / 2; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        if (valid) {
            const float inv = amax > 0.f ? 127.f / amax : 0.f;
            a.out_q[(size_t)l * a.N + grow + j] = (int8_t)__float2int_rn(v * inv);
            if (j == 0)
                a.out_d[(size_t)l * (a.N / G) + grow / G] = amax / 127.f;
        }
    }
    LTF_DETAIL(5);
}
