// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "magpietts_cuda_sampling_device.cuh"
#include "magpietts_lt_fused.h"

// ggml type ids (ggml.h): F32 = 0, F16 = 1, Q8_0 = 8.
static constexpr int LTF_TYPE_F32 = 0;
static constexpr int LTF_TYPE_F16 = 1;
static constexpr int LTF_TYPE_Q8_0 = 8;
static constexpr int LTF_LANES = 2;
#ifndef LTF_THREADS_OVERRIDE
#define LTF_THREADS_OVERRIDE 256
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

struct ltf_layer {
    const void* qkv;
    const void* o;
    const void* ff1;
    const void* ff2;
    const float* norm_self;
    const float* norm_ff;
    int qkv_type, o_type, ff1_type, ff2_type;
    ltf_q8p qkv_p, o_p, ff1_p, ff2_p;
};

struct magpietts_lt_fused {
    magpietts_lt_fused_weights w{};
    std::vector<ltf_layer> layers;
    std::vector<const void*> audio_emb;
    std::vector<const void*> out_w;
    std::vector<const float*> out_b;
    std::vector<ltf_q8p> out_p;
    std::vector<void*> planar_buffers;  // owned device allocations
    // scratch (device)
    float* h_in = nullptr;    // [2][n_embd] round-0 inputs (cond, uncond)
    float* h = nullptr;       // [2][n_embd] residual stream
    float* qkv = nullptr;     // [2][3*n_embd]
    float* ctx = nullptr;     // [2][n_embd]
    float* ff = nullptr;      // [2][n_ff]
    float* logits = nullptr;  // [2][vocab]
    float* kcache = nullptr;  // [layers][2][pos_rows][n_embd]
    float* vcache = nullptr;
    int* sample_counters = nullptr;  // [n_rounds] last-block-done counters
    cudaStream_t capture_stream = nullptr;
    // persistent chain kernel (all rounds in one launch)
    unsigned int* barrier = nullptr;  // monotonic grid-barrier counter (reset before each launch)
    int chain_grid = 0;               // blocks (0 = chain unsupported)
    void* arena = nullptr;  // single allocation holding every scratch buffer (L2 access window)
    size_t arena_bytes = 0;
    bool l2_window = false;  // persisting-L2 window enabled for the arena
};

static void
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

// ---------------------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------------------
enum ltf_epilogue { LTF_EPI_STORE = 0, LTF_EPI_RESIDUAL = 1, LTF_EPI_GELU = 2, LTF_EPI_BIAS = 3 };

struct ltf_gemv_args {
    const float* x;       // [2][K] input (residual stream or activations)
    const float* norm_w;  // LayerNorm weight [K] or nullptr (no LN)
    const void* W;        // [N rows][K] (ggml layout; used by the F16 path)
    int w_type;
    const int8_t* Wq;  // planar Q8 quants [N][K]
    const __half* Wd;  // planar Q8 scales [N][K/32]
    int N, K;
    float* out;         // [2][N] (for RESIDUAL: out == residual stream, updated in place)
    const float* bias;  // [N] or nullptr
    // input assembly (round start): x := (c == 0 ? h_in : emb[codes[c-1]]) + pos[c]; written to
    // x_store
    int assemble;       // 0/1
    int round;          // c
    const float* h_in;  // [2][K]
    const void* emb;    // audio embedding table of codebook c-1 [vocab][K]
    int emb_type;
    const void* pos;  // pos_emb [pos_rows][K]
    int pos_type;
    const int32_t* codes;  // sampled codes of the frame
    float* x_store;        // [2][K] residual stream to initialize when assemble
    float ln_eps;
    // optional K/V cache write for qkv outputs: rows [K,2K) -> kcache, [2K,3K) -> vcache,
    // layout [lane][pos_rows][K]
    float* kcache;
    float* vcache;
    int pos_rows;
    int cache_pos;
    // optional fused sampling tail (output projection only)
    const magpietts_cuda_sampling_config* sample_config;
    int32_t* sample_codes;
    int32_t* sample_argmax;
    int32_t* sample_top_ids;
    float* sample_top_vals;
    int* sample_counter;
    int sample_codebook;
    int sample_vocab;
    int sample_codebook_size;
    int sample_eos_id;
};

// Assembled/raw input element k of CFG lane l (f32), read from global memory.
static __device__ __forceinline__ float
ltf_input_at(const ltf_gemv_args& a, int l, int k) {
    if (!a.assemble) {
        return a.x[(size_t)l * a.K + k];
    }
    const float p = ltf_load_scalar(a.pos, a.pos_type, (size_t)a.round * a.K + k);
    if (a.round == 0) {
        return a.h_in[(size_t)l * a.K + k] + p;
    }
    return ltf_load_scalar(a.emb, a.emb_type, (size_t)a.codes[a.round - 1] * a.K + k) + p;
}

// Q8 variant: no full-vector staging. LayerNorm statistics stream from global memory, the
// 8-lane-group quantizer applies (x - m) * r * w on the fly, and only the int8 activations
// (2K bytes) plus scales live in shared memory, so ~10 blocks fit per SM and the whole grid
// runs in one wave.
template <int EPI, int MB, bool SAMPLE>
static __global__ void
__launch_bounds__(LTF_THREADS) ltf_gemv_q8_kernel(ltf_gemv_args a) {
    __shared__ magpietts_sample_fast_shared s_samp[1];  // used only when SAMPLE
    __shared__ int s_last;
    __shared__ __align__(16) int s_q[LTF_LANES * (LTF_MAX_K / 4)];
    __shared__ float s_d[LTF_LANES * (LTF_MAX_K / 32)];
    __shared__ float s_red[4 * LTF_WARPS];
    __shared__ __align__(16) float s_x[LTF_LANES * LTF_MAX_EMBD];  // staged input for LN kernels
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int K = a.K;
    const int blocks = K >> 5;
    const bool staged = a.norm_w != nullptr;  // LN kernels have K <= LTF_MAX_EMBD

    // Phase A: prefetch this warp's weight blocks (independent of the activations).
    const int row0 = blockIdx.x * LTF_ROWS_PER_BLOCK + warp * LTF_ROWS_PER_WARP;
    ltf_q8_prefetch<LTF_ROWS_PER_WARP, MB> pf;
    ltf_prefetch_q8<LTF_ROWS_PER_WARP, MB>(a.Wq, a.Wd, K, a.N, row0, lane, pf);

    float m0 = 0.f, m1 = 0.f, r0 = 1.f, r1 = 1.f;
    if (a.norm_w) {
        float s0 = 0.f, s1 = 0.f, q0 = 0.f, q1 = 0.f;
#pragma unroll 4
        for (int k = tid; k < K; k += LTF_THREADS) {
            const float x0 = ltf_input_at(a, 0, k);
            const float x1 = ltf_input_at(a, 1, k);
            s_x[k] = x0;
            s_x[K + k] = x1;
            if (a.assemble && blockIdx.x == 0) {
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
            s_red[warp * 4 + 0] = s0;
            s_red[warp * 4 + 1] = s1;
            s_red[warp * 4 + 2] = q0;
            s_red[warp * 4 + 3] = q1;
        }
        __syncthreads();
        float S0 = 0.f, S1 = 0.f, Q0 = 0.f, Q1 = 0.f;
#pragma unroll
        for (int w = 0; w < LTF_WARPS; ++w) {
            S0 += s_red[w * 4 + 0];
            S1 += s_red[w * 4 + 1];
            Q0 += s_red[w * 4 + 2];
            Q1 += s_red[w * 4 + 3];
        }
        m0 = S0 / (float)K;
        m1 = S1 / (float)K;
        r0 = rsqrtf(fmaxf(Q0 / (float)K - m0 * m0, 0.f) + a.ln_eps);
        r1 = rsqrtf(fmaxf(Q1 / (float)K - m1 * m1, 0.f) + a.ln_eps);
    } else if (a.assemble && blockIdx.x == 0) {
#pragma unroll 4
        for (int k = tid; k < K; k += LTF_THREADS) {
            a.x_store[k] = ltf_input_at(a, 0, k);
            a.x_store[K + k] = ltf_input_at(a, 1, k);
        }
    }

    // ---- quantize: 8 lanes per 32-block, 4 blocks per warp iteration, loads for 4 iterations
    // issued before use ----
    {
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
                        v[j] = *reinterpret_cast<const float4*>(s_x + (size_t)l * K + k0);
                    } else if (!a.assemble) {
                        v[j] = __ldg(reinterpret_cast<const float4*>(a.x + (size_t)l * K + k0));
                    } else {
                        v[j] = make_float4(
                            ltf_input_at(a, l, k0), ltf_input_at(a, l, k0 + 1),
                            ltf_input_at(a, l, k0 + 2), ltf_input_at(a, l, k0 + 3));
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
                if (a.norm_w) {
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
                s_q[((size_t)l * (K >> 2)) + b * 8 + sub] =
                    q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
                if (sub == 0)
                    s_d[l * blocks + b] = amax / 127.f;
            }
        }
    }
    __syncthreads();

    float acc0[LTF_ROWS_PER_WARP], acc1[LTF_ROWS_PER_WARP];
    if (row0 < a.N) {
        ltf_dot_q8_prefetched<LTF_ROWS_PER_WARP, MB>(
            pf, K, s_q, s_q + (K >> 2), s_d, s_d + blocks, lane, acc0, acc1);
    }
    if (lane == 0 && row0 < a.N) {
#pragma unroll
        for (int r = 0; r < LTF_ROWS_PER_WARP; ++r) {
            const int row = row0 + r;
            if (row >= a.N)
                break;
            float v0 = acc0[r];
            float v1 = acc1[r];
            if (a.bias) {
                v0 += a.bias[row];
                v1 += a.bias[row];
            }
            if (EPI == LTF_EPI_GELU) {
                v0 = ltf_gelu(v0);
                v1 = ltf_gelu(v1);
            }
            if (EPI == LTF_EPI_RESIDUAL) {
                a.out[row] += v0;
                a.out[a.N + row] += v1;
            } else {
                a.out[row] = v0;
                a.out[a.N + row] = v1;
            }
            if (a.kcache && row >= K) {
                float* cache = row < 2 * K ? a.kcache : a.vcache;
                const int f = row < 2 * K ? row - K : row - 2 * K;
                cache[((size_t)0 * a.pos_rows + a.cache_pos) * K + f] = v0;
                cache[((size_t)1 * a.pos_rows + a.cache_pos) * K + f] = v1;
            }
        }
    }
    if (SAMPLE) {
        // last-block-done: the final block samples from the complete logits
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            const int done = atomicAdd(a.sample_counter, 1);
            s_last = (done == (int)gridDim.x - 1) ? 1 : 0;
        }
        __syncthreads();
        if (!s_last)
            return;
        __threadfence();
        const magpietts_cuda_sampling_config config = *a.sample_config;
        int k = config.top_k < a.sample_vocab ? config.top_k : a.sample_vocab;
        if (k < 1)
            k = 1;
        if (k > MAGPIETTS_CUDA_MAX_FAST_TOPK)
            k = MAGPIETTS_CUDA_MAX_FAST_TOPK;  // fused path limit
        magpietts_sample_codebook_fast_block<MAGPIETTS_CUDA_SMALL_ITEMS_PER_THREAD>(
            a.out, a.out + a.N, a.sample_vocab, a.sample_codebook_size, a.sample_eos_id, config,
            a.sample_codebook, k, a.sample_top_ids, a.sample_top_vals,
            a.sample_codes + a.sample_codebook, a.sample_argmax + a.sample_codebook, s_samp[0]);
        if (tid == 0)
            *a.sample_counter = 0;
    }
}

// F16 variant (f32 activations staged in shared memory).
template <int EPI>
static __global__ void
__launch_bounds__(LTF_THREADS) ltf_gemv_f16_kernel(ltf_gemv_args a) {
    __shared__ __align__(16) float s_x[LTF_LANES * LTF_MAX_K];
    __shared__ __align__(16) float s_xn[LTF_LANES * LTF_MAX_EMBD];
    __shared__ float s_red[4 * LTF_WARPS];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int K = a.K;
#pragma unroll 4
    for (int k = tid; k < K; k += LTF_THREADS) {
        s_x[k] = ltf_input_at(a, 0, k);
        s_x[K + k] = ltf_input_at(a, 1, k);
        if (a.assemble && blockIdx.x == 0) {
            a.x_store[k] = s_x[k];
            a.x_store[K + k] = s_x[K + k];
        }
    }
    __syncthreads();
    const float* xn0 = s_x;
    const float* xn1 = s_x + K;
    if (a.norm_w) {
        ltf_block_layernorm(s_x, s_xn, a.norm_w, K, a.ln_eps, s_red);
        xn0 = s_xn;
        xn1 = s_xn + K;
    }
    static_assert(LTF_ROWS_PER_WARP % 2 == 0, "F16 path handles row pairs");
    for (int pair = 0; pair < LTF_ROWS_PER_WARP / 2; ++pair) {
        const int row0 = blockIdx.x * LTF_ROWS_PER_BLOCK + warp * LTF_ROWS_PER_WARP + pair * 2;
        if (row0 >= a.N)
            return;
        const int row1 = (row0 + 1 < a.N) ? row0 + 1 : -1;
        float acc00, acc01, acc10, acc11;
        ltf_rows_dot_f16(
            reinterpret_cast<const __half*>(a.W), K, row0, row1, xn0, xn1, lane, acc00, acc01,
            acc10, acc11);
        if (lane == 0) {
            float r0v[2] = {acc00, acc01};
            float r1v[2] = {acc10, acc11};
            const int rows[2] = {row0, row1};
            float* accs[2] = {r0v, r1v};
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = rows[r];
                if (row < 0)
                    continue;
                float v0 = accs[r][0];
                float v1 = accs[r][1];
                if (a.bias) {
                    v0 += a.bias[row];
                    v1 += a.bias[row];
                }
                if (EPI == LTF_EPI_GELU) {
                    v0 = ltf_gelu(v0);
                    v1 = ltf_gelu(v1);
                }
                if (EPI == LTF_EPI_RESIDUAL) {
                    a.out[row] += v0;
                    a.out[a.N + row] += v1;
                } else {
                    a.out[row] = v0;
                    a.out[a.N + row] = v1;
                }
                if (a.kcache && row >= K) {
                    float* cache = row < 2 * K ? a.kcache : a.vcache;
                    const int f = row < 2 * K ? row - K : row - 2 * K;
                    cache[((size_t)0 * a.pos_rows + a.cache_pos) * K + f] = v0;
                    cache[((size_t)1 * a.pos_rows + a.cache_pos) * K + f] = v1;
                }
            }
        }
    }
}

template <int EPI, bool SAMPLE = false>
static void
ltf_launch_gemv(const ltf_gemv_args& a, cudaStream_t stream) {
    const int grid = (a.N + LTF_ROWS_PER_BLOCK - 1) / LTF_ROWS_PER_BLOCK;
    if (a.w_type == LTF_TYPE_Q8_0) {
        const int blocks = a.K >> 5;
        if (blocks <= 32)
            ltf_gemv_q8_kernel<EPI, 1, SAMPLE><<<grid, LTF_THREADS, 0, stream>>>(a);
        else if (blocks <= 64)
            ltf_gemv_q8_kernel<EPI, 2, SAMPLE><<<grid, LTF_THREADS, 0, stream>>>(a);
        else
            ltf_gemv_q8_kernel<EPI, 3, SAMPLE><<<grid, LTF_THREADS, 0, stream>>>(a);
    } else {
        ltf_gemv_f16_kernel<EPI><<<grid, LTF_THREADS, 0, stream>>>(a);
    }
}

// Cached self-attention for q_len = 1 over positions 0..pos (inclusive), one warp per
// (head, lane). d_head must be 64 (2 features per CUDA lane).
static __global__ void
ltf_attention_kernel(
    const float* qkv, const float* kcache, const float* vcache, float* ctx, int K, int n_head,
    int pos_rows, int pos, float scale) {
    const int h = blockIdx.x;
    const int l = blockIdx.y;
    const int lane = threadIdx.x;
    const int d_head = K / n_head;  // 64
    const int d0 = h * d_head + lane * 2;
    const float* q = qkv + (size_t)l * 3 * K;
    const float2 q2 = *reinterpret_cast<const float2*>(q + d0);
    float scores[LTF_MAX_POS];
    float m = -INFINITY;
    for (int j = 0; j <= pos; ++j) {
        const float2 k2 =
            *reinterpret_cast<const float2*>(kcache + ((size_t)l * pos_rows + j) * K + d0);
        float s = q2.x * k2.x + q2.y * k2.y;
        s = ltf_warp_sum(s) * scale;
        scores[j] = s;
        m = fmaxf(m, s);
    }
    float sum = 0.0f;
    for (int j = 0; j <= pos; ++j) {
        scores[j] = __expf(scores[j] - m);
        sum += scores[j];
    }
    const float inv = 1.0f / sum;
    float c0 = 0.0f, c1 = 0.0f;
    for (int j = 0; j <= pos; ++j) {
        const float2 v2 =
            *reinterpret_cast<const float2*>(vcache + ((size_t)l * pos_rows + j) * K + d0);
        c0 += scores[j] * v2.x;
        c1 += scores[j] * v2.y;
    }
    float2 out;
    out.x = c0 * inv;
    out.y = c1 * inv;
    *reinterpret_cast<float2*>(ctx + (size_t)l * K + d0) = out;
}


// ---------------------------------------------------------------------------------------
// Persistent chain kernel: every round of the frame in ONE launch. Phases (LN+qkv, attention+
// o+res, LN+ff1+gelu, ff2+res, out-proj, sampling) are separated by a software grid barrier.
// Each block stages the weights of its next GEMV tile into shared memory with cp.async right
// after its current dot product, so the DRAM traffic overlaps the epilogue, the barrier and the
// next prologue instead of a kernel ramp-up. Co-residency of all blocks is verified at create
// time (occupancy query), so the spin barrier cannot deadlock.
// ---------------------------------------------------------------------------------------
static constexpr int LTF_MAX_LAYERS = 4;
static constexpr int LTF_MAX_ROUNDS = LTF_MAX_POS;
static constexpr int LTF_STAGE_BYTES = 24576;  // max tile quants: 32 rows x 768 or 8 rows x 3072
static constexpr int LTF_STAGE_SCALES = 768;   // max tile scale count (rows x K/32)
#ifndef LTF_BARRIER_SLEEP_NS
#define LTF_BARRIER_SLEEP_NS 200
#endif

struct ltf_chain_layer {
    ltf_q8p qkv, o, ff1, ff2;
    const float* norm_self;
    const float* norm_ff;
};

struct ltf_chain_args {
    int K, NFF, H, P, vocab, n_layers, n_rounds;
    float ln_eps, attn_scale;
    ltf_chain_layer layers[LTF_MAX_LAYERS];
    const void* pos;
    int pos_type;
    const void* audio_emb[LTF_MAX_ROUNDS];  // table of codebook c-1 read by round c
    int emb_type;
    ltf_q8p out_w[LTF_MAX_ROUNDS];
    const float* out_b[LTF_MAX_ROUNDS];
    const float* h_in;
    float* h;
    float* qkv;
    float* ctx;
    float* ff;
    float* logits;
    float* kcache;
    float* vcache;
    unsigned int* barrier;
    const magpietts_cuda_sampling_config* sample_config;
    int32_t* codes;
    int32_t* argmax;
    int32_t* top_ids;
    float* top_vals;
    int sample_codebook_size;
    int sample_eos_id;
};

// Shared memory of the chain kernel. The GEMV scratch and the sampler scratch are never live in
// the same phase and alias each other.
struct ltf_chain_gemv_smem {
    __align__(16) int s_q[LTF_LANES * (LTF_MAX_K / 4)];
    float s_d[LTF_LANES * (LTF_MAX_K / 32)];
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
        __threadfence();
        atomicAdd(counter, 1u);
        while (*reinterpret_cast<volatile unsigned int*>(counter) < target) {
#ifndef LTF_BARRIER_NO_SLEEP
            __nanosleep(LTF_BARRIER_SLEEP_NS);
#endif
        }
        __threadfence();
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

// Stage this block's tile (rows [tile*ROWS, +ROWS) of W, contiguous in the planar layout) into
// shared memory asynchronously. Callers must have finished reading the previous stage
// (__syncthreads) before issuing.
template <int R>
static __device__ __forceinline__ void
ltf_chain_stage(const ltf_q8p& W, int K, int N, int tile, ltf_chain_smem& sm) {
    constexpr int ROWS = LTF_WARPS * R;
    const int row_begin = tile * ROWS;
    if (row_begin >= N)
        return;
    const int rows = min(ROWS, N - row_begin);
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
    const float* norm_w;  // LayerNorm weight or nullptr
    int N, K;
    float* out;         // [2][N]
    const float* bias;  // [N] or nullptr
    // round-start input assembly
    int assemble;
    const float* h_in;
    const void* emb_row;  // embedding row of the previous code (F16/F32), or nullptr for round 0
    const void* pos_row;  // positional row
    int emb_type, pos_type;
    float* x_store;  // residual stream initialized by tile 0 when assembling
    float ln_eps;
    // K/V cache write (qkv phase)
    float* kcache;
    float* vcache;
    int pos_rows, cache_pos;
};

static __device__ __forceinline__ float
ltf_chain_input(const ltf_chain_phase_args& a, int l, int k) {
    if (a.x_smem)
        return a.x_smem[(size_t)l * a.K + k];
    if (!a.assemble)
        return a.x[(size_t)l * a.K + k];
    const float p = ltf_load_scalar(a.pos_row, a.pos_type, (size_t)k);
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
    constexpr int ROWS = LTF_WARPS * R;
    ltf_chain_gemv_smem& g = sm.u.g;
    if (tile * ROWS >= a.N)
        return;  // block-uniform: no rows for this block
    const int rows_valid = min(ROWS, a.N - tile * ROWS);
    const int wrow0 = warp * R;
    const int row0 = tile * ROWS + wrow0;
    const bool staged = a.norm_w != nullptr;  // LN phases stage the input in shared memory

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
    if (staged) {
        float s0 = 0.f, s1 = 0.f, q0 = 0.f, q1 = 0.f;
#pragma unroll 3
        for (int k = tid; k < K; k += LTF_THREADS) {
            const float x0 = ltf_chain_input(a, 0, k);
            const float x1 = ltf_chain_input(a, 1, k);
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
    } else if (a.assemble && tile == 0) {
#pragma unroll 3
        for (int k = tid; k < K; k += LTF_THREADS) {
            a.x_store[k] = ltf_chain_input(a, 0, k);
            a.x_store[K + k] = ltf_chain_input(a, 1, k);
        }
    }
    {
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
    ltf_cp_async_wait_all();
    __syncthreads();

    float acc0[R], acc1[R];
    if (wrow0 < rows_valid) {
        ltf_q8_prefetch<R, MB> pf;
        ltf_chain_load_staged<R, MB>(sm, K, rows_valid, wrow0, lane, pf);
        ltf_dot_q8_prefetched<R, MB>(
            pf, K, g.s_q, g.s_q + (K >> 2), g.s_d, g.s_d + blocks, lane, acc0, acc1);
    }
    if (lane == 0 && wrow0 < rows_valid) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row >= a.N)
                break;
            float v0 = acc0[r] + bia[r];
            float v1 = acc1[r] + bia[r];
            if (EPI == LTF_EPI_GELU) {
                v0 = ltf_gelu(v0);
                v1 = ltf_gelu(v1);
            }
            if (EPI == LTF_EPI_RESIDUAL) {
                v0 += res0[r];
                v1 += res1[r];
            }
            a.out[row] = v0;
            a.out[a.N + row] = v1;
            if (a.kcache && row >= K) {
                float* cache = row < 2 * K ? a.kcache : a.vcache;
                const int f = row < 2 * K ? row - K : row - 2 * K;
                cache[((size_t)0 * a.pos_rows + a.cache_pos) * K + f] = v0;
                cache[((size_t)1 * a.pos_rows + a.cache_pos) * K + f] = v1;
            }
        }
    }
    __syncthreads();  // all warps done with the staged weights and the scratch
}

// Cached self-attention for (head h, CFG lane l), split over LTF_ATTN_SPLIT warps of one block:
// warp `chunk` scores positions [chunk*CH, +CH) (each lane owns two features, so every K/V row
// read is one coalesced 256-byte warp access and all of a warp's loads are independent), then
// the partial (max, sum, acc) triples are combined through shared memory. d_head must be 64.
static constexpr int LTF_ATTN_SPLIT = 4;  // warps per task
static constexpr int LTF_ATTN_CH = (LTF_MAX_POS + LTF_ATTN_SPLIT - 1) / LTF_ATTN_SPLIT;  // 8
static constexpr int LTF_ATTN_TASKS_PER_BLOCK = LTF_WARPS / LTF_ATTN_SPLIT;              // 2
struct ltf_attn_partial {
    float m[LTF_ATTN_SPLIT];
    float sum[LTF_ATTN_SPLIT];
    float acc[LTF_ATTN_SPLIT][64];
};
static __device__ __forceinline__ void
ltf_chain_attention_split(
    const float* qkv, const float* kcache, const float* vcache, float* ctx_out, int K, int pos_rows,
    int pos, float scale, int h, int l, int chunk, int lane, ltf_attn_partial& part,
    int task_warp0) {
    const int d_head = 64;
    const int d0 = h * d_head + lane * 2;
    const float2 q2 = *reinterpret_cast<const float2*>(qkv + (size_t)l * 3 * K + d0);
    const float* kbase = kcache + (size_t)l * pos_rows * K + d0;
    const float* vbase = vcache + (size_t)l * pos_rows * K + d0;
    const int j0 = chunk * LTF_ATTN_CH;
    float2 k2[LTF_ATTN_CH], v2[LTF_ATTN_CH];
#pragma unroll
    for (int i = 0; i < LTF_ATTN_CH; ++i) {
        const int j = j0 + i;
        const bool ok = j < pos_rows;
        k2[i] =
            ok ? *reinterpret_cast<const float2*>(kbase + (size_t)j * K) : make_float2(0.f, 0.f);
        v2[i] =
            ok ? *reinterpret_cast<const float2*>(vbase + (size_t)j * K) : make_float2(0.f, 0.f);
    }
    float sc[LTF_ATTN_CH];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < LTF_ATTN_CH; ++i) {
        const float part_dot = ltf_warp_sum(q2.x * k2[i].x + q2.y * k2[i].y) * scale;
        sc[i] = (j0 + i) <= pos ? part_dot : -INFINITY;
        m = fmaxf(m, sc[i]);
    }
    float sum = 0.f, c0 = 0.f, c1 = 0.f;
    if (m > -INFINITY) {
#pragma unroll
        for (int i = 0; i < LTF_ATTN_CH; ++i) {
            const float pj = __expf(sc[i] - m);
            sum += pj;
            c0 += pj * v2[i].x;
            c1 += pj * v2[i].y;
        }
    }
    if (lane == 0) {
        part.m[chunk] = m;
        part.sum[chunk] = sum;
    }
    part.acc[chunk][lane * 2] = c0;
    part.acc[chunk][lane * 2 + 1] = c1;
    __syncthreads();
    if ((threadIdx.x >> 5) == task_warp0) {
        float M = -INFINITY;
#pragma unroll
        for (int c = 0; c < LTF_ATTN_SPLIT; ++c) M = fmaxf(M, part.m[c]);
        float S = 0.f, a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int c = 0; c < LTF_ATTN_SPLIT; ++c) {
            const float w = part.m[c] > -INFINITY ? __expf(part.m[c] - M) : 0.f;
            S += w * part.sum[c];
            a0 += w * part.acc[c][lane * 2];
            a1 += w * part.acc[c][lane * 2 + 1];
        }
        const float inv = 1.0f / S;
        *reinterpret_cast<float2*>(ctx_out + (size_t)l * K + d0) = make_float2(a0 * inv, a1 * inv);
    }
}

static __global__ void
__launch_bounds__(LTF_THREADS, 2) ltf_chain_kernel(const ltf_chain_args a) {
    __shared__ ltf_chain_smem sm;
    __shared__ int s_flag;
    const int tile = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const unsigned int nblocks = gridDim.x;
    unsigned int target = 0;
    const int K = a.K;
    const size_t layer_cache = (size_t)LTF_LANES * a.P * K;
    const size_t pos_elem = a.pos_type == LTF_TYPE_F16 ? 2 : 4;
    const size_t emb_elem = a.emb_type == LTF_TYPE_F16 ? 2 : 4;

    auto sync = [&]() { ltf_grid_barrier(a.barrier, target, nblocks); };

    ltf_chain_stage<4>(a.layers[0].qkv, K, 3 * K, tile, sm);
    for (int c = 0; c < a.n_rounds; ++c) {
        for (int il = 0; il < a.n_layers; ++il) {
            const ltf_chain_layer& L = a.layers[il];
            float* kc = a.kcache + il * layer_cache;
            float* vc = a.vcache + il * layer_cache;
            {  // LN(norm_self) + qkv (+ input assembly on layer 0) + K/V cache write
                ltf_chain_phase_args g{};
                g.x = a.h;
                g.norm_w = L.norm_self;
                g.N = 3 * K;
                g.K = K;
                g.out = a.qkv;
                g.ln_eps = a.ln_eps;
                if (il == 0) {
                    g.assemble = 1;
                    g.h_in = a.h_in;
                    if (c > 0) {
                        const int code = a.codes[c - 1];
                        g.emb_row = reinterpret_cast<const char*>(a.audio_emb[c - 1]) +
                                    (size_t)code * K * emb_elem;
                    }
                    g.pos_row = reinterpret_cast<const char*>(a.pos) + (size_t)c * K * pos_elem;
                    g.emb_type = a.emb_type;
                    g.pos_type = a.pos_type;
                    g.x_store = a.h;
                }
                g.kcache = kc;
                g.vcache = vc;
                g.pos_rows = a.P;
                g.cache_pos = c;
                ltf_chain_gemv<LTF_EPI_STORE, 4, 1>(g, tile, sm);
            }
            ltf_chain_stage<1>(L.o, K, K, tile, sm);
            // attention by the last blocks to finish their qkv rows (K/V of position c complete):
            // LTF_ATTN_SPLIT warps per (head, lane) task, LTF_ATTN_TASKS_PER_BLOCK tasks per block
            {
                const int n_tasks = a.H * LTF_LANES;
                const int attn_blocks =
                    (n_tasks + LTF_ATTN_TASKS_PER_BLOCK - 1) / LTF_ATTN_TASKS_PER_BLOCK;
                const int rank =
                    ltf_chain_arrival_rank(a.barrier + 1, nblocks, attn_blocks, &s_flag);
                const int slot = rank - ((int)nblocks - attn_blocks);
                if (slot >= 0 && slot < attn_blocks) {
                    const int local_task = warp / LTF_ATTN_SPLIT;
                    const int task = slot * LTF_ATTN_TASKS_PER_BLOCK + local_task;
                    ltf_attn_partial* parts = reinterpret_cast<ltf_attn_partial*>(sm.u.g.s_x);
                    if (task < n_tasks) {
                        ltf_chain_attention_split(
                            a.qkv, kc, vc, a.ctx, K, a.P, c, a.attn_scale, task >> 1, task & 1,
                            warp % LTF_ATTN_SPLIT, lane, parts[local_task],
                            local_task * LTF_ATTN_SPLIT);
                    } else {
                        __syncthreads();  // matches the barrier inside the split attention
                    }
                    __syncthreads();
                }
            }
            sync();
            {  // o + residual
                ltf_chain_phase_args g{};
                g.x = a.ctx;
                g.N = K;
                g.K = K;
                g.out = a.h;
                ltf_chain_gemv<LTF_EPI_RESIDUAL, 1, 3>(g, tile, sm);
            }
            ltf_chain_stage<4>(L.ff1, K, a.NFF, tile, sm);
            sync();
            {  // LN(norm_ff) + ff1 + gelu
                ltf_chain_phase_args g{};
                g.x = a.h;
                g.norm_w = L.norm_ff;
                g.N = a.NFF;
                g.K = K;
                g.out = a.ff;
                g.ln_eps = a.ln_eps;
                ltf_chain_gemv<LTF_EPI_GELU, 4, 1>(g, tile, sm);
            }
            ltf_chain_stage<1>(L.ff2, a.NFF, K, tile, sm);
            sync();
            {  // ff2 + residual
                ltf_chain_phase_args g{};
                g.x = a.ff;
                g.N = K;
                g.K = a.NFF;
                g.out = a.h;
                ltf_chain_gemv<LTF_EPI_RESIDUAL, 1, 3>(g, tile, sm);
            }
            if (il + 1 < a.n_layers) {
                ltf_chain_stage<4>(a.layers[il + 1].qkv, K, 3 * K, tile, sm);
            } else {
                ltf_chain_stage<4>(a.out_w[c], K, a.vocab, tile, sm);
            }
            sync();
        }
        {  // output projection + bias -> logits
            ltf_chain_phase_args g{};
            g.x = a.h;
            g.N = a.vocab;
            g.K = K;
            g.out = a.logits;
            g.bias = a.out_b[c];
            ltf_chain_gemv<LTF_EPI_BIAS, 4, 1>(g, tile, sm);
        }
        if (c + 1 < a.n_rounds) {
            ltf_chain_stage<4>(a.layers[0].qkv, K, 3 * K, tile, sm);
        }
        // sampling by the last block to finish its logit rows
        if (ltf_chain_arrival_rank(a.barrier + 1, nblocks, 1, &s_flag) == (int)nblocks - 1) {
            const magpietts_cuda_sampling_config config = *a.sample_config;
            int k = config.top_k < a.vocab ? config.top_k : a.vocab;
            if (k < 1)
                k = 1;
            if (k > MAGPIETTS_CUDA_MAX_FAST_TOPK)
                k = MAGPIETTS_CUDA_MAX_FAST_TOPK;
            magpietts_sample_codebook_fast_block<MAGPIETTS_CUDA_SMALL_ITEMS_PER_THREAD>(
                a.logits, a.logits + a.vocab, a.vocab, a.sample_codebook_size, a.sample_eos_id,
                config, c, k, a.top_ids, a.top_vals, a.codes + c, a.argmax + c, sm.u.samp);
        }
        if (c + 1 < a.n_rounds) {
            sync();
        }
    }
}

// ---------------------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------------------
// Repack a ggml Q8_0 device tensor [N rows][K] into planar quants/scales device buffers.
static bool
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

static bool
ltf_type_ok(int t) {
    return t == LTF_TYPE_F16 || t == LTF_TYPE_Q8_0;
}

// Decide whether the persistent chain kernel applies: all weights planar Q8, shapes within the
// kernel's static limits, small vocab for the in-kernel sampler, and every block co-resident.
static void
ltf_chain_setup(magpietts_lt_fused* f) {
    f->chain_grid = 0;
    const magpietts_lt_fused_weights& w = f->w;
    if (w.n_layers > LTF_MAX_LAYERS || w.n_rounds > LTF_MAX_ROUNDS ||
        w.vocab > MAGPIETTS_CUDA_SMALL_VOCAB || w.out_type != LTF_TYPE_Q8_0) {
        return;
    }
    for (const ltf_layer& L : f->layers) {
        if (L.qkv_type != LTF_TYPE_Q8_0 || L.o_type != LTF_TYPE_Q8_0 ||
            L.ff1_type != LTF_TYPE_Q8_0 || L.ff2_type != LTF_TYPE_Q8_0)
            return;
    }
    const int K = w.n_embd;
    if ((K >> 5) > 32 || (w.n_ff >> 5) > 96)
        return;  // R=4 tiles need MB=1, ff2 tiles MB<=3
    if (LTF_WARPS * 4 * K > LTF_STAGE_BYTES || LTF_WARPS * 1 * w.n_ff > LTF_STAGE_BYTES ||
        LTF_WARPS * 4 * (K >> 5) > LTF_STAGE_SCALES || LTF_WARPS * (w.n_ff >> 5) > LTF_STAGE_SCALES)
        return;
    auto tiles = [](int N, int R) { return (N + LTF_WARPS * R - 1) / (LTF_WARPS * R); };
    int grid = tiles(3 * K, 4);
    grid = std::max(grid, tiles(K, 1));
    grid = std::max(grid, tiles(w.n_ff, 4));
    grid = std::max(grid, tiles(w.vocab, 4));
    grid = std::max(grid, (w.n_head * LTF_LANES + LTF_WARPS - 1) / LTF_WARPS);
    int device = 0, sms = 0, per_sm = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
        return;
    const cudaError_t occ_err =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, ltf_chain_kernel, LTF_THREADS, 0);
    if (occ_err != cudaSuccess)
        return;
    if (per_sm * sms < grid)
        return;
    f->chain_grid = grid;
    // Pin the activation arena (~0.5 MB) in L2 while the 17 MB/round weight stream passes.
    {
        int max_persist = 0, max_window = 0;
        cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, device);
        cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, device);
        if (max_persist > 0 && max_window > 0 && f->arena_bytes <= (size_t)max_window) {
            size_t cur = 0;
            cudaDeviceGetLimit(&cur, cudaLimitPersistingL2CacheSize);
            const size_t want =
                std::min<size_t>((size_t)max_persist, std::max<size_t>(cur, f->arena_bytes * 2));
            if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want) == cudaSuccess)
                f->l2_window = true;
        }
    }
}

magpietts_lt_fused*
magpietts_lt_fused_create(const magpietts_lt_fused_weights& w, char* error, size_t error_size) {
    if (w.n_embd <= 0 || w.n_embd > LTF_MAX_EMBD || (w.n_embd % 64) != 0 || w.n_head <= 0 ||
        (w.n_embd / w.n_head) != 64 || w.n_ff <= 0 || (w.n_ff % 64) != 0 || w.n_ff > LTF_MAX_K ||
        w.n_layers <= 0 || w.vocab <= 0 || w.n_rounds <= 0 || w.pos_rows <= 0 ||
        w.pos_rows > LTF_MAX_POS || w.n_rounds > w.pos_rows) {
        ltf_set_error(error, error_size, "fused local transformer: unsupported shape");
        return nullptr;
    }
    for (int i = 0; i < w.n_layers; ++i) {
        const magpietts_lt_fused_layer_weights& L = w.layers[i];
        if (!L.qkv || !L.o || !L.ff1 || !L.ff2 || !L.norm_self || !L.norm_ff ||
            !ltf_type_ok(L.qkv_type) || !ltf_type_ok(L.o_type) || !ltf_type_ok(L.ff1_type) ||
            !ltf_type_ok(L.ff2_type)) {
            ltf_set_error(
                error, error_size, "fused local transformer: unsupported layer weight types");
            return nullptr;
        }
    }
    if (!ltf_type_ok(w.out_type) ||
        (w.audio_emb_type != LTF_TYPE_F16 && w.audio_emb_type != LTF_TYPE_F32) ||
        (w.pos_type != LTF_TYPE_F16 && w.pos_type != LTF_TYPE_F32)) {
        ltf_set_error(
            error, error_size, "fused local transformer: unsupported embedding/output types");
        return nullptr;
    }
    for (int r = 0; r < w.n_rounds; ++r) {
        if (!w.out_w[r] || (r > 0 && !w.audio_emb[r - 1])) {
            ltf_set_error(
                error, error_size, "fused local transformer: missing per-codebook tensors");
            return nullptr;
        }
    }
    magpietts_lt_fused* f = new magpietts_lt_fused;
    f->w = w;
    f->layers.resize(w.n_layers);
    for (int i = 0; i < w.n_layers; ++i) {
        const auto& L = w.layers[i];
        f->layers[i] = {L.qkv,     L.o,        L.ff1,    L.ff2,      L.norm_self,
                        L.norm_ff, L.qkv_type, L.o_type, L.ff1_type, L.ff2_type};
    }
    f->audio_emb.assign(w.audio_emb, w.audio_emb + w.n_rounds);
    f->out_w.assign(w.out_w, w.out_w + w.n_rounds);
    f->out_b.assign(w.out_b, w.out_b + w.n_rounds);
    f->out_p.resize(w.n_rounds);
    {
        bool ok = true;
        for (int i = 0; i < w.n_layers && ok; ++i) {
            ltf_layer& L = f->layers[i];
            if (L.qkv_type == LTF_TYPE_Q8_0)
                ok = ok && ltf_repack_q8(L.qkv, 3 * w.n_embd, w.n_embd, f->planar_buffers, L.qkv_p);
            if (L.o_type == LTF_TYPE_Q8_0)
                ok = ok && ltf_repack_q8(L.o, w.n_embd, w.n_embd, f->planar_buffers, L.o_p);
            if (L.ff1_type == LTF_TYPE_Q8_0)
                ok = ok && ltf_repack_q8(L.ff1, w.n_ff, w.n_embd, f->planar_buffers, L.ff1_p);
            if (L.ff2_type == LTF_TYPE_Q8_0)
                ok = ok && ltf_repack_q8(L.ff2, w.n_embd, w.n_ff, f->planar_buffers, L.ff2_p);
        }
        for (int r = 0; r < w.n_rounds && ok; ++r) {
            if (w.out_type == LTF_TYPE_Q8_0)
                ok = ok &&
                     ltf_repack_q8(
                         w.out_w[r], w.vocab, w.n_embd, f->planar_buffers, f->out_p[(size_t)r]);
        }
        if (!ok) {
            ltf_set_error(error, error_size, "fused local transformer: Q8 weight repack failed");
            magpietts_lt_fused_free(f);
            return nullptr;
        }
    }
    const size_t K = (size_t)w.n_embd;
    const size_t cache = (size_t)w.n_layers * LTF_LANES * w.pos_rows * K;
    cudaError_t err = cudaSuccess;
    // All scratch lives in one arena so a single L2 access-policy window can pin it.
    size_t total = 0;
    auto reserve = [&](size_t n) {
        const size_t off = total;
        total += ((n * sizeof(float)) + 255) & ~(size_t)255;
        return off;
    };
    const size_t off_h_in = reserve(LTF_LANES * K), off_h = reserve(LTF_LANES * K),
                 off_qkv = reserve(LTF_LANES * 3 * K), off_ctx = reserve(LTF_LANES * K),
                 off_ff = reserve(LTF_LANES * (size_t)w.n_ff),
                 off_logits = reserve(LTF_LANES * (size_t)w.vocab), off_k = reserve(cache),
                 off_v = reserve(cache);
    err = cudaMalloc(&f->arena, total);
    if (err == cudaSuccess)
        err = cudaMemset(f->arena, 0, total);
    if (err == cudaSuccess) {
        f->arena_bytes = total;
        char* base = reinterpret_cast<char*>(f->arena);
        f->h_in = reinterpret_cast<float*>(base + off_h_in);
        f->h = reinterpret_cast<float*>(base + off_h);
        f->qkv = reinterpret_cast<float*>(base + off_qkv);
        f->ctx = reinterpret_cast<float*>(base + off_ctx);
        f->ff = reinterpret_cast<float*>(base + off_ff);
        f->logits = reinterpret_cast<float*>(base + off_logits);
        f->kcache = reinterpret_cast<float*>(base + off_k);
        f->vcache = reinterpret_cast<float*>(base + off_v);
    }
    if (err == cudaSuccess)
        err = cudaMalloc(&f->sample_counters, (size_t)w.n_rounds * sizeof(int));
    if (err == cudaSuccess)
        err = cudaMemset(f->sample_counters, 0, (size_t)w.n_rounds * sizeof(int));
    if (err == cudaSuccess)
        err = cudaStreamCreateWithFlags(&f->capture_stream, cudaStreamNonBlocking);
    if (err == cudaSuccess)
        err = cudaMalloc(&f->barrier, 2 * sizeof(unsigned int));
    if (err == cudaSuccess)
        err = cudaMemset(f->barrier, 0, 2 * sizeof(unsigned int));
    if (err != cudaSuccess) {
        ltf_set_error(error, error_size, "fused local transformer: allocation failed", err);
        magpietts_lt_fused_free(f);
        return nullptr;
    }
    ltf_chain_setup(f);
    if (error && error_size)
        error[0] = '\0';
    return f;
}

void
magpietts_lt_fused_free(magpietts_lt_fused* f) {
    if (!f)
        return;
    if (f->arena) {
        cudaFree(f->arena);
        f->arena = nullptr;
        f->h_in = f->h = f->qkv = f->ctx = f->ff = f->logits = f->kcache = f->vcache = nullptr;
    }
    cudaFree(f->h_in);
    cudaFree(f->h);
    cudaFree(f->qkv);
    cudaFree(f->ctx);
    cudaFree(f->ff);
    cudaFree(f->logits);
    cudaFree(f->kcache);
    cudaFree(f->vcache);
    cudaFree(f->sample_counters);
    cudaFree(f->barrier);
    for (void* b : f->planar_buffers) cudaFree(b);
    if (f->capture_stream)
        cudaStreamDestroy(f->capture_stream);
    delete f;
}

float*
magpietts_lt_fused_input_cond(magpietts_lt_fused* f) {
    return f ? f->h_in : nullptr;
}

float*
magpietts_lt_fused_input_uncond(magpietts_lt_fused* f) {
    return f ? f->h_in + f->w.n_embd : nullptr;
}

static bool
ltf_enqueue_round(
    magpietts_lt_fused* f, int c, const int32_t* codes, cudaStream_t stream, char* error,
    size_t error_size, const magpietts_lt_fused_sampler* sampler) {
    const int K = f->w.n_embd;
    const int NFF = f->w.n_ff;
    const int H = f->w.n_head;
    const int P = f->w.pos_rows;
    const float scale = 1.0f / sqrtf((float)(K / H));
    const size_t layer_cache = (size_t)LTF_LANES * P * K;
    for (int il = 0; il < f->w.n_layers; ++il) {
        const ltf_layer& L = f->layers[il];
        // 1. [assemble +] LN(norm_self) + qkv GEMV
        ltf_gemv_args a{};
        a.x = f->h;
        a.norm_w = L.norm_self;
        a.W = L.qkv;
        a.w_type = L.qkv_type;
        a.Wq = L.qkv_p.qs;
        a.Wd = L.qkv_p.d;
        a.N = 3 * K;
        a.K = K;
        a.out = f->qkv;
        a.ln_eps = f->w.ln_eps;
        if (il == 0) {
            a.assemble = 1;
            a.round = c;
            a.h_in = f->h_in;
            a.emb = c > 0 ? f->audio_emb[c - 1] : nullptr;
            a.emb_type = f->w.audio_emb_type;
            a.pos = f->w.pos_emb;
            a.pos_type = f->w.pos_type;
            a.codes = codes;
            a.x_store = f->h;
        }
        a.kcache = f->kcache + il * layer_cache;
        a.vcache = f->vcache + il * layer_cache;
        a.pos_rows = P;
        a.cache_pos = c;
        ltf_launch_gemv<LTF_EPI_STORE>(a, stream);
        // 2. attention over the cached positions (K/V for position c written by the epilogue)
        ltf_attention_kernel<<<dim3(H, LTF_LANES), 32, 0, stream>>>(
            f->qkv, f->kcache + il * layer_cache, f->vcache + il * layer_cache, f->ctx, K, H, P, c,
            scale);
        // 3. o GEMV + residual
        ltf_gemv_args o{};
        o.x = f->ctx;
        o.W = L.o;
        o.w_type = L.o_type;
        o.Wq = L.o_p.qs;
        o.Wd = L.o_p.d;
        o.N = K;
        o.K = K;
        o.out = f->h;
        ltf_launch_gemv<LTF_EPI_RESIDUAL>(o, stream);
        // 4. LN(norm_ff) + ff1 GEMV + gelu
        ltf_gemv_args f1{};
        f1.x = f->h;
        f1.norm_w = L.norm_ff;
        f1.W = L.ff1;
        f1.w_type = L.ff1_type;
        f1.Wq = L.ff1_p.qs;
        f1.Wd = L.ff1_p.d;
        f1.N = NFF;
        f1.K = K;
        f1.out = f->ff;
        f1.ln_eps = f->w.ln_eps;
        ltf_launch_gemv<LTF_EPI_GELU>(f1, stream);
        // 5. ff2 GEMV + residual
        ltf_gemv_args f2{};
        f2.x = f->ff;
        f2.W = L.ff2;
        f2.w_type = L.ff2_type;
        f2.Wq = L.ff2_p.qs;
        f2.Wd = L.ff2_p.d;
        f2.N = K;
        f2.K = NFF;
        f2.out = f->h;
        ltf_launch_gemv<LTF_EPI_RESIDUAL>(f2, stream);
    }
    // 6. output projection + bias -> logits [2][vocab]
    ltf_gemv_args out{};
    out.x = f->h;
    out.W = f->out_w[c];
    out.w_type = f->w.out_type;
    out.Wq = f->out_p[(size_t)c].qs;
    out.Wd = f->out_p[(size_t)c].d;
    out.N = f->w.vocab;
    out.K = K;
    out.out = f->logits;
    out.bias = f->out_b[c];
    if (sampler && f->w.out_type == LTF_TYPE_Q8_0 && f->w.vocab <= MAGPIETTS_CUDA_SMALL_VOCAB) {
        out.sample_config =
            reinterpret_cast<const magpietts_cuda_sampling_config*>(sampler->config);
        out.sample_codes = sampler->codes;
        out.sample_argmax = sampler->argmax;
        out.sample_top_ids = sampler->top_ids;
        out.sample_top_vals = sampler->top_vals;
        out.sample_counter = f->sample_counters + c;
        out.sample_codebook = c;
        out.sample_vocab = f->w.vocab;
        out.sample_codebook_size = sampler->audio_codebook_size;
        out.sample_eos_id = sampler->audio_eos_id;
        ltf_launch_gemv<LTF_EPI_BIAS, true>(out, stream);
    } else {
        ltf_launch_gemv<LTF_EPI_BIAS>(out, stream);
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ltf_set_error(error, error_size, "fused local transformer: kernel launch failed", err);
        return false;
    }
    return true;
}

bool
magpietts_lt_fused_round(
    magpietts_lt_fused* f, int c, const int32_t* codes, void* stream, const float** logits_cond,
    const float** logits_uncond, char* error, size_t error_size,
    const magpietts_lt_fused_sampler* sampler) {
    if (!f || c < 0 || c >= f->w.n_rounds || (c > 0 && !codes)) {
        ltf_set_error(error, error_size, "fused local transformer: invalid round arguments");
        return false;
    }
    if (f->w.n_ff > LTF_MAX_K) {
        ltf_set_error(error, error_size, "fused local transformer: ff width too large");
        return false;
    }
    if (!ltf_enqueue_round(
            f, c, codes, reinterpret_cast<cudaStream_t>(stream), error, error_size, sampler))
        return false;
    *logits_cond = f->logits;
    *logits_uncond = f->logits + f->w.vocab;
    if (error && error_size)
        error[0] = '\0';
    return true;
}

bool
magpietts_lt_fused_capture_round(
    magpietts_lt_fused* f, int c, const int32_t* codes, void** graph_out, const float** logits_cond,
    const float** logits_uncond, char* error, size_t error_size,
    const magpietts_lt_fused_sampler* sampler) {
    if (!f || !graph_out) {
        ltf_set_error(error, error_size, "fused local transformer: invalid capture arguments");
        return false;
    }
    cudaError_t err = cudaStreamBeginCapture(f->capture_stream, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        ltf_set_error(error, error_size, "fused local transformer: begin capture failed", err);
        return false;
    }
    const bool ok = ltf_enqueue_round(f, c, codes, f->capture_stream, error, error_size, sampler);
    cudaGraph_t graph = nullptr;
    err = cudaStreamEndCapture(f->capture_stream, &graph);
    if (!ok || err != cudaSuccess) {
        if (graph)
            cudaGraphDestroy(graph);
        if (ok)
            ltf_set_error(error, error_size, "fused local transformer: end capture failed", err);
        return false;
    }
    *graph_out = graph;
    *logits_cond = f->logits;
    *logits_uncond = f->logits + f->w.vocab;
    if (error && error_size)
        error[0] = '\0';
    return true;
}

void
magpietts_lt_fused_destroy_graph(void* graph) {
    if (graph)
        cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(graph));
}

bool
magpietts_lt_fused_sampling_supported(const magpietts_lt_fused* f) {
    return f && f->w.out_type == LTF_TYPE_Q8_0 && f->w.vocab <= MAGPIETTS_CUDA_SMALL_VOCAB;
}

bool
magpietts_lt_fused_chain_supported(const magpietts_lt_fused* f) {
    return f && f->chain_grid > 0;
}

static bool
ltf_enqueue_chain(
    magpietts_lt_fused* f, cudaStream_t stream, const magpietts_lt_fused_sampler* sampler,
    char* error, size_t error_size) {
    if (!f->chain_grid || !sampler || !sampler->config || !sampler->codes) {
        ltf_set_error(error, error_size, "fused local transformer: chain unsupported");
        return false;
    }
    ltf_chain_args a{};
    a.K = f->w.n_embd;
    a.NFF = f->w.n_ff;
    a.H = f->w.n_head;
    a.P = f->w.pos_rows;
    a.vocab = f->w.vocab;
    a.n_layers = f->w.n_layers;
    a.n_rounds = f->w.n_rounds;
    a.ln_eps = f->w.ln_eps;
    a.attn_scale = 1.0f / sqrtf((float)(a.K / a.H));
    for (int i = 0; i < a.n_layers; ++i) {
        const ltf_layer& L = f->layers[i];
        a.layers[i] = {L.qkv_p, L.o_p, L.ff1_p, L.ff2_p, L.norm_self, L.norm_ff};
    }
    a.pos = f->w.pos_emb;
    a.pos_type = f->w.pos_type;
    a.emb_type = f->w.audio_emb_type;
    for (int c = 0; c < a.n_rounds; ++c) {
        a.audio_emb[c] = f->audio_emb[(size_t)c];
        a.out_w[c] = f->out_p[(size_t)c];
        a.out_b[c] = f->out_b[(size_t)c];
    }
    a.h_in = f->h_in;
    a.h = f->h;
    a.qkv = f->qkv;
    a.ctx = f->ctx;
    a.ff = f->ff;
    a.logits = f->logits;
    a.kcache = f->kcache;
    a.vcache = f->vcache;
    a.barrier = f->barrier;
    a.sample_config = reinterpret_cast<const magpietts_cuda_sampling_config*>(sampler->config);
    a.codes = sampler->codes;
    a.argmax = sampler->argmax;
    a.top_ids = sampler->top_ids;
    a.top_vals = sampler->top_vals;
    a.sample_codebook_size = sampler->audio_codebook_size;
    a.sample_eos_id = sampler->audio_eos_id;
    cudaError_t err = cudaMemsetAsync(f->barrier, 0, 2 * sizeof(unsigned int), stream);
    if (err != cudaSuccess) {
        ltf_set_error(error, error_size, "fused local transformer: barrier reset failed", err);
        return false;
    }
    if (f->l2_window) {
        cudaStreamAttrValue attr{};
        attr.accessPolicyWindow.base_ptr = f->arena;
        attr.accessPolicyWindow.num_bytes = f->arena_bytes;
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    }
    ltf_chain_kernel<<<f->chain_grid, LTF_THREADS, 0, stream>>>(a);
    if (f->l2_window) {
        cudaStreamAttrValue none{};
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &none);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        ltf_set_error(error, error_size, "fused local transformer: chain launch failed", err);
        return false;
    }
    return true;
}

bool
magpietts_lt_fused_chain(
    magpietts_lt_fused* f, void* stream, const magpietts_lt_fused_sampler* sampler, char* error,
    size_t error_size) {
    if (!f) {
        ltf_set_error(error, error_size, "fused local transformer: invalid chain arguments");
        return false;
    }
    if (!ltf_enqueue_chain(f, reinterpret_cast<cudaStream_t>(stream), sampler, error, error_size))
        return false;
    if (error && error_size)
        error[0] = '\0';
    return true;
}

bool
magpietts_lt_fused_capture_chain(
    magpietts_lt_fused* f, void** graph_out, const magpietts_lt_fused_sampler* sampler, char* error,
    size_t error_size) {
    if (!f || !graph_out) {
        ltf_set_error(error, error_size, "fused local transformer: invalid capture arguments");
        return false;
    }
    cudaError_t err = cudaStreamBeginCapture(f->capture_stream, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        ltf_set_error(error, error_size, "fused local transformer: begin capture failed", err);
        return false;
    }
    const bool ok = ltf_enqueue_chain(f, f->capture_stream, sampler, error, error_size);
    cudaGraph_t graph = nullptr;
    err = cudaStreamEndCapture(f->capture_stream, &graph);
    if (!ok || err != cudaSuccess) {
        if (graph)
            cudaGraphDestroy(graph);
        if (ok)
            ltf_set_error(error, error_size, "fused local transformer: end capture failed", err);
        return false;
    }
    *graph_out = graph;
    if (error && error_size)
        error[0] = '\0';
    return true;
}
