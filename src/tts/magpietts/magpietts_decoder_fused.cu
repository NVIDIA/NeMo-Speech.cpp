// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "magpietts_chain_common.cuh"
#include "magpietts_decoder_fused.h"

// ---------------------------------------------------------------------------------------
// One decoder step in a single persistent launch. Per layer the phases are
//   qkv GEMV (+ K/V append to the ring cache) | self-attention partials (all blocks, split-KV)
//   | o + residual | cross-q GEMV -> cross-attention tail (last blocks to arrive)
//   | cross-o + residual (cond lane) | ff1 + gelu | ff2 + residual
// separated by grid barriers; every block stages the next projection's rows into shared
// memory with cp.async right after its current dot. Activations move between phases as int8
// with per-group scales produced by the phase that computes them; LayerNorm is folded into the
// consumer (see ltf_chain_gemv in magpietts_chain_common.cuh).
// ---------------------------------------------------------------------------------------
static constexpr int DF_MAX_LAYERS = 16;
static constexpr int DF_MAX_CODEBOOKS = 16;
static constexpr int DF_ATTN_SPLITS = 2;  // key-range splits per (lane, head) self-attention item
static constexpr int DF_ATTN_MAX_RANGE =
    2048 / DF_ATTN_SPLITS;                    // keys per split (cache_len + 1 <= 2048)
static constexpr int DF_XATTN_BLOCKS = 8;     // tail blocks sharing the cross-attention keys
static constexpr int DF_XATTN_MAX_KEYS = 64;  // keys per tail block (text_capacity <= 512)
static constexpr int DF_R_CQ = 1;             // cross_dim rows per block: 128 / 64 = 2

struct df_layer {
    ltf_q8p qkv, o, cq, co, ff1, ff2;
    const float* norm_self;
    const float* norm_xq;
    const float* norm_ff;
    const float* qkv_c;  // LayerNorm fold constants (norm_self)
    const float* cq_c;   // (norm_xq)
    const float* ff1_c;  // (norm_ff)
    float* kv_arena;     // [2 planes][2 lanes][cache_len][E]
    int flags;           // 1: has_cross, 2: collect alignment, 4: apply prior
};

struct df_args {
    int E, H, NFF, CD, n_layers, n_codebooks, cache_len, TC;
    float ln_eps, attn_scale, xattn_scale, emb_scale;
    df_layer layers[DF_MAX_LAYERS];
    const void* audio_emb[DF_MAX_CODEBOOKS];
    int emb_type;
    const void* pos_emb;
    int pos_type;
    const float* norm_out;
    const float* cross_k;
    const float* cross_v;
    // step inputs
    const int32_t* tokens;
    int position, ring_head, valid_len, n_collect;
    const float* prior;
    const float* mask;
    float* hidden_cond;
    float* hidden_uncond;
    float* alignment;
    // scratch
    float* h;               // [2][E] residual stream
    int8_t* h_q;            // [2][E] q(h * gamma_next), groups of 4
    float* h_d;             // [2][E/4]
    float* stats;           // [2 lanes][sum, sumsq][grid]
    float* qkv;             // [2][3E]
    float* apart;           // self-attention partials [2][H][SPLITS][66]
    float* qc;              // [2][CD] cross query (lane 0 used)
    float* xpart;           // cross partials [XATTN_BLOCKS][2 + CD]
    float* xscores;         // [TC] cross scores of the current layer (alignment)
    float* align_acc;       // [TC]
    int8_t* ff_q;           // [2][NFF] groups of 16
    float* ff_d;            // [2][NFF/16]
    unsigned int* barrier;  // [0] grid barrier, [1] arrival counter, [2] tail-block counter
};

struct magpietts_decoder_fused {
    magpietts_decoder_fused_weights w{};
    magpietts_decoder_fused_cache cache{};
    std::vector<magpietts_decoder_fused_layer_weights> layer_w;
    std::vector<df_layer> layers;
    std::vector<void*> owned;
    std::vector<const void*> audio_emb;
    void* arena = nullptr;
    size_t arena_bytes = 0;
    df_args a{};
    int grid = 0;
    int n_collect = 0;
    unsigned int* barrier = nullptr;
};

// ---------------------------------------------------------------------------------------
// device: self-attention partials over the ring cache (flash-decoding split), all blocks
// ---------------------------------------------------------------------------------------
static __device__ __forceinline__ void
df_self_attention(
    const df_args& a, float* kbase, float* vbase, int tile, int nblocks, ltf_chain_smem& sm) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int E = a.E;
    const size_t cache_ss = (size_t)a.cache_len * E;
    const int n_keys = a.valid_len + 1;  // cached rows + the row appended this step
    const int per_split = (n_keys + DF_ATTN_SPLITS - 1) / DF_ATTN_SPLITS;
    const int n_items = LTF_LANES * a.H * DF_ATTN_SPLITS;
    float* query = sm.u.g.s_x;                // 64
    float* scores = query + 64;               // DF_ATTN_MAX_RANGE
    float* red = scores + DF_ATTN_MAX_RANGE;  // LTF_WARPS
    float* vpart = red + LTF_WARPS;           // 4 x 64
    for (int item = tile; item < n_items; item += nblocks) {
        const int l = item / (a.H * DF_ATTN_SPLITS);
        const int rem = item - l * a.H * DF_ATTN_SPLITS;
        const int hd = rem / DF_ATTN_SPLITS;
        const int sp = rem - hd * DF_ATTN_SPLITS;
        const int j0 = sp * per_split;
        const int j1 = min(n_keys, j0 + per_split);
        const int range = max(0, j1 - j0);
        const float* Kh = kbase + (size_t)l * cache_ss + hd * 64;
        const float* Vh = vbase + (size_t)l * cache_ss + hd * 64;
        // logical key j -> physical ring row (the appended row lands on ring_head itself)
        const int phys0 = a.ring_head + a.cache_len - a.valid_len;
        __syncthreads();  // previous item's scratch reads are done
        if (tid < 64)
            query[tid] = a.qkv[(size_t)l * 3 * E + hd * 64 + tid];
        __syncthreads();
        const int group = lane >> 3, sub = lane & 7, d8 = sub * 8;
        const float4 qa = *reinterpret_cast<const float4*>(query + d8);
        const float4 qb = *reinterpret_cast<const float4*>(query + d8 + 4);
        float local_max = -INFINITY;
        for (int base = j0 + warp * 4; base < j1; base += LTF_WARPS * 4) {
            const int j = base + group;
            float score = 0.f;
            if (j < j1) {
                int p = phys0 + j;
                p -= (p >= a.cache_len) ? a.cache_len : 0;
                const float4 ka = *reinterpret_cast<const float4*>(Kh + (size_t)p * E + d8);
                const float4 kb = *reinterpret_cast<const float4*>(Kh + (size_t)p * E + d8 + 4);
                score = ka.x * qa.x + ka.y * qa.y + ka.z * qa.z + ka.w * qa.w + kb.x * qb.x +
                        kb.y * qb.y + kb.z * qb.z + kb.w * qb.w;
            }
            score += __shfl_xor_sync(0xffffffffu, score, 1);
            score += __shfl_xor_sync(0xffffffffu, score, 2);
            score += __shfl_xor_sync(0xffffffffu, score, 4);
            if (sub == 0 && j < j1) {
                score *= a.attn_scale;
                scores[j - j0] = score;
                local_max = fmaxf(local_max, score);
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, o));
        if (lane == 0)
            red[warp] = local_max;
        __syncthreads();
        float m = red[0];
#pragma unroll
        for (int w = 1; w < LTF_WARPS; ++w) m = fmaxf(m, red[w]);
        __syncthreads();
        float local_sum = 0.f;
        for (int i = tid; i < range; i += LTF_THREADS) {
            const float wgt = (m > -INFINITY) ? __expf(scores[i] - m) : 0.f;
            scores[i] = wgt;
            local_sum += wgt;
        }
        local_sum = ltf_warp_sum(local_sum);
        if (lane == 0)
            red[warp] = local_sum;
        __syncthreads();
        float s = red[0];
#pragma unroll
        for (int w = 1; w < LTF_WARPS; ++w) s += red[w];
        // context partial: 4 key-parts x 64 features
        const int d = tid & 63, part = tid >> 6;
        float acc = 0.f;
        for (int i = part; i < range; i += 4) {
            int p = phys0 + j0 + i;
            p -= (p >= a.cache_len) ? a.cache_len : 0;
            acc += scores[i] * Vh[(size_t)p * E + d];
        }
        vpart[part * 64 + d] = acc;
        __syncthreads();
        float* out = a.apart + (((size_t)l * a.H + hd) * DF_ATTN_SPLITS + sp) * 66;
        if (tid < 64)
            out[2 + tid] = vpart[tid] + vpart[64 + tid] + vpart[128 + tid] + vpart[192 + tid];
        if (tid == 64) {
            out[0] = m;
            out[1] = s;
        }
    }
}

// Combine the self-attention partials into ctx [2][E] in shared memory (s_x).
static __device__ __forceinline__ void
df_combine_self(const df_args& a, ltf_chain_smem& sm) {
    float* ctx = sm.u.g.s_x;
    const int total = LTF_LANES * a.H * 64;
    for (int idx = threadIdx.x; idx < total; idx += LTF_THREADS) {
        const int l = idx / (a.H * 64);
        const int hd = (idx / 64) % a.H;
        const int d = idx & 63;
        const float* part = a.apart + ((size_t)l * a.H + hd) * DF_ATTN_SPLITS * 66;
        float M = -INFINITY;
#pragma unroll
        for (int sp = 0; sp < DF_ATTN_SPLITS; ++sp) M = fmaxf(M, part[sp * 66]);
        float S = 0.f, acc = 0.f;
#pragma unroll
        for (int sp = 0; sp < DF_ATTN_SPLITS; ++sp) {
            const float m = part[sp * 66];
            const float wgt = (m > -INFINITY) ? __expf(m - M) : 0.f;
            S += wgt * part[sp * 66 + 1];
            acc += wgt * part[sp * 66 + 2 + d];
        }
        ctx[(size_t)l * a.E + hd * 64 + d] = S > 0.f ? acc / S : 0.f;
    }
}

// Cross-attention query rows [slot*CD/8, +CD/8) for the cond lane, computed by a tail block from
// the producer-quantized residual q(h * gamma_xq) with the LayerNorm fold, weights read straight
// from the planar Q8 buffer (2 rows per warp, 12 KB per block). Then the tail blocks synchronize
// among themselves so every one of them sees the whole query.
static __device__ __forceinline__ void
df_cross_q_tail(const df_args& a, int il, int slot, int nblocks, ltf_chain_smem& sm, int* s_flag) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int K = a.E;
    const df_layer& L = a.layers[il];
    float* red = sm.u.g.s_red;
    {  // LN statistics of lane 0 from the producers' partials
        const int grp = tid >> 6;
        float v = 0.f;
        for (int b = tid & 63; b < nblocks; b += 64) v += a.stats[grp * nblocks + b];
        v = ltf_warp_sum(v);
        if (lane == 0)
            red[warp] = v;
    }
    __syncthreads();
    const float S0 = red[0] + red[1], Q0 = red[2] + red[3];
    const float m0 = S0 / (float)K;
    const float r0 = rsqrtf(fmaxf(Q0 / (float)K - m0 * m0, 0.f) + a.ln_eps);
    const int rows_per = a.CD / DF_XATTN_BLOCKS;  // 16
    const int row0 = slot * rows_per + warp * 2;  // 2 rows per warp
    if (warp * 2 < rows_per) {
        ltf_q8_prefetch<2, 1> pf;
        ltf_prefetch_q8<2, 1>(L.cq.qs, L.cq.d, K, a.CD, row0, lane, pf);
        float acc0[2], acc1[2];
        ltf_dot_q8_prefetched_xq<2, 1, 4>(pf, K, a.h_q, a.h_d, lane, acc0, acc1);
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = row0 + r;
                if (row < a.CD)
                    a.qc[row] = r0 * (acc0[r] - m0 * L.cq_c[row]);
            }
        }
    }
    // all DF_XATTN_BLOCKS tail blocks must have written their rows before anyone attends
    ltf_chain_arrival_rank(a.barrier + 2, DF_XATTN_BLOCKS, DF_XATTN_BLOCKS, s_flag);
}

// Cross-attention over this tail block's share of the text rows (single head, cond lane).
static __device__ __forceinline__ void
df_cross_attention(const df_args& a, int il, int slot, ltf_chain_smem& sm) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int CD = a.CD;
    const int keys_per = a.TC / DF_XATTN_BLOCKS;
    const int j0 = slot * keys_per;
    const df_layer& L = a.layers[il];
    const float* bias = (L.flags & 4) ? a.prior : a.mask;
    const float* K = a.cross_k + ((size_t)il * a.TC + j0) * CD;
    const float* V = a.cross_v + ((size_t)il * a.TC + j0) * CD;
    float* sc = sm.u.g.s_x;               // keys_per
    float* red = sc + DF_XATTN_MAX_KEYS;  // LTF_WARPS
    float* q = red + LTF_WARPS;           // CD
    for (int d = tid; d < CD; d += LTF_THREADS) q[d] = a.qc[d];
    __syncthreads();
    float local_max = -INFINITY;
    for (int jj = warp; jj < keys_per; jj += LTF_WARPS) {
        float s = 0.f;
        for (int d = lane; d < CD; d += 32) s += q[d] * K[(size_t)jj * CD + d];
        s = ltf_warp_sum(s);
        s = s * a.xattn_scale + bias[j0 + jj];
        if (lane == 0) {
            sc[jj] = s;
            a.xscores[j0 + jj] = s;
        }
        local_max = fmaxf(local_max, s);
    }
    if (lane == 0)
        red[warp] = local_max;
    __syncthreads();
    float m = red[0];
#pragma unroll
    for (int w = 1; w < LTF_WARPS; ++w) m = fmaxf(m, red[w]);
    __syncthreads();
    float local_sum = 0.f;
    for (int jj = tid; jj < keys_per; jj += LTF_THREADS) {
        const float wgt = __expf(sc[jj] - m);
        sc[jj] = wgt;
        local_sum += wgt;
    }
    local_sum = ltf_warp_sum(local_sum);
    if (lane == 0)
        red[warp] = local_sum;
    __syncthreads();
    float s = red[0];
#pragma unroll
    for (int w = 1; w < LTF_WARPS; ++w) s += red[w];
    float* out = a.xpart + (size_t)slot * (2 + CD);
    for (int d = tid; d < CD; d += LTF_THREADS) {
        float acc = 0.f;
        for (int jj = 0; jj < keys_per; ++jj) acc += sc[jj] * V[(size_t)jj * CD + d];
        out[2 + d] = acc;
    }
    if (tid == 0) {
        out[0] = m;
        out[1] = s;
    }
}

// Combine the cross-attention partials into ctx [CD] (lane 0; lane 1 zero) in s_x; block 0
// accumulates the normalized attention of collect layers into the alignment sum.
static __device__ __forceinline__ void
df_combine_cross(const df_args& a, int il, int tile, ltf_chain_smem& sm) {
    const int tid = threadIdx.x;
    const int CD = a.CD;
    float* ctx = sm.u.g.s_x;
    float M = -INFINITY;
#pragma unroll
    for (int sp = 0; sp < DF_XATTN_BLOCKS; ++sp) M = fmaxf(M, a.xpart[(size_t)sp * (2 + CD)]);
    float S = 0.f;
#pragma unroll
    for (int sp = 0; sp < DF_XATTN_BLOCKS; ++sp) {
        const float* p = a.xpart + (size_t)sp * (2 + CD);
        S += __expf(p[0] - M) * p[1];
    }
    const float inv = S > 0.f ? 1.f / S : 0.f;
    for (int d = tid; d < CD; d += LTF_THREADS) {
        float acc = 0.f;
#pragma unroll
        for (int sp = 0; sp < DF_XATTN_BLOCKS; ++sp) {
            const float* p = a.xpart + (size_t)sp * (2 + CD);
            acc += __expf(p[0] - M) * p[2 + d];
        }
        ctx[d] = acc * inv;
        ctx[CD + d] = 0.f;
    }
    if (tile == 0 && a.alignment && (a.layers[il].flags & 2)) {
        for (int j = tid; j < a.TC; j += LTF_THREADS)
            a.align_acc[j] += __expf(a.xscores[j] - M) * inv;
    }
}

// Final LayerNorm from the producers' partial statistics; hidden outputs and alignment mean.
static __device__ __forceinline__ void
df_final(const df_args& a, int tile, int nblocks, ltf_chain_smem& sm) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int E = a.E;
    float* red = sm.u.g.s_red;
    {
        const int grp = tid >> 6;
        float v = 0.f;
        for (int b = tid & 63; b < nblocks; b += 64) v += a.stats[grp * nblocks + b];
        v = ltf_warp_sum(v);
        if (lane == 0)
            red[warp] = v;
    }
    __syncthreads();
    const float S0 = red[0] + red[1], Q0 = red[2] + red[3], S1 = red[4] + red[5],
                Q1 = red[6] + red[7];
    const float m0 = S0 / (float)E, m1 = S1 / (float)E;
    const float r0 = rsqrtf(fmaxf(Q0 / (float)E - m0 * m0, 0.f) + a.ln_eps);
    const float r1 = rsqrtf(fmaxf(Q1 / (float)E - m1 * m1, 0.f) + a.ln_eps);
    int row_begin, rows;
    ltf_chain_rows(E, tile, nblocks, row_begin, rows);
    for (int i = tid; i < rows; i += LTF_THREADS) {
        const int row = row_begin + i;
        const float g = a.norm_out ? a.norm_out[row] : 1.f;
        a.hidden_cond[row] = (a.h[row] - m0) * r0 * g;
        a.hidden_uncond[row] = (a.h[E + row] - m1) * r1 * g;
    }
    if (tile == 0 && a.alignment) {
        const float inv = a.n_collect > 0 ? 1.f / (float)a.n_collect : 0.f;
        for (int j = tid; j < a.TC; j += LTF_THREADS) a.alignment[j] = a.align_acc[j] * inv;
    }
}

static __global__ void LTF_CHAIN_LAUNCH_BOUNDS
df_kernel(const df_args a) {
    extern __shared__ __align__(16) unsigned char df_dyn_smem[];
    ltf_chain_smem& sm = *reinterpret_cast<ltf_chain_smem*>(df_dyn_smem);
    __shared__ int s_flag;
    __shared__ const void* s_emb[DF_MAX_CODEBOOKS];
    const int tile = blockIdx.x;
    const int tid = threadIdx.x;
    const unsigned int nblocks = gridDim.x;
    unsigned int target = 0;
    const int E = a.E;
    const size_t cache_ss = (size_t)a.cache_len * E;
    const size_t emb_elem = a.emb_type == LTF_TYPE_F16 ? 2 : 4;
    const size_t pos_elem = a.pos_type == LTF_TYPE_F16 ? 2 : 4;
    int phase = 0;  // 6 barriers per layer: qkv | attn | o(+cross-q/attn tail) | co | ff1 | ff2
                    // (timing slots)
#ifdef LTF_CHAIN_TIMING
    int detail_phase = -1;
#endif
    auto sync = [&]() {
        LTF_STAMP(2 + phase * 2);
#ifdef LTF_CHAIN_TIMING
        LTF_ARRIVE(detail_phase);
#endif
        ltf_grid_barrier(a.barrier, target, nblocks);
        LTF_STAMP(2 + phase * 2 + 1);
        ++phase;
#ifdef LTF_CHAIN_TIMING
        if (detail_phase >= 0)
            ++detail_phase;
        if (tile == 0 && threadIdx.x == 0)
            g_ltf_chain_detail_phase = detail_phase;
        __syncthreads();
#endif
    };
    LTF_STAMP(0);

    if (tid < a.n_codebooks) {
        s_emb[tid] =
            reinterpret_cast<const char*>(a.audio_emb[tid]) + (size_t)a.tokens[tid] * E * emb_elem;
    }
    if (tile == 0 && a.alignment) {
        for (int j = tid; j < a.TC; j += LTF_THREADS) a.align_acc[j] = 0.f;
    }
    __syncthreads();

    ltf_chain_stage(a.layers[0].qkv, E, 3 * E, tile, sm);
    for (int il = 0; il < a.n_layers; ++il) {
        const df_layer& L = a.layers[il];
        float* kbase = L.kv_arena;
#ifdef LTF_CHAIN_TIMING
        phase = il * 8;
        detail_phase = il == LTF_TIMING_DETAIL_ROUND ? 0 : -1;
        if (tile == 0 && threadIdx.x == 0)
            g_ltf_chain_detail_phase = detail_phase;
        __syncthreads();
#endif
        float* vbase = kbase + 2 * cache_ss;
        {  // LN(norm_self) + qkv; K/V rows appended to the ring cache at ring_head
            ltf_chain_phase_args g{};
            g.N = 3 * E;
            g.K = E;
            g.out = a.qkv;
            g.ln_eps = a.ln_eps;
            if (il == 0) {
                g.x = a.h;
                g.norm_w = L.norm_self;
                g.assemble = 1;
                g.emb_rows = s_emb;
                g.n_emb_rows = a.n_codebooks;
                g.emb_scale = a.emb_scale;
                g.emb_type = a.emb_type;
                g.pos_type = a.pos_type;
                g.pos_row =
                    reinterpret_cast<const char*>(a.pos_emb) + (size_t)a.position * E * pos_elem;
                g.x_store = a.h;
            } else {
                g.xq = a.h_q;
                g.xd = a.h_d;
                g.xgroup = 4;
                g.stats = a.stats;
                g.ln_c = L.qkv_c;
            }
            g.kcache = kbase;
            g.vcache = vbase;
            g.kv_lane_stride = cache_ss;
            g.kv_row_stride = (size_t)E;
            g.kv_row = a.ring_head;
            ltf_chain_gemv<LTF_EPI_STORE, LTF_R_QKV, 1>(g, tile, sm);
        }
        ltf_chain_stage(L.o, E, E, tile, sm);
        sync();
        df_self_attention(a, kbase, vbase, tile, (int)nblocks, sm);
        sync();
        {  // o + residual; produces q(h * gamma_next) and partial LN stats
            df_combine_self(a, sm);
            __syncthreads();
            ltf_chain_phase_args g{};
            g.x_smem = sm.u.g.s_x;
            g.N = E;
            g.K = E;
            g.out = a.h;
            g.out_q = a.h_q;
            g.out_d = a.h_d;
            g.out_group = 4;
            g.out_keep_f32 = 1;
            g.out_gamma = (L.flags & 1) ? L.norm_xq : L.norm_ff;
            g.out_stats = a.stats;
            ltf_chain_gemv<LTF_EPI_RESIDUAL, LTF_R_O, 1>(g, tile, sm);
        }
        if (L.flags & 1) {
            ltf_chain_stage(L.co, a.CD, E, tile, sm);
            {  // tail of the o+residual phase: the last DF_XATTN_BLOCKS blocks to finish their rows
               // compute the cross-attention query (LN folded) and attend over the text rows
                const int rank =
                    ltf_chain_arrival_rank(a.barrier + 1, nblocks, DF_XATTN_BLOCKS, &s_flag);
                const int slot = rank - ((int)nblocks - DF_XATTN_BLOCKS);
                if (slot >= 0) {
                    df_cross_q_tail(a, il, slot, (int)nblocks, sm, &s_flag);
                    __syncthreads();
                    df_cross_attention(a, il, slot, sm);
                }
            }
            sync();
            {  // cross-o + residual on the cond lane; produces q(h * gamma_ff) + stats
                df_combine_cross(a, il, tile, sm);
                __syncthreads();
                ltf_chain_phase_args g{};
                g.x_smem = sm.u.g.s_x;
                g.N = E;
                g.K = a.CD;
                g.out = a.h;
                g.lane_mask = 1;
                g.out_q = a.h_q;
                g.out_d = a.h_d;
                g.out_group = 4;
                g.out_keep_f32 = 1;
                g.out_gamma = L.norm_ff;
                g.out_stats = a.stats;
                ltf_chain_gemv<LTF_EPI_RESIDUAL, LTF_R_O, 1>(g, tile, sm);
            }
        }
        ltf_chain_stage(L.ff1, E, a.NFF, tile, sm);
        sync();
        {  // LN(norm_ff) + ff1 + gelu -> int8 groups of 16
            ltf_chain_phase_args g{};
            g.xq = a.h_q;
            g.xd = a.h_d;
            g.xgroup = 4;
            g.stats = a.stats;
            g.ln_c = L.ff1_c;
            g.N = a.NFF;
            g.K = E;
            g.out = a.qkv;  // unused (int8 only)
            g.out_q = a.ff_q;
            g.out_d = a.ff_d;
            g.out_group = 16;
            g.ln_eps = a.ln_eps;
            ltf_chain_gemv<LTF_EPI_GELU, LTF_R_FF1, 1>(g, tile, sm);
        }
        ltf_chain_stage(L.ff2, a.NFF, E, tile, sm);
        sync();
        {  // ff2 + residual; q(h * gamma_self[next]) for the next layer, stats for the final norm
            ltf_chain_phase_args g{};
            g.xq = a.ff_q;
            g.xd = a.ff_d;
            g.xgroup = 16;
            g.N = E;
            g.K = a.NFF;
            g.out = a.h;
            g.out_q = a.h_q;
            g.out_d = a.h_d;
            g.out_group = 4;
            g.out_keep_f32 = 1;
            g.out_gamma = (il + 1 < a.n_layers) ? a.layers[il + 1].norm_self : nullptr;
            g.out_stats = a.stats;
            ltf_chain_gemv<LTF_EPI_RESIDUAL, LTF_R_O, 3>(g, tile, sm);
        }
        if (il + 1 < a.n_layers)
            ltf_chain_stage(a.layers[il + 1].qkv, E, 3 * E, tile, sm);
        sync();
    }
    df_final(a, tile, (int)nblocks, sm);
    LTF_STAMP(1);
}

// ---------------------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------------------
static bool
df_repack(const void* w, int type, int N, int K, std::vector<void*>& owned, ltf_q8p& out) {
    if (type != LTF_TYPE_Q8_0 || !w)
        return false;
    return ltf_repack_q8(w, N, K, owned, out);
}

magpietts_decoder_fused*
magpietts_decoder_fused_create(
    const magpietts_decoder_fused_weights& w, const magpietts_decoder_fused_cache& cache,
    char* error, size_t error_size) {
    if (!ltf_kernel_arch_supported(reinterpret_cast<const void*>(df_kernel))) {
        ltf_set_error(
            error, error_size, "fused decoder: needs compute capability 8.0+ and an sm_80+ build");
        return nullptr;
    }
    if (w.n_embd <= 0 || w.n_embd > LTF_MAX_EMBD || w.n_embd % 64 != 0 || w.n_head <= 0 ||
        w.n_embd / w.n_head != 64 || w.n_ff <= 0 || w.n_ff > LTF_MAX_K || w.n_ff % 64 != 0 ||
        w.cross_dim <= 0 || w.cross_dim % 32 != 0 || w.n_layers <= 0 ||
        w.n_layers > DF_MAX_LAYERS || w.n_codebooks <= 0 || w.n_codebooks > DF_MAX_CODEBOOKS ||
        !w.audio_emb || !w.pos_emb || !w.layers || cache.cache_len <= 0 || !cache.kv_arena ||
        cache.text_capacity <= 0 || cache.text_capacity % DF_XATTN_BLOCKS != 0 ||
        cache.text_capacity / DF_XATTN_BLOCKS > DF_XATTN_MAX_KEYS ||
        cache.cache_len + 1 > DF_ATTN_SPLITS * DF_ATTN_MAX_RANGE || !cache.cross_k ||
        !cache.cross_v) {
        ltf_set_error(error, error_size, "fused decoder: unsupported shape");
        return nullptr;
    }
    if ((w.emb_type != LTF_TYPE_F16 && w.emb_type != LTF_TYPE_F32) ||
        (w.pos_type != LTF_TYPE_F16 && w.pos_type != LTF_TYPE_F32)) {
        ltf_set_error(error, error_size, "fused decoder: unsupported embedding types");
        return nullptr;
    }
    magpietts_decoder_fused* f = new magpietts_decoder_fused;
    f->w = w;
    f->cache = cache;
    f->layer_w.assign(w.layers, w.layers + w.n_layers);
    f->audio_emb.assign(w.audio_emb, w.audio_emb + w.n_codebooks);
    f->layers.resize(w.n_layers);
    const int E = w.n_embd, NFF = w.n_ff, CD = w.cross_dim;
    for (int i = 0; i < w.n_layers; ++i) {
        const magpietts_decoder_fused_layer_weights& L = w.layers[i];
        df_layer& d = f->layers[i];
        bool ok = df_repack(L.qkv, L.qkv_type, 3 * E, E, f->owned, d.qkv) &&
                  df_repack(L.o, L.o_type, E, E, f->owned, d.o) &&
                  df_repack(L.ff1, L.ff1_type, NFF, E, f->owned, d.ff1) &&
                  df_repack(L.ff2, L.ff2_type, E, NFF, f->owned, d.ff2) && L.norm_self && L.norm_ff;
        if (ok && L.has_cross)
            ok = df_repack(L.cross_q, L.cross_q_type, CD, E, f->owned, d.cq) &&
                 df_repack(L.cross_o, L.cross_o_type, E, CD, f->owned, d.co) && L.norm_xq;
        if (ok)
            ok = ltf_ln_fold_constants(d.qkv, L.norm_self, 3 * E, E, f->owned, d.qkv_c) &&
                 ltf_ln_fold_constants(d.ff1, L.norm_ff, NFF, E, f->owned, d.ff1_c);
        if (ok && L.has_cross)
            ok = ltf_ln_fold_constants(d.cq, L.norm_xq, CD, E, f->owned, d.cq_c);
        if (!ok) {
            ltf_set_error(
                error, error_size, "fused decoder: Q8_0 projections required (repack failed)");
            magpietts_decoder_fused_free(f);
            return nullptr;
        }
        d.norm_self = L.norm_self;
        d.norm_xq = L.norm_xq;
        d.norm_ff = L.norm_ff;
        d.kv_arena = cache.kv_arena[i];
        d.flags = (L.has_cross ? 1 : 0) | (L.collect_alignment && L.has_cross ? 2 : 0) |
                  (L.apply_prior ? 4 : 0);
        if (L.has_cross && L.collect_alignment)
            ++f->n_collect;
    }
    int device = 0, sms = 0, per_sm = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        ltf_set_error(error, error_size, "fused decoder: device query failed");
        magpietts_decoder_fused_free(f);
        return nullptr;
    }
    if (cudaFuncSetAttribute(
            df_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(ltf_chain_smem)) !=
        cudaSuccess) {
        ltf_set_error(error, error_size, "fused decoder: shared memory attribute failed");
        magpietts_decoder_fused_free(f);
        return nullptr;
    }
    cudaFuncSetAttribute(
        df_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, df_kernel, LTF_THREADS, sizeof(ltf_chain_smem)) != cudaSuccess ||
        per_sm < 1) {
        ltf_set_error(error, error_size, "fused decoder: kernel does not fit one block per SM");
        magpietts_decoder_fused_free(f);
        return nullptr;
    }
    // Same tiling contract as the LT chain: exactly LTF_CHAIN_MIN_GRID blocks, one per SM, on any
    // GPU with at least that many SMs (the rest stay free for the codec).
    const int grid = LTF_CHAIN_MIN_GRID;
    if (sms < grid) {
        ltf_set_error(error, error_size, "fused decoder: needs at least 64 SMs");
        magpietts_decoder_fused_free(f);
        return nullptr;
    }
    auto rows_per_block = [grid](int N) { return (N + grid - 1) / grid; };
    auto fits = [&](int N, int Kdim, int R) {
        return rows_per_block(N) <= LTF_WARPS * R && rows_per_block(N) * Kdim <= LTF_STAGE_BYTES &&
               rows_per_block(N) * (Kdim >> 5) <= LTF_STAGE_SCALES;
    };
    const bool shape_ok = grid >= LTF_CHAIN_MIN_GRID && grid <= LTF_CHAIN_MAX_GRID &&
                          E % grid == 0 && (E / grid) % 4 == 0 && NFF % grid == 0 &&
                          (NFF / grid) % 16 == 0 && CD % grid == 0 && fits(3 * E, E, LTF_R_QKV) &&
                          fits(E, E, LTF_R_O) && fits(CD, E, DF_R_CQ) && fits(E, CD, LTF_R_O) &&
                          fits(NFF, E, LTF_R_FF1) && fits(E, NFF, LTF_R_O) && (NFF >> 5) <= 96 &&
                          DF_XATTN_BLOCKS <= grid;
    if (!shape_ok) {
        ltf_set_error(error, error_size, "fused decoder: shapes do not fit the phase tiling");
        magpietts_decoder_fused_free(f);
        return nullptr;
    }
    f->grid = grid;
    // scratch arena
    size_t total = 0;
    auto reserve = [&](size_t n_floats) {
        const size_t off = total;
        total += (n_floats * sizeof(float) + 255) & ~(size_t)255;
        return off;
    };
    const size_t off_h = reserve(2 * E), off_hq = reserve(2 * E / 4), off_hd = reserve(2 * E / 4),
                 off_stats = reserve(4 * LTF_CHAIN_MAX_GRID), off_qkv = reserve(2 * 3 * E),
                 off_apart = reserve((size_t)2 * w.n_head * DF_ATTN_SPLITS * 66),
                 off_qc = reserve(2 * CD), off_xpart = reserve((size_t)DF_XATTN_BLOCKS * (2 + CD)),
                 off_xs = reserve(cache.text_capacity), off_aa = reserve(cache.text_capacity),
                 off_ffq = reserve(2 * NFF / 4), off_ffd = reserve(2 * NFF / 16);
    if (cudaMalloc(&f->arena, total) != cudaSuccess ||
        cudaMemset(f->arena, 0, total) != cudaSuccess ||
        cudaMalloc(&f->barrier, 3 * sizeof(unsigned int)) != cudaSuccess) {
        ltf_set_error(error, error_size, "fused decoder: allocation failed");
        magpietts_decoder_fused_free(f);
        return nullptr;
    }
    f->arena_bytes = total;
    char* base = reinterpret_cast<char*>(f->arena);
    df_args& a = f->a;
    a.E = E;
    a.H = w.n_head;
    a.NFF = NFF;
    a.CD = CD;
    a.n_layers = w.n_layers;
    a.n_codebooks = w.n_codebooks;
    a.cache_len = cache.cache_len;
    a.TC = cache.text_capacity;
    a.ln_eps = w.ln_eps;
    a.attn_scale = 1.0f / sqrtf(64.0f);
    a.xattn_scale = 1.0f / sqrtf((float)CD);
    a.emb_scale = 1.0f / (float)w.n_codebooks;
    for (int i = 0; i < w.n_layers; ++i) a.layers[i] = f->layers[i];
    for (int c = 0; c < w.n_codebooks; ++c) a.audio_emb[c] = w.audio_emb[c];
    a.emb_type = w.emb_type;
    a.pos_emb = w.pos_emb;
    a.pos_type = w.pos_type;
    a.norm_out = w.norm_out;
    a.cross_k = cache.cross_k;
    a.cross_v = cache.cross_v;
    a.n_collect = f->n_collect;
    a.h = reinterpret_cast<float*>(base + off_h);
    a.h_q = reinterpret_cast<int8_t*>(base + off_hq);
    a.h_d = reinterpret_cast<float*>(base + off_hd);
    a.stats = reinterpret_cast<float*>(base + off_stats);
    a.qkv = reinterpret_cast<float*>(base + off_qkv);
    a.apart = reinterpret_cast<float*>(base + off_apart);
    a.qc = reinterpret_cast<float*>(base + off_qc);
    a.xpart = reinterpret_cast<float*>(base + off_xpart);
    a.xscores = reinterpret_cast<float*>(base + off_xs);
    a.align_acc = reinterpret_cast<float*>(base + off_aa);
    a.ff_q = reinterpret_cast<int8_t*>(base + off_ffq);
    a.ff_d = reinterpret_cast<float*>(base + off_ffd);
    a.barrier = f->barrier;
    if (error && error_size)
        error[0] = '\0';
    return f;
}

void
magpietts_decoder_fused_free(magpietts_decoder_fused* f) {
    if (!f)
        return;
    for (void* p : f->owned) cudaFree(p);
    cudaFree(f->arena);
    cudaFree(f->barrier);
    delete f;
}

bool
magpietts_decoder_fused_supported(const magpietts_decoder_fused* f) {
    return f && f->grid > 0;
}

int
magpietts_decoder_fused_grid(const magpietts_decoder_fused* f) {
    return f ? f->grid : 0;
}

bool
magpietts_decoder_fused_step(
    magpietts_decoder_fused* f, cudaStream_t stream, const magpietts_decoder_fused_step_args& s,
    char* error, size_t error_size) {
    if (!f || f->grid <= 0 || !s.tokens || !s.hidden_cond || !s.hidden_uncond || !s.prior ||
        !s.mask || s.valid_len < 0 || s.valid_len > f->cache.cache_len || s.ring_head < 0 ||
        s.ring_head >= f->cache.cache_len) {
        ltf_set_error(error, error_size, "fused decoder: invalid step arguments");
        return false;
    }
    df_args a = f->a;
    a.tokens = s.tokens;
    a.position = s.position;
    a.ring_head = s.ring_head;
    a.valid_len = s.valid_len;
    a.prior = s.prior;
    a.mask = s.mask;
    a.hidden_cond = s.hidden_cond;
    a.hidden_uncond = s.hidden_uncond;
    a.alignment = s.alignment;
    cudaError_t err = cudaMemsetAsync(f->barrier, 0, 3 * sizeof(unsigned int), stream);
    if (err != cudaSuccess) {
        ltf_set_error(error, error_size, "fused decoder: barrier reset failed", err);
        return false;
    }
    df_kernel<<<f->grid, LTF_THREADS, sizeof(ltf_chain_smem), stream>>>(a);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        ltf_set_error(error, error_size, "fused decoder: launch failed", err);
        return false;
    }
    return true;
}
