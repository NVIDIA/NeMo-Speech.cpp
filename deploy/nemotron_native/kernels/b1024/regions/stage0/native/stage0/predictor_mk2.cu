#include "../../../../../inflight/progress.cuh"
#if MK2_INFLIGHT
#include <curand_kernel.h>
#endif
#include <kittens.cuh>
#include <math_constants.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>
#include <utility>
#include <memory>
#include <fstream>
#include <algorithm>
using namespace kittens;
constexpr int D_MODEL = 1536;
constexpr int D_FFN = 6144;
constexpr int N_HEADS = 12;
constexpr int D_HEAD = 128;
constexpr int N_LAYERS = 3;
constexpr int N_POS = 16;
#ifndef MK2_BATCH
#define MK2_BATCH 32
#endif
constexpr int BATCH = MK2_BATCH;
#ifndef MK2_ACTIVE_EXPERTS
#define MK2_ACTIVE_EXPERTS 0
#endif
#ifndef MK2_SSM_SPLITS
#define MK2_SSM_SPLITS 1
#endif
static_assert(MK2_SSM_SPLITS > 0 && 24 % MK2_SSM_SPLITS == 0);
#define MK2_FUSE_CONV (MK2_SSM_SPLITS == 1)
#ifndef MK2_DIRECT_INPUT
#define MK2_DIRECT_INPUT 0
#endif
#ifndef MK2_B1_LOCAL_BARRIERS
#define MK2_B1_LOCAL_BARRIERS 0
#endif
#ifndef MK2_SKIP_HIDDEN_SNAPSHOT
#define MK2_SKIP_HIDDEN_SNAPSHOT 0
#endif
constexpr int BATCH_PAD = 64;
constexpr int DECODE_TILES = BATCH / BATCH_PAD;
constexpr int DECODE_ROWS = BATCH_PAD;
static_assert(BATCH == 512 || BATCH == 1024 || BATCH == 2048);
static_assert(BATCH % BATCH_PAD == 0);
constexpr int CODEBOOK = 1032;
constexpr int VOCAB_PADDED = 1088;
constexpr int TOP_K = 80;
constexpr int BB_LAYERS = 31;
constexpr float RMS_EPS = 1e-5f;
constexpr int MB_HEADS = 64;
constexpr int MB_HEAD_DIM = 24;
constexpr int MB_STATE = 128;
constexpr int MB_GROUPS = 8;
constexpr int MB_CONV_K = 4;
constexpr int MB_INNER = MB_HEADS * MB_HEAD_DIM;
constexpr int MB_CONV_DIM = MB_INNER + 2 * MB_GROUPS * MB_STATE;
constexpr int MB_IN_PROJ = 2 * MB_INNER + 2 * MB_GROUPS * MB_STATE + MB_HEADS;
constexpr int MB_GROUP_SIZE = MB_INNER / MB_GROUPS;
constexpr int MB_HEADS_PER_GROUP = MB_HEADS / MB_GROUPS;
constexpr int BB_ATT_HEADS = 12;
constexpr int BB_KV_HEADS = 4;
constexpr int BB_HEAD_DIM = 128;
constexpr int BB_Q_SIZE = BB_ATT_HEADS * BB_HEAD_DIM;
constexpr int BB_KV_SIZE = BB_KV_HEADS * BB_HEAD_DIM;
constexpr int BB_QKV = BB_Q_SIZE + 2 * BB_KV_SIZE;
constexpr int BB_T_MAX = 8;
constexpr int BB_EXPERTS = 24;
constexpr int BB_TOP_K = 4;
constexpr int BB_MOE_INTER = 768;
constexpr int BB_SHARED_INTER = 2048;
constexpr float BB_ROUTED_SCALE = 2.5f;
constexpr int BB_PHONEME = 2051;
constexpr int BB_PHON_PADDED = 2112;
#ifndef MK2_PREFILL_MAX_TOKENS
#define MK2_PREFILL_MAX_TOKENS 4288
#endif
constexpr int PREFILL_MAX_TOKENS = MK2_PREFILL_MAX_TOKENS;
constexpr int PREFILL_STORAGE_TOKENS =
    ((PREFILL_MAX_TOKENS + BATCH_PAD - 1) / BATCH_PAD) * BATCH_PAD;
#ifndef PF_SSM_SPLIT_P
#define PF_SSM_SPLIT_P 1
#endif
#ifndef PF_SKIP_MAMBA
#define PF_SKIP_MAMBA 0
#endif
#ifndef PF_SKIP_ATTN
#define PF_SKIP_ATTN 0
#endif
#ifndef PF_SKIP_MOE
#define PF_SKIP_MOE 0
#endif
constexpr int SORT_N = 1024;
constexpr int AUDIO_EOS = 1025;
constexpr int LN_EPS_NUM = 1;
constexpr float LN_EPS = 1e-5f;
constexpr int OUT_ATOM = 64;
#ifndef KCHUNK
#define KCHUNK 256
#endif
#ifdef ABLATE_BB_MAMBA
#define BB_SKIP_MAMBA 1
#else
#define BB_SKIP_MAMBA 0
#endif
#ifdef ABLATE_BB_MOE
#define BB_SKIP_MOE 1
#else
#define BB_SKIP_MOE 0
#endif
#ifdef ABLATE_BB_ATTN
#define BB_SKIP_ATTN 1
#else
#define BB_SKIP_ATTN 0
#endif
#ifndef BB_ATTN_SKIP_KV_WRITE
#define BB_ATTN_SKIP_KV_WRITE 0
#endif
#ifndef BB_ATTN_LEGACY_CUR_KV
#define BB_ATTN_LEGACY_CUR_KV 0
#endif
#ifndef STAGES
#define STAGES 3
#endif
#ifndef NMULT
#define NMULT 1
#endif
#ifndef SSM_PREFETCH_MAX
#define SSM_PREFETCH_MAX 8
#endif
constexpr int NM_MAX = (NMULT > 1 && DECODE_TILES % NMULT == 0) ? NMULT : 1;
#ifndef MK2_OUT_MULT
#define MK2_OUT_MULT 1
#endif
constexpr int MT = MK2_OUT_MULT;
static_assert(MT >= 1 && MT <= NM_MAX, "an item holds at most NM_MAX accumulators");
constexpr int SLOTS = 1 + NM_MAX;
static_assert(MT + 1 <= SLOTS);
constexpr int THREADS = 256;
constexpr int CONSUMER_WARPS = 4;
constexpr int WARPS = THREADS / 32;
constexpr int NORM_SCRATCH_STRIDE = 2 * WARPS + 2;
using weight_tile = st_bf<OUT_ATOM, KCHUNK>;
using act_tile = st_bf<BATCH_PAD, KCHUNK>;
using out_tile = st_bf<OUT_ATOM, BATCH_PAD>;
using out_tile_f = st<float, OUT_ATOM, BATCH_PAD>;
using weight_gl = gl<bf16, 1, 1, -1, -1, weight_tile>;
using act_gl = gl<bf16, 1, 1, -1, -1, act_tile>;
enum Epilogue { EPI_STORE, EPI_RESIDUAL, EPI_GELU, EPI_BIAS,
                EPI_SILU, EPI_MOE_ACC, EPI_ACC_F32 };
#define EPI_ROW(idx) ((OUT_ATOM == 64 && BATCH_PAD == 64) \
    ? (((idx) & 15) | ((((idx) >> 6) & 3) << 4)) : ((idx) / (OUT_ATOM / 8)))
#define EPI_OCT(idx) ((OUT_ATOM == 64 && BATCH_PAD == 64) \
    ? ((((idx) >> 4) & 3) | ((((idx) >> 8) & 1) << 2)) : ((idx) % (OUT_ATOM / 8)))
struct alignas(16) bf16_oct { bf16 x0, x1, x2, x3, x4, x5, x6, x7; };
static_assert(sizeof(bf16_oct) == 16);
#include <cuda_fp8.h>
constexpr int R9_STATE_MODE=1;
using r9_state_t=bf16;
__device__ __forceinline__ int r9_scale_index(int head,int channel,int n) {return (((head/8)*4+(head%8)/2)*4+channel/6)*128+n;}
__device__ __forceinline__ float r9_load(r9_state_t v,float scale) {
return __bfloat162float(v);
}
__device__ __forceinline__ r9_state_t r9_store(float x,float scale) {
return __float2bfloat16(x);
}
struct MambaLayer {
    weight_gl in_proj;
    weight_gl out_proj;
    const bf16 *conv_w;
    const bf16 *conv_b;
    const float *A;
    const float *D;
    const float *dt_bias;
    const bf16 *gnorm_w;
    bf16 *conv_state;
    int conv_slot_stride;
    float *ssm_state;
    int ssm_slot_stride;
    const float *r8_initial_ssm = nullptr;
    const unsigned int *r8_initial_ready = nullptr;
    const int *r8_seq_len = nullptr;
    int r8_initial_positions = 0;
    const float *r9_scales = nullptr;
    const int *r9_map = nullptr;
};
struct Globals {
    weight_gl qkv[N_LAYERS], o[N_LAYERS], ff_in[N_LAYERS], ff_out[N_LAYERS];
    weight_gl head[N_POS];
    act_gl x_v;
    act_gl normed_v;
    act_gl attn_v;
    act_gl ffh_v;
    bf16 *x, *qkv_out, *attn_out, *ff_hidden, *normed;
    bf16 *logits;
    const bf16 *head_bias[N_POS];
    const int8_t *forbidden;
    const float *gumbel;
    int *sampled;
    int *sampled_pos;
    float temperature;
    const float *norm_self[N_LAYERS], *norm_ff[N_LAYERS];
    const bf16 *pos_emb;
    const bf16 *audio_emb[N_POS];
    bf16 *kcache, *vcache, *hidden_rows;
    const int *codes;
    unsigned int *barrier;
    const bf16 *native_embedding = nullptr;
    long long *native_codes = nullptr;
    int *native_seq_len = nullptr;
};
struct AttnLayer {
    weight_gl qkv;
    weight_gl o_proj;
    bf16 *pool_base;
    const int *block_table;
    int max_blocks_per_seq;
    int tokens_per_block;
};
__device__ __forceinline__ bf16 *kv_slot(
    const AttnLayer &L, int seq, int kv_sel, int pos) {
    const int *row = L.block_table +
                     ((size_t)seq * 2 + kv_sel) * L.max_blocks_per_seq;
    const int blk = row[pos / L.tokens_per_block] & 0x7FFFFFFF;
    const size_t elems_per_block =
        (size_t)L.tokens_per_block * BB_KV_HEADS * BB_HEAD_DIM;
    return L.pool_base + (size_t)blk * elems_per_block;
}
__device__ __forceinline__ int kv_offset(const AttnLayer &L, int pos, int head) {
    return head * L.tokens_per_block * BB_HEAD_DIM +
           (pos % L.tokens_per_block) * BB_HEAD_DIM;
}
struct MoeLayer {
    weight_gl exp_up;
    weight_gl exp_down;
    weight_gl sh_up;
    weight_gl sh_down;
    const bf16 *gate_w;
    const bf16 *gate_bias;
};
struct Backbone {
    act_gl normed_v;
    act_gl gated_v;
    const MambaLayer *mamba;
    const bf16 *norm_f_w;
    const AttnLayer *attn;
    const MoeLayer *moe;
    const bf16 *norm_w[BB_LAYERS];
    int mamba_of_layer[BB_LAYERS];
    int attn_of_layer[BB_LAYERS];
    int moe_of_layer[BB_LAYERS];
    int past_len;
    bf16 *h;
    bf16 *r;
    bf16 *normed;
    bf16 *zxbcdt;
    bf16 *gated;
    bf16 *layer_out;
    bf16 *qkv;
    bf16 *moe_h;
    bf16 *moe_sh;
    float *moe_out;
    float *moe_w;
    const float *moe_w_inject;
    float *moe_w_audit;
    bf16 *dec_snapshot;
    bf16 *xbc_act;
    const int *slot_map;
    int n_prefill_tokens;
    int prefill_slot;
    int pf_n_seq;
    int pf_fuse_decode;
    const bf16 *prompt;
    const int *pf_seq_slot;
    bf16 *pf_h, *pf_r, *pf_normed;
    bf16 *pf_zxbcdt;
    bf16 *pf_xbc_act;
    bf16 *pf_gated;
    bf16 *pf_qkv;
    bf16 *pf_moe_h, *pf_moe_sh;
    float *pf_moe_out, *pf_moe_w;
    bf16 *pf_layer_out;
    bf16 *pf_final;
    act_gl pf_normed_v, pf_gated_v, pf_qkv_v, pf_moe_h_v, pf_moe_sh_v;
    weight_gl phoneme_w;
    const bf16 *phoneme_b;
    bf16 *phoneme_logits;
    int *phoneme_out;
    unsigned long long *launches;
    const int *seq_len;
    act_gl qkv_v;
    act_gl moe_h_v;
    act_gl moe_sh_v;
    int native_prefill_reset = 0;
    bf16 *r6_packed = nullptr;
    int *r6_map = nullptr,*r6_counts = nullptr;
    act_gl *r6_views = nullptr;
    int *r6_lut = nullptr;
};
#ifndef BARRIER_BACKOFF_NS
#define BARRIER_BACKOFF_NS 32
#endif
#define BARRIER_GEN_WORD 32
__device__ __forceinline__ void grid_barrier(unsigned int *state) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        constexpr unsigned long long GRID_BLOCKS = 132ull;
        auto *wide_state = reinterpret_cast<unsigned long long *>(state);
        const unsigned long long arrival = atomicAdd(wide_state, 1ull);
        const unsigned int rank = (unsigned int)(arrival % GRID_BLOCKS);
        if (rank != GRID_BLOCKS - 1u) {
            const unsigned long long target = arrival - rank + GRID_BLOCKS;
            volatile unsigned long long *counter = wide_state;
            while (*counter < target) __nanosleep(BARRIER_BACKOFF_NS);
        }
    }
    __syncthreads();
}
__device__ __forceinline__ float gelu_tanh_fp32(float v) {
    const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.f + tanhf(inner));
}
__device__ void layernorm_rows(
    const bf16 *src, const float *weight, bf16 *dst, float *scratch) {
    constexpr int PER_THREAD = D_MODEL / THREADS;
    float wv[PER_THREAD];
    #pragma unroll
    for (int j = 0; j < PER_THREAD; ++j) wv[j] = weight[threadIdx.x + j * THREADS];
    float v[PER_THREAD], v_next[PER_THREAD];
    #pragma unroll
    for (int j = 0; j < PER_THREAD; ++j)
        v[j] = __bfloat162float(
            src[(size_t)blockIdx.x * D_MODEL + threadIdx.x + j * THREADS]);
    for (int row = blockIdx.x, row_iter = 0; row < BATCH;
         row += gridDim.x, ++row_iter) {
        float *row_scratch = scratch + (row_iter & 1) * NORM_SCRATCH_STRIDE;
        const int next_row = row + (int)gridDim.x;
        if (next_row < BATCH) {
            #pragma unroll
            for (int j = 0; j < PER_THREAD; ++j)
                v_next[j] = __bfloat162float(
                    src[(size_t)next_row * D_MODEL + threadIdx.x + j * THREADS]);
        }
        float sum = 0.f, sq = 0.f;
        #pragma unroll
        for (int j = 0; j < PER_THREAD; ++j) {
            sum += v[j];
            sq += v[j] * v[j];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, off);
            sq += __shfl_xor_sync(0xffffffff, sq, off);
        }
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        if (lane == 0) {
            row_scratch[warp] = sum;
            row_scratch[WARPS + warp] = sq;
        }
        __syncthreads();
        float s = 0.f, q = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            s += row_scratch[w];
            q += row_scratch[WARPS + w];
        }
        const float mean = s / D_MODEL;
        const float inv =
            rsqrtf(q / D_MODEL - (s / D_MODEL) * (s / D_MODEL) + LN_EPS);
        #pragma unroll
        for (int j = 0; j < PER_THREAD; ++j) {
            const int i = threadIdx.x + j * THREADS;
            dst[(size_t)row * D_MODEL + i] =
                __float2bfloat16((v[j] - mean) * inv * wv[j]);
        }
        #pragma unroll
        for (int j = 0; j < PER_THREAD; ++j) v[j] = v_next[j];
    }
}
__device__ void bb_norm_residual(
    bf16 *h, bf16 *r, const bf16 *weight, bf16 *out, float *scratch,
    int n_rows = BATCH, bf16 *snapshot = nullptr, int snapshot_layer = -1,
    float *snapshot_fp32 = nullptr, float *out_fp32 = nullptr,
    const float *fold_src = nullptr, const float *fold_routes = nullptr) {
    constexpr int PER_THREAD = D_MODEL / THREADS;
    bf16 wv[PER_THREAD];
    extern __shared__ int __shm[];
    bf16 *const row_stage = reinterpret_cast<bf16 *>(__shm);
    constexpr int ROW_SLOT = 2 * D_MODEL;
    constexpr int ROW_VECTORS = D_MODEL / 8;
    if ((int)blockIdx.x < n_rows && threadIdx.x < ROW_VECTORS) {
        const int i = 8 * threadIdx.x;
        const size_t at = (size_t)blockIdx.x * D_MODEL + i;
        const uint32_t rs = static_cast<uint32_t>(
            __cvta_generic_to_shared(row_stage + D_MODEL + i));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                     :: "r"(rs), "l"(r + at) : "memory");
        if (!fold_src) {
            const uint32_t hs = static_cast<uint32_t>(
                __cvta_generic_to_shared(row_stage + i));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                         :: "r"(hs), "l"(h + at) : "memory");
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    #pragma unroll
    for (int j = 0; j < PER_THREAD; ++j)
        wv[j] = weight[threadIdx.x + j * THREADS];
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    for (int row = blockIdx.x, row_iter = 0; row < n_rows;
         row += gridDim.x, ++row_iter) {
        bf16 *const current = row_stage + (row_iter & 1) * ROW_SLOT;
        const int next_row = row + (int)gridDim.x;
        if (next_row < n_rows) {
            bf16 *const next = row_stage + ((row_iter + 1) & 1) * ROW_SLOT;
            if (threadIdx.x < ROW_VECTORS) {
                const int i = 8 * threadIdx.x;
                const size_t at = (size_t)next_row * D_MODEL + i;
                const uint32_t rs = static_cast<uint32_t>(
                    __cvta_generic_to_shared(next + D_MODEL + i));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                             :: "r"(rs), "l"(r + at) : "memory");
                if (!fold_src) {
                    const uint32_t hs = static_cast<uint32_t>(
                        __cvta_generic_to_shared(next + i));
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                                 :: "r"(hs), "l"(h + at) : "memory");
                }
            }
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }
        float *row_scratch = scratch + (row_iter & 1) * NORM_SCRATCH_STRIDE;
        const size_t base = (size_t)row * D_MODEL;
        float fv[PER_THREAD];
        if (fold_src) {
            unsigned mask = 0u;
            const float *rw = fold_routes + (size_t)row * BB_EXPERTS;
            #pragma unroll
            for (int e = 0; e < BB_EXPERTS; ++e)
                if (rw[e] != 0.f) mask |= 1u << e;
            const float *pe = fold_src
                + (size_t)row * (BB_EXPERTS + 1) * D_MODEL + threadIdx.x;
            #pragma unroll
            for (int j = 0; j < PER_THREAD; ++j) fv[j] = 0.f;
            #pragma unroll
            for (int e = 0; e <= BB_EXPERTS; ++e) {
                const bool take = e == BB_EXPERTS || ((mask >> e) & 1u);
                #pragma unroll
                for (int j = 0; j < PER_THREAD; ++j)
                    fv[j] = __fadd_rn(fv[j],
                        take ? pe[(size_t)e * D_MODEL + j * THREADS] : 0.f);
            }
        }
        float v[PER_THREAD];
        float sq = 0.f;
        #pragma unroll
        for (int j = 0; j < PER_THREAD; ++j) {
            const int i = threadIdx.x + j * THREADS;
            const bf16 hv = fold_src ? __float2bfloat16(fv[j]) : current[i];
            if (fold_src) h[base + i] = hv;
            if (snapshot)
                snapshot[((size_t)snapshot_layer * BATCH + row) * D_MODEL + i] = hv;
            if (snapshot_fp32)
                snapshot_fp32[((size_t)snapshot_layer * BATCH + row) * D_MODEL + i] =
                    __bfloat162float(hv);
            v[j] = __bfloat162float(hv) +
                   __bfloat162float(current[D_MODEL + i]);
            r[base + i] = __float2bfloat16(v[j]);
            sq += v[j] * v[j];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sq += __shfl_xor_sync(0xffffffff, sq, off);
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        if (lane == 0) row_scratch[warp] = sq;
        __syncthreads();
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) s += row_scratch[w];
        const float inv = rsqrtf(s / D_MODEL + RMS_EPS);
        #pragma unroll
        for (int j = 0; j < PER_THREAD; ++j) {
            const int i = threadIdx.x + j * THREADS;
            const bf16 normed = __float2bfloat16(
                __bfloat162float(wv[j]) * __bfloat162float(__float2bfloat16(v[j] * inv)));
            out[base + i] = normed;
            if (out_fp32) out_fp32[base + i] = __bfloat162float(normed);
        }
        if (next_row < n_rows) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
        }
    }
}
__device__ __forceinline__ bf16 mamba_conv_channel(
    const MambaLayer &L, bf16 *state_slot, const bf16 *zrow, int c) {
    const unsigned ch = (unsigned)c;
    const bf16 *w = L.conv_w + ch * (unsigned)MB_CONV_K;
    bf16 *state = state_slot + ch * (unsigned)(MB_CONV_K - 1);
    const bf16 rawb = zrow[c];
    const float raw = __bfloat162float(rawb);
    float acc = __bfloat162float(L.conv_b[c]);
    #pragma unroll
    for (int k = 0; k < MB_CONV_K - 1; ++k)
        acc += __bfloat162float(w[k]) * __bfloat162float(state[k]);
    acc += __bfloat162float(w[MB_CONV_K - 1]) * raw;
    acc = acc / (1.f + __expf(-acc));
    #pragma unroll
    for (int k = 0; k < MB_CONV_K - 2; ++k) state[k] = state[k + 1];
    state[MB_CONV_K - 2] = rawb;
    return __float2bfloat16(acc);
}
__device__ void mamba_conv(
    const MambaLayer &L, const bf16 *zxbcdt, bf16 *xbc_act, const int *slot_map MK2_PROGRESS_PARAMETER(progress_mask)) {
    const int total = BATCH * MB_CONV_DIM;
    for (int idx = blockIdx.x * THREADS + threadIdx.x; idx < total;
         idx += gridDim.x * THREADS) {
        const int row = idx / MB_CONV_DIM;
        MK2_PROGRESS_GUARD(progress_mask, row);
        const int c = idx - row * MB_CONV_DIM;
        xbc_act[(size_t)row * MB_CONV_DIM + c] = mamba_conv_channel(
            L, L.conv_state + (size_t)slot_map[row] * L.conv_slot_stride,
            zxbcdt + (size_t)row * MB_IN_PROJ + MB_INNER, c);
    }
}
__device__ __forceinline__ float mamba_softplus(float x) {
    return x > 20.f ? x : log1pf(__expf(x));
}
__device__ void mamba_ssm(
    const MambaLayer &L, const bf16 *zxbcdt, const bf16 *xbc_act, bf16 *gated,
    const int *slot_map MK2_PROGRESS_PARAMETER(progress_mask)) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int warps_total = gridDim.x * WARPS;
    const int sb4 = (lane >> 4) & 1, sb3 = (lane >> 3) & 1;
    const int sb2 = (lane >> 2) & 1, sb1 = (lane >> 1) & 1;
    const int ssm_own_ch = 6 * sb4 + 3 * sb3 + 2 * sb2 + sb1;
    const bool ssm_owner = (2 * sb2 + sb1) < 3 && (lane & 1) == 0;
    constexpr int split = MK2_SSM_SPLITS;
    constexpr int channels = MB_HEAD_DIM / split;
    constexpr bool FUSE_GNORM = (split == 1);
    extern __shared__ int __shm[];
    bf16 (*const gn_stage)[MB_GROUP_SIZE] =
        reinterpret_cast<bf16 (*)[MB_GROUP_SIZE]>(&__shm[0]);
    constexpr bool FUSE_CONV = (MK2_FUSE_CONV != 0);
    bf16 (*const conv_x)[MB_GROUP_SIZE] = gn_stage + WARPS;
    constexpr int SSM_NSTATE = MB_STATE / 32;
    constexpr int SSM_PF = channels < SSM_PREFETCH_MAX ? channels : SSM_PREFETCH_MAX;
    constexpr int SSM_GROUP_BYTES = SSM_PF * MB_STATE * (int)sizeof(r9_state_t);
    constexpr bool SSM_SMEM_OK = R9_STATE_MODE == 1 && channels == 2 * SSM_PF &&
        SSM_GROUP_BYTES % 512 == 0;
    constexpr int SSM_RING = 4;
    constexpr int SSM_GROUPS_PER_HEAD = channels / SSM_PF;
    constexpr int SSM_PAIRS = MB_HEADS_PER_GROUP * SSM_GROUPS_PER_HEAD;
    static_assert(!SSM_SMEM_OK || (SSM_RING & (SSM_RING - 1)) == 0,
                  "the flat (head, channel group) ring index is masked");
    static_assert(!SSM_SMEM_OK || SSM_PAIRS >= SSM_RING,
                  "the prologue must be able to fill the ring");
    static_assert(!SSM_SMEM_OK ||
        2 * WARPS * MB_GROUP_SIZE * sizeof(bf16) +
        WARPS * SSM_RING * SSM_GROUP_BYTES <= STAGES * SLOTS * sizeof(weight_tile));
    constexpr int SSM_F32_RING = 2;
    constexpr int SSM_F32_GROUP_BYTES = SSM_PF * MB_STATE * (int)sizeof(float);
    constexpr bool SSM_F32_OK = SSM_SMEM_OK && SSM_F32_GROUP_BYTES % 512 == 0 &&
        (SSM_F32_RING & (SSM_F32_RING - 1)) == 0 && SSM_PAIRS >= SSM_F32_RING &&
        SSM_F32_RING * SSM_F32_GROUP_BYTES <= SSM_RING * SSM_GROUP_BYTES;
    char *const ssm_stage = reinterpret_cast<char *>(conv_x + WARPS);
    const int pairs = BATCH * MB_GROUPS * split;
    for (int pair = blockIdx.x * WARPS + warp; pair < pairs; pair += warps_total) {
        const int row = pair / (MB_GROUPS * split);
        MK2_PROGRESS_GUARD(progress_mask, row);
        const int group = (pair / split) % MB_GROUPS;
        const int first_channel = (pair % split) * channels;
        const bf16 *xa = xbc_act + (size_t)row * MB_CONV_DIM;
        const bf16 *Bsec = xa + MB_INNER + group * MB_STATE;
        const bf16 *Csec = xa + MB_INNER + MB_GROUPS * MB_STATE + group * MB_STATE;
        const bf16 *dtrow = zxbcdt + (size_t)row * MB_IN_PROJ + MB_INNER + MB_CONV_DIM;
        const bool r8_initial = L.r8_initial_ssm && L.r8_initial_ready &&
            *L.r8_initial_ready == 1 && L.r8_seq_len[row] == L.r8_initial_positions;
        r9_state_t *st = reinterpret_cast<r9_state_t *>(L.ssm_state) +
                         (size_t)slot_map[row] * L.ssm_slot_stride;
        char *const warp_ssm =
            ssm_stage + (size_t)warp * SSM_RING * SSM_GROUP_BYTES;
        auto issue_ssm = [&](int head_in_group, int channel0, int buffer) {
            const int head = group * MB_HEADS_PER_GROUP + head_in_group;
            const r9_state_t *src = st +
                (size_t)(head * MB_HEAD_DIM + channel0) * MB_STATE;
            const uint32_t dst = (uint32_t)__cvta_generic_to_shared(
                warp_ssm + (size_t)buffer * SSM_GROUP_BYTES + lane * 16);
            #pragma unroll
            for (int chunk = 0; chunk < SSM_GROUP_BYTES / 512; ++chunk)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                    "r"(dst + (uint32_t)chunk * 512),
                    "l"(reinterpret_cast<const char *>(src) +
                        (size_t)chunk * 512 + lane * 16) : "memory");
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        };
        auto issue_ssm_f32 = [&](int head_in_group, int channel0, int buffer) {
            const int head = group * MB_HEADS_PER_GROUP + head_in_group;
            const float *src = L.r8_initial_ssm +
                (size_t)(head * MB_HEAD_DIM + channel0) * MB_STATE;
            const uint32_t dst = (uint32_t)__cvta_generic_to_shared(
                warp_ssm + (size_t)buffer * SSM_F32_GROUP_BYTES + lane * 16);
            #pragma unroll
            for (int chunk = 0; chunk < SSM_F32_GROUP_BYTES / 512; ++chunk)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                    "r"(dst + (uint32_t)chunk * 512),
                    "l"(reinterpret_cast<const char *>(src) +
                        (size_t)chunk * 512 + lane * 16) : "memory");
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        };
        if constexpr (SSM_SMEM_OK) {
            if (!r8_initial) {
                #pragma unroll
                for (int k = 0; k < SSM_RING; ++k)
                    issue_ssm(k / SSM_GROUPS_PER_HEAD,
                              (k % SSM_GROUPS_PER_HEAD) * SSM_PF, k);
            } else if constexpr (SSM_F32_OK) {
                #pragma unroll
                for (int k = 0; k < SSM_F32_RING; ++k)
                    issue_ssm_f32(k / SSM_GROUPS_PER_HEAD,
                                  (k % SSM_GROUPS_PER_HEAD) * SSM_PF, k);
            }
        }
        float Braw[MB_STATE / 32];
        float Cv[MB_STATE / 32];
        const bf16 *xrow;
        if constexpr (FUSE_CONV) {
            bf16 *const cslot =
                L.conv_state + (size_t)slot_map[row] * L.conv_slot_stride;
            const bf16 *const zrow =
                zxbcdt + (size_t)row * MB_IN_PROJ + MB_INNER;
            static_assert(MB_CONV_K == 4, "the state record is three 32-bit words");
            static_assert(MB_GROUP_SIZE % 64 == 0 && MB_STATE % 64 == 0,
                          "one paired round covers 64 channels of a warp");
            auto wide_lo = [](unsigned int v) { return __uint_as_float(v << 16); };
            auto wide_hi = [](unsigned int v) { return __uint_as_float(v & 0xffff0000u); };
            struct ConvCursor {
                const uint2 *w; unsigned int *s;
                const unsigned int *z; const unsigned int *b;
            };
            auto conv_cursor = [&](int c) {
                const unsigned int cu = (unsigned int)c;
                return ConvCursor{
                    reinterpret_cast<const uint2 *>(
                        L.conv_w + cu * (unsigned int)MB_CONV_K),
                    reinterpret_cast<unsigned int *>(
                        cslot + cu * (unsigned int)(MB_CONV_K - 1)),
                    reinterpret_cast<const unsigned int *>(zrow + cu),
                    reinterpret_cast<const unsigned int *>(L.conv_b + cu)};
            };
            auto conv_two = [&](uint2 w0, uint2 w1, unsigned int s0,
                                unsigned int s1, unsigned int s2,
                                unsigned int zp, unsigned int bp, float &a0,
                                float &a1, unsigned int &n0, unsigned int &n1,
                                unsigned int &n2) {
                a0 = wide_lo(bp);
                a0 += wide_lo(w0.x) * wide_lo(s0);
                a0 += wide_hi(w0.x) * wide_hi(s0);
                a0 += wide_lo(w0.y) * wide_lo(s1);
                a0 += wide_hi(w0.y) * wide_lo(zp);
                a0 = a0 / (1.f + __expf(-a0));
                a1 = wide_hi(bp);
                a1 += wide_lo(w1.x) * wide_hi(s1);
                a1 += wide_hi(w1.x) * wide_lo(s2);
                a1 += wide_lo(w1.y) * wide_hi(s2);
                a1 += wide_hi(w1.y) * wide_hi(zp);
                a1 = a1 / (1.f + __expf(-a1));
                n0 = (s0 >> 16) | (s1 << 16);
                n1 = (zp & 0xffffu) | (s2 << 16);
                n2 = (s2 >> 16) | (zp & 0xffff0000u);
            };
            struct ConvPairIn {
                uint2 w0, w1;
                unsigned int s0, s1, s2, zp, bp;
            };
            auto conv_pair_fetch = [](const ConvCursor &q, int r) {
                const uint2 *const wp = q.w + 64 * r;
                const unsigned int *const sp = q.s + 96 * r;
                return ConvPairIn{wp[0], wp[1], sp[0], sp[1], sp[2],
                                  q.z[32 * r], q.b[32 * r]};
            };
            auto conv_pair_apply = [&](unsigned int *const sp,
                                       const ConvPairIn &v, float &a0,
                                       float &a1) {
                unsigned int n0, n1, n2;
                conv_two(v.w0, v.w1, v.s0, v.s1, v.s2, v.zp, v.bp, a0, a1,
                         n0, n1, n2);
                sp[0] = n0;
                sp[1] = n1;
                sp[2] = n2;
            };
            constexpr int QUAD_SPAN = 4 * 32;
            static_assert(MB_INNER % 4 == 0 && MB_STATE % 4 == 0 &&
                          MB_GROUP_SIZE % 4 == 0,
                          "every quad base is a four-channel boundary");
            static_assert(MB_STATE == QUAD_SPAN &&
                          MB_GROUP_SIZE == QUAD_SPAN + 2 * 32,
                          "one quad round covers B and C; x adds a pair round");
            struct ConvQuad {
                const uint4 *w; uint2 *s; const uint2 *z; const uint2 *b;
            };
            auto conv_quad_cursor = [&](int c) {
                const unsigned int cu = (unsigned int)c;
                return ConvQuad{
                    reinterpret_cast<const uint4 *>(
                        L.conv_w + cu * (unsigned int)MB_CONV_K),
                    reinterpret_cast<uint2 *>(
                        cslot + cu * (unsigned int)(MB_CONV_K - 1)),
                    reinterpret_cast<const uint2 *>(zrow + cu),
                    reinterpret_cast<const uint2 *>(L.conv_b + cu)};
            };
            struct ConvQuadIn {
                uint4 w01, w23;
                uint2 sa, sb, sc, zp, bp;
            };
            auto conv_quad_fetch = [](const ConvQuad &q) {
                const uint2 *const sp = q.s;
                return ConvQuadIn{q.w[0], q.w[1], sp[0], sp[1], sp[2],
                                  q.z[0], q.b[0]};
            };
            auto conv_quad_apply = [&](uint2 *const sp, const ConvQuadIn &v,
                                       float &a0, float &a1, float &a2,
                                       float &a3) {
                unsigned int n0, n1, n2, n3, n4, n5;
                conv_two(make_uint2(v.w01.x, v.w01.y),
                         make_uint2(v.w01.z, v.w01.w),
                         v.sa.x, v.sa.y, v.sb.x, v.zp.x, v.bp.x,
                         a0, a1, n0, n1, n2);
                conv_two(make_uint2(v.w23.x, v.w23.y),
                         make_uint2(v.w23.z, v.w23.w),
                         v.sb.y, v.sc.x, v.sc.y, v.zp.y, v.bp.y,
                         a2, a3, n3, n4, n5);
                sp[0] = make_uint2(n0, n1);
                sp[1] = make_uint2(n2, n3);
                sp[2] = make_uint2(n4, n5);
            };
            auto conv_pack = [](float a, float b) {
                return (unsigned int)__bfloat16_as_ushort(__float2bfloat16(a)) |
                    ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(b)) << 16);
            };
            const ConvQuad qb = conv_quad_cursor(
                MB_INNER + group * MB_STATE + 4 * lane);
            const ConvQuad qc = conv_quad_cursor(
                MB_INNER + MB_GROUPS * MB_STATE + group * MB_STATE + 4 * lane);
            static_assert(MB_GROUP_SIZE * (int)sizeof(bf16) % 8 == 0,
                          "each warp's staging row starts 8-byte aligned");
            const ConvQuad qx = conv_quad_cursor(
                group * MB_GROUP_SIZE + 4 * lane);
            const ConvCursor qt = conv_cursor(
                group * MB_GROUP_SIZE + QUAD_SPAN + 2 * lane);
            const ConvQuadIn vb = conv_quad_fetch(qb);
            const ConvQuadIn vc = conv_quad_fetch(qc);
            const ConvQuadIn vx = conv_quad_fetch(qx);
            const ConvPairIn vt = conv_pair_fetch(qt, 0);
            uint2 bpk, cpk;
            {
                float a0, a1, a2, a3;
                conv_quad_apply(qb.s, vb, a0, a1, a2, a3);
                bpk = make_uint2(conv_pack(a0, a1), conv_pack(a2, a3));
                conv_quad_apply(qc.s, vc, a0, a1, a2, a3);
                cpk = make_uint2(conv_pack(a0, a1), conv_pack(a2, a3));
            }
            {
                float a0, a1, a2, a3;
                conv_quad_apply(qx.s, vx, a0, a1, a2, a3);
                *reinterpret_cast<uint2 *>(&conv_x[warp][4 * lane]) =
                    make_uint2(conv_pack(a0, a1), conv_pack(a2, a3));
            }
            {
                float a0, a1;
                conv_pair_apply(qt.s, vt, a0, a1);
                conv_x[warp][QUAD_SPAN + 2 * lane] = __float2bfloat16(a0);
                conv_x[warp][QUAD_SPAN + 2 * lane + 1] = __float2bfloat16(a1);
            }
            const bool qword = ((lane >> 1) & 1) != 0;
            const bool qhalf = (lane & 1) != 0;
            #pragma unroll
            for (int j = 0; j < MB_STATE / 32; ++j) {
                const int src = (lane >> 2) + 8 * j;
                const unsigned int bx = __shfl_sync(0xffffffffu, bpk.x, src);
                const unsigned int by = __shfl_sync(0xffffffffu, bpk.y, src);
                const unsigned int cx = __shfl_sync(0xffffffffu, cpk.x, src);
                const unsigned int cy = __shfl_sync(0xffffffffu, cpk.y, src);
                const unsigned int bg = qword ? by : bx;
                const unsigned int cg = qword ? cy : cx;
                Braw[j] = qhalf ? wide_hi(bg) : wide_lo(bg);
                Cv[j] = qhalf ? wide_hi(cg) : wide_lo(cg);
            }
            __syncwarp();
            xrow = &conv_x[warp][0];
        } else {
            #pragma unroll
            for (int j = 0; j < MB_STATE / 32; ++j) {
                Braw[j] = __bfloat162float(Bsec[lane + j * 32]);
                Cv[j] = __bfloat162float(Csec[lane + j * 32]);
            }
            xrow = xa + (size_t)group * MB_GROUP_SIZE;
        }
        const int head_lane = group * MB_HEADS_PER_GROUP +
            (lane < MB_HEADS_PER_GROUP ? lane : 0);
        const float dt_lane = mamba_softplus(
            __bfloat162float(dtrow[head_lane]) + L.dt_bias[head_lane]);
        const float dA_lane = __expf(L.A[head_lane] * dt_lane);
        const float D_lane = L.D[head_lane];
        #pragma unroll 1
        for (int head_in_group = 0; head_in_group < MB_HEADS_PER_GROUP;
             ++head_in_group) {
        const int head = group * MB_HEADS_PER_GROUP + head_in_group;
        const float dt = __shfl_sync(0xffffffffu, dt_lane, head_in_group);
        const float dA = __shfl_sync(0xffffffffu, dA_lane, head_in_group);
        const float Dh = __shfl_sync(0xffffffffu, D_lane, head_in_group);
        const float *st_in = L.r8_initial_ssm +
                             (size_t)head * MB_HEAD_DIM * MB_STATE;
        float Bv[MB_STATE / 32];
        #pragma unroll
        for (int j = 0; j < MB_STATE / 32; ++j)
            Bv[j] = Braw[j] * dt;
        constexpr int NSTATE = SSM_NSTATE;
        constexpr int SSM_PREFETCH = SSM_PF;
        static_assert(channels % SSM_PREFETCH == 0);
        for (int g0 = 0; g0 < channels; g0 += SSM_PREFETCH) {
            float sv[SSM_PREFETCH][NSTATE];
            float xvr[SSM_PREFETCH];
            if (r8_initial && SSM_F32_OK) {
                asm volatile("cp.async.wait_group 1;\n" ::: "memory");
                __syncwarp();
                const int pair = head_in_group * SSM_GROUPS_PER_HEAD
                               + g0 / SSM_PREFETCH;
                const int buffer = pair & (SSM_F32_RING - 1);
                const float *const smf = reinterpret_cast<const float *>(
                    warp_ssm + (size_t)buffer * SSM_F32_GROUP_BYTES);
                static_assert(SSM_PREFETCH % 2 == 0 && MB_HEAD_DIM % 2 == 0,
                              "x staging vectorises in element pairs");
                const __nv_bfloat162 *const xq =
                    reinterpret_cast<const __nv_bfloat162 *>(
                        xrow + head_in_group * MB_HEAD_DIM +
                        first_channel + g0);
                #pragma unroll
                for (int gq = 0; gq < SSM_PREFETCH / 2; ++gq) {
                    const float2 xp = __bfloat1622float2(xq[gq]);
                    xvr[2 * gq] = xp.x;
                    xvr[2 * gq + 1] = xp.y;
                }
                #pragma unroll
                for (int gi = 0; gi < SSM_PREFETCH; ++gi) {
                    #pragma unroll
                    for (int j = 0; j < NSTATE; ++j)
                        sv[gi][j] = smf[(size_t)gi * MB_STATE + lane + j * 32];
                }
                __syncwarp();
                const int ahead = pair + SSM_F32_RING;
                if (ahead < SSM_PAIRS)
                    issue_ssm_f32(ahead / SSM_GROUPS_PER_HEAD,
                                  (ahead % SSM_GROUPS_PER_HEAD) * SSM_PF,
                                  buffer);
                else
                    asm volatile("cp.async.commit_group;\n" ::: "memory");
            } else if (r8_initial) {
                #pragma unroll
                for (int gi = 0; gi < SSM_PREFETCH; ++gi) {
                    const size_t prow = (size_t)(first_channel + g0 + gi) * MB_STATE;
                    xvr[gi] = __bfloat162float(
                        xrow[head_in_group * MB_HEAD_DIM +
                             first_channel + g0 + gi]);
                    #pragma unroll
                    for (int j = 0; j < NSTATE; ++j)
                        sv[gi][j] = st_in[prow + lane + j * 32];
                }
            } else if constexpr (SSM_SMEM_OK) {
                asm volatile("cp.async.wait_group 3;\n" ::: "memory");
                __syncwarp();
                const int pair = head_in_group * SSM_GROUPS_PER_HEAD
                               + g0 / SSM_PREFETCH;
                const int buffer = pair & (SSM_RING - 1);
                const r9_state_t *sm = reinterpret_cast<const r9_state_t *>(
                    warp_ssm + (size_t)buffer * SSM_GROUP_BYTES);
                static_assert(SSM_PREFETCH % 2 == 0 && MB_HEAD_DIM % 2 == 0,
                              "x staging vectorises in element pairs");
                const __nv_bfloat162 *const xq =
                    reinterpret_cast<const __nv_bfloat162 *>(
                        xrow + head_in_group * MB_HEAD_DIM +
                        first_channel + g0);
                #pragma unroll
                for (int gq = 0; gq < SSM_PREFETCH / 2; ++gq) {
                    const float2 xp = __bfloat1622float2(xq[gq]);
                    xvr[2 * gq] = xp.x;
                    xvr[2 * gq + 1] = xp.y;
                }
                #pragma unroll
                for (int gi = 0; gi < SSM_PREFETCH; ++gi) {
                    #pragma unroll
                    for (int j = 0; j < NSTATE; ++j)
                        sv[gi][j] = r9_load(
                            sm[(size_t)gi * MB_STATE + lane + j * 32], 1.f);
                }
                __syncwarp();
                const int ahead = pair + SSM_RING;
                if (ahead < SSM_PAIRS)
                    issue_ssm(ahead / SSM_GROUPS_PER_HEAD,
                              (ahead % SSM_GROUPS_PER_HEAD) * SSM_PF, buffer);
                else
                    asm volatile("cp.async.commit_group;\n" ::: "memory");
            } else {
                #pragma unroll
                for (int gi = 0; gi < SSM_PREFETCH; ++gi) {
                    xvr[gi] = __bfloat162float(
                        xrow[head_in_group * MB_HEAD_DIM +
                             first_channel + g0 + gi]);
                    #pragma unroll
                    for (int j = 0; j < NSTATE; ++j)
                    {
                        int p = first_channel+g0+gi;
                        int mapped = R9_STATE_MODE==4 ? L.r9_map[head*24+p] : head*24+p;
                        int n = lane+j*32;
                        float scale = R9_STATE_MODE==1 ? 1.f : L.r9_scales[r9_scale_index(mapped/24,mapped%24,n)];
                        sv[gi][j] = r9_load(st[(size_t)mapped*128+n],scale);
                    }
                }
            }
            if constexpr (SSM_PREFETCH == 12) {
                float v6[6];
                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    float pair_acc[2];
                    #pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int gi = k + 6 * half;
                        const int p = first_channel + g0 + gi;
                        const float xv = xvr[gi];
                        float acc = 0.f;
                        #pragma unroll
                        for (int j = 0; j < NSTATE; ++j) {
                            const int n = lane + j * 32;
                            const float s = sv[gi][j] * dA + Bv[j] * xv;
                            const int mapped = R9_STATE_MODE == 4
                                ? L.r9_map[head * 24 + p] : head * 24 + p;
                            const float scale = R9_STATE_MODE == 1 ? 1.f
                                : L.r9_scales[r9_scale_index(
                                      mapped / 24, mapped % 24, n)];
                            st[(size_t)mapped * 128 + n] = r9_store(s, scale);
                            acc += s * Cv[j];
                        }
                        pair_acc[half] = acc;
                    }
                    const float mine = sb4 ? pair_acc[1] : pair_acc[0];
                    const float away = sb4 ? pair_acc[0] : pair_acc[1];
                    v6[k] = mine + __shfl_xor_sync(0xffffffff, away, 16);
                }
                float v4[4];
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    const float mine = sb3 ? v6[3 + k] : v6[k];
                    const float away = sb3 ? v6[k] : v6[3 + k];
                    v4[k] = mine + __shfl_xor_sync(0xffffffff, away, 8);
                }
                v4[3] = 0.f;
                float v2[2];
                #pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const float mine = sb2 ? v4[2 + k] : v4[k];
                    const float away = sb2 ? v4[k] : v4[2 + k];
                    v2[k] = mine + __shfl_xor_sync(0xffffffff, away, 4);
                }
                float acc = (sb1 ? v2[1] : v2[0]) +
                    __shfl_xor_sync(0xffffffff, sb1 ? v2[0] : v2[1], 2);
                acc += __shfl_xor_sync(0xffffffff, acc, 1);
                if (ssm_owner) {
                    const int p = first_channel + g0 + ssm_own_ch;
                    const float xv = __bfloat162float(
                        xrow[head_in_group * MB_HEAD_DIM + p]);
                    const bf16 outv = __float2bfloat16(acc + xv * Dh);
                    if constexpr (FUSE_GNORM)
                        gn_stage[warp][head_in_group * MB_HEAD_DIM + p] = outv;
                    else
                        gated[(size_t)row * MB_INNER + head * MB_HEAD_DIM + p] = outv;
                }
            } else {
                float cacc[SSM_PREFETCH];
                #pragma unroll
                for (int gi = 0; gi < SSM_PREFETCH; ++gi) {
                    const int p = first_channel + g0 + gi;
                    const float xv = xvr[gi];
                    float acc = 0.f;
                    #pragma unroll
                    for (int j = 0; j < NSTATE; ++j) {
                        const int n = lane + j * 32;
                        const float s = sv[gi][j] * dA + Bv[j] * xv;
                        const int mapped = R9_STATE_MODE == 4
                            ? L.r9_map[head * 24 + p] : head * 24 + p;
                        const float scale = R9_STATE_MODE == 1 ? 1.f
                            : L.r9_scales[r9_scale_index(
                                  mapped / 24, mapped % 24, n)];
                        st[(size_t)mapped * 128 + n] = r9_store(s, scale);
                        acc += s * Cv[j];
                    }
                    cacc[gi] = acc;
                }
                #pragma unroll
                for (int gi = 0; gi < SSM_PREFETCH; ++gi) {
                    const int p = first_channel + g0 + gi;
                    float acc = cacc[gi];
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1)
                        acc += __shfl_xor_sync(0xffffffff, acc, off);
                    if (lane == 0) {
                        const bf16 outv = __float2bfloat16(acc + xvr[gi] * Dh);
                        if constexpr (FUSE_GNORM)
                            gn_stage[warp][head_in_group * MB_HEAD_DIM + p] = outv;
                        else
                            gated[(size_t)row * MB_INNER + head * MB_HEAD_DIM + p] = outv;
                    }
                }
            }
        }
        }
        if constexpr (SSM_SMEM_OK) {
            if (!r8_initial || SSM_F32_OK) {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                __syncwarp();
            }
        }
        if constexpr (FUSE_GNORM) {
            __syncwarp();
            const bf16 *zrow = zxbcdt + (size_t)row * MB_IN_PROJ;
            bf16 *grow = gated + (size_t)row * MB_INNER;
            float sq = 0.f;
            float held[MB_GROUP_SIZE / 32], wg[MB_GROUP_SIZE / 32];
            #pragma unroll
            for (int j = 0; j < MB_GROUP_SIZE / 32; ++j) {
                const int idx = group * MB_GROUP_SIZE + lane + j * 32;
                const float z = __bfloat162float(zrow[idx]);
                const float gv = __bfloat162float(gn_stage[warp][lane + j * 32]) *
                                 (z / (1.f + __expf(-z)));
                wg[j] = __bfloat162float(L.gnorm_w[idx]);
                held[j] = gv;
                sq += gv * gv;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                sq += __shfl_xor_sync(0xffffffff, sq, off);
            const float inv = rsqrtf(sq / MB_GROUP_SIZE + RMS_EPS);
            #pragma unroll
            for (int j = 0; j < MB_GROUP_SIZE / 32; ++j) {
                const int idx = group * MB_GROUP_SIZE + lane + j * 32;
                grow[idx] = __float2bfloat16(held[j] * inv * wg[j]);
            }
            __syncwarp();
        }
    }
}
__device__ void mamba_conv_prefill(
    const MambaLayer &L, const bf16 *zxbcdt, bf16 *xbc_act, int n_tokens,
    int slot, int n_seq = 1, const int *seq_slot = nullptr) {
    const int total = n_tokens * MB_CONV_DIM;
    for (int idx = blockIdx.x * THREADS + threadIdx.x; idx < total;
         idx += gridDim.x * THREADS) {
        const int t = idx / MB_CONV_DIM;
        const int c = idx - t * MB_CONV_DIM;
        const int pos = t / n_seq;
        const int seq = t - pos * n_seq;
        const int my_slot = seq_slot ? seq_slot[seq] : slot;
        const bf16 *w = L.conv_w + (size_t)c * MB_CONV_K;
        const bf16 *state = L.conv_state + (size_t)my_slot * L.conv_slot_stride +
                            (size_t)c * (MB_CONV_K - 1);
        float acc = __bfloat162float(L.conv_b[c]);
        #pragma unroll
        for (int k = 0; k < MB_CONV_K; ++k) {
            const int back = (MB_CONV_K - 1) - k;
            const int src = (pos - back >= 0) ? t - back * n_seq : -1;
            float v;
#ifdef PF_CONV_ZERO_STATE
            if (src < 0) v = 0.f;
#else
            if (src < 0) v = __bfloat162float(state[pos + k]);
#endif
            else v = __bfloat162float(
                zxbcdt[(size_t)src * MB_IN_PROJ + MB_INNER + c]);
            acc += __bfloat162float(w[k]) * v;
        }
#ifndef PF_CONV_NO_SILU
        acc = acc / (1.f + __expf(-acc));
#endif
        xbc_act[(size_t)t * MB_CONV_DIM + c] = __float2bfloat16(acc);
    }
}
__device__ void mamba_conv_prefill_state(
    const MambaLayer &L, const bf16 *zxbcdt, int n_tokens, int slot,
    int n_seq = 1, const int *seq_slot = nullptr) {
    if (n_seq > 1) {
        const int n_pos = n_tokens / n_seq;
        for (int idx = blockIdx.x * THREADS + threadIdx.x;
             idx < n_seq * MB_CONV_DIM; idx += gridDim.x * THREADS) {
            const int seq = idx / MB_CONV_DIM;
            const int c = idx - seq * MB_CONV_DIM;
            const int my_slot = seq_slot ? seq_slot[seq] : slot;
            bf16 *state = L.conv_state + (size_t)my_slot * L.conv_slot_stride +
                          (size_t)c * (MB_CONV_K - 1);
            bf16 old[MB_CONV_K - 1];
            #pragma unroll
            for (int k = 0; k < MB_CONV_K - 1; ++k) old[k] = state[k];
            #pragma unroll
            for (int k = 0; k < MB_CONV_K - 1; ++k) {
                const int p = n_pos - (MB_CONV_K - 1) + k;
                state[k] = p >= 0
                    ? zxbcdt[((size_t)p * n_seq + seq) * MB_IN_PROJ + MB_INNER + c]
                    : old[k + n_pos];
            }
        }
        return;
    }
    for (int c = blockIdx.x * THREADS + threadIdx.x; c < MB_CONV_DIM;
         c += gridDim.x * THREADS) {
        bf16 *state = L.conv_state + (size_t)slot * L.conv_slot_stride +
                      (size_t)c * (MB_CONV_K - 1);
        bf16 old[MB_CONV_K - 1];
        #pragma unroll
        for (int k = 0; k < MB_CONV_K - 1; ++k) old[k] = state[k];
        #pragma unroll
        for (int k = 0; k < MB_CONV_K - 1; ++k) {
            const int src = n_tokens - (MB_CONV_K - 1) + k;
#ifdef PF_CONV_STATE_NO_SHIFT
            if (src >= 0)
                state[k] = zxbcdt[(size_t)src * MB_IN_PROJ + MB_INNER + c];
#else
#ifdef PF_CONV_STATE_NO_SHIFT
            if (src >= 0)
                state[k] = zxbcdt[(size_t)src * MB_IN_PROJ + MB_INNER + c];
#else
#ifdef PF_CONV_STATE_NO_SHIFT
            if (src >= 0)
                state[k] = zxbcdt[(size_t)src * MB_IN_PROJ + MB_INNER + c];
#else
            state[k] = src >= 0
                           ? zxbcdt[(size_t)src * MB_IN_PROJ + MB_INNER + c]
                           : old[k + n_tokens];
#endif
#endif
#endif
        }
    }
}
__device__ void mamba_ssm_prefill(
    const MambaLayer &L, const bf16 *zxbcdt, const bf16 *xbc_act, bf16 *gated,
    int n_tokens, int slot, int n_seq = 1, const int *seq_slot = nullptr) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int warps_total = gridDim.x * WARPS;
    if (n_seq > 1) {
        const int n_pos = n_tokens / n_seq;
        const int units = n_seq * MB_HEADS * MB_HEAD_DIM;
        for (int unit = blockIdx.x * WARPS + warp; unit < units;
             unit += warps_total) {
            const int seq = unit / (MB_HEADS * MB_HEAD_DIM);
            const int rem = unit - seq * (MB_HEADS * MB_HEAD_DIM);
            const int head = rem / MB_HEAD_DIM;
            const int p = rem - head * MB_HEAD_DIM;
            const int group = head / MB_HEADS_PER_GROUP;
            const int my_slot = seq_slot ? seq_slot[seq] : slot;
            const float Ah = L.A[head], Dh = L.D[head], dtb = L.dt_bias[head];
            float *const st = L.ssm_state + (size_t)my_slot * L.ssm_slot_stride +
                              (size_t)head * MB_HEAD_DIM * MB_STATE +
                              (size_t)p * MB_STATE;
            float sreg[MB_STATE / 32];
            #pragma unroll
            for (int j = 0; j < MB_STATE / 32; ++j) sreg[j] = st[lane + j * 32];
            for (int pos = 0; pos < n_pos; ++pos) {
                const size_t t = (size_t)pos * n_seq + seq;
                const bf16 *xa = xbc_act + t * MB_CONV_DIM;
                const bf16 *Bsec = xa + MB_INNER + group * MB_STATE;
                const bf16 *Csec = xa + MB_INNER + MB_GROUPS * MB_STATE +
                                   group * MB_STATE;
                const bf16 *dtrow = zxbcdt + t * MB_IN_PROJ + MB_INNER + MB_CONV_DIM;
                const float dt = mamba_softplus(
                    __bfloat162float(dtrow[head]) + dtb);
                const float dA = __expf(Ah * dt);
                const float xv = __bfloat162float(xa[head * MB_HEAD_DIM + p]);
                float acc = 0.f;
                #pragma unroll
                for (int j = 0; j < MB_STATE / 32; ++j) {
                    const int n = lane + j * 32;
                    sreg[j] = sreg[j] * dA + __bfloat162float(Bsec[n]) * dt * xv;
                    acc += sreg[j] * __bfloat162float(Csec[n]);
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    acc += __shfl_xor_sync(0xffffffff, acc, off);
                if (lane == 0)
                    gated[t * MB_INNER + head * MB_HEAD_DIM + p] =
                        __float2bfloat16(acc + xv * Dh);
            }
            #pragma unroll
            for (int j = 0; j < MB_STATE / 32; ++j) st[lane + j * 32] = sreg[j];
        }
        return;
    }
#if PF_SSM_SPLIT_P
    const int units = MB_HEADS * MB_HEAD_DIM;
    for (int unit = blockIdx.x * WARPS + warp; unit < units; unit += warps_total) {
        const int head = unit / MB_HEAD_DIM;
        const int p_only = unit - head * MB_HEAD_DIM;
        const int group = head / MB_HEADS_PER_GROUP;
#else
    for (int head = blockIdx.x * WARPS + warp; head < MB_HEADS;
         head += warps_total) {
        const int group = head / MB_HEADS_PER_GROUP;
#endif
        const float Ah = L.A[head], Dh = L.D[head], dtb = L.dt_bias[head];
        float *st = L.ssm_state + (size_t)slot * L.ssm_slot_stride +
                    (size_t)head * MB_HEAD_DIM * MB_STATE
#if PF_SSM_SPLIT_P
                    + (size_t)p_only * MB_STATE
#endif
            ;
        for (int t = 0; t < n_tokens; ++t) {
            const bf16 *xa = xbc_act + (size_t)t * MB_CONV_DIM;
            const bf16 *Bsec = xa + MB_INNER + group * MB_STATE;
            const bf16 *Csec = xa + MB_INNER + MB_GROUPS * MB_STATE + group * MB_STATE;
            const bf16 *dtrow =
                zxbcdt + (size_t)t * MB_IN_PROJ + MB_INNER + MB_CONV_DIM;
            const float dt = mamba_softplus(
                __bfloat162float(dtrow[head]) + dtb);
            const float dA = __expf(Ah * dt);
            float Bv[MB_STATE / 32], Cv[MB_STATE / 32];
            #pragma unroll
            for (int j = 0; j < MB_STATE / 32; ++j) {
                Bv[j] = __bfloat162float(Bsec[lane + j * 32]) * dt;
                Cv[j] = __bfloat162float(Csec[lane + j * 32]);
            }
#if PF_SSM_SPLIT_P
            {
                const int p = p_only;
#else
            for (int p = 0; p < MB_HEAD_DIM; ++p) {
#endif
                const float xv = __bfloat162float(xa[head * MB_HEAD_DIM + p]);
                float acc = 0.f;
                #pragma unroll
                for (int j = 0; j < MB_STATE / 32; ++j) {
#if PF_SSM_SPLIT_P
                    const size_t at = (size_t)lane + j * 32;
#else
                    const size_t at = (size_t)p * MB_STATE + lane + j * 32;
#endif
                    const float s = st[at] * dA + Bv[j] * xv;
                    st[at] = s;
                    acc += s * Cv[j];
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    acc += __shfl_xor_sync(0xffffffff, acc, off);
                if (lane == 0)
                    gated[(size_t)t * MB_INNER + head * MB_HEAD_DIM + p] =
                        __float2bfloat16(acc + xv * Dh);
            }
        }
    }
}
__device__ void mamba_gated_norm(
    const MambaLayer &L, const bf16 *zxbcdt, bf16 *gated, int n_rows = BATCH) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int warps_total = gridDim.x * WARPS;
    const int pairs = n_rows * MB_GROUPS;
    for (int pair = blockIdx.x * WARPS + warp; pair < pairs; pair += warps_total) {
        const int row = pair / MB_GROUPS;
        const int group = pair - row * MB_GROUPS;
        const bf16 *zrow = zxbcdt + (size_t)row * MB_IN_PROJ;
        bf16 *grow = gated + (size_t)row * MB_INNER;
        float sq = 0.f;
        float held[MB_GROUP_SIZE / 32], wg[MB_GROUP_SIZE / 32];
        #pragma unroll
        for (int j = 0; j < MB_GROUP_SIZE / 32; ++j) {
            const int idx = group * MB_GROUP_SIZE + lane + j * 32;
            const float z = __bfloat162float(zrow[idx]);
            const float g = __bfloat162float(grow[idx]) * (z / (1.f + __expf(-z)));
            wg[j] = __bfloat162float(L.gnorm_w[idx]);
            held[j] = g;
            sq += g * g;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sq += __shfl_xor_sync(0xffffffff, sq, off);
        const float inv = rsqrtf(sq / MB_GROUP_SIZE + RMS_EPS);
        #pragma unroll
        for (int j = 0; j < MB_GROUP_SIZE / 32; ++j) {
            const int idx = group * MB_GROUP_SIZE + lane + j * 32;
            grow[idx] = __float2bfloat16(held[j] * inv * wg[j]);
        }
    }
}
__device__ void bb_attention(
    const AttnLayer &L, const bf16 *qkv, bf16 *out, int past_len_scalar,
    const int *seq_len MK2_PROGRESS_PARAMETER(progress_mask)) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int Q_PER_WARP = 3;
    constexpr int GROUPS_PER_KV = 1;
    const int pairs = BATCH * BB_KV_HEADS * GROUPS_PER_KV;
    const float scale = rsqrtf((float)BB_HEAD_DIM);
    const int kv_per_q = BB_ATT_HEADS / BB_KV_HEADS;
    for (int pair = blockIdx.x + warp * gridDim.x; pair < pairs;
         pair += gridDim.x * WARPS) {
        const int row = pair / (BB_KV_HEADS * GROUPS_PER_KV);
        MK2_PROGRESS_GUARD(progress_mask, row);
        const int within_row = pair % (BB_KV_HEADS * GROUPS_PER_KV);
        const int kv = within_row / GROUPS_PER_KV;
        const int q_group = within_row % GROUPS_PER_KV;
        const int q_first = q_group * Q_PER_WARP;
        const int q_count = q_group == 0 ? Q_PER_WARP : 1;
        const int past_len = seq_len ? seq_len[row] : past_len_scalar;
        const int T = past_len + 1;
        const bf16 *base = qkv + (size_t)row * BB_QKV;
        const bf16 *k_new = base + BB_Q_SIZE + kv * BB_HEAD_DIM;
        const bf16 *v_new = base + BB_Q_SIZE + BB_KV_SIZE + kv * BB_HEAD_DIM;
#if !BB_ATTN_SKIP_KV_WRITE
        if (q_group == 0) {
            bf16 *kc = kv_slot(L, row, 0, past_len) + kv_offset(L, past_len, kv);
            bf16 *vc = kv_slot(L, row, 1, past_len) + kv_offset(L, past_len, kv);
            for (int i = lane; i < BB_HEAD_DIM; i += 32) {
                kc[i] = k_new[i];
                vc[i] = v_new[i];
            }
        }
#endif
        __syncwarp();
        constexpr int PER_LANE = BB_HEAD_DIM / 32;
        float acc[Q_PER_WARP][PER_LANE];
        #pragma unroll
        for (int h = 0; h < Q_PER_WARP; ++h) {
            #pragma unroll
            for (int j = 0; j < PER_LANE; ++j) acc[h][j] = 0.f;
        }
        float m[Q_PER_WARP], denom[Q_PER_WARP];
        #pragma unroll
        for (int h = 0; h < Q_PER_WARP; ++h) {
            m[h] = -CUDART_INF_F;
            denom[h] = 0.f;
        }
        constexpr int PF = 8;
        const int tpb = L.tokens_per_block;
        const size_t elems_per_block =
            (size_t)tpb * BB_KV_HEADS * BB_HEAD_DIM;
        const int *bt_k = L.block_table + (size_t)(row * 2) * L.max_blocks_per_seq;
        const int *bt_v = bt_k + L.max_blocks_per_seq;
        const int head_base = kv * tpb * BB_HEAD_DIM;
        float qr[Q_PER_WARP][PER_LANE];
        #pragma unroll
        for (int h = 0; h < Q_PER_WARP; ++h) {
            if (h >= q_count) break;
            const bf16 *q = base + (kv * kv_per_q + q_first + h) * BB_HEAD_DIM;
            #pragma unroll
            for (int j = 0; j < PER_LANE; ++j)
                qr[h][j] = __bfloat162float(q[lane + j * 32]);
        }
        for (int page0 = 0; page0 < T; page0 += tpb) {
            const int k_page = bt_k[page0 / tpb] & 0x7FFFFFFF;
            const int v_page = bt_v[page0 / tpb] & 0x7FFFFFFF;
            const bf16 *kbase = L.pool_base +
                (size_t)k_page * elems_per_block +
                head_base;
            const bf16 *vbase = L.pool_base +
                (size_t)v_page * elems_per_block +
                head_base;
            const int tend = page0 + tpb < T ? page0 + tpb : T;
            for (int t0 = page0; t0 < tend; t0 += PF) {
                const int navail = tend - t0 < PF ? tend - t0 : PF;
                float kr[PF][PER_LANE], vr[PF][PER_LANE];
                #pragma unroll
                for (int u = 0; u < PF; ++u) {
                    const int tpos = t0 + (u < navail ? u : navail - 1);
#if BB_ATTN_LEGACY_CUR_KV
                    const bf16 *kt = kbase + (tpos - page0) * BB_HEAD_DIM;
                    const bf16 *vt = vbase + (tpos - page0) * BB_HEAD_DIM;
#else
                    const bool cur_token = tpos == past_len;
                    const bf16 *kt = cur_token
                        ? k_new : kbase + (tpos - page0) * BB_HEAD_DIM;
                    const bf16 *vt = cur_token
                        ? v_new : vbase + (tpos - page0) * BB_HEAD_DIM;
#endif
                    #pragma unroll
                    for (int j = 0; j < PER_LANE; ++j) {
                        const int i = lane + j * 32;
                        kr[u][j] = __bfloat162float(kt[i]);
                        vr[u][j] = __bfloat162float(vt[i]);
                    }
                }
                #pragma unroll
                for (int u = 0; u < PF; ++u) {
                    if (u >= navail) break;
                    #pragma unroll
                    for (int h = 0; h < Q_PER_WARP; ++h) {
                        if (h >= q_count) break;
                        float dot = 0.f;
                        #pragma unroll
                        for (int j = 0; j < PER_LANE; ++j)
                            dot += qr[h][j] * kr[u][j];
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1)
                            dot += __shfl_xor_sync(0xffffffff, dot, off);
                        const float s = dot * scale;
                        const float m_new = fmaxf(m[h], s);
                        const float rescale = __expf(m[h] - m_new);
                        const float w = __expf(s - m_new);
                        denom[h] = denom[h] * rescale + w;
                        #pragma unroll
                        for (int j = 0; j < PER_LANE; ++j)
                            acc[h][j] = acc[h][j] * rescale + w * vr[u][j];
                        m[h] = m_new;
                    }
                }
            }
        }
        #pragma unroll
        for (int h = 0; h < Q_PER_WARP; ++h) {
            if (h >= q_count) break;
            const float inv = 1.f / denom[h];
            const int head = kv * kv_per_q + q_first + h;
            #pragma unroll
            for (int j = 0; j < PER_LANE; ++j) {
                const int i = lane + j * 32;
                out[(size_t)row * D_MODEL + head * BB_HEAD_DIM + i] =
                    __float2bfloat16(acc[h][j] * inv);
            }
        }
    }
}
__device__ void bb_attention_prefill_kv(
    const AttnLayer &L, const bf16 *qkv, int n_tokens, int seq_base, int tok0,
    int n_seq = 1, const int *seq_slot = nullptr) {
    const int total = n_tokens * BB_KV_HEADS;
    for (int idx = blockIdx.x * THREADS + threadIdx.x; idx < total;
         idx += gridDim.x * THREADS) {
        const int t = idx / BB_KV_HEADS;
        const int kv = idx - t * BB_KV_HEADS;
        const bf16 *base = qkv + (size_t)t * BB_QKV;
        const bf16 *k_new = base + BB_Q_SIZE + kv * BB_HEAD_DIM;
        const bf16 *v_new = base + BB_Q_SIZE + BB_KV_SIZE + kv * BB_HEAD_DIM;
        const int pos = n_seq > 1 ? t / n_seq : tok0 + t;
        const int seq = n_seq > 1
            ? (seq_slot ? seq_slot[t - (t / n_seq) * n_seq] : t % n_seq)
            : seq_base;
        bf16 *kc = kv_slot(L, seq, 0, pos) + kv_offset(L, pos, kv);
        bf16 *vc = kv_slot(L, seq, 1, pos) + kv_offset(L, pos, kv);
        for (int i = 0; i < BB_HEAD_DIM; ++i) {
            kc[i] = k_new[i];
            vc[i] = v_new[i];
        }
    }
}
__device__ void bb_attention_prefill(
    const AttnLayer &L, const bf16 *qkv, bf16 *out, int n_tokens, int seq_base,
    int tok0, int n_seq = 1, const int *seq_slot = nullptr) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int warps_total = gridDim.x * WARPS;
    const int pairs = n_tokens * BB_ATT_HEADS;
    const float scale = rsqrtf((float)BB_HEAD_DIM);
    const int kv_per_q = BB_ATT_HEADS / BB_KV_HEADS;
    constexpr int PER_LANE = BB_HEAD_DIM / 32;
    for (int pair = blockIdx.x * WARPS + warp; pair < pairs; pair += warps_total) {
        const int t = pair / BB_ATT_HEADS;
        const int head = pair - t * BB_ATT_HEADS;
        const int kv = head / kv_per_q;
        const bf16 *q = qkv + (size_t)t * BB_QKV + head * BB_HEAD_DIM;
        float acc[PER_LANE];
        #pragma unroll
        for (int j = 0; j < PER_LANE; ++j) acc[j] = 0.f;
        float m = -CUDART_INF_F, denom = 0.f;
        const int pos = n_seq > 1 ? t / n_seq : tok0 + t;
        const int seq = n_seq > 1
            ? (seq_slot ? seq_slot[t - (t / n_seq) * n_seq] : t % n_seq)
            : seq_base;
        for (int s = 0; s <= pos; ++s) {
            const bf16 *kt = kv_slot(L, seq, 0, s) + kv_offset(L, s, kv);
            float dot = 0.f;
            #pragma unroll
            for (int j = 0; j < PER_LANE; ++j) {
                const int i = lane + j * 32;
                dot += __bfloat162float(q[i]) * __bfloat162float(kt[i]);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, off);
            const float sc = dot * scale;
            const float m_new = fmaxf(m, sc);
            const float rescale = __expf(m - m_new);
            const float wt = __expf(sc - m_new);
            denom = denom * rescale + wt;
            const bf16 *vt = kv_slot(L, seq, 1, s) + kv_offset(L, s, kv);
            #pragma unroll
            for (int j = 0; j < PER_LANE; ++j) {
                const int i = lane + j * 32;
                acc[j] = acc[j] * rescale + wt * __bfloat162float(vt[i]);
            }
            m = m_new;
        }
        const float inv = 1.f / denom;
        #pragma unroll
        for (int j = 0; j < PER_LANE; ++j) {
            const int i = lane + j * 32;
            out[(size_t)t * D_MODEL + head * BB_HEAD_DIM + i] =
                __float2bfloat16(acc[j] * inv);
        }
    }
}
__device__ void bb_moe_route(
    const bf16 *normed, const bf16 *gate_w, const bf16 *gate_bias, float *moe_w,
    int n_rows = BATCH, bf16 *row_stage = nullptr) {
    __shared__ float score[BB_EXPERTS];
    constexpr int CH = D_MODEL / 8;
    for (int row = blockIdx.x; row < n_rows; row += gridDim.x) {
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        const char *gsrc = reinterpret_cast<const char *>(
            normed + (size_t)row * D_MODEL);
        char *gdst = reinterpret_cast<char *>(row_stage);
        const uint32_t act_s = (uint32_t)__cvta_generic_to_shared(gdst);
        for (int i = threadIdx.x; i < CH; i += THREADS) {
            const uint32_t dst = act_s + (uint32_t)i * 16;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(dst), "l"(gsrc + (size_t)i * 16) : "memory");
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        if ((int)threadIdx.x < BB_EXPERTS)
            moe_w[(size_t)row * BB_EXPERTS + threadIdx.x] = 0.f;
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f;
        const bf16 *w0 = gate_w + (size_t)warp * D_MODEL;
        const bf16 *w1 = gate_w + (size_t)(warp + WARPS) * D_MODEL;
        const bf16 *w2 = gate_w + (size_t)(warp + 2 * WARPS) * D_MODEL;
        for (int i = lane; i < D_MODEL; i += 128) {
            bf16 sx0, sx1;
            move<bf16>::lds(sx0, act_s + (uint32_t)i * sizeof(bf16));
            move<bf16>::lds(sx1, act_s + (uint32_t)(i + 32) * sizeof(bf16));
            float x0 = __bfloat162float(sx0);
            float x1 = __bfloat162float(sx1);
            acc0 += x0 * __bfloat162float(w0[i]);
            acc1 += x0 * __bfloat162float(w1[i]);
            acc2 += x0 * __bfloat162float(w2[i]);
            acc0 += x1 * __bfloat162float(w0[i + 32]);
            acc1 += x1 * __bfloat162float(w1[i + 32]);
            acc2 += x1 * __bfloat162float(w2[i + 32]);
            move<bf16>::lds(sx0, act_s + (uint32_t)(i + 64) * sizeof(bf16));
            move<bf16>::lds(sx1, act_s + (uint32_t)(i + 96) * sizeof(bf16));
            x0 = __bfloat162float(sx0);
            x1 = __bfloat162float(sx1);
            acc0 += x0 * __bfloat162float(w0[i + 64]);
            acc1 += x0 * __bfloat162float(w1[i + 64]);
            acc2 += x0 * __bfloat162float(w2[i + 64]);
            acc0 += x1 * __bfloat162float(w0[i + 96]);
            acc1 += x1 * __bfloat162float(w1[i + 96]);
            acc2 += x1 * __bfloat162float(w2[i + 96]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, off);
            acc1 += __shfl_xor_sync(0xffffffffu, acc1, off);
            acc2 += __shfl_xor_sync(0xffffffffu, acc2, off);
        }
        if (lane == 0) {
            score[warp] = 1.f / (1.f + __expf(-acc0));
            score[warp + WARPS] = 1.f / (1.f + __expf(-acc1));
            score[warp + 2 * WARPS] = 1.f / (1.f + __expf(-acc2));
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            float remaining = (lane < BB_EXPERTS)
                                  ? score[lane] + __bfloat162float(gate_bias[lane])
                                  : -CUDART_INF_F;
            int pick[BB_TOP_K];
            float total = 0.f;
            #pragma unroll
            for (int k = 0; k < BB_TOP_K; ++k) {
                float bv = remaining;
                int bi = lane;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    const float ov = __shfl_xor_sync(0xffffffff, bv, off);
                    const int oi = __shfl_xor_sync(0xffffffff, bi, off);
                    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                }
                pick[k] = bi;
                total += score[bi];
                if (lane == bi) remaining = -CUDART_INF_F;
            }
            const float norm = BB_ROUTED_SCALE / (total + 1e-20f);
            #pragma unroll
            for (int k = 0; k < BB_TOP_K; ++k)
                if (lane == pick[k]) moe_w[row * BB_EXPERTS + lane] = score[lane] * norm;
        }
        __syncthreads();
    }
}
__device__ __forceinline__ bf16 bb_moe_fold_deterministic(const float *parts, int idx) {
    const size_t base = (size_t)(idx / D_MODEL) * (BB_EXPERTS + 1) * D_MODEL + idx % D_MODEL;
    float value = 0.f;
    #pragma unroll
    for (int expert = 0; expert <= BB_EXPERTS; ++expert)
        value = __fadd_rn(value, parts[base + (size_t)expert * D_MODEL]);
    return __float2bfloat16(value);
}
__device__ __forceinline__ bf16 bb_moe_fold_selected(
        const float *parts, const float *weights, int idx) {
    const int row = idx / D_MODEL;
    const size_t base = (size_t)row * (BB_EXPERTS + 1) * D_MODEL + idx % D_MODEL;
    float value = 0.f;
    #pragma unroll
    for (int expert = 0; expert < BB_EXPERTS; ++expert) {
        const float part = weights[row * BB_EXPERTS + expert] != 0.f
            ? parts[base + (size_t)expert * D_MODEL] : 0.f;
        value = __fadd_rn(value, part);
    }
    value = __fadd_rn(value, parts[base + (size_t)BB_EXPERTS * D_MODEL]);
    return __float2bfloat16(value);
}
__device__ __forceinline__ uint32_t float_to_ordered(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float ordered_to_float(uint32_t u) {
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
}
__device__ __forceinline__ void sample_argmax_merge(
        float &value, int &rank, float other, int other_rank) {
    if (other > value || (other == value && other_rank < rank)) {
        value = other;
        rank = other_rank;
    }
}
__device__ __forceinline__ int phoneme_warp_argmax(const bf16 *lg) {
    const int lane = threadIdx.x & 31;
    float best = -CUDART_INF_F;
    int arg = 0;
    for (int n = lane; n < BB_PHONEME; n += 32) {
        const float v = __bfloat162float(lg[n]);
        if (v > best) { best = v; arg = n; }
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffffu, best, off);
        const int other_arg = __shfl_down_sync(0xffffffffu, arg, off);
        sample_argmax_merge(best, arg, other, other_arg);
    }
    return arg;
}
__device__ __forceinline__ int sample_block_argmax(float value, int rank) {
    __shared__ float warp_value[WARPS];
    __shared__ int warp_rank[WARPS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffffu, value, off);
        const int other_rank = __shfl_down_sync(0xffffffffu, rank, off);
        sample_argmax_merge(value, rank, other, other_rank);
    }
    if (lane == 0) { warp_value[warp] = value; warp_rank[warp] = rank; }
    __syncthreads();
    if (warp == 0) {
        value = lane < WARPS ? warp_value[lane] : -CUDART_INF_F;
        rank = lane < WARPS ? warp_rank[lane] : INT_MAX;
        #pragma unroll
        for (int off = 16; off; off >>= 1) {
            const float other = __shfl_down_sync(0xffffffffu, value, off);
            const int other_rank = __shfl_down_sync(0xffffffffu, rank, off);
            sample_argmax_merge(value, rank, other, other_rank);
        }
    }
    return rank;
}
__device__ __forceinline__ int sample_eos_insertion_rank(
        const uint64_t *keys, float eos) {
    int lo = 0, hi = TOP_K;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (eos > ordered_to_float((uint32_t)(keys[mid] >> 32))) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}
__device__ __forceinline__ void sample_topk_serial(
        const uint64_t *keys, float eos, const float *noise, int *out) {
    float best_noised = -CUDART_INF_F;
    int best_code = 0;
    int a = 0;
    bool eos_used = false;
    for (int j = 0; j < TOP_K; ++j) {
        const float av = ordered_to_float((uint32_t)(keys[a] >> 32));
        float value;
        int code;
        if (!eos_used && eos > av) {
            value = eos;
            code = AUDIO_EOS;
            eos_used = true;
        } else {
            value = av;
            code = (int)(0xFFFFFFFFu - (uint32_t)(keys[a] & 0xFFFFFFFFu));
            ++a;
        }
        const float noised = value + noise[j];
        if (noised > best_noised) {
            best_noised = noised;
            best_code = code;
        }
    }
    *out = best_code;
}
__device__ __forceinline__ void sample_topk_parallel(
        const uint64_t *keys, float eos, const float *noise, int *out) {
    const int j = threadIdx.x;
    const bool sorted_nan = j < TOP_K &&
        isnan(ordered_to_float((uint32_t)(keys[j] >> 32)));
    if (__syncthreads_or(sorted_nan)) {
        if (threadIdx.x == 0) sample_topk_serial(keys, eos, noise, out);
        return;
    }
    const int lane = threadIdx.x & 31;
    int insertion = lane == 0 ? sample_eos_insertion_rank(keys, eos) : 0;
    insertion = __shfl_sync(0xffffffffu, insertion, 0);
    float value = -CUDART_INF_F;
    int rank = INT_MAX;
    if (j < TOP_K) {
        const int a = j - (j > insertion);
        const float v = j == insertion
            ? eos : ordered_to_float((uint32_t)(keys[a] >> 32));
        const float noised = v + noise[j];
        if (noised > -CUDART_INF_F) { value = noised; rank = j; }
    }
    const int winner = sample_block_argmax(value, rank);
    if (threadIdx.x == 0) {
        if (winner == INT_MAX) *out = 0;
        else if (winner == insertion) *out = AUDIO_EOS;
        else {
            const int a = winner - (winner > insertion);
            *out = (int)(0xFFFFFFFFu - (uint32_t)(keys[a] & 0xFFFFFFFFu));
        }
    }
}
constexpr int SORT_ITEMS_PER_THREAD = SORT_N / THREADS;
struct sampler_storage {
    uint64_t keys[2][SORT_N];
};
__device__ void sample_codes(
    const bf16 *logits, const bf16 *bias_unused, const int8_t *forbidden,
    const float *gumbel, int pos, int *out_codes,
    sampler_storage &storage, float temperature) {
#ifdef HEAD_BF16
    const bf16 *bias = bias_unused;
#else
    (void)bias_unused;
#endif
    for (int row = blockIdx.x; row < BATCH; row += gridDim.x) {
        uint64_t thread_keys[SORT_ITEMS_PER_THREAD];
        #pragma unroll
        for (int item = 0; item < SORT_ITEMS_PER_THREAD; ++item) {
            const int n = threadIdx.x + item * THREADS;
            float v = __bfloat162float(logits[(size_t)row * VOCAB_PADDED + n]);
#ifdef HEAD_BF16
            v += __bfloat162float(bias[n]);
#endif
            v /= temperature;
            thread_keys[item] = ((uint64_t)float_to_ordered(v) << 32) |
                                (uint64_t)(0xFFFFFFFFu - (uint32_t)n);
        }
        __shared__ float eos_v;
        if (threadIdx.x == 0) {
            float v = -CUDART_INF_F;
            if (!forbidden[AUDIO_EOS]) {
                v = __bfloat162float(logits[(size_t)row * VOCAB_PADDED + AUDIO_EOS]);
#ifdef HEAD_BF16
                v += __bfloat162float(bias[AUDIO_EOS]);
#endif
                v /= temperature;
            }
            eos_v = v;
        }
        int cross_stage = 0;
        for (int k = 2; k <= SORT_N; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                if (j >= 32) {
                    const int bank = cross_stage++ & 1;
                    #pragma unroll
                    for (int item = 0; item < SORT_ITEMS_PER_THREAD; ++item) {
                        const int n = threadIdx.x + item * THREADS;
                        storage.keys[bank][n] = thread_keys[item];
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int item = 0; item < SORT_ITEMS_PER_THREAD; ++item) {
                        const int n = threadIdx.x + item * THREADS;
                        const uint64_t other = storage.keys[bank][n ^ j];
                        const bool take_hi = ((n & k) == 0) == ((n & j) == 0);
                        const uint64_t mine = thread_keys[item];
                        thread_keys[item] = take_hi
                            ? (mine > other ? mine : other)
                            : (mine < other ? mine : other);
                    }
                } else {
                    #pragma unroll
                    for (int item = 0; item < SORT_ITEMS_PER_THREAD; ++item) {
                        const int n = threadIdx.x + item * THREADS;
                        const uint64_t other = __shfl_xor_sync(
                            0xffffffffu, thread_keys[item], j);
                        const bool take_hi = ((n & k) == 0) == ((n & j) == 0);
                        const uint64_t mine = thread_keys[item];
                        thread_keys[item] = take_hi
                            ? (mine > other ? mine : other)
                            : (mine < other ? mine : other);
                    }
                }
            }
        }
        const int top_bank = cross_stage & 1;
        if (threadIdx.x < TOP_K)
            storage.keys[top_bank][threadIdx.x] = thread_keys[0];
        __syncthreads();
        sample_topk_parallel(
            storage.keys[top_bank], eos_v,
            gumbel + ((size_t)row * N_POS + pos) * TOP_K,
            out_codes + row);
        __syncthreads();
    }
}
__device__ __forceinline__ int mk2_expert_tile(int work, unsigned mask,
                                             int tiles_per_expert) {
    if (!tiles_per_expert) return work;
    int rank = work / tiles_per_expert;
    while (rank-- > 0) mask &= mask - 1;
    return (__ffs(mask) - 1) * tiles_per_expert + work % tiles_per_expert;
}
__device__ void r6_sparse_map(const float* routes,int* map,int* counts
                             MK2_PROGRESS_PARAMETER(progress_mask)) {
    __shared__ int r7_warp_counts[THREADS/32];
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    const unsigned lower=(1u<<lane)-1u;
    for(int e=blockIdx.x;e<BB_EXPERTS;e+=gridDim.x) {
        int count=0;
        for(int base=0;base<BATCH;base+=THREADS) {
            const int row=base+threadIdx.x;
            const bool selected=row<BATCH && MK2_PROGRESS_ENABLED(progress_mask,row) &&
                routes[row*BB_EXPERTS+e]!=0.f;
            const unsigned mask=__ballot_sync(0xffffffffu,selected);
            if(lane==0)r7_warp_counts[warp]=__popc(mask);
            __syncthreads();
            int prefix=0,total=0;
            for(int w=0;w<THREADS/32;++w) {
                const int n=r7_warp_counts[w];
                if(w<warp)prefix+=n;
                total+=n;
            }
            if(selected)map[e*BATCH+count+prefix+__popc(mask&lower)]=row;
            count+=total;
            __syncthreads();
        }
        if(threadIdx.x==0)counts[e]=count;
        for(int row=count+threadIdx.x;row<((count+BATCH_PAD-1)/BATCH_PAD)*BATCH_PAD;row+=THREADS)
            map[e*BATCH+row]=-1;
    }
}
__device__ void r6_sparse_gather(const bf16* source,bf16* packed,const int* map,
                                 const int* counts,int* lut) {
    if(blockIdx.x==0&&threadIdx.x==0) {
        int n=0;
        for(int e=0;e<BB_EXPERTS;++e) {
            const int t_e=(counts[e]+BATCH_PAD-1)/BATCH_PAD;
            for(int t=0;t+1<t_e;t+=2) {
                lut[1+n++]=e*DECODE_TILES+t;
                lut[1+n++]=e*DECODE_TILES+t+1;
            }
        }
        const int paired=n>>1;
        for(int e=0;e<BB_EXPERTS;++e) {
            const int t_e=(counts[e]+BATCH_PAD-1)/BATCH_PAD;
            if(t_e&1) lut[1+n++]=e*DECODE_TILES+t_e-1;
        }
        lut[0]=n|(paired<<12);
    }
    constexpr int vector_width=sizeof(uint4)/sizeof(bf16);
    constexpr int row_vectors=D_MODEL/vector_width;
    for(int e=0;e<BB_EXPERTS;++e) {
        const int rows=((counts[e]+BATCH_PAD-1)/BATCH_PAD)*BATCH_PAD;
        for(int i=blockIdx.x*THREADS+threadIdx.x;i<rows*row_vectors;i+=gridDim.x*THREADS) {
            const int row=map[e*BATCH+i/row_vectors];
            reinterpret_cast<uint4*>(packed)[(size_t)e*BATCH*row_vectors+i]=
                row<0?make_uint4(0,0,0,0):reinterpret_cast<const uint4*>(source)[(size_t)row*row_vectors+i%row_vectors];
        }
    }
}
__device__ void r6_tile(int work,int output_tiles,const int* lut,int &tile,int &tok,
                        int mtg) {
    const int groups=output_tiles/mtg;
    tok=lut[1+work/groups];
    const int e=tok/DECODE_TILES;
    tile=e*output_tiles+(work%groups)*mtg;
}
__device__ __forceinline__ void mk2_group_tile(
    int work, int grouped, int &tile, int &tok, int &here) {
    here = work < grouped ? NM_MAX : 1;
    const int unit = work < grouped ? work * NM_MAX : grouped * (NM_MAX - 1) + work;
    tile = unit / DECODE_TILES;
    tok = unit - tile * DECODE_TILES;
}
__device__ __forceinline__ void gemm(
    const weight_gl &w, const act_gl &a, int n_out, int k_total,
    bf16 *dest, int dest_stride, bf16 *residual, Epilogue epi,
    weight_tile (&p_s)[STAGES][SLOTS], out_tile &o_s,
    semaphore *filled, semaphore *drained, out_tile_f &o_sf, int &iter_io,
    int &tile_io,
    const bf16 *bias = nullptr,
    int out_tiles_per_expert = 0, int act_k_tiles_per_expert = 0,
    const float *moe_w = nullptr, float *moe_out = nullptr,
    int n_tok = DECODE_TILES,
    int rows_valid = DECODE_ROWS,
    const float *active_routes = nullptr, int active_expert_tiles = 0,
    const int *r6_lut = nullptr,const int *r6_map = nullptr) {
    unsigned active_mask = 0;
    int compact_tiles = 0;
    if constexpr (MK2_ACTIVE_EXPERTS && BATCH == 1) {
        if (active_routes && active_expert_tiles) {
            for (int expert = 0; expert < BB_EXPERTS; ++expert)
                if (active_routes[expert] != 0.f) active_mask |= 1u << expert;
            compact_tiles = active_expert_tiles;
        }
    }
    constexpr int QSLOT = MT + NM_MAX;
    constexpr bool WIDE = MT > 1 && NM_MAX > 1 && STAGES * SLOTS >= 2 * QSLOT;
    constexpr int PSTAGES = WIDE ? 2 : STAGES;
    constexpr int PSLOT = WIDE ? QSLOT : SLOTS;
    constexpr int ACCN = WIDE ? MT * NM_MAX : NM_MAX;
    constexpr int HEX_MT = 3, HEX_NM = 2;
    constexpr int HEX_SLOT = HEX_MT + HEX_NM;
    auto epi_publish = [](out_tile &st, const rt_fl<16, BATCH_PAD> &src) {
        using acc_t = rt_fl<16, BATCH_PAD>;
        static_assert(acc_t::height == 1, "one row block per warp");
        static_assert(out_tile::rows == BATCH_PAD && out_tile::cols == OUT_ATOM,
                      "staging tile must be square for an in-place transpose");
        const int t = ::kittens::laneid();
        const uint32_t base =
            static_cast<uint32_t>(__cvta_generic_to_shared(&st.data[0]));
        const int row0 = 8 * (t >> 4) + (t & 7);
        const int col0 = 16 * warpgroup::warpid() + 8 * ((t >> 3) & 1);
        #pragma unroll
        for (int j = 0; j < acc_t::width; ++j) {
            bf16_2 q0 = base_types::convertor<bf16_2, float2>::convert(
                src.tiles[0][j].data[0]);
            bf16_2 q1 = base_types::convertor<bf16_2, float2>::convert(
                src.tiles[0][j].data[1]);
            bf16_2 q2 = base_types::convertor<bf16_2, float2>::convert(
                src.tiles[0][j].data[2]);
            bf16_2 q3 = base_types::convertor<bf16_2, float2>::convert(
                src.tiles[0][j].data[3]);
            move<bf16_2>::stsm4t(
                out_tile::idx(base, {16 * j + row0, col0}), q0, q1, q2, q3);
        }
    };
    weight_tile *const pf = &p_s[0][0];
    const bool arena10 =
        reinterpret_cast<char *>(&o_sf) ==
        reinterpret_cast<char *>(pf + STAGES * SLOTS);
    const bool two_d = !r6_lut && out_tiles_per_expert == 0
        && n_tok == DECODE_TILES && DECODE_TILES % NM_MAX == 0;
    int n_tiles = compact_tiles ? __popc(active_mask)*compact_tiles : n_out/OUT_ATOM;
    constexpr int QTOK = DECODE_TILES / NM_MAX;
    const int out_tiles = n_tiles;
    const bool tall3 = WIDE && two_d && out_tiles == 24;
    const bool split72 = WIDE && two_d && DECODE_TILES == 16 && out_tiles == 72;
    const bool split81 = WIDE && two_d && DECODE_TILES == 16 && out_tiles == 81;
    const bool split17 = WIDE && two_d && DECODE_TILES == 16 && out_tiles == 17;
    const bool mixed40 = WIDE && two_d && DECODE_TILES == 16 && out_tiles == 40;
    const int n_tiles_hex_all = (out_tiles / HEX_MT) * (DECODE_TILES / HEX_NM);
    const bool hex = WIDE && two_d && arena10 && DECODE_TILES % HEX_NM == 0 &&
        (out_tiles == 96 || out_tiles == 81 || out_tiles == 72 ||
         out_tiles == 40) &&
        (epi == EPI_STORE || epi == EPI_GELU) && !r6_lut &&
        out_tiles_per_expert == 0;
    const int hex_tok = DECODE_TILES / HEX_NM;
    const int hex_wide = (out_tiles == 72 || out_tiles == 81)
        ? 128 : n_tiles_hex_all;
    const int pslot = hex ? HEX_SLOT : PSLOT;
    int q_units = 0, q_rem_base = 0;
    int grouped = 0;
    if constexpr (WIDE) {
        if (two_d) {
            q_units = (out_tiles / MT) * QTOK;
            q_rem_base = out_tiles - out_tiles % MT;
        }
    } else if (two_d) {
        const int g = (int)gridDim.x;
        const int units = n_tiles * DECODE_TILES;
        const int rounds = (units + g - 1) / g;
        const int cap = units / NM_MAX;
        grouped = g * (rounds / NM_MAX);
        if (grouped > cap) grouped = cap;
    }
    const int r6_output_tiles = r6_lut ? n_out/BB_EXPERTS/OUT_ATOM : 0;
    const int r6_mtg = (MT > 1 && r6_lut && r6_output_tiles % MT == 0) ? MT : 1;
    const int r6_groups = r6_lut ? r6_output_tiles/r6_mtg : 0;
    const int r6_rows = r6_lut ? (r6_lut[0] & 0xFFF) : 0;
    const int r6_pairs = (r6_lut && WIDE && r6_mtg == MT) ? (r6_lut[0] >> 12) : 0;
    const int r6_pair_items = r6_pairs * r6_groups;
    const int r6_single_mtg = (r6_lut && WIDE && r6_output_tiles % 3 == 0)
        ? 3 : r6_mtg;
    const int r6_single_groups = r6_lut ? r6_output_tiles/r6_single_mtg : 0;
    const int r6_single_items = r6_lut
        ? (r6_rows - 2 * r6_pairs) * r6_single_groups : 0;
    if(r6_lut) {
        n_tiles=r6_pair_items+r6_single_items;
        n_tok=1;
    } else if(two_d) {
        if constexpr (WIDE) {
            if (hex) n_tiles = (out_tiles == 72) ? 256
                                    : (out_tiles == 81) ? 264
                                    : (out_tiles == 40)
                                        ? n_tiles_hex_all + hex_tok
                                    : n_tiles_hex_all;
            else if (mixed40) n_tiles = 244;
            else if (split72) n_tiles = 360;
            else if (split81) n_tiles = 384;
            else if (split17) n_tiles = 132;
            else if (tall3) n_tiles = (out_tiles / 3) * DECODE_TILES;
            else n_tiles = q_units + (out_tiles % MT) * QTOK;
        }
        else n_tiles = grouped + (n_tiles*DECODE_TILES - grouped*NM_MAX);
        n_tok=1; }
    const int r6_extra = r6_lut ? n_tiles % (int)gridDim.x : 0;
    const int r6_skew = r6_lut
        ? (r6_single_items < r6_extra ? r6_single_items : r6_extra) : 0;
    const int iters = k_total / KCHUNK;
    int tile_start = (int)blockIdx.x - tile_io % (int)gridDim.x;
    if (tile_start < 0) tile_start += (int)gridDim.x;
    tile_io += n_tiles;
    int tiles_here = 0;
    for (int t = tile_start; t < n_tiles; t += gridDim.x) ++tiles_here;
    const int iter_base = iter_io;
    iter_io += tiles_here * iters * n_tok;
    const int my_tiles = (n_tiles - tile_start + gridDim.x - 1) / gridDim.x;
    if (warpgroup::groupid() == 1) {
        if (hex) {
            int iter = iter_base;
            for (int work = tile_start; work < n_tiles; work += gridDim.x) {
                if (out_tiles == 81 && work >= hex_wide) {
                    const int j = work - hex_wide;
                    const bool pair = j >= 128;
                    const int nt = pair ? 80 : 48 + 2 * (j / hex_tok);
                    const int ntok = HEX_NM * (j % hex_tok);
                    if (pair) {
                        for (int i = 0; i < iters; ++i, ++iter) {
                            const int s = iter % PSTAGES;
                            wait(drained[s], (iter / PSTAGES) & 1);
                            if (warpgroup::laneid() == 0) {
                                tma::expect_bytes(filled[s], sizeof(weight_tile)
                                                           + HEX_NM * sizeof(act_tile));
                                tma::load_async<2, cache_policy::EVICT_FIRST>(
                                    pf[s * pslot], w, {nt, i}, filled[s]);
                                #pragma unroll
                                for (int m = 0; m < HEX_NM; ++m)
                                    tma::load_async<2, cache_policy::EVICT_LAST>(
                                        pf[s * pslot + 1 + m], a,
                                        {ntok + m, i}, filled[s]);
                            }
                        }
                    } else {
                        for (int i = 0; i < iters; ++i, ++iter) {
                            const int s = iter % PSTAGES;
                            wait(drained[s], (iter / PSTAGES) & 1);
                            if (warpgroup::laneid() == 0) {
                                tma::expect_bytes(filled[s], 2 * sizeof(weight_tile)
                                                           + HEX_NM * sizeof(act_tile));
                                #pragma unroll
                                for (int c = 0; c < 2; ++c)
                                    tma::load_async<2, cache_policy::EVICT_FIRST>(
                                        pf[s * pslot + c], w, {nt + c, i}, filled[s]);
                                #pragma unroll
                                for (int m = 0; m < HEX_NM; ++m)
                                    tma::load_async<2, cache_policy::EVICT_LAST>(
                                        pf[s * pslot + 2 + m], a,
                                        {ntok + m, i}, filled[s]);
                            }
                        }
                    }
                    continue;
                }
                if (out_tiles == 40 && work >= hex_wide) {
                    const int ntok = HEX_NM * (work - hex_wide);
                    for (int i = 0; i < iters; ++i, ++iter) {
                        const int s = iter % PSTAGES;
                        wait(drained[s], (iter / PSTAGES) & 1);
                        if (warpgroup::laneid() == 0) {
                            tma::expect_bytes(filled[s], sizeof(weight_tile)
                                                       + HEX_NM * sizeof(act_tile));
                            tma::load_async<2, cache_policy::EVICT_FIRST>(
                                pf[s * pslot], w, {out_tiles - 1, i}, filled[s]);
                            #pragma unroll
                            for (int m = 0; m < HEX_NM; ++m)
                                tma::load_async<2, cache_policy::EVICT_LAST>(
                                    pf[s * pslot + 1 + m], a,
                                    {ntok + m, i}, filled[s]);
                        }
                    }
                    continue;
                }
                if (work >= hex_wide) {
                    const int j = work - hex_wide;
                    const int nt = HEX_MT * ((hex_wide + (j >> 1)) / hex_tok);
                    const int ntok = HEX_NM * ((hex_wide + (j >> 1)) % hex_tok) + (j & 1);
                    for (int i = 0; i < iters; ++i, ++iter) {
                        const int s = iter % PSTAGES;
                        wait(drained[s], (iter / PSTAGES) & 1);
                        if (warpgroup::laneid() == 0) {
                            tma::expect_bytes(filled[s], HEX_MT * sizeof(weight_tile)
                                                       + sizeof(act_tile));
                            #pragma unroll
                            for (int c = 0; c < HEX_MT; ++c)
                                tma::load_async<2, cache_policy::EVICT_FIRST>(
                                    pf[s * pslot + c], w, {nt + c, i}, filled[s]);
                            tma::load_async<2, cache_policy::EVICT_LAST>(
                                pf[s * pslot + HEX_MT], a, {ntok, i}, filled[s]);
                        }
                    }
                    continue;
                }
                const int tile_id = HEX_MT * (work / hex_tok);
                const int tok = HEX_NM * (work % hex_tok);
                rt_fl<16, BATCH_PAD> pacc[HEX_NM];
                #pragma unroll
                for (int u = 0; u < HEX_NM; ++u) warp::zero(pacc[u]);
                int pend = -1, pend_ph = 0;
                for (int i = 0; i < iters; ++i, ++iter) {
                    const int s = iter % PSTAGES;
                    const int phase = (iter / PSTAGES) & 1;
                    wait(drained[s], phase);
                    if (warpgroup::laneid() == 0) {
                        tma::expect_bytes(filled[s], HEX_MT * sizeof(weight_tile)
                                                   + HEX_NM * sizeof(act_tile));
                        #pragma unroll
                        for (int c = 0; c < HEX_MT; ++c)
                            tma::load_async<2, cache_policy::EVICT_FIRST>(
                                pf[s * pslot + c], w, {tile_id + c, i}, filled[s]);
                        #pragma unroll
                        for (int m = 0; m < HEX_NM; ++m)
                            tma::load_async<2, cache_policy::EVICT_LAST>(
                                pf[s * pslot + HEX_MT + m], a, {tok + m, i}, filled[s]);
                    }
                    if (pend >= 0) { {
                        wait(filled[pend], pend_ph);
                        warpgroup::mma_ABt(pacc[0], pf[pend * pslot + HEX_MT - 1],
                                           pf[pend * pslot + HEX_MT]);
                        warpgroup::mma_ABt<rt_fl<16, BATCH_PAD>, weight_tile,
                                           weight_tile, 0>(
                            pacc[1], pf[pend * pslot + HEX_MT - 1],
                            pf[pend * pslot + HEX_MT + 1]);
                        warpgroup::mma_async_wait();
                    } }
                    pend = s; pend_ph = phase;
                }
                {
                        wait(filled[pend], pend_ph);
                        warpgroup::mma_ABt(pacc[0], pf[pend * pslot + HEX_MT - 1],
                                           pf[pend * pslot + HEX_MT]);
                        warpgroup::mma_ABt<rt_fl<16, BATCH_PAD>, weight_tile,
                                           weight_tile, 0>(
                            pacc[1], pf[pend * pslot + HEX_MT - 1],
                            pf[pend * pslot + HEX_MT + 1]);
                        warpgroup::mma_async_wait();
                    }
                const int es = iter % PSTAGES;
                wait(drained[es], (iter / PSTAGES) & 1);
                out_tile_f &q_sf =
                    *reinterpret_cast<out_tile_f *>(&pf[es * pslot]);
                out_tile &q_s = *reinterpret_cast<out_tile *>(&q_sf);
                const int qbase = (tile_id + HEX_MT - 1) * OUT_ATOM;
                constexpr int OUT_OCTS = OUT_ATOM / 8;
                #pragma unroll
                for (int u = 0; u < HEX_NM; ++u) {
                    const int grow0 = (tok + u) * BATCH_PAD;
                    epi_publish(q_s, pacc[u]);
                    warpgroup::sync(3);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    for (int idx = warpgroup::laneid();
                         idx < OUT_OCTS * rows_valid; idx += 128) {
                        const int row = idx >> 3, col = (idx & 7) << 3;
                        const size_t at =
                            (size_t)(grow0 + row) * dest_stride + qbase + col;
                        const bf16_oct q = *reinterpret_cast<const bf16_oct *>(
                            &q_s[int2{row, col}]);
                        if (epi == EPI_STORE) {
                            *reinterpret_cast<bf16_oct *>(dest + at) = q;
                        } else {
                            *reinterpret_cast<bf16_oct *>(dest + at) = bf16_oct{
                                __float2bfloat16(gelu_tanh_fp32(__bfloat162float(q.x0))),
                                __float2bfloat16(gelu_tanh_fp32(__bfloat162float(q.x1))),
                                __float2bfloat16(gelu_tanh_fp32(__bfloat162float(q.x2))),
                                __float2bfloat16(gelu_tanh_fp32(__bfloat162float(q.x3))),
                                __float2bfloat16(gelu_tanh_fp32(__bfloat162float(q.x4))),
                                __float2bfloat16(gelu_tanh_fp32(__bfloat162float(q.x5))),
                                __float2bfloat16(gelu_tanh_fp32(__bfloat162float(q.x6))),
                                __float2bfloat16(gelu_tanh_fp32(__bfloat162float(q.x7)))};
                        }
                    }
                    warpgroup::sync(3);
                }
            }
        } else if (warpgroup::laneid() == 0) {
            int iter = iter_base;
            for (int work = tile_start; work < n_tiles; work += gridDim.x) {
                int tile_id = mk2_expert_tile(work, active_mask, compact_tiles), r6_tok=0;
                int nm = 1, mt = 1;
                if(r6_lut) {
                    int rwork = work - r6_skew;
                    if (rwork < 0) rwork += n_tiles;
                    const bool pair = rwork < r6_pair_items;
                    const int q = pair ? rwork / r6_groups
                                       : (rwork-r6_pair_items) / r6_single_groups;
                    const int group = pair ? rwork-q*r6_groups
                                           : rwork-r6_pair_items-q*r6_single_groups;
                    const int base_item = pair ? 2*q : 2*r6_pairs+q;
                    const int ltok = r6_lut[1 + base_item];
                    r6_tok = ltok;
                    tile_id = (ltok / DECODE_TILES) * r6_output_tiles
                            + group * (pair ? r6_mtg : r6_single_mtg);
                    mt = pair ? r6_mtg : r6_single_mtg;
                    nm = pair ? NM_MAX : 1;
                }
                else if(two_d) {
                    if constexpr (WIDE) {
                        if (mixed40) {
                            const bool quad = work < 132;
                            const int item = quad ? work : work - 132;
                            if (quad && item < 84) {
                                tile_id = 2 * (item / 7);
                                r6_tok = 2 * (item % 7);
                            } else if (quad) {
                                const int tail = item - 84;
                                tile_id = 2 * (12 + tail / 6);
                                r6_tok = 2 * (tail % 6);
                            } else if (item < 48) {
                                tile_id = 2 * (item / 4) + (item % 4) / 2;
                                r6_tok = 14 + item % 2;
                            } else {
                                const int tail = item - 48;
                                tile_id = 2 * (12 + tail / 8) + (tail % 8) / 4;
                                r6_tok = 12 + tail % 4;
                            }
                            mt = quad ? MT : 1;
                            nm = quad ? NM_MAX : 1;
                        } else if (split72) {
                            if (work < 264) {
                                tile_id = 2 * (work / 8);
                                r6_tok = 2 * (work % 8);
                                mt = MT; nm = NM_MAX;
                            } else {
                                const int i = work - 264, c = i >> 2, q = i & 3;
                                tile_id = 2 * (33 + (c >> 3)) + (q >> 1);
                                r6_tok = 2 * (c & 7) + (q & 1);
                                mt = 1; nm = 1;
                            }
                        } else if (split81) {
                            if (work < 264) {
                                tile_id = 2 * (work / 8);
                                r6_tok = 2 * (work % 8);
                                mt = MT; nm = NM_MAX;
                            } else if (work < 376) {
                                const int i = work - 264;
                                tile_id = 2 * (33 + i / 16);
                                r6_tok = i % 16;
                                mt = MT; nm = 1;
                            } else {
                                tile_id = 80;
                                r6_tok = 2 * (work - 376);
                                mt = 1; nm = NM_MAX;
                            }
                        } else if (split17) {
                            if (work < 8) {
                                tile_id = 14;
                                r6_tok = work;
                                mt = 3; nm = 1;
                            } else if (work < 120) {
                                const int i = work - 8;
                                tile_id = i >> 3;
                                r6_tok = 2 * (i & 7);
                                mt = 1; nm = NM_MAX;
                            } else {
                                const int i = work - 120;
                                tile_id = 14 + (i >> 2);
                                r6_tok = 8 + 2 * (i & 3);
                                mt = 1; nm = NM_MAX;
                            }
                        } else if (tall3) {
                            tile_id = 3 * (work / DECODE_TILES);
                            r6_tok = work % DECODE_TILES;
                            mt = 3;
                            nm = 1;
                        } else {
                            const int j = work < q_units ? work : work - q_units;
                            tile_id = work < q_units ? MT * (j / QTOK)
                                                     : q_rem_base + j / QTOK;
                            r6_tok = NM_MAX * (j % QTOK);
                            mt = work < q_units ? MT : 1;
                            nm = NM_MAX;
                        }
                    } else mk2_group_tile(work,grouped,tile_id,r6_tok,nm);
                }
                for (int token_iteration = 0; token_iteration < n_tok; ++token_iteration)
                for (int i = 0; i < iters; ++i, ++iter) {
                    const int tok = (r6_lut || two_d) ? r6_tok : token_iteration;
                    const int s = iter % PSTAGES;
                    const int phase = (iter / PSTAGES) & 1;
                    wait(drained[s], phase);
                    tma::expect_bytes(filled[s],
                                      mt * sizeof(weight_tile) + nm * sizeof(act_tile));
                    if (mt == 1) {
                        tma::load_async<2, cache_policy::EVICT_FIRST>(
                            pf[s * pslot], w, {tile_id, i}, filled[s]);
                    } else if (mt == 3) {
                        #pragma unroll
                        for (int c = 0; c < 3; ++c)
                            tma::load_async<2, cache_policy::EVICT_FIRST>(
                                pf[s * pslot + c], w, {tile_id + c, i}, filled[s]);
                    } else {
                        #pragma unroll
                        for (int c = 0; c < MT; ++c)
                            tma::load_async<2, cache_policy::EVICT_FIRST>(
                                pf[s * pslot + c], w, {tile_id + c, i}, filled[s]);
                    }
                    const int a_col = out_tiles_per_expert && !r6_lut
                        ? (tile_id / out_tiles_per_expert) * act_k_tiles_per_expert + i
                        : i;
                    if (NM_MAX == 1 || nm == NM_MAX) {
                        #pragma unroll
                        for (int m = 0; m < NM_MAX; ++m)
                            tma::load_async<2, cache_policy::EVICT_LAST>(
                                pf[s * pslot + mt + m], a, {tok + m, a_col}, filled[s]);
                    } else {
                        tma::load_async<2, cache_policy::EVICT_LAST>(
                            pf[s * pslot + mt], a, {tok, a_col}, filled[s]);
                    }
                }
            }
        }
    } else {
        int iter = iter_base;
        for (int work = tile_start; work < n_tiles; work += gridDim.x)
        for (int token_iteration = 0; token_iteration < n_tok; ++token_iteration) {
            int tile_id=mk2_expert_tile(work, active_mask, compact_tiles),tok=token_iteration;
            int nm = 1, mt = 1;
            if(r6_lut) {
                    int rwork = work - r6_skew;
                    if (rwork < 0) rwork += n_tiles;
                    const bool pair = rwork < r6_pair_items;
                    const int q = pair ? rwork / r6_groups
                                       : (rwork-r6_pair_items) / r6_single_groups;
                    const int group = pair ? rwork-q*r6_groups
                                           : rwork-r6_pair_items-q*r6_single_groups;
                    const int base_item = pair ? 2*q : 2*r6_pairs+q;
                    const int ltok = r6_lut[1 + base_item];
                    tok = ltok;
                    tile_id = (ltok / DECODE_TILES) * r6_output_tiles
                            + group * (pair ? r6_mtg : r6_single_mtg);
                    mt = pair ? r6_mtg : r6_single_mtg;
                    nm = pair ? NM_MAX : 1;
                }
            else if(two_d) {
                if constexpr (WIDE) {
                    if (hex) {
                        if (out_tiles == 81 && work >= hex_wide) {
                            const int j = work - hex_wide;
                            const bool pair = j >= 128;
                            tile_id = pair ? 80 : 48 + 2 * (j / hex_tok);
                            tok = HEX_NM * (j % hex_tok);
                            mt = pair ? 1 : 2;
                            nm = HEX_NM;
                        } else if (out_tiles == 40 && work >= hex_wide) {
                            tile_id = out_tiles - 1;
                            tok = HEX_NM * (work - hex_wide);
                            mt = 1;
                            nm = HEX_NM;
                        } else {
                            const int hw = work < hex_wide
                                ? work : hex_wide + ((work - hex_wide) >> 1);
                            tile_id = HEX_MT * (hw / hex_tok);
                            tok = HEX_NM * (hw % hex_tok)
                                + (work < hex_wide ? 0 : ((work - hex_wide) & 1));
                            mt = HEX_MT; nm = work < hex_wide ? HEX_NM : 1;
                        }
                    } else if (mixed40) {
                        const bool quad = work < 132;
                        const int item = quad ? work : work - 132;
                        if (quad && item < 84) {
                            tile_id = 2 * (item / 7);
                            tok = 2 * (item % 7);
                        } else if (quad) {
                            const int tail = item - 84;
                            tile_id = 2 * (12 + tail / 6);
                            tok = 2 * (tail % 6);
                        } else if (item < 48) {
                            tile_id = 2 * (item / 4) + (item % 4) / 2;
                            tok = 14 + item % 2;
                        } else {
                            const int tail = item - 48;
                            tile_id = 2 * (12 + tail / 8) + (tail % 8) / 4;
                            tok = 12 + tail % 4;
                        }
                        mt = quad ? MT : 1;
                        nm = quad ? NM_MAX : 1;
                    } else if (split72) {
                        if (work < 264) {
                            tile_id = 2 * (work / 8);
                            tok = 2 * (work % 8);
                            mt = MT; nm = NM_MAX;
                        } else {
                            const int i = work - 264, c = i >> 2, q = i & 3;
                            tile_id = 2 * (33 + (c >> 3)) + (q >> 1);
                            tok = 2 * (c & 7) + (q & 1);
                            mt = 1; nm = 1;
                        }
                    } else if (split81) {
                        if (work < 264) {
                            tile_id = 2 * (work / 8);
                            tok = 2 * (work % 8);
                            mt = MT; nm = NM_MAX;
                        } else if (work < 376) {
                            const int i = work - 264;
                            tile_id = 2 * (33 + i / 16);
                            tok = i % 16;
                            mt = MT; nm = 1;
                        } else {
                            tile_id = 80;
                            tok = 2 * (work - 376);
                            mt = 1; nm = NM_MAX;
                        }
                    } else if (split17) {
                        if (work < 8) {
                            tile_id = 14;
                            tok = work;
                            mt = 3; nm = 1;
                        } else if (work < 120) {
                            const int i = work - 8;
                            tile_id = i >> 3;
                            tok = 2 * (i & 7);
                            mt = 1; nm = NM_MAX;
                        } else {
                            const int i = work - 120;
                            tile_id = 14 + (i >> 2);
                            tok = 8 + 2 * (i & 3);
                            mt = 1; nm = NM_MAX;
                        }
                    } else if (tall3) {
                        tile_id = 3 * (work / DECODE_TILES);
                        tok = work % DECODE_TILES;
                        mt = 3;
                        nm = 1;
                    } else {
                        const int j = work < q_units ? work : work - q_units;
                        tile_id = work < q_units ? MT * (j / QTOK)
                                                 : q_rem_base + j / QTOK;
                        tok = NM_MAX * (j % QTOK);
                        mt = work < q_units ? MT : 1;
                        nm = NM_MAX;
                    }
                } else mk2_group_tile(work,grouped,tile_id,tok,nm);
            }
            int epi_hold = -1;
            rt_fl<16, BATCH_PAD> accum[ACCN];
            if (hex && nm == 1) {
                #pragma unroll
                for (int u = 0; u < 3; ++u) warp::zero(accum[u]);
                for (int i = 0; i < iters; ++i, ++iter) {
                    const int s = iter % PSTAGES;
                    const int phase = (iter / PSTAGES) & 1;
                    wait(filled[s], phase);
                    warpgroup::mma_ABt(accum[0], pf[s * pslot], pf[s * pslot + HEX_MT]);
                    warpgroup::mma_ABt<rt_fl<16, BATCH_PAD>, weight_tile,
                                       weight_tile, 0>(
                        accum[1], pf[s * pslot + 1], pf[s * pslot + HEX_MT]);
                    warpgroup::mma_ABt<rt_fl<16, BATCH_PAD>, weight_tile,
                                       weight_tile, 0>(
                        accum[2], pf[s * pslot + 2], pf[s * pslot + HEX_MT]);
                    warpgroup::mma_async_wait();
                    if (i + 1 < iters) {
                        if (warpgroup::laneid() == 0)
                            arrive(drained[s], CONSUMER_WARPS);
                    } else epi_hold = s;
                }
            } else if (hex && mt == HEX_MT) {
                #pragma unroll
                for (int u = 0; u < ACCN; ++u) warp::zero(accum[u]);
                for (int i = 0; i < iters; ++i, ++iter) {
                    const int s = iter % PSTAGES;
                    const int phase = (iter / PSTAGES) & 1;
                    wait(filled[s], phase);
                    warpgroup::mma_ABt(accum[0], pf[s * pslot],
                                       pf[s * pslot + HEX_MT]);
                    #pragma unroll
                    for (int u = 1; u < ACCN; ++u)
                        warpgroup::mma_ABt<rt_fl<16, BATCH_PAD>, weight_tile,
                                           weight_tile, 0>(
                            accum[u], pf[s * pslot + u / HEX_NM],
                            pf[s * pslot + HEX_MT + u % HEX_NM]);
                    warpgroup::mma_async_wait();
                    if (i + 1 < iters) {
                        if (warpgroup::laneid() == 0)
                            arrive(drained[s], CONSUMER_WARPS);
                    } else epi_hold = s;
                }
            } else if (WIDE && mt == 3) {
                #pragma unroll
                for (int u = 0; u < 3; ++u) warp::zero(accum[u]);
                for (int i = 0; i < iters; ++i, ++iter) {
                    const int s = iter % PSTAGES;
                    const int phase = (iter / PSTAGES) & 1;
                    wait(filled[s], phase);
                    warpgroup::mma_ABt(accum[0], pf[s * pslot], pf[s * pslot + 3]);
                    warpgroup::mma_ABt<rt_fl<16, BATCH_PAD>, weight_tile,
                                       weight_tile, 0>(
                        accum[1], pf[s * pslot + 1], pf[s * pslot + 3]);
                    warpgroup::mma_ABt<rt_fl<16, BATCH_PAD>, weight_tile,
                                       weight_tile, 0>(
                        accum[2], pf[s * pslot + 2], pf[s * pslot + 3]);
                    warpgroup::mma_async_wait();
                    if (warpgroup::laneid() == 0) arrive(drained[s], CONSUMER_WARPS);
                }
            } else if (WIDE && mt == MT && nm == NM_MAX) {
                #pragma unroll
                for (int u = 0; u < ACCN; ++u) warp::zero(accum[u]);
                for (int i = 0; i < iters; ++i, ++iter) {
                    const int s = iter % PSTAGES;
                    const int phase = (iter / PSTAGES) & 1;
                    wait(filled[s], phase);
                    warpgroup::mma_ABt(accum[0], pf[s * pslot], pf[s * pslot + MT]);
                    #pragma unroll
                    for (int u = 1; u < ACCN; ++u)
                        warpgroup::mma_ABt<rt_fl<16, BATCH_PAD>, weight_tile,
                                           weight_tile, 0>(
                            accum[u], pf[s * pslot + u / NM_MAX],
                            pf[s * pslot + MT + u % NM_MAX]);
                    warpgroup::mma_async_wait();
                    if (warpgroup::laneid() == 0) arrive(drained[s], CONSUMER_WARPS);
                }
            } else if (MT > 1 && mt == MT) {
                #pragma unroll
                for (int c = 0; c < MT; ++c) warp::zero(accum[c]);
                for (int i = 0; i < iters; ++i, ++iter) {
                    const int s = iter % PSTAGES;
                    const int phase = (iter / PSTAGES) & 1;
                    wait(filled[s], phase);
                    warpgroup::mma_ABt(accum[0], pf[s * pslot], pf[s * pslot + MT]);
                    #pragma unroll
                    for (int c = 1; c < MT; ++c)
                        warpgroup::mma_ABt<rt_fl<16, BATCH_PAD>, weight_tile,
                                           weight_tile, 0>(
                            accum[c], pf[s * pslot + c], pf[s * pslot + MT]);
                    warpgroup::mma_async_wait();
                    if (warpgroup::laneid() == 0) arrive(drained[s], CONSUMER_WARPS);
                }
            } else if (NM_MAX == 1 || nm == NM_MAX) {
                #pragma unroll
                for (int m = 0; m < NM_MAX; ++m) warp::zero(accum[m]);
                for (int i = 0; i < iters; ++i, ++iter) {
                    const int s = iter % PSTAGES;
                    const int phase = (iter / PSTAGES) & 1;
                    wait(filled[s], phase);
                    warpgroup::mma_ABt(accum[0], pf[s * pslot], pf[s * pslot + 1]);
                    #pragma unroll
                    for (int m = 1; m < NM_MAX; ++m)
                        warpgroup::mma_ABt<rt_fl<16, BATCH_PAD>, weight_tile,
                                           weight_tile, 0>(
                            accum[m], pf[s * pslot], pf[s * pslot + 1 + m]);
                    warpgroup::mma_async_wait();
                    if (warpgroup::laneid() == 0) arrive(drained[s], CONSUMER_WARPS);
                }
            } else {
                warp::zero(accum[0]);
                for (int i = 0; i < iters; ++i, ++iter) {
                    const int s = iter % PSTAGES;
                    const int phase = (iter / PSTAGES) & 1;
                    wait(filled[s], phase);
                    warpgroup::mma_ABt(accum[0], pf[s * pslot], pf[s * pslot + 1]);
                    warpgroup::mma_async_wait();
                    if (warpgroup::laneid() == 0) arrive(drained[s], CONSUMER_WARPS);
                }
            }
            out_tile_f &e_sf = epi_hold >= 0
                ? *reinterpret_cast<out_tile_f *>(&pf[epi_hold * pslot])
                : o_sf;
            out_tile *const e_sp = reinterpret_cast<out_tile *>(&e_sf);
            const int nacc = mt * nm;
            if (epi != EPI_BIAS) {
                epi_publish(e_sp[0], accum[0]);
                warpgroup::sync(1);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            }
            #pragma unroll
            for (int u = 0; u < ACCN; ++u) {
                if (u >= mt * nm) break;
                const int c = (mt > 1) ? ((nm > 1) ? (u / NM_MAX) : u) : 0;
                const int m = (nm > 1) ? (u % NM_MAX) : 0;
                const int base = (r6_lut ? (tile_id + c)%r6_output_tiles
                                            : (tile_id + c)) * OUT_ATOM;
                const int grow0 = (tok + m) * BATCH_PAD;
                if (epi == EPI_BIAS) {
                    const int bw = 16 * warpgroup::warpid()
                                 + (::kittens::laneid() >> 2);
                    const float b0 = __bfloat162float(bias[base + bw]);
                    const float b1 = __bfloat162float(bias[base + bw + 8]);
                    #pragma unroll
                    for (int j = 0; j < rt_fl<16, BATCH_PAD>::width; ++j) {
                        accum[u].tiles[0][j].data[0].x += b0;
                        accum[u].tiles[0][j].data[0].y += b0;
                        accum[u].tiles[0][j].data[1].x += b1;
                        accum[u].tiles[0][j].data[1].y += b1;
                        accum[u].tiles[0][j].data[2].x += b0;
                        accum[u].tiles[0][j].data[2].y += b0;
                        accum[u].tiles[0][j].data[3].x += b1;
                        accum[u].tiles[0][j].data[3].y += b1;
                    }
                    epi_publish(e_sp[0], accum[u]);
                    warpgroup::sync(1);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    constexpr int OUT_OCTS = OUT_ATOM / 8;
                    for (int idx = warpgroup::laneid(); idx < OUT_OCTS * rows_valid;
                         idx += 128) {
                        const int row = idx >> 3, col = (idx & 7) << 3;
                        const size_t at = (size_t)(grow0 + row) * dest_stride + base + col;
                        *reinterpret_cast<bf16_oct *>(dest + at) =
                            *reinterpret_cast<const bf16_oct *>(
                                &e_sp[0][int2{row, col}]);
                    }
                    warpgroup::sync(1);
                    continue;
                }
                if (u + 1 < ACCN && u + 1 < nacc)
                    epi_publish(e_sp[(u + 1) & 1],
                                accum[u + 1 < ACCN ? u + 1 : 0]);
                const out_tile &e_sr = e_sp[u & 1];
                constexpr int OUT_OCTS = OUT_ATOM / 8;
                for (int idx = warpgroup::laneid(); idx < OUT_OCTS * rows_valid;
                     idx += 128) {
                    const int row = idx >> 3;
                    const int col = (idx & 7) << 3;
                    const int grow = grow0 + row;
                    const size_t at =
                        (size_t)grow * dest_stride + base + col;
                    const bf16_oct src = *reinterpret_cast<const bf16_oct *>(
                        &e_sr[int2{row, col}]);
                    if (epi == EPI_STORE) {
                        *reinterpret_cast<bf16_oct *>(dest + at) = src;
                        continue;
                    }
                    const float v0 = __bfloat162float(src.x0);
                    const float v1 = __bfloat162float(src.x1);
                    const float v2 = __bfloat162float(src.x2);
                    const float v3 = __bfloat162float(src.x3);
                    const float v4 = __bfloat162float(src.x4);
                    const float v5 = __bfloat162float(src.x5);
                    const float v6 = __bfloat162float(src.x6);
                    const float v7 = __bfloat162float(src.x7);
                    switch (epi) {
                        case EPI_STORE: break;
                        case EPI_SILU:
                            *reinterpret_cast<bf16_oct *>(dest + at) = bf16_oct{
                                __float2bfloat16(v0 / (1.f + __expf(-v0))),
                                __float2bfloat16(v1 / (1.f + __expf(-v1))),
                                __float2bfloat16(v2 / (1.f + __expf(-v2))),
                                __float2bfloat16(v3 / (1.f + __expf(-v3))),
                                __float2bfloat16(v4 / (1.f + __expf(-v4))),
                                __float2bfloat16(v5 / (1.f + __expf(-v5))),
                                __float2bfloat16(v6 / (1.f + __expf(-v6))),
                                __float2bfloat16(v7 / (1.f + __expf(-v7)))};
                            break;
                        case EPI_MOE_ACC: {
                            const int expert = (tile_id + c) / out_tiles_per_expert;
                            const int local = ((tile_id + c) % out_tiles_per_expert) * OUT_ATOM
                                              + col;
                            const int original_row=r6_map?r6_map[grow]:grow;
                            if(original_row<0) break;
                            const float wt = moe_w[original_row * BB_EXPERTS + expert];
                            if (wt != 0.f) {
                                const size_t dst =
                                    ((size_t)original_row * (BB_EXPERTS + 1) + expert) *
                                    D_MODEL + local;
                                *reinterpret_cast<float4 *>(moe_out + dst) = make_float4(
                                    __fmul_rn(wt, v0), __fmul_rn(wt, v1),
                                    __fmul_rn(wt, v2), __fmul_rn(wt, v3));
                                *reinterpret_cast<float4 *>(moe_out + dst + 4) = make_float4(
                                    __fmul_rn(wt, v4), __fmul_rn(wt, v5),
                                    __fmul_rn(wt, v6), __fmul_rn(wt, v7));
                            }
                            break;
                        }
                        case EPI_ACC_F32:
                            *reinterpret_cast<float4 *>(moe_out +
                                ((size_t)grow * (BB_EXPERTS + 1) + BB_EXPERTS) *
                                D_MODEL + base + col) = make_float4(v0, v1, v2, v3);
                            *reinterpret_cast<float4 *>(moe_out +
                                ((size_t)grow * (BB_EXPERTS + 1) + BB_EXPERTS) *
                                D_MODEL + base + col + 4) = make_float4(v4, v5, v6, v7);
                            break;
                        case EPI_BIAS: break;
                        case EPI_GELU:
                            *reinterpret_cast<bf16_oct *>(dest + at) = bf16_oct{
                                __float2bfloat16(gelu_tanh_fp32(v0)),
                                __float2bfloat16(gelu_tanh_fp32(v1)),
                                __float2bfloat16(gelu_tanh_fp32(v2)),
                                __float2bfloat16(gelu_tanh_fp32(v3)),
                                __float2bfloat16(gelu_tanh_fp32(v4)),
                                __float2bfloat16(gelu_tanh_fp32(v5)),
                                __float2bfloat16(gelu_tanh_fp32(v6)),
                                __float2bfloat16(gelu_tanh_fp32(v7))};
                            break;
                        case EPI_RESIDUAL: {
                            const bf16_oct prior =
                                *reinterpret_cast<const bf16_oct *>(residual + at);
                            *reinterpret_cast<bf16_oct *>(residual + at) = bf16_oct{
                                __float2bfloat16(__bfloat162float(prior.x0) + v0),
                                __float2bfloat16(__bfloat162float(prior.x1) + v1),
                                __float2bfloat16(__bfloat162float(prior.x2) + v2),
                                __float2bfloat16(__bfloat162float(prior.x3) + v3),
                                __float2bfloat16(__bfloat162float(prior.x4) + v4),
                                __float2bfloat16(__bfloat162float(prior.x5) + v5),
                                __float2bfloat16(__bfloat162float(prior.x6) + v6),
                                __float2bfloat16(__bfloat162float(prior.x7) + v7)};
                            break;
                        }
                    }
                }
                if (u + 1 < nacc) {
                    warpgroup::sync(1);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
            }
            if (epi != EPI_BIAS) warpgroup::sync(1);
            if (epi_hold >= 0 && warpgroup::laneid() == 0)
                arrive(drained[epi_hold], CONSUMER_WARPS);
        }
        (void)my_tiles;
    }
}
__global__ __launch_bounds__(THREADS, 1) void predictor_mk2(
    const __grid_constant__ Globals g, const Backbone *bb, int n_layers,
    int n_pos, int n_backbone_layers, const int *run_predicate = nullptr,
    const unsigned char *native_predictor_active_mask = nullptr
    MK2_PROGRESS_PARAMETER(native_progress_mask)) {
    if (!bb && run_predicate && *run_predicate == 0) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < BATCH * N_POS; i += gridDim.x * blockDim.x)
            g.sampled[i] = 0;
        return;
    }
    if (bb && n_backbone_layers > 0 && bb->launches && blockIdx.x == 0 &&
        threadIdx.x == 0) {
        atomicAdd(bb->launches, 1ull);
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        bb->launches[2] = t0;
    }
    extern __shared__ int __shm[];
    tma_swizzle_allocator alloc(&__shm[0]);
    weight_tile (&p_s)[STAGES][SLOTS] = alloc.allocate<weight_tile, STAGES, SLOTS>();
    out_tile_f (&o_sf) = alloc.allocate<out_tile_f>();
    out_tile &o_s = *reinterpret_cast<out_tile *>(&o_sf);
    bf16 *const route_stage = reinterpret_cast<bf16 *>(&o_sf);
    __shared__ semaphore filled[STAGES], drained[STAGES];
    __shared__ float logits[N_POS];
    __shared__ float ln_scratch[2 * NORM_SCRATCH_STRIDE];
    static_assert(STAGES * SLOTS * sizeof(weight_tile) >= sizeof(sampler_storage));
    sampler_storage &sort_storage = *reinterpret_cast<sampler_storage *>(&__shm[0]);
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            init_semaphore(filled[s], 0, 1);
            init_semaphore(drained[s], CONSUMER_WARPS, 0);
            arrive(drained[s], CONSUMER_WARPS);
        }
    }
    int gemm_iter = 0;
    int gemm_tile = 0;
    __syncthreads();
    const bool prefilling = (bb && bb->n_prefill_tokens > 0);
    if (g.native_embedding) {
        for (int i = blockIdx.x * THREADS + threadIdx.x;
             i < BATCH * D_MODEL; i += gridDim.x * THREADS) {
            bb->h[i] = g.native_embedding[i];
            bb->r[i] = __float2bfloat16(0.f);
        }
        grid_barrier(g.barrier);
    }
    const float *fold_src = nullptr, *fold_routes = nullptr;
    for (int layer = 0; !prefilling && layer < n_backbone_layers; ++layer) {
        bb_norm_residual(bb->h, bb->r, bb->norm_w[layer], bb->normed, ln_scratch,
                         BATCH, nullptr, -1, nullptr, nullptr, fold_src, fold_routes);
        fold_src = nullptr;
        fold_routes = nullptr;
        grid_barrier(g.barrier);
        const int mi = bb->mamba_of_layer[layer];
        if (mi >= 0) {
            if (BB_SKIP_MAMBA) {
                grid_barrier(g.barrier);
                grid_barrier(g.barrier);
                grid_barrier(g.barrier);
#if !MK2_FUSE_CONV
                grid_barrier(g.barrier);
#endif
            } else {
            const MambaLayer &L = bb->mamba[mi];
            gemm(L.in_proj, bb->normed_v, MB_IN_PROJ, D_MODEL,
                 bb->zxbcdt, MB_IN_PROJ, nullptr, EPI_STORE,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
#if !MK2_FUSE_CONV
            mamba_conv(L, bb->zxbcdt, bb->xbc_act, bb->slot_map
                       MK2_PROGRESS_ARGUMENT(native_progress_mask));
            grid_barrier(g.barrier);
#endif
            mamba_ssm(L, bb->zxbcdt, bb->xbc_act, bb->gated, bb->slot_map
                      MK2_PROGRESS_ARGUMENT(native_progress_mask));
            grid_barrier(g.barrier);
            gemm(L.out_proj, bb->gated_v, D_MODEL, MB_INNER,
                 bb->h, D_MODEL, nullptr, EPI_STORE,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
            }
        } else if (bb->attn_of_layer[layer] >= 0) {
            if (BB_SKIP_ATTN) {
                grid_barrier(g.barrier);
                grid_barrier(g.barrier);
                grid_barrier(g.barrier);
            } else {
            const AttnLayer &L = bb->attn[bb->attn_of_layer[layer]];
            gemm(L.qkv, bb->normed_v, BB_QKV, D_MODEL, bb->qkv, BB_QKV,
                 nullptr, EPI_STORE, p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
            bb_attention(L, bb->qkv, bb->gated, bb->past_len, bb->seq_len
                         MK2_PROGRESS_ARGUMENT(native_progress_mask));
            grid_barrier(g.barrier);
            gemm(L.o_proj, bb->gated_v, D_MODEL, D_MODEL, bb->h, D_MODEL,
                 nullptr, EPI_STORE, p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
            }
        } else if (BB_SKIP_MOE) {
            grid_barrier(g.barrier);
            grid_barrier(g.barrier);
            grid_barrier(g.barrier);
            grid_barrier(g.barrier);
            grid_barrier(g.barrier);
        } else {
            const int mo = bb->moe_of_layer[layer];
            const MoeLayer &L = bb->moe[mo];
            if (bb->moe_w_audit) {
                bb_moe_route(bb->normed, L.gate_w, L.gate_bias,
                             bb->moe_w_audit + (size_t)mo * BATCH * BB_EXPERTS,
                             BATCH, route_stage);
                grid_barrier(g.barrier);
            }
            if (bb->moe_w_inject) {
                for (int idx = blockIdx.x * THREADS + threadIdx.x;
                     idx < BATCH * BB_EXPERTS; idx += gridDim.x * THREADS)
                    bb->moe_w[idx] =
                        bb->moe_w_inject[(size_t)mo * BATCH * BB_EXPERTS + idx];
            } else {
                bb_moe_route(bb->normed, L.gate_w, L.gate_bias, bb->moe_w, BATCH,
                             route_stage);
            }
            if constexpr (MK2_ACTIVE_EXPERTS && BATCH == 1) {
                for (int idx = blockIdx.x * THREADS + threadIdx.x;
                     idx < BATCH * (BB_EXPERTS + 1) * D_MODEL; idx += gridDim.x * THREADS)
                    bb->moe_out[idx] = 0.f;
            }
            grid_barrier(g.barrier);
            r6_sparse_map(bb->moe_w,bb->r6_map,bb->r6_counts
                          MK2_PROGRESS_ARGUMENT(native_progress_mask));
            grid_barrier(g.barrier);
            r6_sparse_gather(bb->normed,bb->r6_packed,bb->r6_map,bb->r6_counts,
                             bb->r6_lut);
            grid_barrier(g.barrier);
            gemm(L.exp_up, bb->r6_views[0], BB_EXPERTS * BB_MOE_INTER, D_MODEL,
                 bb->moe_h, BB_MOE_INTER, nullptr, EPI_SILU,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, nullptr,
                 0,0,nullptr,nullptr,DECODE_TILES,DECODE_ROWS,
                 nullptr,0,bb->r6_lut,bb->r6_map);
            gemm(L.sh_up, bb->normed_v, BB_SHARED_INTER, D_MODEL,
                 bb->moe_sh, BB_SHARED_INTER, nullptr, EPI_SILU,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
            gemm(L.exp_down, bb->r6_views[1], BB_EXPERTS * D_MODEL, BB_MOE_INTER,
                 nullptr, D_MODEL, nullptr, EPI_MOE_ACC,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, nullptr,
                 D_MODEL / OUT_ATOM, BB_MOE_INTER / KCHUNK, bb->moe_w, bb->moe_out,
                 DECODE_TILES, DECODE_ROWS, nullptr,0,bb->r6_lut,bb->r6_map);
            gemm(L.sh_down, bb->moe_sh_v, D_MODEL, BB_SHARED_INTER,
                 nullptr, D_MODEL, nullptr, EPI_ACC_F32,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, nullptr,
                 0, 0, nullptr, bb->moe_out);
            grid_barrier(g.barrier);
            if (bb->layer_out) {
                for (int idx = blockIdx.x * THREADS + threadIdx.x;
                     idx < BATCH * D_MODEL; idx += gridDim.x * THREADS)
                    bb->h[idx] = bb_moe_fold_selected(bb->moe_out, bb->moe_w, idx);
                grid_barrier(g.barrier);
            } else {
                fold_src = bb->moe_out;
                fold_routes = bb->moe_w;
            }
        }
        if (bb->layer_out) {
            for (int row = blockIdx.x; row < BATCH; row += gridDim.x)
                for (int i = threadIdx.x; i < D_MODEL; i += THREADS)
                    bb->layer_out[((size_t)layer * BATCH + row) * D_MODEL + i] =
                        bb->h[(size_t)row * D_MODEL + i];
            grid_barrier(g.barrier);
        }
    }
    if (bb && bb->n_prefill_tokens > 0) {
        const int n_total = bb->n_prefill_tokens;
        const int slot = bb->prefill_slot;
        if (bb->native_prefill_reset) {
            constexpr int CV = MB_CONV_DIM * (MB_CONV_K - 1);
            constexpr int SS = MB_HEADS * MB_HEAD_DIM * MB_STATE;
            for (int i = blockIdx.x * THREADS + threadIdx.x; i < 14 * BATCH * CV;
                 i += gridDim.x * THREADS) {
                const int cell = i % CV;
                const int row_layer = i / CV;
                const int row = row_layer % BATCH;
                const MambaLayer &L = bb->mamba[row_layer / BATCH];
                L.conv_state[(size_t)row * L.conv_slot_stride + cell] = __float2bfloat16(0.f);
            }
            for (int i = blockIdx.x * THREADS + threadIdx.x; i < 14 * BATCH * SS;
                 i += gridDim.x * THREADS) {
                const int cell = i % SS;
                const int row_layer = i / SS;
                const int row = row_layer % BATCH;
                const MambaLayer &L = bb->mamba[row_layer / BATCH];
                L.ssm_state[(size_t)row * L.ssm_slot_stride + cell] = 0.f;
            }
            for (int row = blockIdx.x * THREADS + threadIdx.x; row < BATCH;
                 row += gridDim.x * THREADS)
                const_cast<int *>(bb->seq_len)[row] = n_total / BATCH;
            grid_barrier(g.barrier);
        }
        for (int idx = blockIdx.x * THREADS + threadIdx.x; idx < n_total * D_MODEL;
             idx += gridDim.x * THREADS) {
            bb->pf_h[idx] = bb->prompt[idx];
            bb->pf_r[idx] = __float2bfloat16(0.f);
        }
        grid_barrier(g.barrier);
        const int n_tok = (n_total + BATCH_PAD - 1) / BATCH_PAD;
        const int n_seq = bb->pf_n_seq > 0 ? bb->pf_n_seq : 1;
        for (int layer = 0; layer < n_backbone_layers; ++layer) {
        const int T = n_total;
        const int tok0 = 0;
        const size_t off = 0;
        {
            bb_norm_residual(bb->pf_h + off, bb->pf_r + off, bb->norm_w[layer],
                             bb->pf_normed, ln_scratch, T);
            grid_barrier(g.barrier);
            const int mi = bb->mamba_of_layer[layer];
            const int ai = bb->attn_of_layer[layer];
            if (mi >= 0 && PF_SKIP_MAMBA) {
                for (int i = 0; i < 5; ++i) grid_barrier(g.barrier);
            } else if (mi >= 0) {
                const MambaLayer &L = bb->mamba[mi];
                gemm(L.in_proj, bb->pf_normed_v, MB_IN_PROJ, D_MODEL,
                     bb->pf_zxbcdt, MB_IN_PROJ, nullptr, EPI_STORE,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile,
                     nullptr, 0, 0, nullptr, nullptr, n_tok, BATCH_PAD);
                grid_barrier(g.barrier);
                mamba_conv_prefill(L, bb->pf_zxbcdt, bb->pf_xbc_act, T, slot,
                                   n_seq, bb->pf_seq_slot);
                grid_barrier(g.barrier);
                mamba_ssm_prefill(L, bb->pf_zxbcdt, bb->pf_xbc_act, bb->pf_gated,
                                  T, slot, n_seq, bb->pf_seq_slot);
                grid_barrier(g.barrier);
                mamba_conv_prefill_state(L, bb->pf_zxbcdt, T, slot, n_seq,
                                         bb->pf_seq_slot);
                mamba_gated_norm(L, bb->pf_zxbcdt, bb->pf_gated, T);
                grid_barrier(g.barrier);
                gemm(L.out_proj, bb->pf_gated_v, D_MODEL, MB_INNER,
                     bb->pf_h + off, D_MODEL, nullptr, EPI_STORE,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile,
                     nullptr, 0, 0, nullptr, nullptr, n_tok, BATCH_PAD);
                grid_barrier(g.barrier);
            } else if (ai >= 0 && PF_SKIP_ATTN) {
                for (int i = 0; i < 4; ++i) grid_barrier(g.barrier);
            } else if (ai >= 0) {
                const AttnLayer &L = bb->attn[ai];
                gemm(L.qkv, bb->pf_normed_v, BB_QKV, D_MODEL,
                     bb->pf_qkv, BB_QKV, nullptr, EPI_STORE,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile,
                     nullptr, 0, 0, nullptr, nullptr, n_tok, BATCH_PAD);
                grid_barrier(g.barrier);
                bb_attention_prefill_kv(L, bb->pf_qkv, T, slot, tok0, n_seq,
                                        bb->pf_seq_slot);
                grid_barrier(g.barrier);
                bb_attention_prefill(L, bb->pf_qkv, bb->pf_gated, T, slot, tok0,
                                     n_seq, bb->pf_seq_slot);
                grid_barrier(g.barrier);
                gemm(L.o_proj, bb->pf_gated_v, D_MODEL, D_MODEL,
                     bb->pf_h + off, D_MODEL, nullptr, EPI_STORE,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile,
                     nullptr, 0, 0, nullptr, nullptr, n_tok, BATCH_PAD);
                grid_barrier(g.barrier);
            } else if (PF_SKIP_MOE) {
                for (int i = 0; i < 5; ++i) grid_barrier(g.barrier);
            } else {
                const MoeLayer &L = bb->moe[bb->moe_of_layer[layer]];
                const int mo_audit = bb->moe_of_layer[layer];
                if (bb->moe_w_audit) {
                    bb_moe_route(bb->pf_normed, L.gate_w, L.gate_bias,
                                 bb->moe_w_audit +
                                     ((size_t)mo_audit * PREFILL_MAX_TOKENS + tok0) *
                                         BB_EXPERTS,
                                 T, route_stage);
                    grid_barrier(g.barrier);
                }
                if (bb->moe_w_inject) {
                    const int mo = bb->moe_of_layer[layer];
                    for (int idx = blockIdx.x * THREADS + threadIdx.x;
                         idx < T * BB_EXPERTS; idx += gridDim.x * THREADS) {
                        const int tk = idx / BB_EXPERTS;
                        const int e = idx - tk * BB_EXPERTS;
                        bb->pf_moe_w[idx] = bb->moe_w_inject[
                            ((size_t)mo * bb->n_prefill_tokens + tok0 + tk) *
                            BB_EXPERTS + e];
                    }
                } else {
                    bb_moe_route(bb->pf_normed, L.gate_w, L.gate_bias, bb->pf_moe_w, T,
                                 route_stage);
                }
                for (int idx = blockIdx.x * THREADS + threadIdx.x; idx < T * (BB_EXPERTS + 1) * D_MODEL;
                     idx += gridDim.x * THREADS)
                    bb->pf_moe_out[idx] = 0.f;
                grid_barrier(g.barrier);
                gemm(L.exp_up, bb->pf_normed_v, BB_EXPERTS * BB_MOE_INTER, D_MODEL,
                     bb->pf_moe_h, BB_EXPERTS * BB_MOE_INTER, nullptr, EPI_SILU,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile,
                     nullptr, 0, 0, nullptr, nullptr, n_tok, BATCH_PAD);
                gemm(L.sh_up, bb->pf_normed_v, BB_SHARED_INTER, D_MODEL,
                     bb->pf_moe_sh, BB_SHARED_INTER, nullptr, EPI_SILU,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile,
                     nullptr, 0, 0, nullptr, nullptr, n_tok, BATCH_PAD);
                grid_barrier(g.barrier);
                gemm(L.exp_down, bb->pf_moe_h_v, BB_EXPERTS * D_MODEL, BB_MOE_INTER,
                     nullptr, D_MODEL, nullptr, EPI_MOE_ACC,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, nullptr,
                     D_MODEL / OUT_ATOM, BB_MOE_INTER / KCHUNK,
                     bb->pf_moe_w, bb->pf_moe_out, n_tok, BATCH_PAD);
                grid_barrier(g.barrier);
                gemm(L.sh_down, bb->pf_moe_sh_v, D_MODEL, BB_SHARED_INTER,
                     nullptr, D_MODEL, nullptr, EPI_ACC_F32,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, nullptr,
                     0, 0, nullptr, bb->pf_moe_out, n_tok, BATCH_PAD);
                grid_barrier(g.barrier);
                for (int idx = blockIdx.x * THREADS + threadIdx.x; idx < T * D_MODEL;
                     idx += gridDim.x * THREADS)
                    bb->pf_h[off + idx] = bb_moe_fold_deterministic(bb->pf_moe_out, idx);
                grid_barrier(g.barrier);
            }
            if (bb->pf_layer_out && tok0 + T <= PREFILL_MAX_TOKENS) {
                for (int idx = blockIdx.x * THREADS + threadIdx.x; idx < T * D_MODEL;
                     idx += gridDim.x * THREADS)
                    bb->pf_layer_out[(size_t)layer * PREFILL_MAX_TOKENS * D_MODEL +
                                     off + idx] = bb->pf_h[off + idx];
                grid_barrier(g.barrier);
            }
        }
        }
        bb_norm_residual(bb->pf_h, bb->pf_r, bb->norm_f_w, bb->pf_final, ln_scratch,
                         n_total);
        grid_barrier(g.barrier);
        if (!bb->pf_fuse_decode) return;
        {
            const int n_pos = n_total / n_seq;
            const size_t last = (size_t)(n_pos - 1) * n_seq;
            for (int idx = blockIdx.x * THREADS + threadIdx.x;
                 idx < BATCH * D_MODEL; idx += gridDim.x * THREADS) {
                const int row = idx / D_MODEL;
                const int i = idx - row * D_MODEL;
                const size_t src = (last + (row < n_seq ? row : n_seq - 1)) * D_MODEL + i;
                bb->h[idx] = bb->pf_h[src];
                bb->r[idx] = bb->pf_r[src];
            }
            grid_barrier(g.barrier);
        }
    }
    if (n_backbone_layers > 0 && bb->launches && blockIdx.x == 0 &&
        threadIdx.x == 0)
        atomicAdd(bb->launches + 1, 1ull);
    if (n_backbone_layers > 0) {
        bb_norm_residual(bb->h, bb->r, bb->norm_f_w, g.x, ln_scratch,
                         BATCH, nullptr, -1, nullptr, nullptr, fold_src, fold_routes);
        grid_barrier(g.barrier);
        if (bb->phoneme_out) {
            gemm(bb->phoneme_w, g.x_v, BB_PHON_PADDED, D_MODEL,
                 bb->phoneme_logits, BB_PHON_PADDED, nullptr, EPI_BIAS,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, bb->phoneme_b);
            grid_barrier(g.barrier);
            const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
            for (int row = blockIdx.x + warp * gridDim.x; row < BATCH;
                 row += gridDim.x * WARPS) {
                const bf16 *lg = bb->phoneme_logits + (size_t)row * BB_PHON_PADDED;
                MK2_PROGRESS_GUARD(native_progress_mask, row);
                const int arg = phoneme_warp_argmax(lg);
                if (lane == 0) bb->phoneme_out[row] = arg;
            }
            grid_barrier(g.barrier);
        }
        if (bb->dec_snapshot) {
            for (int idx = blockIdx.x * THREADS + threadIdx.x;
                 idx < BATCH * D_MODEL; idx += gridDim.x * THREADS)
                bb->dec_snapshot[idx] = g.x[idx];
            grid_barrier(g.barrier);
        }
    }
    __shared__ int native_run_predictor;
    if (native_predictor_active_mask && bb && g.native_codes) {
        if (threadIdx.x == 0) {
            int any_active = 0;
            for (int row = 0; row < BATCH; ++row)
                any_active |= int(native_predictor_active_mask[row] != 0 &&
                                  MK2_PROGRESS_ENABLED(native_progress_mask, row));
            native_run_predictor = any_active;
        }
        __syncthreads();
    }
    const bool skip_native_predictor = native_predictor_active_mask && bb &&
        g.native_codes && native_run_predictor == 0;
    if (skip_native_predictor) {
        for (int index = blockIdx.x * THREADS + threadIdx.x;
             index < BATCH * N_POS; index += gridDim.x * THREADS) {
            MK2_PROGRESS_GUARD(native_progress_mask, index / N_POS);
            g.sampled[index] = 0;
            g.native_codes[index] = 0;
        }
        grid_barrier(g.barrier);
    } else {
    for (int row = blockIdx.x; row < BATCH; row += gridDim.x) {
        for (int i = threadIdx.x; i < D_MODEL; i += THREADS) {
            const size_t at = (size_t)row * D_MODEL + i;
            g.x[at] = __float2bfloat16(__bfloat162float(g.x[at]) +
                                       __bfloat162float(g.pos_emb[i]));
        }
    }
    if constexpr (!(MK2_B1_LOCAL_BARRIERS && BATCH == 1))
        grid_barrier(g.barrier);
    for (int pos = 0; pos < n_pos; ++pos) {
        for (int layer = 0; layer < n_layers; ++layer) {
#ifndef ABLATE_NORM
            layernorm_rows(g.x, g.norm_self[layer], g.normed, ln_scratch);
#endif
            grid_barrier(g.barrier);
            gemm(g.qkv[layer], g.normed_v, 3 * D_MODEL, D_MODEL,
                 g.qkv_out, 3 * D_MODEL, nullptr, EPI_STORE,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
            const int pairs = BATCH * N_HEADS;
#ifndef ABLATE_ATTN
            {
                const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
                const int warps_total = gridDim.x * WARPS;
                const float scale = rsqrtf((float)D_HEAD);
                for (int pair = blockIdx.x * WARPS + warp; pair < pairs;
                     pair += warps_total) {
                    const int row = pair / N_HEADS;
                    const int head = pair % N_HEADS;
                    const bf16 *qkv_row = g.qkv_out + (size_t)row * 3 * D_MODEL;
                    const bf16 *q = qkv_row + head * D_HEAD;
                    const bf16 *k_new = qkv_row + D_MODEL + head * D_HEAD;
                    const bf16 *v_new = qkv_row + 2 * D_MODEL + head * D_HEAD;
                    const size_t cache_base =
                        (((size_t)layer * BATCH + row) * N_HEADS + head) * N_POS *
                        D_HEAD;
                    constexpr int DPL = D_HEAD / 32;
                    float qv[DPL], kn[DPL], vn[DPL];
                    #pragma unroll
                    for (int j = 0; j < DPL; ++j) {
                        const int i = lane + j * 32;
                        qv[j] = __bfloat162float(q[i]);
                        kn[j] = __bfloat162float(k_new[i]);
                        vn[j] = __bfloat162float(v_new[i]);
                    }
                    #pragma unroll
                    for (int j = 0; j < DPL; ++j) {
                        const int i = lane + j * 32;
                        g.kcache[cache_base + pos * D_HEAD + i] = k_new[i];
                        g.vcache[cache_base + pos * D_HEAD + i] = v_new[i];
                    }
                    __syncwarp();
                    float sc[N_POS];
                    #pragma unroll 4
                    for (int t = 0; t < pos; ++t) {
                        const bf16 *kt = g.kcache + cache_base + t * D_HEAD;
                        float acc = 0.f;
                        #pragma unroll
                        for (int j = 0; j < D_HEAD / 32; ++j) {
                            const int i = lane + j * 32;
                            acc += qv[j] * __bfloat162float(kt[i]);
                        }
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1)
                            acc += __shfl_xor_sync(0xffffffff, acc, off);
                        sc[t] = acc * scale;
                    }
                    {
                        float acc = 0.f;
                        #pragma unroll
                        for (int j = 0; j < DPL; ++j) acc += qv[j] * kn[j];
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1)
                            acc += __shfl_xor_sync(0xffffffff, acc, off);
                        sc[pos] = acc * scale;
                    }
                    float m = sc[0];
                    for (int t = 1; t <= pos; ++t) m = fmaxf(m, sc[t]);
                    float s = 0.f;
                    for (int t = 0; t <= pos; ++t) {
                        sc[t] = __expf(sc[t] - m);
                        s += sc[t];
                    }
                    const float inv = 1.f / s;
                    for (int t = 0; t <= pos; ++t) sc[t] *= inv;
                    float ov[DPL] = {};
                    #pragma unroll 4
                    for (int t = 0; t < pos; ++t) {
                        const bf16 *vt = g.vcache + cache_base + t * D_HEAD;
                        const float s = sc[t];
                        #pragma unroll
                        for (int j = 0; j < DPL; ++j)
                            ov[j] += s * __bfloat162float(vt[lane + j * 32]);
                    }
                    #pragma unroll
                    for (int j = 0; j < DPL; ++j) {
                        ov[j] += sc[pos] * vn[j];
                        g.attn_out[(size_t)row * D_MODEL + head * D_HEAD +
                                   lane + j * 32] = __float2bfloat16(ov[j]);
                    }
                }
            }
#endif
            grid_barrier(g.barrier);
            gemm(g.o[layer], g.attn_v, D_MODEL, D_MODEL,
                 nullptr, D_MODEL, g.x, EPI_RESIDUAL,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
#ifndef ABLATE_NORM
            layernorm_rows(g.x, g.norm_ff[layer], g.normed, ln_scratch);
#endif
            grid_barrier(g.barrier);
            gemm(g.ff_in[layer], g.normed_v, D_FFN, D_MODEL,
                 g.ff_hidden, D_FFN, nullptr, EPI_GELU,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
            gemm(g.ff_out[layer], g.ffh_v, D_MODEL, D_FFN,
                 nullptr, D_MODEL, g.x, EPI_RESIDUAL,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
        }
        if constexpr (!MK2_SKIP_HIDDEN_SNAPSHOT)
        for (int row = blockIdx.x; row < BATCH; row += gridDim.x) {
            for (int i = threadIdx.x; i < D_MODEL; i += THREADS)
                g.hidden_rows[((size_t)pos * BATCH + row) * D_MODEL + i] =
                    g.x[(size_t)row * D_MODEL + i];
        }
#ifdef ABLATE_HEAD
        (void)0;
#elif defined(HEAD_BF16)
        gemm(g.head[pos], g.x_v, VOCAB_PADDED, D_MODEL,
             g.logits, VOCAB_PADDED, nullptr, EPI_STORE,
             p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, nullptr);
#else
        gemm(g.head[pos], g.x_v, VOCAB_PADDED, D_MODEL,
             g.logits, VOCAB_PADDED, nullptr, EPI_BIAS,
             p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, g.head_bias[pos]);
#endif
        grid_barrier(g.barrier);
#ifndef ABLATE_SAMPLE
        sample_codes(g.logits, g.head_bias[pos], g.forbidden, g.gumbel, pos,
                     g.sampled_pos, sort_storage, g.temperature);
#else
        if (threadIdx.x < BATCH && blockIdx.x == 0)
            g.sampled_pos[threadIdx.x] = g.codes[threadIdx.x * N_POS + pos];
#endif
#ifdef ABLATE_SAMPLE
        if constexpr (MK2_B1_LOCAL_BARRIERS && BATCH == 1) {
            __syncthreads();
        } else {
            grid_barrier(g.barrier);
        }
#endif
        if (pos + 1 < n_pos) {
            for (int row = blockIdx.x; row < BATCH; row += gridDim.x) {
                const int code = g.sampled_pos[row];
                const bf16 *emb = g.audio_emb[pos] + (size_t)code * D_MODEL;
                const bf16 *pe = g.pos_emb + (pos + 1) * D_MODEL;
                for (int i = threadIdx.x; i < D_MODEL; i += THREADS)
                    g.x[(size_t)row * D_MODEL + i] =
                        __float2bfloat16(__bfloat162float(emb[i]) +
                                         __bfloat162float(pe[i]));
            }
        }
        if (threadIdx.x == 0) {
            for (int row = blockIdx.x; row < BATCH; row += gridDim.x) {
                const int code = g.sampled_pos[row];
                g.sampled[row * N_POS + pos] = code;
                if (g.native_codes && MK2_PROGRESS_ENABLED(native_progress_mask, row)) {
#if MK2_INFLIGHT
                    g.native_codes[row * N_POS + pos] =
                        native_predictor_active_mask && !native_predictor_active_mask[row]
                            ? 0 : (long long)code;
#else
                    g.native_codes[row * N_POS + pos] = (long long)code;
#endif
                }
            }
        }
#ifdef ABLATE_SAMPLE
        grid_barrier(g.barrier);
#else
        if (pos + 1 == n_pos) grid_barrier(g.barrier);
#endif
    }
    }
    if (g.native_seq_len && blockIdx.x == 0)
        for (int row = threadIdx.x; row < BATCH; row += blockDim.x) {
            MK2_PROGRESS_GUARD(native_progress_mask, row);
            g.native_seq_len[row] += 1;
        }
    if (bb && n_backbone_layers > 0 && bb->launches && blockIdx.x == 0 &&
        threadIdx.x == 0) {
        unsigned long long t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        const unsigned long long t0 = bb->launches[2];
        if (t1 > t0) {
            const unsigned long long dt = t1 - t0;
            atomicAdd(bb->launches + 3, dt);
            atomicAdd(bb->launches + 4, 1ull);
            atomicMax(bb->launches + 5, dt);
        }
    }
}
__global__ __launch_bounds__(THREADS, 1) void predictor_mk2_startup(
    const __grid_constant__ Globals g, const Backbone *bb, int n_layers,
    int n_pos, int n_backbone_layers, const int *run_predicate = nullptr,
    const unsigned char *native_predictor_active_mask = nullptr
    MK2_PROGRESS_PARAMETER(native_progress_mask)) {
    if (!bb && run_predicate && *run_predicate == 0) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < BATCH * N_POS; i += gridDim.x * blockDim.x)
            g.sampled[i] = 0;
        return;
    }
    if (bb && n_backbone_layers > 0 && bb->launches && blockIdx.x == 0 &&
        threadIdx.x == 0) {
        atomicAdd(bb->launches, 1ull);
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        bb->launches[2] = t0;
    }
    extern __shared__ int __shm[];
    tma_swizzle_allocator alloc(&__shm[0]);
    weight_tile (&p_s)[STAGES][SLOTS] = alloc.allocate<weight_tile, STAGES, SLOTS>();
    out_tile_f (&o_sf) = alloc.allocate<out_tile_f>();
    out_tile &o_s = *reinterpret_cast<out_tile *>(&o_sf);
    bf16 *const route_stage = reinterpret_cast<bf16 *>(&o_sf);
    __shared__ semaphore filled[STAGES], drained[STAGES];
    __shared__ float logits[N_POS];
    __shared__ float ln_scratch[2 * NORM_SCRATCH_STRIDE];
    static_assert(STAGES * SLOTS * sizeof(weight_tile) >= sizeof(sampler_storage));
    sampler_storage &sort_storage = *reinterpret_cast<sampler_storage *>(&__shm[0]);
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            init_semaphore(filled[s], 0, 1);
            init_semaphore(drained[s], CONSUMER_WARPS, 0);
            arrive(drained[s], CONSUMER_WARPS);
        }
    }
    int gemm_iter = 0;
    int gemm_tile = 0;
    __syncthreads();
    constexpr bool prefilling = false;
    if (g.native_embedding) {
        for (int i = blockIdx.x * THREADS + threadIdx.x;
             i < BATCH * D_MODEL; i += gridDim.x * THREADS) {
            bb->h[i] = g.native_embedding[i];
            bb->r[i] = __float2bfloat16(0.f);
        }
        grid_barrier(g.barrier);
    }
    const float *fold_src = nullptr, *fold_routes = nullptr;
    for (int layer = 0; !prefilling && layer < n_backbone_layers; ++layer) {
        bb_norm_residual(bb->h, bb->r, bb->norm_w[layer], bb->normed, ln_scratch,
                         BATCH, nullptr, -1, nullptr, nullptr, fold_src, fold_routes);
        fold_src = nullptr;
        fold_routes = nullptr;
        grid_barrier(g.barrier);
        const int mi = bb->mamba_of_layer[layer];
        if (mi >= 0) {
            if (BB_SKIP_MAMBA) {
                grid_barrier(g.barrier);
                grid_barrier(g.barrier);
                grid_barrier(g.barrier);
#if !MK2_FUSE_CONV
                grid_barrier(g.barrier);
#endif
            } else {
            const MambaLayer &L = bb->mamba[mi];
            gemm(L.in_proj, bb->normed_v, MB_IN_PROJ, D_MODEL,
                 bb->zxbcdt, MB_IN_PROJ, nullptr, EPI_STORE,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
#if !MK2_FUSE_CONV
            mamba_conv(L, bb->zxbcdt, bb->xbc_act, bb->slot_map
                       MK2_PROGRESS_ARGUMENT(native_progress_mask));
            grid_barrier(g.barrier);
#endif
            mamba_ssm(L, bb->zxbcdt, bb->xbc_act, bb->gated, bb->slot_map
                      MK2_PROGRESS_ARGUMENT(native_progress_mask));
            grid_barrier(g.barrier);
            gemm(L.out_proj, bb->gated_v, D_MODEL, MB_INNER,
                 bb->h, D_MODEL, nullptr, EPI_STORE,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
            }
        } else if (bb->attn_of_layer[layer] >= 0) {
            if (BB_SKIP_ATTN) {
                grid_barrier(g.barrier);
                grid_barrier(g.barrier);
                grid_barrier(g.barrier);
            } else {
            const AttnLayer &L = bb->attn[bb->attn_of_layer[layer]];
            gemm(L.qkv, bb->normed_v, BB_QKV, D_MODEL, bb->qkv, BB_QKV,
                 nullptr, EPI_STORE, p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
            bb_attention(L, bb->qkv, bb->gated, bb->past_len, bb->seq_len
                         MK2_PROGRESS_ARGUMENT(native_progress_mask));
            grid_barrier(g.barrier);
            gemm(L.o_proj, bb->gated_v, D_MODEL, D_MODEL, bb->h, D_MODEL,
                 nullptr, EPI_STORE, p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
            }
        } else if (BB_SKIP_MOE) {
            grid_barrier(g.barrier);
            grid_barrier(g.barrier);
            grid_barrier(g.barrier);
            grid_barrier(g.barrier);
            grid_barrier(g.barrier);
        } else {
            const int mo = bb->moe_of_layer[layer];
            const MoeLayer &L = bb->moe[mo];
            if (bb->moe_w_audit) {
                bb_moe_route(bb->normed, L.gate_w, L.gate_bias,
                             bb->moe_w_audit + (size_t)mo * BATCH * BB_EXPERTS,
                             BATCH, route_stage);
                grid_barrier(g.barrier);
            }
            if (bb->moe_w_inject) {
                for (int idx = blockIdx.x * THREADS + threadIdx.x;
                     idx < BATCH * BB_EXPERTS; idx += gridDim.x * THREADS)
                    bb->moe_w[idx] =
                        bb->moe_w_inject[(size_t)mo * BATCH * BB_EXPERTS + idx];
            } else {
                bb_moe_route(bb->normed, L.gate_w, L.gate_bias, bb->moe_w, BATCH,
                             route_stage);
            }
            if constexpr (MK2_ACTIVE_EXPERTS && BATCH == 1) {
                for (int idx = blockIdx.x * THREADS + threadIdx.x;
                     idx < BATCH * (BB_EXPERTS + 1) * D_MODEL; idx += gridDim.x * THREADS)
                    bb->moe_out[idx] = 0.f;
            }
            grid_barrier(g.barrier);
            r6_sparse_map(bb->moe_w,bb->r6_map,bb->r6_counts
                          MK2_PROGRESS_ARGUMENT(native_progress_mask));
            grid_barrier(g.barrier);
            r6_sparse_gather(bb->normed,bb->r6_packed,bb->r6_map,bb->r6_counts,
                             bb->r6_lut);
            grid_barrier(g.barrier);
            gemm(L.exp_up, bb->r6_views[0], BB_EXPERTS * BB_MOE_INTER, D_MODEL,
                 bb->moe_h, BB_MOE_INTER, nullptr, EPI_SILU,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, nullptr,
                 0,0,nullptr,nullptr,DECODE_TILES,DECODE_ROWS,
                 nullptr,0,bb->r6_lut,bb->r6_map);
            gemm(L.sh_up, bb->normed_v, BB_SHARED_INTER, D_MODEL,
                 bb->moe_sh, BB_SHARED_INTER, nullptr, EPI_SILU,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
            gemm(L.exp_down, bb->r6_views[1], BB_EXPERTS * D_MODEL, BB_MOE_INTER,
                 nullptr, D_MODEL, nullptr, EPI_MOE_ACC,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, nullptr,
                 D_MODEL / OUT_ATOM, BB_MOE_INTER / KCHUNK, bb->moe_w, bb->moe_out,
                 DECODE_TILES, DECODE_ROWS, nullptr,0,bb->r6_lut,bb->r6_map);
            gemm(L.sh_down, bb->moe_sh_v, D_MODEL, BB_SHARED_INTER,
                 nullptr, D_MODEL, nullptr, EPI_ACC_F32,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, nullptr,
                 0, 0, nullptr, bb->moe_out);
            grid_barrier(g.barrier);
            if (bb->layer_out) {
                for (int idx = blockIdx.x * THREADS + threadIdx.x;
                     idx < BATCH * D_MODEL; idx += gridDim.x * THREADS)
                    bb->h[idx] = bb_moe_fold_selected(bb->moe_out, bb->moe_w, idx);
                grid_barrier(g.barrier);
            } else {
                fold_src = bb->moe_out;
                fold_routes = bb->moe_w;
            }
        }
        if (bb->layer_out) {
            for (int row = blockIdx.x; row < BATCH; row += gridDim.x)
                for (int i = threadIdx.x; i < D_MODEL; i += THREADS)
                    bb->layer_out[((size_t)layer * BATCH + row) * D_MODEL + i] =
                        bb->h[(size_t)row * D_MODEL + i];
            grid_barrier(g.barrier);
        }
    }
    if constexpr (false) {
        const int n_total = bb->n_prefill_tokens;
        const int slot = bb->prefill_slot;
        if (bb->native_prefill_reset) {
            constexpr int CV = MB_CONV_DIM * (MB_CONV_K - 1);
            constexpr int SS = MB_HEADS * MB_HEAD_DIM * MB_STATE;
            for (int i = blockIdx.x * THREADS + threadIdx.x; i < 14 * BATCH * CV;
                 i += gridDim.x * THREADS) {
                const int cell = i % CV;
                const int row_layer = i / CV;
                const int row = row_layer % BATCH;
                const MambaLayer &L = bb->mamba[row_layer / BATCH];
                L.conv_state[(size_t)row * L.conv_slot_stride + cell] = __float2bfloat16(0.f);
            }
            for (int i = blockIdx.x * THREADS + threadIdx.x; i < 14 * BATCH * SS;
                 i += gridDim.x * THREADS) {
                const int cell = i % SS;
                const int row_layer = i / SS;
                const int row = row_layer % BATCH;
                const MambaLayer &L = bb->mamba[row_layer / BATCH];
                L.ssm_state[(size_t)row * L.ssm_slot_stride + cell] = 0.f;
            }
            for (int row = blockIdx.x * THREADS + threadIdx.x; row < BATCH;
                 row += gridDim.x * THREADS)
                const_cast<int *>(bb->seq_len)[row] = n_total / BATCH;
            grid_barrier(g.barrier);
        }
        for (int idx = blockIdx.x * THREADS + threadIdx.x; idx < n_total * D_MODEL;
             idx += gridDim.x * THREADS) {
            bb->pf_h[idx] = bb->prompt[idx];
            bb->pf_r[idx] = __float2bfloat16(0.f);
        }
        grid_barrier(g.barrier);
        const int n_tok = (n_total + BATCH_PAD - 1) / BATCH_PAD;
        const int n_seq = bb->pf_n_seq > 0 ? bb->pf_n_seq : 1;
        for (int layer = 0; layer < n_backbone_layers; ++layer) {
        const int T = n_total;
        const int tok0 = 0;
        const size_t off = 0;
        {
            bb_norm_residual(bb->pf_h + off, bb->pf_r + off, bb->norm_w[layer],
                             bb->pf_normed, ln_scratch, T);
            grid_barrier(g.barrier);
            const int mi = bb->mamba_of_layer[layer];
            const int ai = bb->attn_of_layer[layer];
            if (mi >= 0 && PF_SKIP_MAMBA) {
                for (int i = 0; i < 5; ++i) grid_barrier(g.barrier);
            } else if (mi >= 0) {
                const MambaLayer &L = bb->mamba[mi];
                gemm(L.in_proj, bb->pf_normed_v, MB_IN_PROJ, D_MODEL,
                     bb->pf_zxbcdt, MB_IN_PROJ, nullptr, EPI_STORE,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile,
                     nullptr, 0, 0, nullptr, nullptr, n_tok, BATCH_PAD);
                grid_barrier(g.barrier);
                mamba_conv_prefill(L, bb->pf_zxbcdt, bb->pf_xbc_act, T, slot,
                                   n_seq, bb->pf_seq_slot);
                grid_barrier(g.barrier);
                mamba_ssm_prefill(L, bb->pf_zxbcdt, bb->pf_xbc_act, bb->pf_gated,
                                  T, slot, n_seq, bb->pf_seq_slot);
                grid_barrier(g.barrier);
                mamba_conv_prefill_state(L, bb->pf_zxbcdt, T, slot, n_seq,
                                         bb->pf_seq_slot);
                mamba_gated_norm(L, bb->pf_zxbcdt, bb->pf_gated, T);
                grid_barrier(g.barrier);
                gemm(L.out_proj, bb->pf_gated_v, D_MODEL, MB_INNER,
                     bb->pf_h + off, D_MODEL, nullptr, EPI_STORE,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile,
                     nullptr, 0, 0, nullptr, nullptr, n_tok, BATCH_PAD);
                grid_barrier(g.barrier);
            } else if (ai >= 0 && PF_SKIP_ATTN) {
                for (int i = 0; i < 4; ++i) grid_barrier(g.barrier);
            } else if (ai >= 0) {
                const AttnLayer &L = bb->attn[ai];
                gemm(L.qkv, bb->pf_normed_v, BB_QKV, D_MODEL,
                     bb->pf_qkv, BB_QKV, nullptr, EPI_STORE,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile,
                     nullptr, 0, 0, nullptr, nullptr, n_tok, BATCH_PAD);
                grid_barrier(g.barrier);
                bb_attention_prefill_kv(L, bb->pf_qkv, T, slot, tok0, n_seq,
                                        bb->pf_seq_slot);
                grid_barrier(g.barrier);
                bb_attention_prefill(L, bb->pf_qkv, bb->pf_gated, T, slot, tok0,
                                     n_seq, bb->pf_seq_slot);
                grid_barrier(g.barrier);
                gemm(L.o_proj, bb->pf_gated_v, D_MODEL, D_MODEL,
                     bb->pf_h + off, D_MODEL, nullptr, EPI_STORE,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile,
                     nullptr, 0, 0, nullptr, nullptr, n_tok, BATCH_PAD);
                grid_barrier(g.barrier);
            } else if (PF_SKIP_MOE) {
                for (int i = 0; i < 5; ++i) grid_barrier(g.barrier);
            } else {
                const MoeLayer &L = bb->moe[bb->moe_of_layer[layer]];
                const int mo_audit = bb->moe_of_layer[layer];
                if (bb->moe_w_audit) {
                    bb_moe_route(bb->pf_normed, L.gate_w, L.gate_bias,
                                 bb->moe_w_audit +
                                     ((size_t)mo_audit * PREFILL_MAX_TOKENS + tok0) *
                                         BB_EXPERTS,
                                 T, route_stage);
                    grid_barrier(g.barrier);
                }
                if (bb->moe_w_inject) {
                    const int mo = bb->moe_of_layer[layer];
                    for (int idx = blockIdx.x * THREADS + threadIdx.x;
                         idx < T * BB_EXPERTS; idx += gridDim.x * THREADS) {
                        const int tk = idx / BB_EXPERTS;
                        const int e = idx - tk * BB_EXPERTS;
                        bb->pf_moe_w[idx] = bb->moe_w_inject[
                            ((size_t)mo * bb->n_prefill_tokens + tok0 + tk) *
                            BB_EXPERTS + e];
                    }
                } else {
                    bb_moe_route(bb->pf_normed, L.gate_w, L.gate_bias, bb->pf_moe_w, T,
                                 route_stage);
                }
                for (int idx = blockIdx.x * THREADS + threadIdx.x; idx < T * (BB_EXPERTS + 1) * D_MODEL;
                     idx += gridDim.x * THREADS)
                    bb->pf_moe_out[idx] = 0.f;
                grid_barrier(g.barrier);
                gemm(L.exp_up, bb->pf_normed_v, BB_EXPERTS * BB_MOE_INTER, D_MODEL,
                     bb->pf_moe_h, BB_EXPERTS * BB_MOE_INTER, nullptr, EPI_SILU,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile,
                     nullptr, 0, 0, nullptr, nullptr, n_tok, BATCH_PAD);
                gemm(L.sh_up, bb->pf_normed_v, BB_SHARED_INTER, D_MODEL,
                     bb->pf_moe_sh, BB_SHARED_INTER, nullptr, EPI_SILU,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile,
                     nullptr, 0, 0, nullptr, nullptr, n_tok, BATCH_PAD);
                grid_barrier(g.barrier);
                gemm(L.exp_down, bb->pf_moe_h_v, BB_EXPERTS * D_MODEL, BB_MOE_INTER,
                     nullptr, D_MODEL, nullptr, EPI_MOE_ACC,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, nullptr,
                     D_MODEL / OUT_ATOM, BB_MOE_INTER / KCHUNK,
                     bb->pf_moe_w, bb->pf_moe_out, n_tok, BATCH_PAD);
                grid_barrier(g.barrier);
                gemm(L.sh_down, bb->pf_moe_sh_v, D_MODEL, BB_SHARED_INTER,
                     nullptr, D_MODEL, nullptr, EPI_ACC_F32,
                     p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, nullptr,
                     0, 0, nullptr, bb->pf_moe_out, n_tok, BATCH_PAD);
                grid_barrier(g.barrier);
                for (int idx = blockIdx.x * THREADS + threadIdx.x; idx < T * D_MODEL;
                     idx += gridDim.x * THREADS)
                    bb->pf_h[off + idx] = bb_moe_fold_deterministic(bb->pf_moe_out, idx);
                grid_barrier(g.barrier);
            }
            if (bb->pf_layer_out && tok0 + T <= PREFILL_MAX_TOKENS) {
                for (int idx = blockIdx.x * THREADS + threadIdx.x; idx < T * D_MODEL;
                     idx += gridDim.x * THREADS)
                    bb->pf_layer_out[(size_t)layer * PREFILL_MAX_TOKENS * D_MODEL +
                                     off + idx] = bb->pf_h[off + idx];
                grid_barrier(g.barrier);
            }
        }
        }
        bb_norm_residual(bb->pf_h, bb->pf_r, bb->norm_f_w, bb->pf_final, ln_scratch,
                         n_total);
        grid_barrier(g.barrier);
        if (!bb->pf_fuse_decode) return;
        {
            const int n_pos = n_total / n_seq;
            const size_t last = (size_t)(n_pos - 1) * n_seq;
            for (int idx = blockIdx.x * THREADS + threadIdx.x;
                 idx < BATCH * D_MODEL; idx += gridDim.x * THREADS) {
                const int row = idx / D_MODEL;
                const int i = idx - row * D_MODEL;
                const size_t src = (last + (row < n_seq ? row : n_seq - 1)) * D_MODEL + i;
                bb->h[idx] = bb->pf_h[src];
                bb->r[idx] = bb->pf_r[src];
            }
            grid_barrier(g.barrier);
        }
    }
    if (n_backbone_layers > 0 && bb->launches && blockIdx.x == 0 &&
        threadIdx.x == 0)
        atomicAdd(bb->launches + 1, 1ull);
    if (n_backbone_layers > 0) {
        bb_norm_residual(bb->h, bb->r, bb->norm_f_w, g.x, ln_scratch,
                         BATCH, nullptr, -1, nullptr, nullptr, fold_src, fold_routes);
        grid_barrier(g.barrier);
        if (bb->phoneme_out) {
            gemm(bb->phoneme_w, g.x_v, BB_PHON_PADDED, D_MODEL,
                 bb->phoneme_logits, BB_PHON_PADDED, nullptr, EPI_BIAS,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, bb->phoneme_b);
            grid_barrier(g.barrier);
            const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
            for (int row = blockIdx.x + warp * gridDim.x; row < BATCH;
                 row += gridDim.x * WARPS) {
                const bf16 *lg = bb->phoneme_logits + (size_t)row * BB_PHON_PADDED;
                MK2_PROGRESS_GUARD(native_progress_mask, row);
                const int arg = phoneme_warp_argmax(lg);
                if (lane == 0) bb->phoneme_out[row] = arg;
            }
            grid_barrier(g.barrier);
        }
        if (bb->dec_snapshot) {
            for (int idx = blockIdx.x * THREADS + threadIdx.x;
                 idx < BATCH * D_MODEL; idx += gridDim.x * THREADS)
                bb->dec_snapshot[idx] = g.x[idx];
            grid_barrier(g.barrier);
        }
    }
    __shared__ int native_run_predictor;
    if (native_predictor_active_mask && bb && g.native_codes) {
        if (threadIdx.x == 0) {
            int any_active = 0;
            for (int row = 0; row < BATCH; ++row)
                any_active |= int(native_predictor_active_mask[row] != 0 &&
                                  MK2_PROGRESS_ENABLED(native_progress_mask, row));
            native_run_predictor = any_active;
        }
        __syncthreads();
    }
    const bool skip_native_predictor = native_predictor_active_mask && bb &&
        g.native_codes && native_run_predictor == 0;
    if constexpr (true) {
        for (int index = blockIdx.x * THREADS + threadIdx.x;
             index < BATCH * N_POS; index += gridDim.x * THREADS) {
            MK2_PROGRESS_GUARD(native_progress_mask, index / N_POS);
            g.sampled[index] = 0;
            g.native_codes[index] = 0;
        }
        grid_barrier(g.barrier);
    } else {
    for (int row = blockIdx.x; row < BATCH; row += gridDim.x) {
        for (int i = threadIdx.x; i < D_MODEL; i += THREADS) {
            const size_t at = (size_t)row * D_MODEL + i;
            g.x[at] = __float2bfloat16(__bfloat162float(g.x[at]) +
                                       __bfloat162float(g.pos_emb[i]));
        }
    }
    if constexpr (!(MK2_B1_LOCAL_BARRIERS && BATCH == 1))
        grid_barrier(g.barrier);
    for (int pos = 0; pos < n_pos; ++pos) {
        for (int layer = 0; layer < n_layers; ++layer) {
#ifndef ABLATE_NORM
            layernorm_rows(g.x, g.norm_self[layer], g.normed, ln_scratch);
#endif
            grid_barrier(g.barrier);
            gemm(g.qkv[layer], g.normed_v, 3 * D_MODEL, D_MODEL,
                 g.qkv_out, 3 * D_MODEL, nullptr, EPI_STORE,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
            const int pairs = BATCH * N_HEADS;
#ifndef ABLATE_ATTN
            {
                const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
                const int warps_total = gridDim.x * WARPS;
                const float scale = rsqrtf((float)D_HEAD);
                for (int pair = blockIdx.x * WARPS + warp; pair < pairs;
                     pair += warps_total) {
                    const int row = pair / N_HEADS;
                    const int head = pair % N_HEADS;
                    const bf16 *qkv_row = g.qkv_out + (size_t)row * 3 * D_MODEL;
                    const bf16 *q = qkv_row + head * D_HEAD;
                    const bf16 *k_new = qkv_row + D_MODEL + head * D_HEAD;
                    const bf16 *v_new = qkv_row + 2 * D_MODEL + head * D_HEAD;
                    const size_t cache_base =
                        (((size_t)layer * BATCH + row) * N_HEADS + head) * N_POS *
                        D_HEAD;
                    constexpr int DPL = D_HEAD / 32;
                    float qv[DPL], kn[DPL], vn[DPL];
                    #pragma unroll
                    for (int j = 0; j < DPL; ++j) {
                        const int i = lane + j * 32;
                        qv[j] = __bfloat162float(q[i]);
                        kn[j] = __bfloat162float(k_new[i]);
                        vn[j] = __bfloat162float(v_new[i]);
                    }
                    #pragma unroll
                    for (int j = 0; j < DPL; ++j) {
                        const int i = lane + j * 32;
                        g.kcache[cache_base + pos * D_HEAD + i] = k_new[i];
                        g.vcache[cache_base + pos * D_HEAD + i] = v_new[i];
                    }
                    __syncwarp();
                    float sc[N_POS];
                    #pragma unroll 4
                    for (int t = 0; t < pos; ++t) {
                        const bf16 *kt = g.kcache + cache_base + t * D_HEAD;
                        float acc = 0.f;
                        #pragma unroll
                        for (int j = 0; j < D_HEAD / 32; ++j) {
                            const int i = lane + j * 32;
                            acc += qv[j] * __bfloat162float(kt[i]);
                        }
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1)
                            acc += __shfl_xor_sync(0xffffffff, acc, off);
                        sc[t] = acc * scale;
                    }
                    {
                        float acc = 0.f;
                        #pragma unroll
                        for (int j = 0; j < DPL; ++j) acc += qv[j] * kn[j];
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1)
                            acc += __shfl_xor_sync(0xffffffff, acc, off);
                        sc[pos] = acc * scale;
                    }
                    float m = sc[0];
                    for (int t = 1; t <= pos; ++t) m = fmaxf(m, sc[t]);
                    float s = 0.f;
                    for (int t = 0; t <= pos; ++t) {
                        sc[t] = __expf(sc[t] - m);
                        s += sc[t];
                    }
                    const float inv = 1.f / s;
                    for (int t = 0; t <= pos; ++t) sc[t] *= inv;
                    float ov[DPL] = {};
                    #pragma unroll 4
                    for (int t = 0; t < pos; ++t) {
                        const bf16 *vt = g.vcache + cache_base + t * D_HEAD;
                        const float s = sc[t];
                        #pragma unroll
                        for (int j = 0; j < DPL; ++j)
                            ov[j] += s * __bfloat162float(vt[lane + j * 32]);
                    }
                    #pragma unroll
                    for (int j = 0; j < DPL; ++j) {
                        ov[j] += sc[pos] * vn[j];
                        g.attn_out[(size_t)row * D_MODEL + head * D_HEAD +
                                   lane + j * 32] = __float2bfloat16(ov[j]);
                    }
                }
            }
#endif
            grid_barrier(g.barrier);
            gemm(g.o[layer], g.attn_v, D_MODEL, D_MODEL,
                 nullptr, D_MODEL, g.x, EPI_RESIDUAL,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
#ifndef ABLATE_NORM
            layernorm_rows(g.x, g.norm_ff[layer], g.normed, ln_scratch);
#endif
            grid_barrier(g.barrier);
            gemm(g.ff_in[layer], g.normed_v, D_FFN, D_MODEL,
                 g.ff_hidden, D_FFN, nullptr, EPI_GELU,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
            gemm(g.ff_out[layer], g.ffh_v, D_MODEL, D_FFN,
                 nullptr, D_MODEL, g.x, EPI_RESIDUAL,
                 p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile);
            grid_barrier(g.barrier);
        }
        if constexpr (!MK2_SKIP_HIDDEN_SNAPSHOT)
        for (int row = blockIdx.x; row < BATCH; row += gridDim.x) {
            for (int i = threadIdx.x; i < D_MODEL; i += THREADS)
                g.hidden_rows[((size_t)pos * BATCH + row) * D_MODEL + i] =
                    g.x[(size_t)row * D_MODEL + i];
        }
#ifdef ABLATE_HEAD
        (void)0;
#elif defined(HEAD_BF16)
        gemm(g.head[pos], g.x_v, VOCAB_PADDED, D_MODEL,
             g.logits, VOCAB_PADDED, nullptr, EPI_STORE,
             p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, nullptr);
#else
        gemm(g.head[pos], g.x_v, VOCAB_PADDED, D_MODEL,
             g.logits, VOCAB_PADDED, nullptr, EPI_BIAS,
             p_s, o_s, filled, drained, o_sf, gemm_iter, gemm_tile, g.head_bias[pos]);
#endif
        grid_barrier(g.barrier);
#ifndef ABLATE_SAMPLE
        sample_codes(g.logits, g.head_bias[pos], g.forbidden, g.gumbel, pos,
                     g.sampled_pos, sort_storage, g.temperature);
#else
        if (threadIdx.x < BATCH && blockIdx.x == 0)
            g.sampled_pos[threadIdx.x] = g.codes[threadIdx.x * N_POS + pos];
#endif
#ifdef ABLATE_SAMPLE
        if constexpr (MK2_B1_LOCAL_BARRIERS && BATCH == 1) {
            __syncthreads();
        } else {
            grid_barrier(g.barrier);
        }
#endif
        if (pos + 1 < n_pos) {
            for (int row = blockIdx.x; row < BATCH; row += gridDim.x) {
                const int code = g.sampled_pos[row];
                const bf16 *emb = g.audio_emb[pos] + (size_t)code * D_MODEL;
                const bf16 *pe = g.pos_emb + (pos + 1) * D_MODEL;
                for (int i = threadIdx.x; i < D_MODEL; i += THREADS)
                    g.x[(size_t)row * D_MODEL + i] =
                        __float2bfloat16(__bfloat162float(emb[i]) +
                                         __bfloat162float(pe[i]));
            }
        }
        if (threadIdx.x == 0) {
            for (int row = blockIdx.x; row < BATCH; row += gridDim.x) {
                const int code = g.sampled_pos[row];
                g.sampled[row * N_POS + pos] = code;
                if (g.native_codes && MK2_PROGRESS_ENABLED(native_progress_mask, row)) {
#if MK2_INFLIGHT
                    g.native_codes[row * N_POS + pos] =
                        native_predictor_active_mask && !native_predictor_active_mask[row]
                            ? 0 : (long long)code;
#else
                    g.native_codes[row * N_POS + pos] = (long long)code;
#endif
                }
            }
        }
#ifdef ABLATE_SAMPLE
        grid_barrier(g.barrier);
#else
        if (pos + 1 == n_pos) grid_barrier(g.barrier);
#endif
    }
    }
    if (g.native_seq_len && blockIdx.x == 0)
        for (int row = threadIdx.x; row < BATCH; row += blockDim.x) {
            MK2_PROGRESS_GUARD(native_progress_mask, row);
            g.native_seq_len[row] += 1;
        }
    if (bb && n_backbone_layers > 0 && bb->launches && blockIdx.x == 0 &&
        threadIdx.x == 0) {
        unsigned long long t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        const unsigned long long t0 = bb->launches[2];
        if (t1 > t0) {
            const unsigned long long dt = t1 - t0;
            atomicAdd(bb->launches + 3, dt);
            atomicAdd(bb->launches + 4, 1ull);
            atomicMax(bb->launches + 5, dt);
        }
    }
}
#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("FAIL %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
__global__ void fp16_to_bf16(const __half *src, bf16 *dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(__half2float(src[i]));
}
__global__ void fp16_to_bf16_masked(const __half *src, bf16 *dst, int n,
                                   const unsigned char *audio_valid, int *run_predicate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(__half2float(src[i]));
    if (i == 0) {
        int run = 0;
        for (int row = 0; row < BATCH; ++row) run |= audio_valid[row] != 0;
        *run_predicate = run;
    }
}
__global__ void codes_to_int64(const int *src, long long *dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (long long)src[i];
}
static std::vector<char> read_file(const std::string &path, size_t expected) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { printf("FAIL cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    const size_t bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (expected && bytes != expected) {
        printf("FAIL %s is %zu bytes, expected %zu\n", path.c_str(), bytes, expected);
        exit(1);
    }
    std::vector<char> data(bytes);
    if (fread(data.data(), 1, bytes, f) != bytes) { printf("FAIL short read\n"); exit(1); }
    fclose(f);
    return data;
}
template <typename T>
static T *upload(const std::string &path, size_t count) {
    auto host = read_file(path, count * sizeof(T));
    T *device = nullptr;
    CHECK(cudaMalloc(&device, host.size()));
    CHECK(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice));
    return device;
}
#ifndef MK2_NO_MAIN
int main(int argc, char **argv) {
    setbuf(stdout, nullptr);
    const std::string root = argc > 1 ? argv[1] : "/out/predictor_ref_b32";
    bool bench = false, verify = true;
    const char *dump_rows = nullptr;
    int n_backbone = 0;
    int past_len = 7;
    const char *dump_moe_w = nullptr;
    bool inject_moe = false;
    bool stage0 = false;
    bool shuffle_slots = false;
    bool router_audit = false;
    std::string prefill_dir;
    int prefill_bench = 0;
    int prefill_seq = 1;
    int prefill_fuse = 0;
    int bench_iters = 200;
    bool poison_cur_kv = false;
    int decode_steps = 1;
    const char *dump_final = nullptr;
    int n_layers = N_LAYERS, n_pos = N_POS;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bench") bench = true;
        else if (a == "--no-verify") verify = false;
        else if (a == "--dump-rows" && i + 1 < argc) dump_rows = argv[++i];
        else if (a == "--backbone-layers" && i + 1 < argc) n_backbone = atoi(argv[++i]);
        else if (a == "--past-len" && i + 1 < argc) past_len = atoi(argv[++i]);
        else if (a == "--dump-moe-w" && i + 1 < argc) dump_moe_w = argv[++i];
        else if (a == "--inject-moe-w") inject_moe = true;
        else if (a == "--stage0") stage0 = true;
        else if (a == "--shuffle-slots") shuffle_slots = true;
        else if (a == "--router-audit") router_audit = true;
        else if (a == "--prefill" && i + 1 < argc) prefill_dir = argv[++i];
        else if (a == "--prefill-bench" && i + 1 < argc) prefill_bench = atoi(argv[++i]);
        else if (a == "--prefill-seq" && i + 1 < argc) prefill_seq = atoi(argv[++i]);
        else if (a == "--prefill-fuse") prefill_fuse = 1;
        else if (a == "--bench-iters" && i + 1 < argc) bench_iters = atoi(argv[++i]);
        else if (a == "--poison-cur-kv") poison_cur_kv = true;
        else if (a == "--decode-steps" && i + 1 < argc) decode_steps = atoi(argv[++i]);
        else if (a == "--dump-final" && i + 1 < argc) dump_final = argv[++i];
        else if (a == "--layers" && i + 1 < argc) n_layers = atoi(argv[++i]);
        else if (a == "--positions" && i + 1 < argc) n_pos = atoi(argv[++i]);
    }
    bf16 *w_qkv[N_LAYERS], *w_o[N_LAYERS], *w_ffin[N_LAYERS], *w_ffout[N_LAYERS];
    const float *ns[N_LAYERS], *nf[N_LAYERS];
    for (int l = 0; l < N_LAYERS; ++l) {
        const std::string p = root + "/l" + std::to_string(l) + "_";
        w_qkv[l] = upload<bf16>(p + "qkv.bin", (size_t)3 * D_MODEL * D_MODEL);
        w_o[l] = upload<bf16>(p + "o.bin", (size_t)D_MODEL * D_MODEL);
        w_ffin[l] = upload<bf16>(p + "ff_in.bin", (size_t)D_FFN * D_MODEL);
        w_ffout[l] = upload<bf16>(p + "ff_out.bin", (size_t)D_MODEL * D_FFN);
        ns[l] = upload<float>(p + "norm_self.bin", D_MODEL);
        nf[l] = upload<float>(p + "norm_ff.bin", D_MODEL);
    }
    const bf16 *pos_emb = upload<bf16>(root + "/position_embeddings.bin",
                                       (size_t)18 * D_MODEL);
    const bf16 *ae[N_POS];
    for (int k = 0; k < N_POS; ++k)
        ae[k] = upload<bf16>(root + "/audio_emb_" + std::to_string(k) + ".bin", 0);
    bf16 *x = nullptr;
    if (stage0) {
        CHECK(cudaMalloc(&x, (size_t)BATCH * D_MODEL * sizeof(bf16)));
        CHECK(cudaMemset(x, 0, (size_t)BATCH * D_MODEL * sizeof(bf16)));
    } else {
        x = upload<bf16>(root + "/dec_hidden.bin", (size_t)BATCH * D_MODEL);
    }
    bf16 *x_initial = nullptr;
    CHECK(cudaMalloc(&x_initial, (size_t)BATCH * D_MODEL * sizeof(bf16)));
    CHECK(cudaMemcpy(x_initial, x, (size_t)BATCH * D_MODEL * sizeof(bf16),
                     cudaMemcpyDeviceToDevice));
    const int *codes = upload<int>(root + "/codes.bin", (size_t)BATCH * N_POS);
    bf16 *qkv_out, *attn_out, *ff_hidden, *normed, *kcache, *vcache, *hidden_rows;
    unsigned int *barrier;
    CHECK(cudaMalloc(&qkv_out, (size_t)BATCH * 3 * D_MODEL * sizeof(bf16)));
    CHECK(cudaMalloc(&attn_out, (size_t)BATCH * D_MODEL * sizeof(bf16)));
    CHECK(cudaMalloc(&ff_hidden, (size_t)BATCH * D_FFN * sizeof(bf16)));
    CHECK(cudaMalloc(&normed, (size_t)BATCH * D_MODEL * sizeof(bf16)));
    const size_t cache_elems = (size_t)N_LAYERS * BATCH * N_HEADS * N_POS * D_HEAD;
    CHECK(cudaMalloc(&kcache, cache_elems * sizeof(bf16)));
    CHECK(cudaMalloc(&vcache, cache_elems * sizeof(bf16)));
    CHECK(cudaMalloc(&hidden_rows, (size_t)N_POS * BATCH * D_MODEL * sizeof(bf16)));
    CHECK(cudaMalloc(&barrier, 64 * sizeof(unsigned int)));
    CHECK(cudaMemset(barrier, 0, 64 * sizeof(unsigned int)));
    bf16 *head_w[N_POS];
    const bf16 *head_b[N_POS];
    for (int k = 0; k < N_POS; ++k) {
        auto raw = read_file(root + "/out_proj_" + std::to_string(k) + ".bin",
                             (size_t)CODEBOOK * D_MODEL * sizeof(bf16));
        CHECK(cudaMalloc(&head_w[k], (size_t)VOCAB_PADDED * D_MODEL * sizeof(bf16)));
        CHECK(cudaMemset(head_w[k], 0, (size_t)VOCAB_PADDED * D_MODEL * sizeof(bf16)));
        CHECK(cudaMemcpy(head_w[k], raw.data(), raw.size(), cudaMemcpyHostToDevice));
        auto rb = read_file(root + "/out_bias_" + std::to_string(k) + ".bin",
                           (size_t)CODEBOOK * sizeof(bf16));
        bf16 *b_dev = nullptr;
        CHECK(cudaMalloc(&b_dev, (size_t)VOCAB_PADDED * sizeof(bf16)));
        CHECK(cudaMemset(b_dev, 0, (size_t)VOCAB_PADDED * sizeof(bf16)));
        CHECK(cudaMemcpy(b_dev, rb.data(), rb.size(), cudaMemcpyHostToDevice));
        head_b[k] = b_dev;
    }
    std::vector<int8_t> forb(VOCAB_PADDED, 0);
    for (int n = CODEBOOK - 8; n < VOCAB_PADDED; ++n) forb[n] = 1;
    forb[1025] = 0;
    int8_t *forbidden = nullptr;
    CHECK(cudaMalloc(&forbidden, forb.size()));
    CHECK(cudaMemcpy(forbidden, forb.data(), forb.size(), cudaMemcpyHostToDevice));
    const float *gumbel = upload<float>(root + "/gumbel.bin",
                                       (size_t)BATCH * N_POS * TOP_K);
    bf16 *logits_buf = nullptr;
    int *sampled = nullptr, *sampled_pos = nullptr;
    CHECK(cudaMalloc(&logits_buf, (size_t)BATCH * VOCAB_PADDED * sizeof(bf16)));
    CHECK(cudaMalloc(&sampled, (size_t)BATCH * N_POS * sizeof(int)));
    CHECK(cudaMalloc(&sampled_pos, (size_t)BATCH * sizeof(int)));
    const size_t DM = D_MODEL, DF = D_FFN, B = BATCH, QKV = 3 * D_MODEL;
    const size_t VP = VOCAB_PADDED;
    Globals g{
        {weight_gl{w_qkv[0], nullptr, nullptr, QKV, DM},
         weight_gl{w_qkv[1], nullptr, nullptr, QKV, DM},
         weight_gl{w_qkv[2], nullptr, nullptr, QKV, DM}},
        {weight_gl{w_o[0], nullptr, nullptr, DM, DM},
         weight_gl{w_o[1], nullptr, nullptr, DM, DM},
         weight_gl{w_o[2], nullptr, nullptr, DM, DM}},
        {weight_gl{w_ffin[0], nullptr, nullptr, DF, DM},
         weight_gl{w_ffin[1], nullptr, nullptr, DF, DM},
         weight_gl{w_ffin[2], nullptr, nullptr, DF, DM}},
        {weight_gl{w_ffout[0], nullptr, nullptr, DM, DF},
         weight_gl{w_ffout[1], nullptr, nullptr, DM, DF},
         weight_gl{w_ffout[2], nullptr, nullptr, DM, DF}},
        {weight_gl{head_w[0], nullptr, nullptr, VP, DM},
         weight_gl{head_w[1], nullptr, nullptr, VP, DM},
         weight_gl{head_w[2], nullptr, nullptr, VP, DM},
         weight_gl{head_w[3], nullptr, nullptr, VP, DM},
         weight_gl{head_w[4], nullptr, nullptr, VP, DM},
         weight_gl{head_w[5], nullptr, nullptr, VP, DM},
         weight_gl{head_w[6], nullptr, nullptr, VP, DM},
         weight_gl{head_w[7], nullptr, nullptr, VP, DM},
         weight_gl{head_w[8], nullptr, nullptr, VP, DM},
         weight_gl{head_w[9], nullptr, nullptr, VP, DM},
         weight_gl{head_w[10], nullptr, nullptr, VP, DM},
         weight_gl{head_w[11], nullptr, nullptr, VP, DM},
         weight_gl{head_w[12], nullptr, nullptr, VP, DM},
         weight_gl{head_w[13], nullptr, nullptr, VP, DM},
         weight_gl{head_w[14], nullptr, nullptr, VP, DM},
         weight_gl{head_w[15], nullptr, nullptr, VP, DM}},
        act_gl{x, nullptr, nullptr, B, DM},
        act_gl{normed, nullptr, nullptr, B, DM},
        act_gl{attn_out, nullptr, nullptr, B, DM},
        act_gl{ff_hidden, nullptr, nullptr, B, DF},
        x, qkv_out, attn_out, ff_hidden, normed,
        logits_buf,
        {head_b[0], head_b[1], head_b[2], head_b[3], head_b[4], head_b[5],
         head_b[6], head_b[7], head_b[8], head_b[9], head_b[10], head_b[11],
         head_b[12], head_b[13], head_b[14], head_b[15]},
        forbidden, gumbel, sampled, sampled_pos, 0.7f,
        {ns[0], ns[1], ns[2]},
        {nf[0], nf[1], nf[2]},
        pos_emb,
        {ae[0], ae[1], ae[2], ae[3], ae[4], ae[5], ae[6], ae[7],
         ae[8], ae[9], ae[10], ae[11], ae[12], ae[13], ae[14], ae[15]},
        kcache, vcache, hidden_rows, codes, barrier,
    };
    Backbone *d_bb = nullptr;
    bf16 *bb_layer_out_host = nullptr;
    float *bb_moe_w_host = nullptr;
    bf16 *bb_dec_snapshot_host = nullptr;
    int *ph_out_verify = nullptr;
    float *moe_w_audit_host = nullptr;
    unsigned long long *launch_counter_host = nullptr;
    Backbone *bb_saved = nullptr;
    std::vector<MambaLayer> mamba_saved;
    bf16 *pf_layer_out_host = nullptr;
    if (n_backbone > 0) {
        const std::string bb_root = stage0 ? std::string(root)
                                          : std::string(root) + "/../backbone";
        static const char PATTERN[] = "MEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";
        std::vector<MambaLayer> mamba_host;
        int mamba_of_layer[BB_LAYERS];
        const bf16 *norm_w[BB_LAYERS];
        for (int layer = 0; layer < BB_LAYERS; ++layer) {
            const std::string lp = bb_root + "/l" + std::to_string(layer) + "_";
            norm_w[layer] = upload<bf16>(lp + "norm.bin", D_MODEL);
            if (PATTERN[layer] != 'M') { mamba_of_layer[layer] = -1; continue; }
            mamba_of_layer[layer] = (int)mamba_host.size();
            bf16 *in_p = upload<bf16>(lp + "in_proj.bin",
                                      (size_t)MB_IN_PROJ * D_MODEL);
            bf16 *out_p = upload<bf16>(lp + "out_proj.bin",
                                       (size_t)D_MODEL * D_MODEL);
            bf16 *conv_state = nullptr;
            float *ssm_state = nullptr;
            const size_t cs = (size_t)BATCH * MB_CONV_DIM * (MB_CONV_K - 1);
            const size_t ss = (size_t)BATCH * MB_HEADS * MB_HEAD_DIM * MB_STATE;
            CHECK(cudaMalloc(&conv_state, cs * sizeof(bf16)));
            CHECK(cudaMalloc(&ssm_state, ss * sizeof(float)));
            auto csr = read_file(bb_root + "/conv_state_in_" +
                                 std::to_string(layer) + ".bin", cs * sizeof(bf16));
            auto ssr = read_file(bb_root + "/ssm_state_in_" +
                                 std::to_string(layer) + ".bin", ss * sizeof(float));
            const size_t cs_per = (size_t)MB_CONV_DIM * (MB_CONV_K - 1);
            const size_t ss_per = (size_t)MB_HEADS * MB_HEAD_DIM * MB_STATE;
            std::vector<char> csp(csr.size()), ssp(ssr.size());
            for (int i = 0; i < BATCH; ++i) {
                const int slot = shuffle_slots ? BATCH - 1 - i : i;
                memcpy(csp.data() + (size_t)slot * cs_per * sizeof(bf16),
                       csr.data() + (size_t)i * cs_per * sizeof(bf16),
                       cs_per * sizeof(bf16));
                memcpy(ssp.data() + (size_t)slot * ss_per * sizeof(float),
                       ssr.data() + (size_t)i * ss_per * sizeof(float),
                       ss_per * sizeof(float));
            }
            CHECK(cudaMemcpy(conv_state, csp.data(), csp.size(),
                             cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(ssm_state, ssp.data(), ssp.size(),
                             cudaMemcpyHostToDevice));
            mamba_host.push_back(MambaLayer{
                weight_gl{in_p, nullptr, nullptr, (size_t)MB_IN_PROJ,
                          (size_t)D_MODEL},
                weight_gl{out_p, nullptr, nullptr, (size_t)D_MODEL,
                          (size_t)D_MODEL},
                upload<bf16>(lp + "conv_w.bin", (size_t)MB_CONV_DIM * MB_CONV_K),
                upload<bf16>(lp + "conv_b.bin", MB_CONV_DIM),
                upload<float>(lp + "A.bin", MB_HEADS),
                upload<float>(lp + "D.bin", MB_HEADS),
                upload<float>(lp + "dt_bias.bin", MB_HEADS),
                upload<bf16>(lp + "gnorm.bin", MB_INNER),
                conv_state,
                MB_CONV_DIM * (MB_CONV_K - 1),
                ssm_state,
                MB_HEADS * MB_HEAD_DIM * MB_STATE,
            });
        }
        std::vector<AttnLayer> attn_host;
        std::vector<MoeLayer> moe_host;
        int attn_of_layer[BB_LAYERS], moe_of_layer[BB_LAYERS];
        for (int layer = 0; layer < BB_LAYERS; ++layer) {
            attn_of_layer[layer] = -1;
            moe_of_layer[layer] = -1;
            const std::string lp = bb_root + "/l" + std::to_string(layer) + "_";
            if (PATTERN[layer] == '*') {
                attn_of_layer[layer] = (int)attn_host.size();
                const int TPB = 32;
                const int kv_span = max(BB_T_MAX, PREFILL_MAX_TOKENS);
                const int blocks_per_seq = (kv_span + TPB - 1) / TPB;
                const size_t elems_per_block =
                    (size_t)TPB * BB_KV_HEADS * BB_HEAD_DIM;
                const size_t kvn = (size_t)BATCH * BB_T_MAX * BB_KV_SIZE;
                auto kr = read_file(bb_root + "/k_cache_" + std::to_string(layer) +
                                    ".bin", kvn * sizeof(bf16));
                auto vr = read_file(bb_root + "/v_cache_" + std::to_string(layer) +
                                    ".bin", kvn * sizeof(bf16));
                const bf16 *kh = reinterpret_cast<const bf16 *>(kr.data());
                const bf16 *vh = reinterpret_cast<const bf16 *>(vr.data());
                const int total_blocks = BATCH * blocks_per_seq * 2;
                std::vector<bf16> pool(total_blocks * elems_per_block, __float2bfloat16(0.f));
                std::vector<int> table((size_t)BATCH * 2 * blocks_per_seq, 0);
                for (int s = 0; s < BATCH; ++s) {
                    for (int b = 0; b < blocks_per_seq; ++b) {
                        const int kblk = (BATCH - 1 - s) * blocks_per_seq + b;
                        const int vblk = BATCH * blocks_per_seq + s * blocks_per_seq + b;
                        table[((size_t)s * 2 + 0) * blocks_per_seq + b] = kblk;
                        table[((size_t)s * 2 + 1) * blocks_per_seq + b] = vblk;
                        for (int p = b * TPB; p < BB_T_MAX && p < (b + 1) * TPB; ++p) {
                            for (int h = 0; h < BB_KV_HEADS; ++h) {
                                for (int d = 0; d < BB_HEAD_DIM; ++d) {
                                    const size_t src =
                                        (((size_t)s * BB_T_MAX + p) * BB_KV_HEADS + h) *
                                        BB_HEAD_DIM + d;
                                    const size_t off =
                                        (size_t)h * TPB * BB_HEAD_DIM +
                                        (size_t)(p % TPB) * BB_HEAD_DIM + d;
                                    pool[(size_t)kblk * elems_per_block + off] = kh[src];
                                    pool[(size_t)vblk * elems_per_block + off] = vh[src];
                                }
                            }
                        }
                    }
                }
                if (poison_cur_kv && past_len >= 0 && past_len < kv_span) {
                    const int b = past_len / TPB;
                    for (int s = 0; s < BATCH; ++s) {
                        const int kblk = (BATCH - 1 - s) * blocks_per_seq + b;
                        const int vblk =
                            BATCH * blocks_per_seq + s * blocks_per_seq + b;
                        for (int h = 0; h < BB_KV_HEADS; ++h)
                            for (int d = 0; d < BB_HEAD_DIM; ++d) {
                                const size_t off =
                                    (size_t)h * TPB * BB_HEAD_DIM +
                                    (size_t)(past_len % TPB) * BB_HEAD_DIM + d;
                                pool[(size_t)kblk * elems_per_block + off] =
                                    __float2bfloat16(7.0f);
                                pool[(size_t)vblk * elems_per_block + off] =
                                    __float2bfloat16(-13.0f);
                            }
                    }
                }
                bf16 *d_pool = nullptr;
                int *d_table = nullptr;
                CHECK(cudaMalloc(&d_pool, pool.size() * sizeof(bf16)));
                CHECK(cudaMemcpy(d_pool, pool.data(), pool.size() * sizeof(bf16),
                                 cudaMemcpyHostToDevice));
                CHECK(cudaMalloc(&d_table, table.size() * sizeof(int)));
                CHECK(cudaMemcpy(d_table, table.data(), table.size() * sizeof(int),
                                 cudaMemcpyHostToDevice));
                attn_host.push_back(AttnLayer{
                    weight_gl{upload<bf16>(lp + "qkv.bin",
                                           (size_t)BB_QKV * D_MODEL),
                              nullptr, nullptr, (size_t)BB_QKV, (size_t)D_MODEL},
                    weight_gl{upload<bf16>(lp + "o_proj.bin",
                                           (size_t)D_MODEL * D_MODEL),
                              nullptr, nullptr, (size_t)D_MODEL, (size_t)D_MODEL},
                    d_pool, d_table, blocks_per_seq, TPB,
                });
            } else if (PATTERN[layer] == 'E') {
                moe_of_layer[layer] = (int)moe_host.size();
                moe_host.push_back(MoeLayer{
                    weight_gl{upload<bf16>(lp + "exp_up.bin",
                              (size_t)BB_EXPERTS * BB_MOE_INTER * D_MODEL),
                              nullptr, nullptr,
                              (size_t)(BB_EXPERTS * BB_MOE_INTER), (size_t)D_MODEL},
                    weight_gl{upload<bf16>(lp + "exp_down.bin",
                              (size_t)BB_EXPERTS * D_MODEL * BB_MOE_INTER),
                              nullptr, nullptr,
                              (size_t)(BB_EXPERTS * D_MODEL), (size_t)BB_MOE_INTER},
                    weight_gl{upload<bf16>(lp + "sh_up.bin",
                              (size_t)BB_SHARED_INTER * D_MODEL),
                              nullptr, nullptr,
                              (size_t)BB_SHARED_INTER, (size_t)D_MODEL},
                    weight_gl{upload<bf16>(lp + "sh_down.bin",
                              (size_t)D_MODEL * BB_SHARED_INTER),
                              nullptr, nullptr,
                              (size_t)D_MODEL, (size_t)BB_SHARED_INTER},
                    upload<bf16>(lp + "gate.bin", (size_t)BB_EXPERTS * D_MODEL),
                    upload<bf16>(lp + "gate_bias.bin", BB_EXPERTS),
                });
            }
        }
        AttnLayer *d_attn = nullptr;
        MoeLayer *d_moe = nullptr;
        CHECK(cudaMalloc(&d_attn, attn_host.size() * sizeof(AttnLayer)));
        CHECK(cudaMemcpy(d_attn, attn_host.data(),
                         attn_host.size() * sizeof(AttnLayer),
                         cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&d_moe, moe_host.size() * sizeof(MoeLayer)));
        CHECK(cudaMemcpy(d_moe, moe_host.data(), moe_host.size() * sizeof(MoeLayer),
                         cudaMemcpyHostToDevice));
        MambaLayer *d_mamba = nullptr;
        CHECK(cudaMalloc(&d_mamba, mamba_host.size() * sizeof(MambaLayer)));
        CHECK(cudaMemcpy(d_mamba, mamba_host.data(),
                         mamba_host.size() * sizeof(MambaLayer),
                         cudaMemcpyHostToDevice));
        bf16 *bb_h = upload<bf16>(bb_root + "/hidden_in.bin",
                                  (size_t)BATCH * D_MODEL);
        bf16 *bb_r, *bb_normed, *bb_zxbcdt, *bb_gated, *bb_layer_out;
        CHECK(cudaMalloc(&bb_r, (size_t)BATCH * D_MODEL * sizeof(bf16)));
        CHECK(cudaMemset(bb_r, 0, (size_t)BATCH * D_MODEL * sizeof(bf16)));
        CHECK(cudaMalloc(&bb_normed, (size_t)BATCH * D_MODEL * sizeof(bf16)));
        CHECK(cudaMalloc(&bb_zxbcdt, (size_t)BATCH * MB_IN_PROJ * sizeof(bf16)));
        CHECK(cudaMalloc(&bb_gated, (size_t)BATCH * MB_INNER * sizeof(bf16)));
        CHECK(cudaMalloc(&bb_layer_out,
                         (size_t)BB_LAYERS * BATCH * D_MODEL * sizeof(bf16)));
        bb_layer_out_host = bb_layer_out;
        bf16 *ph_w_dev = nullptr, *ph_b_dev_nc = nullptr, *ph_logits_dev = nullptr;
        int *ph_out_dev = nullptr;
        {
            auto wr = read_file(bb_root + "/phoneme_w.bin",
                                (size_t)BB_PHONEME * D_MODEL * sizeof(bf16));
            auto br = read_file(bb_root + "/phoneme_b.bin",
                                (size_t)BB_PHONEME * sizeof(bf16));
            CHECK(cudaMalloc(&ph_w_dev,
                             (size_t)BB_PHON_PADDED * D_MODEL * sizeof(bf16)));
            CHECK(cudaMemset(ph_w_dev, 0,
                             (size_t)BB_PHON_PADDED * D_MODEL * sizeof(bf16)));
            CHECK(cudaMemcpy(ph_w_dev, wr.data(), wr.size(),
                             cudaMemcpyHostToDevice));
            CHECK(cudaMalloc(&ph_b_dev_nc, (size_t)BB_PHON_PADDED * sizeof(bf16)));
            CHECK(cudaMemset(ph_b_dev_nc, 0, (size_t)BB_PHON_PADDED * sizeof(bf16)));
            CHECK(cudaMemcpy(ph_b_dev_nc, br.data(), br.size(),
                             cudaMemcpyHostToDevice));
            CHECK(cudaMalloc(&ph_logits_dev,
                             (size_t)BATCH * BB_PHON_PADDED * sizeof(bf16)));
            CHECK(cudaMalloc(&ph_out_dev, (size_t)BATCH * sizeof(int)));
        }
        const bf16 *ph_b_dev = ph_b_dev_nc;
        bf16 *bb_qkv, *bb_moe_h, *bb_moe_sh;
        float *bb_moe_out, *bb_moe_w;
        CHECK(cudaMalloc(&bb_qkv, (size_t)BATCH * BB_QKV * sizeof(bf16)));
        CHECK(cudaMalloc(&bb_moe_h,
                         (size_t)BATCH * BB_EXPERTS * BB_MOE_INTER * sizeof(bf16)));
        CHECK(cudaMalloc(&bb_moe_sh,
                         (size_t)BATCH * BB_SHARED_INTER * sizeof(bf16)));
        CHECK(cudaMalloc(&bb_moe_out, (size_t)BATCH * (BB_EXPERTS + 1) * D_MODEL * sizeof(float)));
        CHECK(cudaMalloc(&bb_moe_w, (size_t)BATCH * BB_EXPERTS * sizeof(float)));
        bb_moe_w_host = bb_moe_w;
        bf16 *pf_h, *pf_r, *pf_normed, *pf_zxbcdt, *pf_xbc, *pf_gated_b, *pf_qkv_b;
        bf16 *pf_moe_h_b, *pf_moe_sh_b, *pf_layer_out_b, *pf_final_b;
        float *pf_moe_out_b, *pf_moe_w_b;
        {
            const size_t P = PREFILL_STORAGE_TOKENS;
            const size_t T1 = PREFILL_STORAGE_TOKENS;
            CHECK(cudaMalloc(&pf_h, P * D_MODEL * sizeof(bf16)));
            CHECK(cudaMalloc(&pf_r, P * D_MODEL * sizeof(bf16)));
            CHECK(cudaMalloc(&pf_final_b, P * D_MODEL * sizeof(bf16)));
            CHECK(cudaMalloc(&pf_normed, T1 * D_MODEL * sizeof(bf16)));
            CHECK(cudaMalloc(&pf_zxbcdt, T1 * MB_IN_PROJ * sizeof(bf16)));
            CHECK(cudaMalloc(&pf_xbc, T1 * MB_CONV_DIM * sizeof(bf16)));
            CHECK(cudaMalloc(&pf_gated_b, T1 * MB_INNER * sizeof(bf16)));
            CHECK(cudaMalloc(&pf_qkv_b, T1 * BB_QKV * sizeof(bf16)));
            CHECK(cudaMalloc(&pf_moe_h_b,
                             T1 * BB_EXPERTS * BB_MOE_INTER * sizeof(bf16)));
            CHECK(cudaMalloc(&pf_moe_sh_b, T1 * BB_SHARED_INTER * sizeof(bf16)));
            CHECK(cudaMalloc(&pf_moe_out_b, T1 * (BB_EXPERTS + 1) * D_MODEL * sizeof(float)));
            CHECK(cudaMalloc(&pf_moe_w_b, T1 * BB_EXPERTS * sizeof(float)));
            CHECK(cudaMalloc(&pf_layer_out_b,
                             (size_t)BB_LAYERS * P * D_MODEL * sizeof(bf16)));
            for (auto pr : std::vector<std::pair<void *, size_t>>{
                     {pf_h, P * D_MODEL * sizeof(bf16)},
                     {pf_r, P * D_MODEL * sizeof(bf16)},
                     {pf_final_b, P * D_MODEL * sizeof(bf16)},
                     {pf_normed, T1 * D_MODEL * sizeof(bf16)},
                     {pf_zxbcdt, T1 * MB_IN_PROJ * sizeof(bf16)},
                     {pf_xbc, T1 * MB_CONV_DIM * sizeof(bf16)},
                     {pf_gated_b, T1 * MB_INNER * sizeof(bf16)},
                     {pf_qkv_b, T1 * BB_QKV * sizeof(bf16)},
                     {pf_moe_h_b, T1 * BB_EXPERTS * BB_MOE_INTER * sizeof(bf16)},
                     {pf_moe_sh_b, T1 * BB_SHARED_INTER * sizeof(bf16)},
                     {pf_moe_out_b, T1 * (BB_EXPERTS + 1) * D_MODEL * sizeof(float)},
                     {pf_moe_w_b, T1 * BB_EXPERTS * sizeof(float)},
                     {pf_layer_out_b, (size_t)BB_LAYERS * P * D_MODEL * sizeof(bf16)}})
                CHECK(cudaMemset(pr.first, 0, pr.second));
        }
        unsigned long long *bb_launches = nullptr;
        CHECK(cudaMalloc(&bb_launches, 6 * sizeof(unsigned long long)));
        CHECK(cudaMemset(bb_launches, 0, 6 * sizeof(unsigned long long)));
        launch_counter_host = bb_launches;
        Backbone host_bb{
            act_gl{bb_normed, nullptr, nullptr, (size_t)BATCH, (size_t)D_MODEL},
            act_gl{bb_gated, nullptr, nullptr, (size_t)BATCH, (size_t)MB_INNER},
            d_mamba,
            upload<bf16>(bb_root + "/norm_f.bin", D_MODEL),
            d_attn,
            d_moe,
            {},
            {}, {}, {},
            past_len,
            bb_h, bb_r, bb_normed, bb_zxbcdt, bb_gated, bb_layer_out,
            bb_qkv, bb_moe_h, bb_moe_sh, bb_moe_out, bb_moe_w,
            inject_moe ? upload<float>(bb_root + "/moe_w.bin",
                                      (size_t)14 * BATCH * BB_EXPERTS)
                       : nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            0, 0, 1, 0,
            nullptr,
            nullptr,
            pf_h, pf_r, pf_normed, pf_zxbcdt, pf_xbc, pf_gated_b, pf_qkv_b,
            pf_moe_h_b, pf_moe_sh_b, pf_moe_out_b, pf_moe_w_b, pf_layer_out_b,
            pf_final_b,
            act_gl{pf_normed, nullptr, nullptr, (size_t)PREFILL_STORAGE_TOKENS,
                   (size_t)D_MODEL},
            act_gl{pf_gated_b, nullptr, nullptr, (size_t)PREFILL_STORAGE_TOKENS,
                   (size_t)MB_INNER},
            act_gl{pf_qkv_b, nullptr, nullptr, (size_t)PREFILL_STORAGE_TOKENS,
                   (size_t)BB_QKV},
            act_gl{pf_moe_h_b, nullptr, nullptr, (size_t)PREFILL_STORAGE_TOKENS,
                   (size_t)(BB_EXPERTS * BB_MOE_INTER)},
            act_gl{pf_moe_sh_b, nullptr, nullptr, (size_t)PREFILL_STORAGE_TOKENS,
                   (size_t)BB_SHARED_INTER},
            weight_gl{ph_w_dev, nullptr, nullptr, (size_t)BB_PHON_PADDED,
                      (size_t)D_MODEL},
            ph_b_dev, ph_logits_dev, ph_out_dev,
            bb_launches,
            nullptr,
            act_gl{bb_qkv, nullptr, nullptr, (size_t)BATCH, (size_t)BB_QKV},
            act_gl{bb_moe_h, nullptr, nullptr, (size_t)BATCH,
                   (size_t)(BB_EXPERTS * BB_MOE_INTER)},
            act_gl{bb_moe_sh, nullptr, nullptr, (size_t)BATCH,
                   (size_t)BB_SHARED_INTER},
        };
        CHECK(cudaMalloc(&bb_dec_snapshot_host,
                         (size_t)BATCH * D_MODEL * sizeof(bf16)));
        host_bb.dec_snapshot = bb_dec_snapshot_host;
        bf16 *bb_xbc = nullptr;
        CHECK(cudaMalloc(&bb_xbc, (size_t)BATCH * MB_CONV_DIM * sizeof(bf16)));
        host_bb.xbc_act = bb_xbc;
        std::vector<int> slots_host(BATCH);
        for (int i = 0; i < BATCH; ++i) slots_host[i] = i;
        if (shuffle_slots) {
            for (int i = 0; i < BATCH; ++i) slots_host[i] = BATCH - 1 - i;
        }
        int *d_slots = nullptr;
        CHECK(cudaMalloc(&d_slots, BATCH * sizeof(int)));
        CHECK(cudaMemcpy(d_slots, slots_host.data(), BATCH * sizeof(int),
                         cudaMemcpyHostToDevice));
        host_bb.slot_map = d_slots;
        ph_out_verify = ph_out_dev;
        if (router_audit) {
            CHECK(cudaMalloc(&moe_w_audit_host,
                             (size_t)14 * BATCH * BB_EXPERTS * sizeof(float)));
            CHECK(cudaMemset(moe_w_audit_host, 0,
                             (size_t)14 * BATCH * BB_EXPERTS * sizeof(float)));
            host_bb.moe_w_audit = moe_w_audit_host;
        }
        for (int i = 0; i < BB_LAYERS; ++i) {
            host_bb.norm_w[i] = norm_w[i];
            host_bb.mamba_of_layer[i] = mamba_of_layer[i];
            host_bb.attn_of_layer[i] = attn_of_layer[i];
            host_bb.moe_of_layer[i] = moe_of_layer[i];
        }
        CHECK(cudaMalloc(&d_bb, sizeof(Backbone)));
        CHECK(cudaMemcpy(d_bb, &host_bb, sizeof(Backbone), cudaMemcpyHostToDevice));
        bb_saved = new Backbone(host_bb);
        mamba_saved = mamba_host;
        pf_layer_out_host = pf_layer_out_b;
        printf("  backbone: %zu Mamba2 layers loaded, %d layers requested\n",
               mamba_host.size(), n_backbone);
    }
    const int smem = STAGES * SLOTS * sizeof(weight_tile)
                     + sizeof(out_tile_f) + 2048;
    CHECK(cudaFuncSetAttribute(predictor_mk2,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int max_blocks = 0, sms = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, predictor_mk2, THREADS, smem));
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    if (max_blocks < 1) {
        printf("FAIL no block is resident per SM at %d B shared\n", smem);
        return 1;
    }
    int grid = min(sms * max_blocks, D_FFN / OUT_ATOM);
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--grid" && i + 1 < argc) grid = atoi(argv[i + 1]);
    printf("  grid=%d blocks x %d threads  smem=%d B  KCHUNK=%d STAGES=%d"
           "  (%d resident/SM, %d SMs)\n",
           grid, THREADS, smem, (int)KCHUNK, (int)STAGES, max_blocks, sms);
    if (prefill_bench > 0) {
        if (d_bb == nullptr) {
            printf("  --prefill-bench needs --backbone-layers > 0\n");
            return 1;
        }
        const int T = prefill_bench;
        if (T > PREFILL_MAX_TOKENS) {
            printf("  T=%d exceeds PREFILL_MAX_TOKENS=%d\n", T, PREFILL_MAX_TOKENS);
            return 1;
        }
        std::vector<uint16_t> prompt_h((size_t)T * D_MODEL);
        for (size_t i = 0; i < prompt_h.size(); ++i)
            prompt_h[i] = 0x3D00 | (uint16_t)(i & 0xFF);
        bf16 *d_prompt = nullptr;
        CHECK(cudaMalloc(&d_prompt, prompt_h.size() * sizeof(uint16_t)));
        CHECK(cudaMemcpy(d_prompt, prompt_h.data(),
                         prompt_h.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        if (prefill_seq > 1 && T % prefill_seq) {
            printf("  T=%d is not a multiple of --prefill-seq %d\n", T, prefill_seq);
            return 1;
        }
        Backbone hb = *bb_saved;
        hb.n_prefill_tokens = T;
        hb.prefill_slot = 0;
        hb.pf_n_seq = prefill_seq;
        hb.pf_fuse_decode = prefill_fuse;
        hb.prompt = d_prompt;
        if (prefill_seq > 1) {
            std::vector<int> slots(prefill_seq);
            for (int i = 0; i < prefill_seq; ++i) slots[i] = i;
            int *d_slots = nullptr;
            CHECK(cudaMalloc(&d_slots, slots.size() * sizeof(int)));
            CHECK(cudaMemcpy(d_slots, slots.data(), slots.size() * sizeof(int),
                             cudaMemcpyHostToDevice));
            hb.pf_seq_slot = d_slots;
        }
        hb.pf_layer_out = nullptr;
        hb.moe_w_inject = nullptr;
        hb.moe_w_audit = nullptr;
        CHECK(cudaMemcpy(d_bb, &hb, sizeof(Backbone), cudaMemcpyHostToDevice));
        const int warm = 3, iters = 10;
        for (int i = 0; i < warm; ++i)
            predictor_mk2<<<grid, THREADS, smem>>>(g, d_bb, n_layers, n_pos, n_backbone);
        CHECK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0));
        CHECK(cudaEventCreate(&e1));
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i)
            predictor_mk2<<<grid, THREADS, smem>>>(g, d_bb, n_layers, n_pos, n_backbone);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaDeviceSynchronize());
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        const int tiles = (T + BATCH_PAD - 1) / BATCH_PAD;
        printf("  BENCH T=%d, %d layers, %d tiles, %d seq%s: %.3f ms/launch"
               "  (%.3f ms/tile)\n",
               T, n_backbone, tiles, prefill_seq,
               prefill_fuse ? ", FUSED prefill+decode" : "", ms / iters,
               ms / iters / tiles);
        if (prefill_fuse) {
            auto fresh_state = [&]() {
                for (const MambaLayer &L : mamba_saved) {
                    CHECK(cudaMemset(L.conv_state, 0,
                                     (size_t)BATCH * MB_CONV_DIM * (MB_CONV_K - 1) *
                                         sizeof(bf16)));
                    CHECK(cudaMemset(L.ssm_state, 0,
                                     (size_t)BATCH * MB_HEADS * MB_HEAD_DIM * MB_STATE *
                                         sizeof(float)));
                }
            };
            std::vector<int> fused_codes((size_t)BATCH * N_POS);
            std::vector<int> split_codes((size_t)BATCH * N_POS);
            fresh_state();
            CHECK(cudaMemcpy(d_bb, &hb, sizeof(Backbone), cudaMemcpyHostToDevice));
            predictor_mk2<<<grid, THREADS, smem>>>(g, d_bb, n_layers, n_pos, n_backbone);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(fused_codes.data(), (const void *)g.codes,
                             fused_codes.size() * sizeof(int),
                             cudaMemcpyDeviceToHost));
            fresh_state();
            Backbone pb = hb;
            pb.pf_fuse_decode = 0;
            CHECK(cudaMemcpy(d_bb, &pb, sizeof(Backbone), cudaMemcpyHostToDevice));
            predictor_mk2<<<grid, THREADS, smem>>>(g, d_bb, n_layers, n_pos, n_backbone);
            CHECK(cudaDeviceSynchronize());
            const int n_pos_pf = T / prefill_seq;
            for (int row = 0; row < BATCH; ++row) {
                const int s = row < prefill_seq ? row : prefill_seq - 1;
                const size_t src = ((size_t)(n_pos_pf - 1) * prefill_seq + s) * D_MODEL;
                CHECK(cudaMemcpy(pb.h + (size_t)row * D_MODEL, pb.pf_h + src,
                                 D_MODEL * sizeof(bf16), cudaMemcpyDeviceToDevice));
                CHECK(cudaMemcpy(pb.r + (size_t)row * D_MODEL, pb.pf_r + src,
                                 D_MODEL * sizeof(bf16), cudaMemcpyDeviceToDevice));
            }
            Backbone db = pb;
            db.n_prefill_tokens = 0;
            CHECK(cudaMemcpy(d_bb, &db, sizeof(Backbone), cudaMemcpyHostToDevice));
            predictor_mk2<<<grid, THREADS, smem>>>(g, d_bb, n_layers, n_pos, n_backbone);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(split_codes.data(), (const void *)g.codes,
                             split_codes.size() * sizeof(int),
                             cudaMemcpyDeviceToHost));
            int diff = 0;
            for (size_t i = 0; i < fused_codes.size(); ++i)
                if (fused_codes[i] != split_codes[i]) ++diff;
            printf("  %s fused vs split: %d of %zu codes differ\n",
                   diff ? "FAIL" : "PASS", diff, fused_codes.size());
            if (diff) return 1;
        }
        return 0;
    }
    if (!prefill_dir.empty()) {
        if (d_bb == nullptr) {
            printf("  --prefill needs --backbone-layers > 0\n");
            return 1;
        }
        auto meta = read_file(prefill_dir + "/meta.json", 0);
        const std::string ms(meta.begin(), meta.end());
        const size_t at = ms.find("\"tokens\"");
        const int T = at == std::string::npos ? 0 : atoi(ms.c_str() + ms.find(':', at) + 1);
        if (T <= 0 || T > PREFILL_MAX_TOKENS) {
            printf("  fixture tokens=%d outside 1..%d\n", T, PREFILL_MAX_TOKENS);
            return 1;
        }
        bf16 *d_prompt = upload<bf16>(prefill_dir + "/prompt.bin",
                                      (size_t)T * D_MODEL);
        Backbone hb = *bb_saved;
        hb.n_prefill_tokens = T;
        hb.prefill_slot = 0;
        hb.prompt = d_prompt;
        if (inject_moe) {
            const int n_moe = 14;
            hb.moe_w_inject = upload<float>(
                prefill_dir + "/moe_w.bin", (size_t)n_moe * T * BB_EXPERTS);
            printf("  routing: oracle-injected (isolates the MoE math from near-tie "
                   "flips)\n");
        } else {
            hb.moe_w_inject = nullptr;
            printf("  routing: kernel's own router\n");
        }
        float *pf_audit = nullptr;
        {
            const size_t n = (size_t)14 * PREFILL_MAX_TOKENS * BB_EXPERTS;
            CHECK(cudaMalloc(&pf_audit, n * sizeof(float)));
            CHECK(cudaMemset(pf_audit, 0, n * sizeof(float)));
            hb.moe_w_audit = pf_audit;
        }
        for (const MambaLayer &L : mamba_saved) {
            CHECK(cudaMemset(L.conv_state + (size_t)hb.prefill_slot * L.conv_slot_stride,
                             0, (size_t)MB_CONV_DIM * (MB_CONV_K - 1) * sizeof(bf16)));
            CHECK(cudaMemset(L.ssm_state + (size_t)hb.prefill_slot * L.ssm_slot_stride,
                             0,
                             (size_t)MB_HEADS * MB_HEAD_DIM * MB_STATE * sizeof(float)));
        }
        CHECK(cudaMemcpy(d_bb, &hb, sizeof(Backbone), cudaMemcpyHostToDevice));
        {
            std::vector<uint16_t> cs(MB_CONV_DIM * (MB_CONV_K - 1));
            CHECK(cudaMemcpy(cs.data(),
                             mamba_saved[0].conv_state +
                                 (size_t)hb.prefill_slot * mamba_saved[0].conv_slot_stride,
                             cs.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            int nz = 0;
            for (uint16_t v : cs) if (v) ++nz;
            if (nz) {
                printf("  FAIL conv_state[layer0] slot %d has %d of %zu non-zero\n",
                       hb.prefill_slot, nz, cs.size());
                return 1;
            }
        }
        static const char PF_PATTERN[] = "MEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";
        printf("  prefill: T=%d, %d chunks of %d, %d layers\n",
               T, (T + BATCH_PAD - 1) / BATCH_PAD, BATCH_PAD, n_backbone);
        predictor_mk2<<<grid, THREADS, smem>>>(g, d_bb, n_layers, n_pos, n_backbone);
        CHECK(cudaDeviceSynchronize());
        {
            const int warm = 3, iters = 20;
            for (int i = 0; i < warm; ++i)
                predictor_mk2<<<grid, THREADS, smem>>>(g, d_bb, n_layers, n_pos,
                                                       n_backbone);
            CHECK(cudaDeviceSynchronize());
            cudaEvent_t e0, e1;
            CHECK(cudaEventCreate(&e0));
            CHECK(cudaEventCreate(&e1));
            CHECK(cudaEventRecord(e0));
            for (int i = 0; i < iters; ++i)
                predictor_mk2<<<grid, THREADS, smem>>>(g, d_bb, n_layers, n_pos,
                                                       n_backbone);
            CHECK(cudaEventRecord(e1));
            CHECK(cudaDeviceSynchronize());
            float ms = 0.f;
            CHECK(cudaEventElapsedTime(&ms, e0, e1));
            const double per = ms / iters;
            printf("  TIMING prefill T=%d, %d layers: %.3f ms/launch"
                   "  (%.3f ms/chunk over %d chunks)\n",
                   T, n_backbone, per,
                   per / ((T + BATCH_PAD - 1) / BATCH_PAD),
                   (T + BATCH_PAD - 1) / BATCH_PAD);
            for (const MambaLayer &L : mamba_saved) {
                CHECK(cudaMemset(L.conv_state +
                                     (size_t)hb.prefill_slot * L.conv_slot_stride,
                                 0,
                                 (size_t)MB_CONV_DIM * (MB_CONV_K - 1) * sizeof(bf16)));
                CHECK(cudaMemset(L.ssm_state +
                                     (size_t)hb.prefill_slot * L.ssm_slot_stride,
                                 0,
                                 (size_t)MB_HEADS * MB_HEAD_DIM * MB_STATE *
                                     sizeof(float)));
            }
            predictor_mk2<<<grid, THREADS, smem>>>(g, d_bb, n_layers, n_pos, n_backbone);
            CHECK(cudaDeviceSynchronize());
        }
        std::vector<bf16> got((size_t)BB_LAYERS * PREFILL_MAX_TOKENS * D_MODEL);
        CHECK(cudaMemcpy(got.data(), pf_layer_out_host, got.size() * sizeof(bf16),
                         cudaMemcpyDeviceToHost));
        int bad_layers = 0;
        for (int layer = 0; layer < n_backbone; ++layer) {
            char name[64];
            snprintf(name, sizeof(name), "/layer%02d.bin", layer);
            auto want_raw = read_file(prefill_dir + name,
                                      (size_t)T * D_MODEL * sizeof(float));
            const float *want = reinterpret_cast<const float *>(want_raw.data());
            double worst_cos = 1.0;
            int worst_tok = -1;
            int bad_tokens = 0;
            for (int tok = 0; tok < T; ++tok) {
                double dot = 0, na = 0, nb = 0;
                for (int i = 0; i < D_MODEL; ++i) {
                    const double a = __bfloat162float(
                        got[((size_t)layer * PREFILL_MAX_TOKENS + tok) * D_MODEL + i]);
                    const double b = want[(size_t)tok * D_MODEL + i];
                    dot += a * b; na += a * a; nb += b * b;
                }
                const double cos = dot / (sqrt(na) * sqrt(nb) + 1e-30);
                if (cos < worst_cos) { worst_cos = cos; worst_tok = tok; }
                if (cos < 0.995) ++bad_tokens;
            }
            const bool ok = worst_cos > 0.995;
            if (!ok && bad_layers == 0) {
                printf("       kernel[0..5]:");
                for (int i = 0; i < 6; ++i)
                    printf(" %+8.4f", __bfloat162float(
                        got[((size_t)layer * PREFILL_MAX_TOKENS + 0) * D_MODEL + i]));
                printf("\n       oracle[0..5]:");
                for (int i = 0; i < 6; ++i) printf(" %+8.4f", want[i]);
                printf("\n       ratio       :");
                for (int i = 0; i < 6; ++i) {
                    const float a = __bfloat162float(
                        got[((size_t)layer * PREFILL_MAX_TOKENS + 0) * D_MODEL + i]);
                    printf(" %+8.4f", want[i] == 0.f ? 0.f : a / want[i]);
                }
                printf("\n");
            }
            if (!ok) ++bad_layers;
            printf("  %5d  %c  %s  worst cos %.6f at token %d  (%d/%d tokens bad)\n",
                   layer, PF_PATTERN[layer], ok ? "PASS" : "FAIL", worst_cos,
                   worst_tok, bad_tokens, T);
        }
        {
            auto dump = [&](const char *name, const void *dev, size_t bytes) {
                std::vector<char> host(bytes);
                CHECK(cudaMemcpy(host.data(), dev, bytes, cudaMemcpyDeviceToHost));
                const std::string path = std::string("/out/pf_") + name + ".bin";
                FILE *f = fopen(path.c_str(), "wb");
                if (f) { fwrite(host.data(), 1, bytes, f); fclose(f); }
            };
            const size_t rows = (size_t)min(T, BATCH_PAD);
            dump("normed", bb_saved->pf_normed, rows * D_MODEL * sizeof(bf16));
            dump("zxbcdt", bb_saved->pf_zxbcdt, rows * MB_IN_PROJ * sizeof(bf16));
            dump("xbc_act", bb_saved->pf_xbc_act, rows * MB_CONV_DIM * sizeof(bf16));
            dump("gated", bb_saved->pf_gated, rows * MB_INNER * sizeof(bf16));
            printf("  dumped last-tile intermediates to /out/pf_*.bin (%zu rows)\n",
                   rows);
        }
        {
            int bad_state = 0;
            double worst_state_cos = 1.0;
            int worst_state_layer = -1;
            for (int layer = 0; layer < n_backbone; ++layer) {
                if (PF_PATTERN[layer] != 'M') continue;
                const int mi = bb_saved->mamba_of_layer[layer];
                const size_t n = (size_t)MB_CONV_DIM * (MB_CONV_K - 1);
                std::vector<uint16_t> mine(n);
                CHECK(cudaMemcpy(mine.data(),
                                 mamba_saved[mi].conv_state +
                                     (size_t)hb.prefill_slot *
                                         mamba_saved[mi].conv_slot_stride,
                                 n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
                char nm[64];
                snprintf(nm, sizeof(nm), "/conv_state_out_%d.bin", layer);
                auto want_raw = read_file(prefill_dir + nm, n * sizeof(uint16_t));
                const uint16_t *want =
                    reinterpret_cast<const uint16_t *>(want_raw.data());
                double dot = 0, na = 0, nb = 0;
                for (size_t i = 0; i < n; ++i) {
                    const float a = __bfloat162float(*(const bf16 *)&mine[i]);
                    const float b = __bfloat162float(*(const bf16 *)&want[i]);
                    dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
                }
                const double cs = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb)
                                                     : (na == nb ? 1.0 : 0.0);
                if (cs < worst_state_cos) { worst_state_cos = cs; worst_state_layer = layer; }
                if (cs < 0.999) ++bad_state;
            }
            printf("  %s conv state out: worst cos %.6f at layer %d (%d bad)\n",
                   bad_state ? "FAIL" : "PASS", worst_state_cos, worst_state_layer,
                   bad_state);
            if (bad_state) ++bad_layers;
        }
        {
            const size_t n = (size_t)14 * PREFILL_MAX_TOKENS * BB_EXPERTS;
            std::vector<float> mine(n);
            CHECK(cudaMemcpy(mine.data(), pf_audit, n * sizeof(float),
                             cudaMemcpyDeviceToHost));
            auto want_raw = read_file(prefill_dir + "/moe_w.bin",
                                     (size_t)14 * T * BB_EXPERTS * sizeof(float));
            const float *want = reinterpret_cast<const float *>(want_raw.data());
            int rows_diff = 0, total_rows = 0;
            double worst_gap = 1e30;
            int worst_mo = -1, worst_tok = -1;
            for (int mo = 0; mo < 14; ++mo) {
                for (int tok = 0; tok < T; ++tok) {
                    const float *a = mine.data() +
                                     ((size_t)mo * PREFILL_MAX_TOKENS + tok) * BB_EXPERTS;
                    const float *b = want + ((size_t)mo * T + tok) * BB_EXPERTS;
                    ++total_rows;
                    bool same = true;
                    for (int e = 0; e < BB_EXPERTS; ++e)
                        if ((a[e] != 0.f) != (b[e] != 0.f)) { same = false; break; }
                    if (same) continue;
                    ++rows_diff;
                    double mine_only = 0.0, theirs_only = 0.0;
                    for (int e = 0; e < BB_EXPERTS; ++e) {
                        if (a[e] != 0.f && b[e] == 0.f) mine_only = std::max(mine_only, (double)a[e]);
                        if (b[e] != 0.f && a[e] == 0.f) theirs_only = std::max(theirs_only, (double)b[e]);
                    }
                    const double gap = std::abs(mine_only - theirs_only);
                    if (gap < worst_gap) { worst_gap = gap; worst_mo = mo; worst_tok = tok; }
                    if (rows_diff <= 3) {
                        printf("          flip mo=%d tok=%d\n            kernel:", mo, tok);
                        for (int e = 0; e < BB_EXPERTS; ++e)
                            if (a[e] != 0.f) printf(" e%d=%.4f", e, a[e]);
                        printf("\n            oracle:");
                        for (int e = 0; e < BB_EXPERTS; ++e)
                            if (b[e] != 0.f) printf(" e%d=%.4f", e, b[e]);
                        printf("\n");
                    }
                }
            }
            printf("  router: %d of %d (layer, token) rows pick a different expert set"
                   " (%.2f%%)\n", rows_diff, total_rows,
                   100.0 * rows_diff / total_rows);
            if (rows_diff)
                printf("          widest disagreement at MoE layer %d token %d: the two "
                       "contested experts differ by %.6g\n",
                       worst_mo, worst_tok, worst_gap);
        }
        printf("%s  prefill: %d/%d layers match the oracle\n",
               bad_layers ? "FAIL" : "PASS", n_backbone - bad_layers, n_backbone);
        return bad_layers ? 1 : 0;
    }
    if (decode_steps > 1 && !bb_saved) {
        printf("FAIL --decode-steps needs the backbone (--stage0 --backbone-layers)\n");
        return 1;
    }
    for (int step = 0; step < decode_steps; ++step) {
        if (step > 0) {
            Backbone nb = *bb_saved;
            nb.past_len = past_len + step;
            CHECK(cudaMemcpy(d_bb, &nb, sizeof(Backbone), cudaMemcpyHostToDevice));
        }
        predictor_mk2<<<grid, THREADS, smem>>>(g, d_bb, n_layers, n_pos, n_backbone);
        CHECK(cudaDeviceSynchronize());
    }
    if (launch_counter_host) {
        unsigned long long lc[6] = {0, 0, 0, 0, 0, 0};
        CHECK(cudaMemcpy(lc, launch_counter_host, sizeof(lc),
                         cudaMemcpyDeviceToHost));
        const bool ok = lc[0] == (unsigned long long)decode_steps &&
                        lc[1] == (unsigned long long)decode_steps;
        printf("%s  launch counter: entered=%llu completed=%llu, expected %d\n",
               ok ? "PASS" : "FAIL", lc[0], lc[1], decode_steps);
    }
    bool pass = true;
    if (n_backbone > 0 && dump_moe_w) {
        std::vector<float> mw((size_t)BATCH * BB_EXPERTS);
        CHECK(cudaMemcpy(mw.data(), bb_moe_w_host, mw.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));
        FILE *f = fopen(dump_moe_w, "wb");
        if (f) { fwrite(mw.data(), sizeof(float), mw.size(), f); fclose(f); }
        printf("  moe_w -> %s\n", dump_moe_w);
    }
    if (n_backbone > 0 && verify) {
        std::vector<bf16> got((size_t)BB_LAYERS * BATCH * D_MODEL);
        CHECK(cudaMemcpy(got.data(), bb_layer_out_host, got.size() * sizeof(bf16),
                         cudaMemcpyDeviceToHost));
        static const char PAT[] = "MEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";
        const std::string bb_root = stage0 ? std::string(root)
                                          : std::string(root) + "/../backbone";
        printf("  layer  kind   cos        max_abs/rms\n");
        for (int layer = 0; layer < n_backbone; ++layer) {
            char name[64];
            snprintf(name, sizeof(name), "/layer%02d_%c.bin", layer, PAT[layer]);
            auto want_raw = read_file(bb_root + name,
                                      (size_t)BATCH * D_MODEL * sizeof(float));
            const float *want = reinterpret_cast<const float *>(want_raw.data());
            double dot = 0, aa = 0, ee = 0, max_abs = 0, sq = 0;
            for (int i = 0; i < BATCH * D_MODEL; ++i) {
                const double a = __bfloat162float(
                    got[(size_t)layer * BATCH * D_MODEL + i]);
                const double e = want[i];
                dot += a * e; aa += a * a; ee += e * e;
                max_abs = fmax(max_abs, fabs(a - e));
                sq += e * e;
            }
            const double cos = dot / fmax(sqrt(aa * ee), 1e-30);
            const double rms = sqrt(sq / (BATCH * D_MODEL));
            const bool ok = cos > 0.999;
            printf("  %5d  %c    %s %.6f   %.4f\n", layer, PAT[layer],
                   ok ? "PASS" : "FAIL", cos, max_abs / fmax(rms, 1e-9));
            if (!ok) {
                printf("  first divergence at layer %d (%c); stopping\n",
                       layer, PAT[layer]);
                pass = false;
                break;
            }
        }
        if (moe_w_audit_host) {
            const size_t n = (size_t)14 * BATCH * BB_EXPERTS;
            std::vector<float> mine(n);
            CHECK(cudaMemcpy(mine.data(), moe_w_audit_host, n * sizeof(float),
                             cudaMemcpyDeviceToHost));
            auto want_raw = read_file(bb_root + "/moe_w.bin", n * sizeof(float));
            const float *want = reinterpret_cast<const float *>(want_raw.data());
            int rows_diff = 0, total_rows = 0;
            double worst_gap = 1e30;
            int worst_layer = -1;
            for (int mo = 0; mo < 14; ++mo) {
                for (int row = 0; row < BATCH; ++row) {
                    const size_t off = ((size_t)mo * BATCH + row) * BB_EXPERTS;
                    ++total_rows;
                    bool differ = false;
                    for (int e = 0; e < BB_EXPERTS; ++e) {
                        const bool a = mine[off + e] != 0.f;
                        const bool b = want[off + e] != 0.f;
                        if (a != b) { differ = true; break; }
                    }
                    if (!differ) continue;
                    ++rows_diff;
                    float mine_only = 0.f, want_only = 0.f;
                    for (int e = 0; e < BB_EXPERTS; ++e) {
                        if (mine[off + e] != 0.f && want[off + e] == 0.f)
                            mine_only = fmaxf(mine_only, mine[off + e]);
                        if (want[off + e] != 0.f && mine[off + e] == 0.f)
                            want_only = fmaxf(want_only, want[off + e]);
                    }
                    const double rel = fabs(mine_only - want_only) /
                                       fmax(1e-9, (double)fmax(mine_only, want_only));
                    printf("    MoE layer %2d row %2d: kernel's disputed pick weighs"
                           " %.4f, oracle's weighs %.4f  (%.2f%% apart)\n",
                           mo, row, mine_only, want_only, 100.0 * rel);
                    if (rel < worst_gap) { worst_gap = rel; worst_layer = mo; }
                }
            }
            printf("  router audit: %d of %d rows select a different expert set"
                   "  (%.2f%%)\n", rows_diff, total_rows,
                   100.0 * rows_diff / total_rows);
            if (rows_diff)
                printf("    uniform share for reference: %.4f"
                       "  (routed mass %.1f over top-%d)\n",
                       BB_ROUTED_SCALE / 4.0, BB_ROUTED_SCALE, 4);
        }
        if (n_backbone == BB_LAYERS && ph_out_verify) {
            std::vector<int> mine(BATCH);
            CHECK(cudaMemcpy(mine.data(), ph_out_verify, BATCH * sizeof(int),
                             cudaMemcpyDeviceToHost));
            auto want_raw = read_file(bb_root + "/phoneme_argmax.bin",
                                      (size_t)BATCH * sizeof(int));
            const int *want = reinterpret_cast<const int *>(want_raw.data());
            std::vector<float> margin((size_t)BATCH, 1e30f);
            bool have_margin = false;
            {
                const std::string mp = bb_root + "/phoneme_margin.bin";
                if (FILE *probe = fopen(mp.c_str(), "rb")) {
                    fclose(probe);
                    auto raw = read_file(mp, (size_t)BATCH * sizeof(float));
                    memcpy(margin.data(), raw.data(), raw.size());
                    have_margin = true;
                }
            }
            int bad = 0, tie = 0;
            for (int i = 0; i < BATCH; ++i) {
                if (mine[i] == want[i]) continue;
                if (margin[i] <= 1.0f) ++tie; else ++bad;
            }
            printf("%s  phoneme argmax: %d/%d exact", bad == 0 ? "PASS" : "FAIL",
                   BATCH - bad - tie, BATCH);
            if (tie)
                printf(", %d near-tie%s not graded", tie, tie == 1 ? "" : "s");
            if (!have_margin && (bad || tie))
                printf("  (no phoneme_margin.bin; every row treated as decidable)");
            printf("\n");
            for (int i = 0; i < BATCH; ++i) {
                if (mine[i] == want[i]) continue;
                printf("      row %d: oracle %d, kernel %d, oracle top-2 margin "
                       "%.3fx the bf16 accumulation scale -> %s\n",
                       i, want[i], mine[i], margin[i],
                       margin[i] <= 1.0f ? "NEAR-TIE, not a defect" : "DEFECT");
            }
            pass = pass && bad == 0;
        }
        if (n_backbone == BB_LAYERS) {
            std::vector<bf16> dh((size_t)BATCH * D_MODEL);
            CHECK(cudaMemcpy(dh.data(), bb_dec_snapshot_host,
                             dh.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
            auto want_raw = read_file(bb_root + "/dec_hidden.bin",
                                      (size_t)BATCH * D_MODEL * sizeof(float));
            const float *want = reinterpret_cast<const float *>(want_raw.data());
            double dot = 0, aa = 0, ee = 0;
            for (int i = 0; i < BATCH * D_MODEL; ++i) {
                const double a = __bfloat162float(dh[i]), e = want[i];
                dot += a * e; aa += a * a; ee += e * e;
            }
            const double cos = dot / fmax(sqrt(aa * ee), 1e-30);
            printf("  %s  dec_hidden (backbone -> predictor handoff): cos=%.6f\n",
                   cos > 0.9995 ? "PASS" : "FAIL", cos);
            pass = pass && cos > 0.9995;
        }
    }
    {
        std::vector<int> mine((size_t)BATCH * N_POS);
        CHECK(cudaMemcpy(mine.data(), sampled, mine.size() * sizeof(int),
                         cudaMemcpyDeviceToHost));
        auto refc = read_file(root + "/sampled_codes.bin",
                              (size_t)BATCH * N_POS * sizeof(int));
        const int *want_c = reinterpret_cast<const int *>(refc.data());
        int bad = 0, first_bad_pos = -1;
        std::vector<int> per_pos(n_pos, 0);
        for (int row = 0; row < BATCH; ++row)
            for (int k = 0; k < n_pos; ++k)
                if (mine[row * N_POS + k] != want_c[row * N_POS + k]) {
                    ++bad;
                    ++per_pos[k];
                    if (first_bad_pos < 0) first_bad_pos = k;
                }
        printf("%s  sampled codes: %d/%d exact",
               bad == 0 ? "SAMPLE-PASS" : "SAMPLE-FAIL",
               BATCH * n_pos - bad, BATCH * n_pos);
        if (bad) {
            printf("   first divergence at position %d, per-position:",
                   first_bad_pos);
            for (int k = 0; k < n_pos; ++k) printf(" %d", per_pos[k]);
        }
        printf("\n");
        pass = pass && bad == 0;
    }
    if (dump_final) {
        std::vector<bf16> dh((size_t)BATCH * D_MODEL);
        std::vector<int> cd((size_t)BATCH * N_POS);
        if (bb_dec_snapshot_host)
            CHECK(cudaMemcpy(dh.data(), bb_dec_snapshot_host,
                             dh.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(cd.data(), sampled, cd.size() * sizeof(int),
                         cudaMemcpyDeviceToHost));
        FILE *f = fopen(dump_final, "wb");
        if (!f) { printf("FAIL cannot write %s\n", dump_final); return 1; }
        fwrite(dh.data(), sizeof(bf16), dh.size(), f);
        fwrite(cd.data(), sizeof(int), cd.size(), f);
        fclose(f);
        printf("  final state -> %s (%zu B dec_hidden + %zu B codes)\n", dump_final,
               dh.size() * sizeof(bf16), cd.size() * sizeof(int));
    }
    std::vector<bf16> got((size_t)N_POS * BATCH * D_MODEL);
    CHECK(cudaMemcpy(got.data(), g.hidden_rows, got.size() * sizeof(bf16),
                     cudaMemcpyDeviceToHost));
    if (dump_rows) {
        FILE *f = fopen(dump_rows, "wb");
        if (!f) { printf("FAIL cannot write %s\n", dump_rows); return 1; }
        fwrite(got.data(), sizeof(bf16), (size_t)n_pos * BATCH * D_MODEL, f);
        fclose(f);
        printf("  rows -> %s\n", dump_rows);
    }
    if (verify && !stage0) {
    auto want_raw = read_file(root + "/hidden_rows.bin",
                              (size_t)BATCH * n_pos * D_MODEL * sizeof(float));
    const float *want = reinterpret_cast<const float *>(want_raw.data());
    double worst_ratio = 0.0, all_dot = 0.0, all_aa = 0.0, all_ee = 0.0;
    int worst_pos = -1;
    for (int pos = 0; pos < n_pos; ++pos) {
        double max_abs = 0.0, sq = 0.0, dot = 0.0, aa = 0.0, ee = 0.0;
        for (int row = 0; row < BATCH; ++row) {
            for (int i = 0; i < D_MODEL; ++i) {
                const double a =
                    __bfloat162float(got[((size_t)pos * BATCH + row) * D_MODEL + i]);
                const double e = want[((size_t)row * n_pos + pos) * D_MODEL + i];
                max_abs = fmax(max_abs, fabs(a - e));
                sq += e * e; dot += a * e; aa += a * a; ee += e * e;
            }
        }
        const double rms = sqrt(sq / (BATCH * D_MODEL));
        const double ratio = max_abs / fmax(rms, 1e-9);
        all_dot += dot; all_aa += aa; all_ee += ee;
        if (ratio > worst_ratio) { worst_ratio = ratio; worst_pos = pos; }
        printf("  pos %2d  max_abs=%.4f  rms=%.4f  max_abs/rms=%.4f  cos=%.6f\n",
               pos, max_abs, rms, ratio, dot / fmax(sqrt(aa * ee), 1e-30));
    }
    const double cosine_all = all_dot / fmax(sqrt(all_aa * all_ee), 1e-30);
    pass = cosine_all > 0.9995;
    printf("%s  worst max_abs/rms=%.4f (pos %d)  cos_all=%.6f\n",
           pass ? "PASS" : "FAIL", worst_ratio, worst_pos, cosine_all);
    {
        std::vector<char> refc = read_file(root + "/ref_codes.bin",
                                          (size_t)BATCH * n_pos * sizeof(int));
        const int *want_codes = reinterpret_cast<const int *>(refc.data());
        int mismatches = 0, checked_codes = 0;
        double worst_margin = 1e30;
        int self_mismatch = 0;
        double max_flip_gap = 0.0;
        double sum_flip_gap = 0.0;
        for (int k = 0; k < n_pos; ++k) {
            auto pw = read_file(root + "/out_proj_" + std::to_string(k) + ".bin",
                                (size_t)CODEBOOK * D_MODEL * sizeof(bf16));
            auto pb = read_file(root + "/out_bias_" + std::to_string(k) + ".bin",
                                (size_t)CODEBOOK * sizeof(bf16));
            const bf16 *w = reinterpret_cast<const bf16 *>(pw.data());
            const bf16 *b = reinterpret_cast<const bf16 *>(pb.data());
            for (int row = 0; row < BATCH; ++row) {
                const int ref_code = want_codes[row * n_pos + k];
                const bf16 *h = &got[((size_t)k * BATCH + row) * D_MODEL];
                int best = -1;
                float best_v = -1e30f, second = -1e30f, at_ref = 0.f;
                for (int n = 0; n < CODEBOOK; ++n) {
                    float acc = 0.f;
                    const bf16 *wr = w + (size_t)n * D_MODEL;
                    for (int i = 0; i < D_MODEL; ++i)
                        acc += __bfloat162float(h[i]) * __bfloat162float(wr[i]);
                    acc += __bfloat162float(b[n]);
                    if (n == ref_code) at_ref = acc;
                    if (acc > best_v) { second = best_v; best_v = acc; best = n; }
                    else if (acc > second) { second = acc; }
                }
                worst_margin = fmin(worst_margin, (double)(best_v - second));
                ++checked_codes;
                int sbest = -1;
                float sbest_v = -1e30f;
                for (int n = 0; n < CODEBOOK; ++n) {
                    float acc = 0.f;
                    const bf16 *wr = w + (size_t)n * D_MODEL;
                    for (int i = 0; i < D_MODEL; ++i)
                        acc += __bfloat162float(__float2bfloat16(
                                   want[((size_t)row * n_pos + k) * D_MODEL + i])) *
                               __bfloat162float(wr[i]);
                    acc += __bfloat162float(b[n]);
                    if (acc > sbest_v) { sbest_v = acc; sbest = n; }
                }
                if (sbest != ref_code) ++self_mismatch;
                if (best != sbest) {
                    ++mismatches;
                    const double gap = best_v - at_ref;
                    max_flip_gap = fmax(max_flip_gap, fabs(gap));
                    sum_flip_gap += fabs(gap);
                }
            }
        }
        const double agree = 100.0 * (checked_codes - mismatches) / checked_codes;
        printf("  host-vs-torch projection noise: %d/%d (%.2f%%) - not the gate\n",
               self_mismatch, checked_codes, 100.0 * self_mismatch / checked_codes);
        printf("%s  sampled codes vs reference rows: %d/%d agree (%.2f%%), %d flip, "
               "min argmax margin %.5f\n",
               mismatches == 0 ? "GATE-PASS" : "GATE-FAIL",
               checked_codes - mismatches, checked_codes, agree, mismatches,
               worst_margin);
        if (mismatches)
            printf("  flipped decisions: mean logit gap %.5f, max %.5f"
                   "  (small gaps = genuine near-ties)\n",
                   sum_flip_gap / mismatches, max_flip_gap);
        pass = pass && mismatches == 0;
    }
    }
    if (bench) {
        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0));
        CHECK(cudaEventCreate(&t1));
        for (int i = 0; i < 3; ++i) {
            CHECK(cudaMemcpy(x, x_initial, (size_t)BATCH * D_MODEL * sizeof(bf16),
                             cudaMemcpyDeviceToDevice));
            predictor_mk2<<<grid, THREADS, smem>>>(g, d_bb, n_layers, n_pos, n_backbone);
        }
        CHECK(cudaDeviceSynchronize());
        const int iters = bench_iters;
        std::vector<cudaEvent_t> marks(iters + 1);
        for (auto &ev : marks) CHECK(cudaEventCreate(&ev));
        CHECK(cudaEventRecord(marks[0]));
        for (int i = 0; i < iters; ++i) {
            predictor_mk2<<<grid, THREADS, smem>>>(g, d_bb, n_layers, n_pos, n_backbone);
            CHECK(cudaEventRecord(marks[i + 1]));
        }
        CHECK(cudaEventSynchronize(marks[iters]));
        std::vector<double> samples;
        samples.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            float one = 0.f;
            CHECK(cudaEventElapsedTime(&one, marks[i], marks[i + 1]));
            samples.push_back(one);
        }
        float span = 0.f;
        CHECK(cudaEventElapsedTime(&span, marks[0], marks[iters]));
        for (auto &ev : marks) CHECK(cudaEventDestroy(ev));
        double total = 0.0;
        for (double s : samples) total += s;
        std::sort(samples.begin(), samples.end());
        const double per_step = samples[iters / 2];
        printf("  bench: %d iters   median %.4f ms   mean %.4f ms   "
               "aggregate %.4f ms   min %.4f   p05 %.4f   p95 %.4f   max %.4f   "
               "IQR %.4f\n",
               iters, per_step, total / iters, span / iters, samples.front(),
               samples[(size_t)(0.05 * iters)], samples[(size_t)(0.95 * iters)],
               samples.back(),
               samples[(size_t)(0.75 * iters)] - samples[(size_t)(0.25 * iters)]);
        const double weight_bytes = (double)n_pos * n_layers *
            ((3.0 * D_MODEL * D_MODEL) + (double)D_MODEL * D_MODEL +
             (double)D_FFN * D_MODEL + (double)D_MODEL * D_FFN) * 2.0;
        printf("  per decode step: %.3f ms   weights %.2f GB   %.2f TB/s"
               "   floor at 3.35 TB/s: %.3f ms\n",
               per_step, weight_bytes / 1e9, weight_bytes / (per_step * 1e-3) / 1e12,
               weight_bytes / 3.35e12 * 1e3);
        printf("  for comparison: mma.sync megakernel 9.1 ms, TensorRT 3.414 ms\n");
    }
    return pass ? 0 : 1;
}
#endif
#include "mk2_abi.cuh"
namespace {
thread_local std::string g_last_error;
struct MK2State {
    Globals g;
    int batch;
    int grid;
    int smem;
    bf16 *hidden_bf16;
    int kind;
    int device;
    std::vector<void *> owned;
    int *run_predicate = nullptr;
    const unsigned char *native_predictor_active_mask = nullptr;
    MK2_PROGRESS_MEMBER
    int maximum_grid = 0;
    bool grid_reset_pending = false;
};
static void mk2_check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct MK2Allocations {
    std::vector<void *> ptrs;
    ~MK2Allocations() { for (void *p : ptrs) cudaFree(p); }
    template <typename T> cudaError_t alloc(T **out, size_t n) {
        cudaError_t e = cudaMalloc(out, n);
        if (e == cudaSuccess) {
            try { ptrs.push_back(*out); }
            catch (...) { cudaFree(*out); throw; }
        }
        return e;
    }
    template <typename T> T *upload_raw(const std::string &path, size_t count) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("cannot open " + path);
        const auto end = f.tellg();
        if (end <= 0 || size_t(end) % sizeof(T) ||
            (count && size_t(end) != count * sizeof(T)))
            throw std::runtime_error("wrong byte count: " + path);
        std::vector<char> data((size_t(end)));
        f.seekg(0);
        if (!f.read(data.data(), data.size()))
            throw std::runtime_error("short read: " + path);
        T *d = nullptr;
        mk2_check(alloc(&d, data.size()));
        mk2_check(cudaMemcpy(d, data.data(), data.size(), cudaMemcpyHostToDevice));
        return d;
    }
};
}
extern "C" int mk2_build_info(MK2BuildInfo *out) {
    if (!out) { g_last_error = "null build info"; return 1; }
    *out = {1, BATCH, BATCH_PAD, STAGES, KCHUNK, MK2_DIRECT_INPUT,
            MK2_B1_LOCAL_BARRIERS, MK2_SKIP_HIDDEN_SNAPSHOT, THREADS,
            int(STAGES * SLOTS * sizeof(weight_tile) +
                sizeof(out_tile_f) + 2048), 0, 0, 0, 0, 0, 0};
    try {
        mk2_check(cudaGetDevice(&out->device));
        cudaFuncAttributes attr{};
        mk2_check(cudaFuncGetAttributes(&attr, predictor_mk2));
        out->static_smem = int(attr.sharedSizeBytes);
        out->registers = attr.numRegs;
        int optin = 0;
        mk2_check(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                        out->device));
        mk2_check(cudaDeviceGetAttribute(&out->sms, cudaDevAttrMultiProcessorCount,
                                        out->device));
        if (out->dynamic_smem + out->static_smem > optin)
            throw std::runtime_error("shared memory exceeds device limit: dynamic=" +
                std::to_string(out->dynamic_smem) + " static=" +
                std::to_string(out->static_smem) + " limit=" + std::to_string(optin));
        mk2_check(cudaFuncSetAttribute(predictor_mk2,
            cudaFuncAttributeMaxDynamicSharedMemorySize, out->dynamic_smem));
        mk2_check(cudaFuncSetAttribute(predictor_mk2_startup,
            cudaFuncAttributeMaxDynamicSharedMemorySize, out->dynamic_smem));
        mk2_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &out->blocks_per_sm, predictor_mk2, THREADS, out->dynamic_smem));
        if (out->blocks_per_sm < 1)
            throw std::runtime_error("no resident block; grid barrier cannot launch");
        out->grid = std::min(out->sms * out->blocks_per_sm, 4096);
        if (const char *e = getenv("NEMOTRON_STAGE0_MEGAKERNEL_GRID")) {
            const int want = atoi(e);
            if (want > 0) out->grid = std::min(out->grid, want);
        }
        g_last_error.clear();
        return 0;
    } catch (const std::exception &e) { g_last_error = e.what(); return 1; }
}
extern "C" MK2Predictor *mk2_create_from_weights(const MK2Weights *w, int batch) {
    if (batch != BATCH) {
        g_last_error = "this build supports batch " + std::to_string(BATCH) +
                       " only; got " + std::to_string(batch);
        return nullptr;
    }
    if (!w) { g_last_error = "null weights"; return nullptr; }
    MK2BuildInfo info{};
    if (mk2_build_info(&info)) return nullptr;
    for (int i = 0; i < N_LAYERS; ++i)
        if (!w->qkv[i] || !w->o[i] || !w->ff_in[i] || !w->ff_out[i] ||
            !w->norm_self[i] || !w->norm_ff[i]) {
            g_last_error = "null layer weight"; return nullptr;
        }
    if (!w->pos_emb) { g_last_error = "null position embedding"; return nullptr; }
    for (int i = 0; i < N_POS; ++i)
        if (!w->head_w[i] || !w->head_b[i] || !w->audio_emb[i]) {
            g_last_error = "null head/audio embedding"; return nullptr;
        }
    try {
    MK2Allocations owned;
    bf16 *head_w[N_POS];
    const bf16 *head_b[N_POS];
    for (int k = 0; k < N_POS; ++k) {
        mk2_check(owned.alloc(&head_w[k], (size_t)VOCAB_PADDED * D_MODEL * sizeof(bf16)));
        mk2_check(cudaMemset(head_w[k], 0, (size_t)VOCAB_PADDED * D_MODEL * sizeof(bf16)));
        mk2_check(cudaMemcpy(head_w[k], w->head_w[k],
                         (size_t)CODEBOOK * D_MODEL * sizeof(bf16),
                         cudaMemcpyDeviceToDevice));
        bf16 *bd = nullptr;
        mk2_check(owned.alloc(&bd, (size_t)VOCAB_PADDED * sizeof(bf16)));
        mk2_check(cudaMemset(bd, 0, (size_t)VOCAB_PADDED * sizeof(bf16)));
        mk2_check(cudaMemcpy(bd, w->head_b[k], (size_t)CODEBOOK * sizeof(bf16),
                         cudaMemcpyDeviceToDevice));
        head_b[k] = bd;
    }
    std::vector<int8_t> forb(VOCAB_PADDED, 0);
    for (int n = CODEBOOK - 8; n < VOCAB_PADDED; ++n) forb[n] = 1;
    forb[AUDIO_EOS] = 0;
    int8_t *forbidden = nullptr;
    mk2_check(owned.alloc(&forbidden, forb.size()));
    mk2_check(cudaMemcpy(forbidden, forb.data(), forb.size(), cudaMemcpyHostToDevice));
    bf16 *x, *qkv_out, *attn_out, *ff_hidden, *normed, *kcache, *vcache, *hidden_rows;
    bf16 *logits_buf, *hidden_staging;
    int *sampled, *sampled_pos, *codes_unused;
    unsigned int *barrier;
    int *run_predicate;
    mk2_check(owned.alloc(&run_predicate, sizeof(int)));
    mk2_check(owned.alloc(&x, (size_t)BATCH * D_MODEL * sizeof(bf16)));
    mk2_check(owned.alloc(&qkv_out, (size_t)BATCH * 3 * D_MODEL * sizeof(bf16)));
    mk2_check(owned.alloc(&attn_out, (size_t)BATCH * D_MODEL * sizeof(bf16)));
    mk2_check(owned.alloc(&ff_hidden, (size_t)BATCH * D_FFN * sizeof(bf16)));
    mk2_check(owned.alloc(&normed, (size_t)BATCH * D_MODEL * sizeof(bf16)));
    const size_t cache_elems = (size_t)N_LAYERS * BATCH * N_HEADS * N_POS * D_HEAD;
    mk2_check(owned.alloc(&kcache, cache_elems * sizeof(bf16)));
    mk2_check(owned.alloc(&vcache, cache_elems * sizeof(bf16)));
    mk2_check(owned.alloc(&hidden_rows, (size_t)N_POS * BATCH * D_MODEL * sizeof(bf16)));
    mk2_check(owned.alloc(&logits_buf, (size_t)BATCH * VOCAB_PADDED * sizeof(bf16)));
    mk2_check(owned.alloc(&sampled, (size_t)BATCH * N_POS * sizeof(int)));
    mk2_check(owned.alloc(&sampled_pos, (size_t)BATCH * sizeof(int)));
    mk2_check(owned.alloc(&barrier, 64 * sizeof(unsigned int)));
    mk2_check(cudaMemset(barrier, 0, 64 * sizeof(unsigned int)));
    hidden_staging = nullptr;
    if constexpr (!MK2_DIRECT_INPUT)
        mk2_check(owned.alloc(&hidden_staging, (size_t)BATCH * D_MODEL * sizeof(bf16)));
    mk2_check(owned.alloc(&codes_unused, (size_t)BATCH * N_POS * sizeof(int)));
    mk2_check(cudaMemset(codes_unused, 0, (size_t)BATCH * N_POS * sizeof(int)));
    const size_t DM = D_MODEL, DF = D_FFN, B = BATCH, QKV = 3 * D_MODEL;
    const size_t VP = VOCAB_PADDED;
    auto W = [&](int i, const void *const *arr, size_t rows, size_t cols) {
        return weight_gl{(bf16 *)arr[i], nullptr, nullptr, rows, cols};
    };
    Globals g{
        {W(0, w->qkv, QKV, DM), W(1, w->qkv, QKV, DM), W(2, w->qkv, QKV, DM)},
        {W(0, w->o, DM, DM), W(1, w->o, DM, DM), W(2, w->o, DM, DM)},
        {W(0, w->ff_in, DF, DM), W(1, w->ff_in, DF, DM), W(2, w->ff_in, DF, DM)},
        {W(0, w->ff_out, DM, DF), W(1, w->ff_out, DM, DF), W(2, w->ff_out, DM, DF)},
        {weight_gl{head_w[0], nullptr, nullptr, VP, DM},
         weight_gl{head_w[1], nullptr, nullptr, VP, DM},
         weight_gl{head_w[2], nullptr, nullptr, VP, DM},
         weight_gl{head_w[3], nullptr, nullptr, VP, DM},
         weight_gl{head_w[4], nullptr, nullptr, VP, DM},
         weight_gl{head_w[5], nullptr, nullptr, VP, DM},
         weight_gl{head_w[6], nullptr, nullptr, VP, DM},
         weight_gl{head_w[7], nullptr, nullptr, VP, DM},
         weight_gl{head_w[8], nullptr, nullptr, VP, DM},
         weight_gl{head_w[9], nullptr, nullptr, VP, DM},
         weight_gl{head_w[10], nullptr, nullptr, VP, DM},
         weight_gl{head_w[11], nullptr, nullptr, VP, DM},
         weight_gl{head_w[12], nullptr, nullptr, VP, DM},
         weight_gl{head_w[13], nullptr, nullptr, VP, DM},
         weight_gl{head_w[14], nullptr, nullptr, VP, DM},
         weight_gl{head_w[15], nullptr, nullptr, VP, DM}},
        act_gl{x, nullptr, nullptr, B, DM},
        act_gl{normed, nullptr, nullptr, B, DM},
        act_gl{attn_out, nullptr, nullptr, B, DM},
        act_gl{ff_hidden, nullptr, nullptr, B, DF},
        x, qkv_out, attn_out, ff_hidden, normed,
        logits_buf,
        {head_b[0], head_b[1], head_b[2], head_b[3], head_b[4], head_b[5],
         head_b[6], head_b[7], head_b[8], head_b[9], head_b[10], head_b[11],
         head_b[12], head_b[13], head_b[14], head_b[15]},
        forbidden, nullptr, sampled, sampled_pos, 0.7f,
        {(const float *)w->norm_self[0], (const float *)w->norm_self[1],
         (const float *)w->norm_self[2]},
        {(const float *)w->norm_ff[0], (const float *)w->norm_ff[1],
         (const float *)w->norm_ff[2]},
        (const bf16 *)w->pos_emb,
        {(const bf16 *)w->audio_emb[0], (const bf16 *)w->audio_emb[1],
         (const bf16 *)w->audio_emb[2], (const bf16 *)w->audio_emb[3],
         (const bf16 *)w->audio_emb[4], (const bf16 *)w->audio_emb[5],
         (const bf16 *)w->audio_emb[6], (const bf16 *)w->audio_emb[7],
         (const bf16 *)w->audio_emb[8], (const bf16 *)w->audio_emb[9],
         (const bf16 *)w->audio_emb[10], (const bf16 *)w->audio_emb[11],
         (const bf16 *)w->audio_emb[12], (const bf16 *)w->audio_emb[13],
         (const bf16 *)w->audio_emb[14], (const bf16 *)w->audio_emb[15]},
        kcache, vcache, hidden_rows, codes_unused, barrier,
    };
    auto *st = new MK2State{g, BATCH, info.grid, info.dynamic_smem,
                            hidden_staging, 0, info.device, {}, run_predicate};
    st->maximum_grid = info.sms * info.blocks_per_sm;
    st->owned.swap(owned.ptrs);
    return reinterpret_cast<MK2Predictor *>(st);
    } catch (const std::exception &e) { g_last_error = e.what(); return nullptr; }
}
extern "C" MK2Predictor *mk2_create(const char *raw_dir, int batch) {
    if (!raw_dir || batch != BATCH) {
        g_last_error = "null raw_dir or wrong batch"; return nullptr;
    }
    MK2BuildInfo info{};
    if (mk2_build_info(&info)) return nullptr;
    MK2Predictor *p = nullptr;
    try {
    MK2Allocations owned;
    const std::string root = raw_dir;
    MK2Weights w{};
    for (int l = 0; l < N_LAYERS; ++l) {
        const std::string p = root + "/l" + std::to_string(l) + "_";
        w.qkv[l] = owned.upload_raw<bf16>(p + "qkv.bin", (size_t)3 * D_MODEL * D_MODEL);
        w.o[l] = owned.upload_raw<bf16>(p + "o.bin", (size_t)D_MODEL * D_MODEL);
        w.ff_in[l] = owned.upload_raw<bf16>(p + "ff_in.bin", (size_t)D_FFN * D_MODEL);
        w.ff_out[l] = owned.upload_raw<bf16>(p + "ff_out.bin", (size_t)D_MODEL * D_FFN);
        w.norm_self[l] = owned.upload_raw<float>(p + "norm_self.bin", D_MODEL);
        w.norm_ff[l] = owned.upload_raw<float>(p + "norm_ff.bin", D_MODEL);
    }
    w.pos_emb = owned.upload_raw<bf16>(root + "/position_embeddings.bin", (size_t)18 * D_MODEL);
    for (int k = 0; k < N_POS; ++k) {
        w.audio_emb[k] =
            owned.upload_raw<bf16>(root + "/audio_emb_" + std::to_string(k) + ".bin", 0);
        w.head_w[k] = owned.upload_raw<bf16>(root + "/out_proj_" + std::to_string(k) + ".bin",
                                   (size_t)CODEBOOK * D_MODEL);
        w.head_b[k] = owned.upload_raw<bf16>(root + "/out_bias_" + std::to_string(k) + ".bin",
                                   (size_t)CODEBOOK);
    }
    p = mk2_create_from_weights(&w, batch);
    if (!p) return nullptr;
    auto *st = reinterpret_cast<MK2State *>(p);
    st->owned.insert(st->owned.end(), owned.ptrs.begin(), owned.ptrs.end());
    owned.ptrs.clear();
    return p;
    } catch (const std::exception &e) {
        if (p) mk2_destroy(p);
        g_last_error = e.what(); return nullptr;
    }
}
extern "C" int mk2_run(MK2Predictor *p, const void *hidden_fp16,
                       const void *gumbel_fp32, float temperature,
                       void *codes_int64, void *stream) {
    auto *st = reinterpret_cast<MK2State *>(p);
    if (!st) { g_last_error = "null handle"; return 1; }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int n_hidden = BATCH * D_MODEL;
    fp16_to_bf16<<<(n_hidden + 255) / 256, 256, 0, s>>>(
        static_cast<const __half *>(hidden_fp16),
        MK2_DIRECT_INPUT ? st->g.x : st->hidden_bf16, n_hidden);
    if constexpr (!MK2_DIRECT_INPUT) {
        const auto copy_error = cudaMemcpyAsync(st->g.x, st->hidden_bf16,
            (size_t)n_hidden * sizeof(bf16), cudaMemcpyDeviceToDevice, s);
        if (copy_error != cudaSuccess) {
            g_last_error = cudaGetErrorString(copy_error); return 1;
        }
    }
    st->g.gumbel = static_cast<const float *>(gumbel_fp32);
    st->g.temperature = temperature;
    predictor_mk2<<<st->grid, THREADS, st->smem, s>>>(st->g, nullptr, N_LAYERS, N_POS, 0);
    codes_to_int64<<<(BATCH * N_POS + 255) / 256, 256, 0, s>>>(
        st->g.sampled, static_cast<long long *>(codes_int64), BATCH * N_POS);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { g_last_error = cudaGetErrorString(e); return 1; }
    return 0;
}
extern "C" int mk2_run_masked(MK2Predictor *p, const void *hidden_fp16,
                              const void *gumbel_fp32, float temperature,
                              const void *audio_valid_bool, void *codes_int64,
                              void *stream) {
    auto *st = reinterpret_cast<MK2State *>(p);
    if (!st || st->kind != 0 || !hidden_fp16 || !gumbel_fp32 ||
        !audio_valid_bool || !codes_int64 || !(temperature > 0.f) ||
        !std::isfinite(temperature)) {
        g_last_error = "masked predictor requires predictor handle, non-null device buffers and finite positive temperature";
        return 1;
    }
    cudaPointerAttributes attributes{};
    const cudaError_t pointer_error = cudaPointerGetAttributes(&attributes, audio_valid_bool);
    if (pointer_error != cudaSuccess) {
        (void)cudaGetLastError();
        g_last_error = "audio_valid must be a device bool buffer";
        return 1;
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != st->device) {
        g_last_error = "audio_valid must be a device bool buffer on the predictor device";
        return 1;
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int n_hidden = BATCH * D_MODEL;
    fp16_to_bf16_masked<<<(n_hidden + 255) / 256, 256, 0, s>>>(
        static_cast<const __half *>(hidden_fp16),
        MK2_DIRECT_INPUT ? st->g.x : st->hidden_bf16, n_hidden,
        static_cast<const unsigned char *>(audio_valid_bool), st->run_predicate);
    if constexpr (!MK2_DIRECT_INPUT) {
        const auto copy_error = cudaMemcpyAsync(st->g.x, st->hidden_bf16,
            (size_t)n_hidden * sizeof(bf16), cudaMemcpyDeviceToDevice, s);
        if (copy_error != cudaSuccess) {
            g_last_error = cudaGetErrorString(copy_error); return 1;
        }
    }
    st->g.gumbel = static_cast<const float *>(gumbel_fp32);
    st->g.temperature = temperature;
    predictor_mk2<<<st->grid, THREADS, st->smem, s>>>(
        st->g, nullptr, N_LAYERS, N_POS, 0, st->run_predicate);
    codes_to_int64<<<(BATCH * N_POS + 255) / 256, 256, 0, s>>>(
        st->g.sampled, static_cast<long long *>(codes_int64), BATCH * N_POS);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { g_last_error = cudaGetErrorString(e); return 1; }
    return 0;
}
namespace {
struct MK2Stage0State {
    MK2State pred;
    Backbone *d_bb;
    Backbone host_bb;
    std::vector<MambaLayer> mb_host;
    std::vector<AttnLayer> at_host;
    Backbone *bb_stage;
    MambaLayer *mb_stage;
    AttnLayer *at_stage;
    std::vector<void *> owned;
    std::vector<void *> pinned;
    MK2Stage0Bindings pinned_bindings{};
    bool bindings_initialized = false;
};
struct MK2Stage0PinnedAllocations {
    std::vector<void *> ptrs;
    ~MK2Stage0PinnedAllocations() {
        for (void *p : ptrs) cudaFreeHost(p);
    }
    template <typename T> cudaError_t alloc(T **out, size_t n) {
        const cudaError_t e = cudaHostAlloc(out, n, cudaHostAllocDefault);
        if (e == cudaSuccess) {
            try { ptrs.push_back(*out); }
            catch (...) { cudaFreeHost(*out); *out = nullptr; throw; }
        }
        return e;
    }
};
struct MK2Stage0BaseDeleter {
    void operator()(MK2State *st) const noexcept {
        if (!st) return;
        for (void *p : st->owned) cudaFree(p);
        delete st;
    }
};
static void mk2_stage0_device_pointer(const void *p, int device,
                                      const char *name) {
    if (!p) throw std::runtime_error(std::string("null Stage 0 binding: ") + name);
    cudaPointerAttributes attributes{};
    const cudaError_t e = cudaPointerGetAttributes(&attributes, p);
    if (e != cudaSuccess) {
        (void)cudaGetLastError();
        throw std::runtime_error(std::string("Stage 0 binding is not a device pointer: ") + name);
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
        throw std::runtime_error(std::string("Stage 0 binding must be on the handle device: ") + name);
}
static bool mk2_stage0_same_pinned_bindings(const MK2Stage0Bindings &a,
                                           const MK2Stage0Bindings &b) {
    for (int i = 0; i < 14; ++i)
        if (a.conv_state[i] != b.conv_state[i] ||
            a.conv_slot_stride[i] != b.conv_slot_stride[i] ||
            a.ssm_state[i] != b.ssm_state[i] ||
            a.ssm_slot_stride[i] != b.ssm_slot_stride[i]) return false;
    for (int i = 0; i < 3; ++i)
        if (a.kv_pool_base[i] != b.kv_pool_base[i] ||
            a.kv_block_table[i] != b.kv_block_table[i]) return false;
    return a.slot_map == b.slot_map && a.seq_len == b.seq_len &&
           a.kv_max_blocks_per_seq == b.kv_max_blocks_per_seq &&
           a.kv_tokens_per_block == b.kv_tokens_per_block &&
           a.phoneme_out == b.phoneme_out &&
           a.moe_w_inject_debug == b.moe_w_inject_debug;
}
template <typename T>
T *upload_layers(const std::vector<T> &v, MK2Allocations &owned) {
    T *d = nullptr;
    mk2_check(owned.alloc(&d, v.size() * sizeof(T)));
    mk2_check(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
}
extern "C" MK2Predictor *mk2_create_stage0(const MK2Weights *pw,
                                           const MK2BackboneWeights *bw,
                                           int batch) {
    try {
    if (!pw || !bw) { g_last_error = "null weights"; return nullptr; }
    if (batch != BATCH) {
        g_last_error = "full Stage 0 build supports batch " + std::to_string(BATCH);
        return nullptr;
    }
    {
        struct Named { const void *p; const char *name; };
        const Named required[] = {
            {bw->norm_f, "norm_f"},
            {bw->phoneme_w, "phoneme_w"},
            {bw->phoneme_b, "phoneme_b"},
        };
        for (const auto &r : required) {
            if (!r.p) {
                g_last_error = std::string("backbone weight '") + r.name +
                               "' is null";
                return nullptr;
            }
        }
        for (int l = 0; l < BB_LAYERS; ++l)
            if (!bw->norm[l]) {
                g_last_error = "backbone weight 'norm[" + std::to_string(l) +
                               "]' is null";
                return nullptr;
            }
        for (int i = 0; i < 14; ++i) {
            const Named per[] = {
                {bw->mb_in_proj[i], "mb_in_proj"},
                {bw->mb_out_proj[i], "mb_out_proj"},
                {bw->mb_conv_w[i], "mb_conv_w"},
                {bw->mb_conv_b[i], "mb_conv_b"},
                {bw->mb_A[i], "mb_A"},
                {bw->mb_D[i], "mb_D"},
                {bw->mb_dt_bias[i], "mb_dt_bias"},
                {bw->mb_gnorm[i], "mb_gnorm"},
                {bw->moe_exp_up[i], "moe_exp_up"},
                {bw->moe_exp_down[i], "moe_exp_down"},
                {bw->moe_sh_up[i], "moe_sh_up"},
                {bw->moe_sh_down[i], "moe_sh_down"},
                {bw->moe_gate[i], "moe_gate"},
            };
            for (const auto &r : per)
                if (!r.p) {
                    g_last_error = std::string("backbone weight '") + r.name + "[" +
                                   std::to_string(i) + "]' is null";
                    return nullptr;
                }
        }
        for (int i = 0; i < 3; ++i) {
            if (!bw->at_qkv[i] || !bw->at_o[i]) {
                g_last_error = "attention weight " + std::to_string(i) + " is null";
                return nullptr;
            }
        }
    }
    if (!bw->pattern || strlen(bw->pattern) != BB_LAYERS) {
        g_last_error = "pattern must be exactly " + std::to_string(BB_LAYERS) +
                       " characters of M/E/*";
        return nullptr;
    }
    int mamba_of[BB_LAYERS], attn_of[BB_LAYERS], moe_of[BB_LAYERS];
    int n_mb = 0, n_at = 0, n_moe = 0;
    for (int l = 0; l < BB_LAYERS; ++l) {
        mamba_of[l] = attn_of[l] = moe_of[l] = -1;
        switch (bw->pattern[l]) {
            case 'M': mamba_of[l] = n_mb++; break;
            case 'E': moe_of[l] = n_moe++; break;
            case '*': attn_of[l] = n_at++; break;
            default:
                g_last_error = std::string("bad pattern character '") +
                               bw->pattern[l] + "' at layer " + std::to_string(l);
                return nullptr;
        }
    }
    if (n_mb != 14 || n_at != 3 || n_moe != 14) {
        g_last_error = "layer pattern mismatch: got " + std::to_string(n_mb) +
                       " Mamba, " + std::to_string(n_at) + " attention, " +
                       std::to_string(n_moe) + " MoE; expected 14/3/14";
        return nullptr;
    }
    std::unique_ptr<MK2State, MK2Stage0BaseDeleter> pred(
        reinterpret_cast<MK2State *>(mk2_create_from_weights(pw, batch)));
    if (!pred) return nullptr;
    MK2Allocations owned;
    MK2Stage0PinnedAllocations pinned;
    std::vector<MambaLayer> mb;
    for (int i = 0; i < 14; ++i) {
        mb.push_back(MambaLayer{
            weight_gl{(bf16 *)bw->mb_in_proj[i], nullptr, nullptr,
                      (size_t)MB_IN_PROJ, (size_t)D_MODEL},
            weight_gl{(bf16 *)bw->mb_out_proj[i], nullptr, nullptr,
                      (size_t)D_MODEL, (size_t)MB_INNER},
            (const bf16 *)bw->mb_conv_w[i], (const bf16 *)bw->mb_conv_b[i],
            (const float *)bw->mb_A[i], (const float *)bw->mb_D[i],
            (const float *)bw->mb_dt_bias[i], (const bf16 *)bw->mb_gnorm[i],
            nullptr, 0, nullptr, 0,
        });
    }
    std::vector<AttnLayer> at;
    for (int i = 0; i < 3; ++i) {
        at.push_back(AttnLayer{
            weight_gl{(bf16 *)bw->at_qkv[i], nullptr, nullptr, (size_t)BB_QKV,
                      (size_t)D_MODEL},
            weight_gl{(bf16 *)bw->at_o[i], nullptr, nullptr, (size_t)D_MODEL,
                      (size_t)D_MODEL},
            nullptr, nullptr, 0, 0,
        });
    }
    std::vector<MoeLayer> mo;
    bf16 *zero_gate_bias = nullptr;
    for (int i = 0; i < 14; ++i)
        if (!bw->moe_gate_bias[i] && !zero_gate_bias) {
            mk2_check(owned.alloc(&zero_gate_bias, BB_EXPERTS * sizeof(bf16)));
            mk2_check(cudaMemset(zero_gate_bias, 0, BB_EXPERTS * sizeof(bf16)));
        }
    for (int i = 0; i < 14; ++i) {
        mo.push_back(MoeLayer{
            weight_gl{(bf16 *)bw->moe_exp_up[i], nullptr, nullptr,
                      (size_t)(BB_EXPERTS * BB_MOE_INTER), (size_t)D_MODEL},
            weight_gl{(bf16 *)bw->moe_exp_down[i], nullptr, nullptr,
                      (size_t)(BB_EXPERTS * D_MODEL), (size_t)BB_MOE_INTER},
            weight_gl{(bf16 *)bw->moe_sh_up[i], nullptr, nullptr,
                      (size_t)BB_SHARED_INTER, (size_t)D_MODEL},
            weight_gl{(bf16 *)bw->moe_sh_down[i], nullptr, nullptr,
                      (size_t)D_MODEL, (size_t)BB_SHARED_INTER},
            (const bf16 *)bw->moe_gate[i],
            bw->moe_gate_bias[i] ? (const bf16 *)bw->moe_gate_bias[i] : zero_gate_bias,
        });
    }
    bf16 *h, *r, *normed, *zxbcdt, *gated, *qkv, *moe_h, *moe_sh, *xbc, *ph_logits;
    float *moe_out, *moe_w;
    int *ph_out;
    mk2_check(owned.alloc(&h, (size_t)BATCH * D_MODEL * sizeof(bf16)));
    mk2_check(owned.alloc(&r, (size_t)BATCH * D_MODEL * sizeof(bf16)));
    mk2_check(owned.alloc(&normed, (size_t)BATCH * D_MODEL * sizeof(bf16)));
    mk2_check(owned.alloc(&zxbcdt, (size_t)BATCH * MB_IN_PROJ * sizeof(bf16)));
    mk2_check(owned.alloc(&gated, (size_t)BATCH * MB_INNER * sizeof(bf16)));
    mk2_check(owned.alloc(&qkv, (size_t)BATCH * BB_QKV * sizeof(bf16)));
    mk2_check(owned.alloc(&moe_h,
                     (size_t)BATCH * BB_EXPERTS * BB_MOE_INTER * sizeof(bf16)));
    mk2_check(owned.alloc(&moe_sh, (size_t)BATCH * BB_SHARED_INTER * sizeof(bf16)));
    mk2_check(owned.alloc(&moe_out, (size_t)BATCH * (BB_EXPERTS + 1) * D_MODEL * sizeof(float)));
    mk2_check(owned.alloc(&moe_w, (size_t)BATCH * BB_EXPERTS * sizeof(float)));
    mk2_check(owned.alloc(&xbc, (size_t)BATCH * MB_CONV_DIM * sizeof(bf16)));
    mk2_check(owned.alloc(&ph_logits, (size_t)BATCH * BB_PHON_PADDED * sizeof(bf16)));
    mk2_check(owned.alloc(&ph_out, (size_t)BATCH * sizeof(int)));
    unsigned long long *launch_counter = nullptr;
    mk2_check(owned.alloc(&launch_counter, 6 * sizeof(unsigned long long)));
    mk2_check(cudaMemset(launch_counter, 0, 6 * sizeof(unsigned long long)));
    bf16 *ph_w = nullptr, *ph_b = nullptr;
    mk2_check(owned.alloc(&ph_w, (size_t)BB_PHON_PADDED * D_MODEL * sizeof(bf16)));
    mk2_check(cudaMemset(ph_w, 0, (size_t)BB_PHON_PADDED * D_MODEL * sizeof(bf16)));
    mk2_check(cudaMemcpy(ph_w, bw->phoneme_w,
                     (size_t)BB_PHONEME * D_MODEL * sizeof(bf16),
                     cudaMemcpyDeviceToDevice));
    mk2_check(owned.alloc(&ph_b, (size_t)BB_PHON_PADDED * sizeof(bf16)));
    mk2_check(cudaMemset(ph_b, 0, (size_t)BB_PHON_PADDED * sizeof(bf16)));
    mk2_check(cudaMemcpy(ph_b, bw->phoneme_b, (size_t)BB_PHONEME * sizeof(bf16),
                     cudaMemcpyDeviceToDevice));
    Backbone hb{
        act_gl{normed, nullptr, nullptr, (size_t)BATCH, (size_t)D_MODEL},
        act_gl{gated, nullptr, nullptr, (size_t)BATCH, (size_t)MB_INNER},
        upload_layers(mb, owned),
        (const bf16 *)bw->norm_f,
        upload_layers(at, owned),
        upload_layers(mo, owned),
        {}, {}, {}, {},
        0,
        h, r, normed, zxbcdt, gated,
        nullptr,
        qkv, moe_h, moe_sh, moe_out, moe_w,
        nullptr,
        nullptr,
        nullptr,
        xbc,
        nullptr,
        0, 0, 1, 0, nullptr, nullptr,
        h, r, normed, zxbcdt, xbc, gated, qkv, moe_h, moe_sh, moe_out, moe_w,
        nullptr,
        nullptr,
        act_gl{normed, nullptr, nullptr, (size_t)BATCH, (size_t)D_MODEL},
        act_gl{gated, nullptr, nullptr, (size_t)BATCH, (size_t)MB_INNER},
        act_gl{qkv, nullptr, nullptr, (size_t)BATCH, (size_t)BB_QKV},
        act_gl{moe_h, nullptr, nullptr, (size_t)BATCH,
               (size_t)(BB_EXPERTS * BB_MOE_INTER)},
        act_gl{moe_sh, nullptr, nullptr, (size_t)BATCH, (size_t)BB_SHARED_INTER},
        weight_gl{ph_w, nullptr, nullptr, (size_t)BB_PHON_PADDED, (size_t)D_MODEL},
        ph_b, ph_logits, ph_out,
        launch_counter,
        nullptr,
        act_gl{qkv, nullptr, nullptr, (size_t)BATCH, (size_t)BB_QKV},
        act_gl{moe_h, nullptr, nullptr, (size_t)BATCH,
               (size_t)(BB_EXPERTS * BB_MOE_INTER)},
        act_gl{moe_sh, nullptr, nullptr, (size_t)BATCH, (size_t)BB_SHARED_INTER},
    };
    for (int l = 0; l < BB_LAYERS; ++l) {
        hb.norm_w[l] = (const bf16 *)bw->norm[l];
        hb.mamba_of_layer[l] = mamba_of[l];
        hb.attn_of_layer[l] = attn_of[l];
        hb.moe_of_layer[l] = moe_of[l];
    }
    mk2_check(owned.alloc(&hb.r6_packed,(size_t)BB_EXPERTS*BATCH*D_MODEL*sizeof(bf16)));
    mk2_check(owned.alloc(&hb.r6_map,(size_t)BB_EXPERTS*BATCH*sizeof(int)));
    mk2_check(owned.alloc(&hb.r6_counts,BB_EXPERTS*sizeof(int)));
    mk2_check(owned.alloc(&hb.r6_lut,(1+BB_EXPERTS*DECODE_TILES)*sizeof(int)));
    act_gl sparse_views[]={
      act_gl{hb.r6_packed,nullptr,nullptr,(size_t)BB_EXPERTS*BATCH,(size_t)D_MODEL},
      act_gl{moe_h,nullptr,nullptr,(size_t)BB_EXPERTS*BATCH,(size_t)BB_MOE_INTER}};
    mk2_check(owned.alloc(&hb.r6_views,sizeof(sparse_views)));
    mk2_check(cudaMemcpy(hb.r6_views,sparse_views,sizeof(sparse_views),cudaMemcpyHostToDevice));
    Backbone *d_bb = nullptr, *bb_stage = nullptr;
    MambaLayer *mb_stage = nullptr;
    AttnLayer *at_stage = nullptr;
    mk2_check(owned.alloc(&d_bb, sizeof(Backbone)));
    mk2_check(pinned.alloc(&bb_stage, sizeof(Backbone)));
    mk2_check(pinned.alloc(&mb_stage, 14 * sizeof(MambaLayer)));
    mk2_check(pinned.alloc(&at_stage, 3 * sizeof(AttnLayer)));
    auto *st = new MK2Stage0State{
        MK2State{pred->g, pred->batch, pred->grid, pred->smem,
                 pred->hidden_bf16, 1, pred->device, {}, pred->run_predicate},
        d_bb, hb, std::move(mb), std::move(at), bb_stage, mb_stage, at_stage,
        {}, {}};
    st->pred.maximum_grid = pred->maximum_grid;
    st->pred.owned.swap(pred->owned);
    st->owned.swap(owned.ptrs);
    st->pinned.swap(pinned.ptrs);
    g_last_error.clear();
    return reinterpret_cast<MK2Predictor *>(st);
    } catch (const std::exception &e) { g_last_error = e.what(); return nullptr; }
      catch (...) { g_last_error = "unknown full Stage 0 construction failure"; return nullptr; }
}
static int mk2_enqueue_stage0(MK2Predictor *p, const MK2Stage0Bindings *b,
                              void *stream, bool native, bool startup = false) {
    try {
    const auto *pred = reinterpret_cast<const MK2State *>(p);
    if (!pred || pred->kind != 1 || pred->batch != BATCH || !b) {
        g_last_error = "full Stage 0 requires a matching Stage 0 handle and bindings";
        return 1;
    }
    int device = -1;
    mk2_check(cudaGetDevice(&device));
    if (device != pred->device)
        throw std::runtime_error("current CUDA device differs from full Stage 0 handle device");
    if (!(b->temperature > 0.f) || !std::isfinite(b->temperature))
        throw std::runtime_error("full Stage 0 temperature must be finite and positive");
    if (b->kv_max_blocks_per_seq <= 0 || b->kv_tokens_per_block <= 0 ||
        b->kv_tokens_per_block > INT_MAX / (BB_KV_HEADS * BB_HEAD_DIM))
        throw std::runtime_error("invalid full Stage 0 KV geometry");
    for (int i = 0; i < 14; ++i) {
        if (b->conv_slot_stride[i] < MB_CONV_DIM * (MB_CONV_K - 1) ||
            b->ssm_slot_stride[i] < MB_HEADS * MB_HEAD_DIM * MB_STATE)
            throw std::runtime_error("full Stage 0 state slot stride is smaller than its state");
        mk2_stage0_device_pointer(b->conv_state[i], device, "conv_state");
        mk2_stage0_device_pointer(b->ssm_state[i], device, "ssm_state");
    }
    for (int i = 0; i < 3; ++i) {
        mk2_stage0_device_pointer(b->kv_pool_base[i], device, "kv_pool_base");
        mk2_stage0_device_pointer(b->kv_block_table[i], device, "kv_block_table");
    }
    mk2_stage0_device_pointer(b->slot_map, device, "slot_map");
    mk2_stage0_device_pointer(b->seq_len, device, "seq_len");
    mk2_stage0_device_pointer(b->embedding, device, "embedding");
    mk2_stage0_device_pointer(b->gumbel, device, "gumbel");
    mk2_stage0_device_pointer(b->codes_out, device, "codes_out");
    mk2_stage0_device_pointer(b->phoneme_out, device, "phoneme_out");
    if (b->moe_w_inject_debug)
        mk2_stage0_device_pointer(b->moe_w_inject_debug, device, "moe_w_inject_debug");
    auto *st = reinterpret_cast<MK2Stage0State *>(p);
    if (st->bindings_initialized && !mk2_stage0_same_pinned_bindings(st->pinned_bindings, *b))
        throw std::runtime_error("full Stage 0 captured state bindings changed; create a new handle");
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    Backbone &hb = st->host_bb;
    const bool first_bind = !st->bindings_initialized;
    if (first_bind) {
        for (int i = 0; i < 14; ++i) {
            st->mb_host[i].conv_state = (bf16 *)b->conv_state[i];
            st->mb_host[i].conv_slot_stride = b->conv_slot_stride[i];
            st->mb_host[i].ssm_state = (float *)b->ssm_state[i];
            st->mb_host[i].ssm_slot_stride = b->ssm_slot_stride[i];
        }
        for (int i = 0; i < 3; ++i) {
            st->at_host[i].pool_base = (bf16 *)b->kv_pool_base[i];
            st->at_host[i].block_table = (const int *)b->kv_block_table[i];
            st->at_host[i].max_blocks_per_seq = b->kv_max_blocks_per_seq;
            st->at_host[i].tokens_per_block = b->kv_tokens_per_block;
        }
        hb.moe_w_inject = (const float *)b->moe_w_inject_debug;
        hb.slot_map = (const int *)b->slot_map;
        hb.seq_len = (const int *)b->seq_len;
        hb.phoneme_out = (int *)b->phoneme_out;
        memcpy(st->mb_stage, st->mb_host.data(), 14 * sizeof(MambaLayer));
        memcpy(st->at_stage, st->at_host.data(), 3 * sizeof(AttnLayer));
        memcpy(st->bb_stage, &hb, sizeof(Backbone));
        st->pinned_bindings = *b;
        st->bindings_initialized = true;
    }
    if (!native || first_bind) {
    mk2_check(cudaMemcpyAsync(const_cast<MambaLayer *>(hb.mamba), st->mb_stage,
                          14 * sizeof(MambaLayer), cudaMemcpyHostToDevice, s));
    mk2_check(cudaMemcpyAsync(const_cast<AttnLayer *>(hb.attn), st->at_stage,
                          3 * sizeof(AttnLayer), cudaMemcpyHostToDevice, s));
    mk2_check(cudaMemcpyAsync(st->d_bb, st->bb_stage, sizeof(Backbone),
                          cudaMemcpyHostToDevice, s));
    }
    if (!native) {
    mk2_check(cudaMemcpyAsync(st->host_bb.h, b->embedding,
                          (size_t)BATCH * D_MODEL * sizeof(bf16),
                          cudaMemcpyDeviceToDevice, s));
    mk2_check(cudaMemsetAsync(st->host_bb.r, 0,
                          (size_t)BATCH * D_MODEL * sizeof(bf16), s));
    }
    st->pred.g.native_embedding = native ? (const bf16 *)b->embedding : nullptr;
    st->pred.g.native_codes = native ? (long long *)b->codes_out : nullptr;
    st->pred.g.native_seq_len = native ? (int *)b->seq_len : nullptr;
    st->pred.g.gumbel = static_cast<const float *>(b->gumbel);
    st->pred.g.temperature = b->temperature;
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    mk2_check(cudaStreamIsCapturing(s, &capture_status));
    if (st->pred.grid_reset_pending || capture_status != cudaStreamCaptureStatusNone) {
        mk2_check(cudaMemsetAsync(st->pred.g.barrier, 0, 64 * sizeof(unsigned int), s));
        st->pred.grid_reset_pending = false;
    }
    if (startup) {
        predictor_mk2_startup<<<st->pred.grid, THREADS, st->pred.smem, s>>>(
            st->pred.g, st->d_bb, N_LAYERS, N_POS, BB_LAYERS, nullptr,
            st->pred.native_predictor_active_mask
            MK2_PROGRESS_ARGUMENT(st->pred.native_progress_mask));
    } else {
    predictor_mk2<<<st->pred.grid, THREADS, st->pred.smem, s>>>(
        st->pred.g, st->d_bb, N_LAYERS, N_POS, BB_LAYERS, nullptr,
        native ? st->pred.native_predictor_active_mask : nullptr
        MK2_PROGRESS_ARGUMENT(native ? st->pred.native_progress_mask : nullptr));
    }
    mk2_check(cudaPeekAtLastError());
    if (!native)
        codes_to_int64<<<(BATCH * N_POS + 255) / 256, 256, 0, s>>>(
            st->pred.g.sampled, static_cast<long long *>(b->codes_out), BATCH * N_POS);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { g_last_error = cudaGetErrorString(e); return 1; }
    g_last_error.clear();
    return 0;
    } catch (const std::exception &e) { g_last_error = e.what(); return 1; }
      catch (...) { g_last_error = "unknown full Stage 0 enqueue failure"; return 1; }
}
extern "C" int mk2_run_stage0(MK2Predictor *p, const MK2Stage0Bindings *b,
                              void *stream) {
    return mk2_enqueue_stage0(p, b, stream, false);
}
extern "C" int mk2_run_native_stage0(MK2Predictor *p, const MK2Stage0Bindings *b,
                                     void *stream) {
    return mk2_enqueue_stage0(p, b, stream, true);
}
extern "C" int mk2_run_native_startup_stage0(MK2Predictor *p, const MK2Stage0Bindings *b, void *stream) {
    return mk2_enqueue_stage0(p, b, stream, true, true);
}
extern "C" int mk2_set_native_predictor_active_mask(MK2Predictor *p,
                                                    const unsigned char *mask) {
    try {
        if (!p || reinterpret_cast<MK2State *>(p)->kind != 1)
            throw std::runtime_error("startup mask requires a full Stage0 handle");
        auto *st = reinterpret_cast<MK2Stage0State *>(p);
        if (st->bindings_initialized && st->pred.native_predictor_active_mask != mask)
            throw std::runtime_error("startup mask pointer cannot change after first Stage0 enqueue");
        if (mask)
            mk2_stage0_device_pointer(mask, st->pred.device, "native_predictor_active_mask");
        st->pred.native_predictor_active_mask = mask;
        g_last_error.clear();
        return 0;
    } catch (const std::exception &e) { g_last_error = e.what(); return 1; }
      catch (...) { g_last_error = "unknown startup mask binding failure"; return 1; }
}
extern "C" int mk2_native_startup_skip_abi_version() { return 1; }
#include "../../../../../inflight/runtime.cuh"
extern "C" unsigned long long mk2_stage0_launches(MK2Predictor *p) {
    auto *st = reinterpret_cast<MK2Stage0State *>(p);
    if (!st || st->pred.kind != 1 || !st->host_bb.launches) return 0ull;
    unsigned long long n[2] = {0ull, 0ull};
    if (cudaMemcpy(n, st->host_bb.launches, sizeof(n),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 0ull;
    return (n[1] << 32) | (n[0] & 0xffffffffull);
}
extern "C" int mk2_stage0_timing(MK2Predictor *p, unsigned long long *out) {
    auto *st = reinterpret_cast<MK2Stage0State *>(p);
    if (!st || st->pred.kind != 1 || !st->host_bb.launches || !out) return -1;
    unsigned long long n[6] = {0, 0, 0, 0, 0, 0};
    if (cudaMemcpy(n, st->host_bb.launches, sizeof(n),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return -2;
    out[0] = n[3];
    out[1] = n[4];
    out[2] = n[5];
    out[3] = n[0];
    return 0;
}
extern "C" void mk2_destroy(MK2Predictor *p) {
    if (!p) return;
    if (reinterpret_cast<MK2State *>(p)->kind == 1) {
        auto *st = reinterpret_cast<MK2Stage0State *>(p);
        cudaError_t first_error = cudaSuccess;
        auto record = [&](cudaError_t e) {
            if (first_error == cudaSuccess && e != cudaSuccess) first_error = e;
        };
        int previous = -1;
        const cudaError_t current = cudaGetDevice(&previous);
        record(current);
        record(cudaSetDevice(st->pred.device));
        for (void *owned : st->owned) record(cudaFree(owned));
        for (void *owned : st->pred.owned) record(cudaFree(owned));
        for (void *pinned : st->pinned) record(cudaFreeHost(pinned));
        if (current == cudaSuccess) record(cudaSetDevice(previous));
        delete st;
        if (first_error != cudaSuccess) g_last_error = cudaGetErrorString(first_error);
    } else {
        auto *st = reinterpret_cast<MK2State *>(p);
        int previous = st->device;
        cudaGetDevice(&previous);
        cudaSetDevice(st->device);
        for (void *owned : st->owned) {
            const auto e = cudaFree(owned);
            if (e != cudaSuccess) g_last_error = cudaGetErrorString(e);
        }
        cudaSetDevice(previous);
        delete st;
    }
}
extern "C" int mk2_batch(const MK2Predictor *p) {
    return p ? reinterpret_cast<const MK2State *>(p)->batch : 0;
}
extern "C" int mk2_positions(const MK2Predictor *) { return N_POS; }
extern "C" const char *mk2_last_error(void) { return g_last_error.c_str(); }
#include "native_prefill_impl.inc"
extern "C" int mk2_set_native_stage0_grid(MK2Predictor *p, int grid) {
    if (!p) { g_last_error = "grid setter requires full Stage0 handle"; return 1; }
    auto *st = reinterpret_cast<MK2State *>(p);
    if (st->kind != 1 || grid < 1 || grid > st->maximum_grid) {
        g_last_error = "requested Stage0 grid outside independently resident capacity";
        return 1;
    }
    if (st->grid != grid) st->grid_reset_pending = true;
    st->grid = grid;
    g_last_error.clear();
    return 0;
}
extern "C" int mk2_r8_bind_initial_state(MK2Predictor *p, const void *const *initial, const void *ready, const void *seq_len, int positions) {
    try {
        auto *st = reinterpret_cast<MK2Stage0State *>(p);
        if (!st || st->pred.kind != 1 || st->bindings_initialized || !initial || !ready || !seq_len || positions <= 0)
            throw std::runtime_error("invalid immutable initial-state binding");
        for (int i = 0; i < 14; ++i) {
            mk2_stage0_device_pointer(initial[i], st->pred.device, "initial state");
            st->mb_host[i].r8_initial_ssm = static_cast<const float *>(initial[i]);
            st->mb_host[i].r8_initial_ready = static_cast<const unsigned int *>(ready);
            st->mb_host[i].r8_seq_len = static_cast<const int *>(seq_len);
            st->mb_host[i].r8_initial_positions = positions;
        }
        return 0;
    } catch (const std::exception &e) { g_last_error = e.what(); return 1; }
}
extern "C" int mk2_r9_state_bind(void* p,const void*const* scales,const void*const* maps) {auto*st=(MK2Stage0State*)p;if(st->bindings_initialized)return 1;for(int i=0;i<14;++i){st->mb_host[i].r9_scales=(const float*)scales[i];st->mb_host[i].r9_map=(const int*)maps[i];}return 0;}
extern "C" int mk2_r9_state_bytes(){return sizeof(r9_state_t);}
