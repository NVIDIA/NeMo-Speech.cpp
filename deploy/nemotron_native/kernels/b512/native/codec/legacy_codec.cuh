#include "kernel_kf_codec.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
using namespace nvcuda;
template<class F> __device__ void prepare_attribute(F,int) {}
#ifndef TC2_QPT
#define TC2_QPT 4
#endif
#define DIVUP(a,b) (((a) + (b) - 1) / (b))
__device__ __forceinline__ float half_snake(float v, int c, int half,
                                            const float *__restrict__ alpha) {
    if (c < half) {
        const float a = alpha[c];
        const float s = __sinf(a * v);
        return __fmaf_rn(s, s / (a + 1e-9f), v);
    }
    return v > 0.f ? v : 0.01f * v;
}
#ifndef WG_ST3
#define WG_ST3 1
#define WG_MB3 3
#define WG_ST7 1
#define WG_MB7 3
#define WG_BN3 128
#define WG_BN7 128
#endif
#define PDL_ENTER() do { (void)0; if (threadIdx.x == 0) (void)0; } while (0)
#define PDL_TRIGGER() do { if (threadIdx.x == 0) (void)0; } while (0)
template <typename KFn, typename... KArgs>
__device__ __forceinline__ void pdl_launch(KFn kern, dim3 grid, dim3 blk, size_t shmem,
                              cudaStream_t s, KArgs... args) {
    native_invoke(kern, grid, blk, shmem, args...);
}
template <typename KFn, typename... KArgs>
__device__ __forceinline__ void pdl_launch_pf(KFn kern, dim3 grid, dim3 blk, size_t shmem,
                              cudaStream_t s, const void *pf, size_t pfb, KArgs... args) {
    native_invoke_pf(kern, grid, blk, shmem, pf, pfb, args...);
}
#include "wg_kernel.inc"
#include "wg2_kernel.inc"
#include "wg3_kernel.inc"
#include "wg4_step_widths.inc"
#include "wg5_step_widths.inc"
#include "repack3.inc"
#include "tail_kernel.inc"
template <int KT, int BM, int BN, int NWARPS, int WM, int WN>
__device__ __noinline__ void dense_conv_kernel(
    const __half *__restrict__ in, const __half *__restrict__ w,
    const float *__restrict__ bias, const float *__restrict__ alpha,
    const __half *__restrict__ resid, __half *__restrict__ out,
    const __half *__restrict__ pad_src, __half *__restrict__ ns,
    int Cin, int Cout, int T, int L, int SCin, int SCout, int N,
    int do_snake, int pad_off, int pad_len) {
    PDL_TRIGGER();
    constexpr int BK = 16 * KT;
    constexpr int NT = NWARPS * 32;
    constexpr int WTM = BM / WM;
    constexpr int WTN = BN / WN;
    constexpr int FM = WTM / 16, FN = WTN / 16;
    constexpr int LDA = BK + 8, LDB = BN + 8;
    constexpr int GEMM_SM = BM * LDA + BK * LDB;
    constexpr int EPI_SM = NWARPS * WTM * WTN * 2;
    constexpr int SM_H = GEMM_SM > EPI_SM ? GEMM_SM : EPI_SM;
    __half *smem=reinterpret_cast<__half*>(native_arena());
    char *colok=reinterpret_cast<char*>(native_aux());
    __half *As = smem;
    __half *Bs = smem + BM * LDA;
    float *Cw = reinterpret_cast<float *>(smem);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int H = KT - 1;
    const int Kg = Cin * KT;
    if (tid < BN) {
        const int n = n0 + tid;
        colok[tid] = (n < N && (n % L) < T) ? 1 : 0;
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);
    constexpr int ATOT = BM * (BK / 8);
    constexpr int AREG = (ATOT + NT - 1) / NT;
    constexpr int BTOT = 16 * (BN / 8);
    constexpr int BREG = (BTOT + NT - 1) / NT;
    int4 ar[AREG];
    int4 br0[BREG], br1[BREG];
#define CODEC_LOADG(kg0) { const int _ci0 = (kg0) / KT; _Pragma("unroll") for (int r = 0; r < AREG; ++r) { const int idx = tid + r * NT; ar[r] = make_int4(0, 0, 0, 0); if (ATOT % NT == 0 || idx < ATOT) { const int m = idx / (BK / 8), v = idx % (BK / 8); const int co = m0 + m, kg = (kg0) + v * 8; if (co < Cout && kg < Kg) ar[r] = *reinterpret_cast<const int4 *>(w + (size_t)co * Kg + kg); } } _Pragma("unroll") for (int r = 0; r < BREG; ++r) { const int idx = tid + r * NT; br0[r] = make_int4(0, 0, 0, 0); br1[r] = make_int4(0, 0, 0, 0); if (BTOT % NT == 0 || idx < BTOT) { const int cil = idx / (BN / 8), nc = idx % (BN / 8); const int ci = _ci0 + cil; if (ci < Cin) { const int4 *src = reinterpret_cast<const int4 *>( in + (size_t)ci * SCin + n0 + nc * 8); br0[r] = src[0]; br1[r] = src[1]; } } } }
    CODEC_LOADG(0)
    for (int kg0 = 0; kg0 < Kg; kg0 += BK) {
        __syncthreads();
#pragma unroll
        for (int r = 0; r < AREG; ++r) {
            const int idx = tid + r * NT;
            if (ATOT % NT == 0 || idx < ATOT)
                *reinterpret_cast<int4 *>(As + (idx / (BK / 8)) * LDA +
                                          (idx % (BK / 8)) * 8) = ar[r];
        }
#pragma unroll
        for (int r = 0; r < BREG; ++r) {
            const int idx = tid + r * NT;
            if (BTOT % NT != 0 && idx >= BTOT) continue;
            const int cil = idx / (BN / 8), nc = idx % (BN / 8);
            unsigned int u[8];
            u[0] = br0[r].x; u[1] = br0[r].y; u[2] = br0[r].z; u[3] = br0[r].w;
            u[4] = br1[r].x; u[5] = br1[r].y; u[6] = br1[r].z; u[7] = br1[r].w;
#pragma unroll
            for (int k = 0; k < KT; ++k) {
                const int kh = k >> 1;
                int4 o;
                if (k & 1) {
                    o.x = __funnelshift_r(u[kh + 0], u[kh + 1], 16);
                    o.y = __funnelshift_r(u[kh + 1], u[kh + 2], 16);
                    o.z = __funnelshift_r(u[kh + 2], u[kh + 3], 16);
                    o.w = __funnelshift_r(u[kh + 3], u[kh + 4], 16);
                } else {
                    o.x = u[kh + 0]; o.y = u[kh + 1];
                    o.z = u[kh + 2]; o.w = u[kh + 3];
                }
                *reinterpret_cast<int4 *>(Bs + (cil * KT + k) * LDB + nc * 8) = o;
            }
        }
        __syncthreads();
        if (kg0 + BK < Kg) CODEC_LOADG(kg0 + BK)
#pragma unroll
        for (int kf = 0; kf < BK; kf += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[FM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf[FN];
#pragma unroll
            for (int i = 0; i < FM; ++i)
                wmma::load_matrix_sync(af[i], As + (wm * WTM + i * 16) * LDA + kf, LDA);
#pragma unroll
            for (int j = 0; j < FN; ++j)
                wmma::load_matrix_sync(bf[j], Bs + kf * LDB + wn * WTN + j * 16, LDB);
#pragma unroll
            for (int i = 0; i < FM; ++i)
#pragma unroll
                for (int j = 0; j < FN; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
    }
#undef CODEC_LOADG
    __syncthreads();
    float *Cme = Cw + warp * WTM * WTN;
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
            wmma::store_matrix_sync(Cme + (i * 16) * WTN + j * 16, acc[i][j], WTN,
                                    wmma::mem_row_major);
    __syncwarp();
    const int half = Cout >> 1;
    const int lane = tid & 31;
    for (int e = lane; e < WTM * WTN; e += 32) {
        const int mm = e / WTN, nn = e % WTN;
        const int co = m0 + wm * WTM + mm;
        const int nl = wn * WTN + nn;
        if (co >= Cout || !colok[nl]) continue;
        const int n = n0 + nl;
        float v = Cme[e] + bias[co];
        if (resid != nullptr)
            v += __half2float(resid[(size_t)co * SCout + n + H]);
        if (do_snake) v = half_snake(v, co, half, alpha);
        out[(size_t)co * SCout + n + H] = __float2half(v);
    }
    if (blockIdx.y == 0) {
        const int nbx = gridDim.x;
        if (ns != nullptr) {
            const int B = N / L;
            const int total = Cin * B * H;
            for (int idx = blockIdx.x * NT + tid; idx < total; idx += nbx * NT) {
                const int j = idx % H, r = idx / H;
                const int b = r % B, ci = r / B;
                ns[((size_t)b * Cin + ci) * H + j] =
                    in[(size_t)ci * SCin + b * L + native_true_tokens(b,T) + j];
            }
        }
        if (pad_src != nullptr) {
            const int B = N / L;
            const int total = Cout * B * pad_len;
            for (int idx = blockIdx.x * NT + tid; idx < total; idx += nbx * NT) {
                const int j = idx % pad_len, r = idx / pad_len;
                const int b = r % B, co = r / B;
                out[(size_t)co * SCout + b * L + pad_off + j] =
                    pad_src[((size_t)b * Cout + co) * pad_len + j];
            }
        }
    }
}
template <int G, int STRIDE, int QPT_ = 2, bool PDL = false>
__device__ __noinline__ void transpose_conv_kernel(
    const __half *__restrict__ in, const __half *__restrict__ w,
    const float *__restrict__ bias, const float *__restrict__ alpha,
    __half *__restrict__ out, const __half *__restrict__ pad_src,
    __half *__restrict__ ns, int Cin, int Cout, int Tin,
    int Lin, int SCin, int jbase, int Lout, int SCout, int Hout, int pad_off,
    int pad_len, int B) {
    if constexpr (PDL) PDL_ENTER();
    constexpr int Ktaps = 2 * STRIDE;
    constexpr int QPT = QPT_;
    constexpr int VPT = QPT * STRIDE;
    static_assert(QPT % 2 == 0, "half2 input loads need an even frame count");
    static_assert(VPT % 2 == 0, "half2 staging needs an even slot");
    float *ws=reinterpret_cast<float*>(native_aux());
    __half *sm=reinterpret_cast<__half*>(native_arena());
    const int co = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x, BLK = blockDim.x;
    if (G != 4 || STRIDE != 7 || native_meta()->lengths[b] > 0) {
        for (int i = tid; i < G * Ktaps; i += BLK)
            ws[i] = __half2float(w[(size_t)(co * G + i / Ktaps) * Ktaps + (i % Ktaps)]);
        __syncthreads();
        const float bs = bias[0 + co];
        const int half = Cout >> 1;
        const bool snake = co < half;
        const float av = snake ? alpha[co] : 0.f;
        const float ri = snake ? __frcp_rn(av + 1e-9f) : 0.f;
        const __half *ic = in + (size_t)(co * G) * SCin + b * Lin + jbase;
        __half *oc = out + (size_t)co * SCout + b * Lout + Hout;
        constexpr bool VEC16 = (VPT % 8) == 0;
        const int ophase =
            VEC16 ? 0 : (int)((((uintptr_t)oc) >> 1) & 7u);
        __half *const sbase = sm + ophase;
        const int hd = (8 - ophase) & 7;
        const int lane = tid & 31;
        for (int qb = 1 + blockIdx.x * BLK * QPT; qb <= Tin;
             qb += gridDim.x * BLK * QPT) {
            const int q = qb + QPT * tid;
            const bool a0 = (q <= Tin);
            float jq[QPT][G], jp0[G];
    #pragma unroll
            for (int g = 0; g < G; ++g) {
    #pragma unroll
                for (int e2 = 0; e2 < QPT / 2; ++e2) {
                    const __half2 v = a0 ? *reinterpret_cast<const __half2 *>(
                                               ic + (size_t)g * SCin + q + 2 * e2)
                                         : __floats2half2_rn(0.f, 0.f);
                    jq[2 * e2][g] = __half2float(v.x);
                    jq[2 * e2 + 1][g] = __half2float(v.y);
                }
            }
    #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float up = __shfl_up_sync(0xffffffffu, jq[QPT - 1][g], 1);
                jp0[g] = (lane == 0)
                             ? (a0 ? __half2float(ic[(size_t)g * SCin + q - 1]) : 0.f)
                             : up;
            }
            __half *const sp = sbase + tid * VPT;
            __align__(16) __half vreg[VPT];
    #pragma unroll
            for (int r = 0; r < STRIDE; ++r) {
                float wc[G], wp[G];
    #pragma unroll
                for (int g = 0; g < G; ++g) {
                    wc[g] = ws[g * Ktaps + r];
                    wp[g] = ws[g * Ktaps + r + STRIDE];
                }
    #pragma unroll
                for (int e = 0; e < QPT; ++e) {
                    float v = bs;
    #pragma unroll
                    for (int g = 0; g < G; ++g) {
                        const float cur = jq[e][g];
                        const float prv = (e == 0) ? jp0[g] : jq[e - 1][g];
                        v = __fmaf_rn(wc[g], cur, __fmaf_rn(wp[g], prv, v));
                    }
                    if (snake) {
                        const float sn = __sinf(av * v);
                        v = __fmaf_rn(sn * sn, ri, v);
                    } else {
                        v = v > 0.f ? v : 0.01f * v;
                    }
                    vreg[e * STRIDE + r] = __float2half(v);
                }
            }
            if constexpr (VEC16) {
    #pragma unroll
                for (int m = 0; m < VPT / 8; ++m)
                    *reinterpret_cast<int4 *>(sp + m * 8) =
                        *reinterpret_cast<const int4 *>(vreg + m * 8);
            } else {
    #pragma unroll
                for (int i = 0; i < VPT; ++i) sp[i] = vreg[i];
            }
            __syncthreads();
            const int nval = min(BLK * QPT, Tin - qb + 1);
            const int tot = nval * STRIDE;
            __half *dst = oc + (size_t)(qb - 1) * STRIDE;
            if constexpr (VEC16) {
                const int tot2 = tot >> 1;
                for (int i = tid; i < tot2; i += BLK)
                    reinterpret_cast<__half2 *>(dst)[i] =
                        reinterpret_cast<const __half2 *>(sbase)[i];
                if ((tot & 1) && tid == 0) dst[tot - 1] = sbase[tot - 1];
            } else {
                const int head = hd < tot ? hd : tot;
                for (int i = tid; i < head; i += BLK) dst[i] = sbase[i];
                const int nb4 = (tot - head) >> 3;
                const int4 *s4 = reinterpret_cast<const int4 *>(sbase + head);
                int4 *d4 = reinterpret_cast<int4 *>(dst + head);
                for (int i = tid; i < nb4; i += BLK) d4[i] = s4[i];
                for (int i = head + (nb4 << 3) + tid; i < tot; i += BLK)
                    dst[i] = sbase[i];
            }
            __syncthreads();
        }
    }
    if (blockIdx.x == 0) {
        if (ns != nullptr) {
            for (int ci = co + tid * Cout; ci < Cin; ci += BLK * Cout)
                ns[(size_t)b * Cin + ci] =
                    in[(size_t)ci * SCin + b * Lin + jbase + native_true_tokens(b,Tin)];
        }
        if (pad_src != nullptr) {
            for (int j = tid; j < pad_len; j += BLK)
                out[(size_t)co * SCout + b * Lout + pad_off + j] =
                    pad_src[((size_t)b * Cout + co) * pad_len + j];
        }
    }
}
template <int G, int STRIDE, int BPACK, int LPB, bool PDL = false>
__device__ __noinline__ void transpose_conv_early_bpack_kernel(
    const __half *__restrict__ in, const __half *__restrict__ w,
    const float *__restrict__ bias, const float *__restrict__ alpha,
    __half *__restrict__ out, const __half *__restrict__ pad_src,
    __half *__restrict__ ns, int Cin, int Cout, int Tin,
    int Lin, int SCin, int jbase, int Lout, int SCout, int Hout, int pad_off,
    int pad_len, int B) {
    if constexpr (PDL) PDL_ENTER();
    static_assert(BPACK <= 32);
    static_assert(LPB == 8 || LPB == 16 || LPB == 32);
    constexpr int Ktaps = 2 * STRIDE;
    constexpr int WARP = 32;
    constexpr int NT = WARP * BPACK;
    constexpr int BPB = WARP / LPB;
    float *ws=reinterpret_cast<float*>(native_aux());
    __half *sm=reinterpret_cast<__half*>(native_arena());
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sub = lane % LPB;
    const int bsub = lane / LPB;
    const int co = blockIdx.y;
    const int b = blockIdx.z * (BPACK * BPB) + warp * BPB + bsub;
    const bool active = b < B;
    for (int i = tid; i < G * Ktaps; i += NT)
        ws[i] = __half2float(w[(size_t)(co * G + i / Ktaps) * Ktaps +
                               (i % Ktaps)]);
    __syncthreads();
    const int qb = 1 + blockIdx.x * LPB;
    const int q = qb + sub;
    __half *sp = sm + (warp * WARP + lane) * STRIDE;
    const bool live = active && q <= Tin;
    {
        const float bs = bias[co];
        const int half = Cout >> 1;
        const bool snake = co < half;
        const float av = snake ? alpha[co] : 0.f;
        const float ri = snake ? __frcp_rn(av + 1e-9f) : 0.f;
        const __half *ic = in + (size_t)(co * G) * SCin + b * Lin + jbase;
        float jq[G], jp[G];
#pragma unroll
        for (int g = 0; g < G; ++g)
            jq[g] = live ? __half2float(ic[(size_t)g * SCin + q]) : 0.f;
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float up = __shfl_up_sync(0xffffffffu, jq[g], 1, LPB);
            jp[g] = (sub == 0)
                        ? (live ? __half2float(ic[(size_t)g * SCin + q - 1]) : 0.f)
                        : up;
        }
        if (live)
#pragma unroll
        for (int r = 0; r < STRIDE; ++r) {
            float v = bs;
#pragma unroll
            for (int g = 0; g < G; ++g)
                v = __fmaf_rn(ws[g * Ktaps + r], jq[g],
                              __fmaf_rn(ws[g * Ktaps + r + STRIDE], jp[g], v));
            if (snake) {
                const float sn = __sinf(av * v);
                v = __fmaf_rn(sn * sn, ri, v);
            } else {
                v = v > 0.f ? v : 0.01f * v;
            }
            sp[r] = __float2half(v);
        }
    }
    __syncwarp();
    if (active) {
        const int nval = min(LPB, Tin - (qb - 1));
        const int tot2 = (nval * STRIDE) >> 1;
        const __half *src = sm + (warp * WARP + bsub * LPB) * STRIDE;
        __half *dst = out + (size_t)co * SCout + b * Lout + Hout +
                      (size_t)(qb - 1) * STRIDE;
        for (int i = sub; i < tot2; i += LPB)
            reinterpret_cast<__half2 *>(dst)[i] =
                reinterpret_cast<const __half2 *>(src)[i];
        if ((nval * STRIDE) & 1)
            if (sub == 0) dst[nval * STRIDE - 1] = src[nval * STRIDE - 1];
    }
    if (blockIdx.x == 0 && active) {
        if (ns != nullptr) {
            for (int ci = co + sub * Cout; ci < Cin; ci += LPB * Cout)
                ns[(size_t)b * Cin + ci] =
                    in[(size_t)ci * SCin + b * Lin + jbase + native_true_tokens(b,Tin)];
        }
        if (pad_src != nullptr) {
            for (int j = sub; j < pad_len; j += LPB)
                out[(size_t)co * SCout + b * Lout + pad_off + j] =
                    pad_src[((size_t)b * Cout + co) * pad_len + j];
        }
    }
}
template <int G, int STRIDE, int BPACK, int LPB, bool PDL = false>
__device__ __noinline__ void transpose_conv_bpack2_kernel(
    const __half *__restrict__ in, const __half *__restrict__ w,
    const float *__restrict__ bias, const float *__restrict__ alpha,
    __half *__restrict__ out, const __half *__restrict__ pad_src,
    __half *__restrict__ ns, int Cin, int Cout, int Tin,
    int Lin, int SCin, int jbase, int Lout, int SCout, int Hout, int pad_off,
    int pad_len, int B) {
    if constexpr (PDL) PDL_ENTER();
    static_assert(BPACK <= 32);
    static_assert(LPB == 8 || LPB == 16 || LPB == 32);
    constexpr int Ktaps = 2 * STRIDE;
    constexpr int WARP = 32;
    constexpr int NT = WARP * BPACK;
    constexpr int BPB = WARP / LPB;
    constexpr int QPT = 2;
    constexpr int VPT = QPT * STRIDE;
    float *ws=reinterpret_cast<float*>(native_aux());
    __half *sm=reinterpret_cast<__half*>(native_arena());
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sub = lane % LPB;
    const int bsub = lane / LPB;
    const int co = blockIdx.y;
    const int b = blockIdx.z * (BPACK * BPB) + warp * BPB + bsub;
    const bool active = b < B;
    for (int i = tid; i < G * Ktaps; i += NT)
        ws[i] = __half2float(w[(size_t)(co * G + i / Ktaps) * Ktaps +
                               (i % Ktaps)]);
    __syncthreads();
    const int qb = 1 + blockIdx.x * (LPB * QPT);
    const int q = qb + sub * QPT;
    __half *sp = sm + (warp * WARP + lane) * VPT;
    const bool a0 = active && (q <= Tin);
    {
        const float bs = bias[co];
        const int half = Cout >> 1;
        const bool snake = co < half;
        const float av = snake ? alpha[co] : 0.f;
        const float ri = snake ? __frcp_rn(av + 1e-9f) : 0.f;
        const __half *ic = in + (size_t)(co * G) * SCin + b * Lin + jbase;
        float jq0[G], jq1[G], jp0[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const __half2 v = a0 ? *reinterpret_cast<const __half2 *>(
                                       ic + (size_t)g * SCin + q)
                                 : __floats2half2_rn(0.f, 0.f);
            jq0[g] = __half2float(v.x);
            jq1[g] = __half2float(v.y);
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float up = __shfl_up_sync(0xffffffffu, jq1[g], 1, LPB);
            jp0[g] = (sub == 0)
                         ? (a0 ? __half2float(ic[(size_t)g * SCin + q - 1])
                               : 0.f)
                         : up;
        }
        if (a0) {
            float vv[VPT];
#pragma unroll
            for (int r = 0; r < STRIDE; ++r) {
                float wc[G], wp[G];
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    wc[g] = ws[g * Ktaps + r];
                    wp[g] = ws[g * Ktaps + r + STRIDE];
                }
#pragma unroll
                for (int e = 0; e < QPT; ++e) {
                    float v = bs;
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        const float cur = (e == 0) ? jq0[g] : jq1[g];
                        const float prv = (e == 0) ? jp0[g] : jq0[g];
                        v = __fmaf_rn(wc[g], cur, __fmaf_rn(wp[g], prv, v));
                    }
                    if (snake) {
                        const float sn = __sinf(av * v);
                        v = __fmaf_rn(sn * sn, ri, v);
                    } else {
                        v = v > 0.f ? v : 0.01f * v;
                    }
                    vv[e * STRIDE + r] = v;
                }
            }
            __half2 *sp2 = reinterpret_cast<__half2 *>(sp);
#pragma unroll
            for (int i = 0; i < VPT / 2; ++i)
                sp2[i] = __floats2half2_rn(vv[2 * i], vv[2 * i + 1]);
        }
    }
    __syncwarp();
    if (active) {
        const int nval = min(LPB * QPT, Tin - (qb - 1));
        const int tot = nval * STRIDE;
        const __half *src = sm + (warp * WARP + bsub * LPB) * VPT;
        __half *dst = out + (size_t)co * SCout + b * Lout + Hout +
                      (size_t)(qb - 1) * STRIDE;
        for (int i = sub; i < (tot >> 1); i += LPB)
            reinterpret_cast<__half2 *>(dst)[i] =
                reinterpret_cast<const __half2 *>(src)[i];
        if (tot & 1)
            if (sub == 0) dst[tot - 1] = src[tot - 1];
    }
    if (blockIdx.x == 0 && active) {
        if (ns != nullptr) {
            for (int ci = co + sub * Cout; ci < Cin; ci += LPB * Cout)
                ns[(size_t)b * Cin + ci] =
                    in[(size_t)ci * SCin + b * Lin + jbase + native_true_tokens(b,Tin)];
        }
        if (pad_src != nullptr) {
            for (int j = sub; j < pad_len; j += LPB)
                out[(size_t)co * SCout + b * Lout + pad_off + j] =
                    pad_src[((size_t)b * Cout + co) * pad_len + j];
        }
    }
}
#define TCONV_BPACK(G,STRIDE,BPACK,LPB,TIN,COUT,BB,...) pdl_launch(transpose_conv_early_bpack_kernel<G, STRIDE, BPACK, LPB, true>, dim3(DIVUP(TIN, LPB), COUT, DIVUP(BB, BPACK * (32 / LPB))), dim3(32 * BPACK), (size_t)(32 * (BPACK)) * (STRIDE) * 2 + 64, s, __VA_ARGS__)
#define TCONV_BP2(G,STRIDE,LPB,TIN,COUT,BB,...) pdl_launch(transpose_conv_bpack2_kernel<G, STRIDE, LPB, LPB, true>, dim3(DIVUP(TIN, (LPB) * 2), COUT, DIVUP(BB, 32)), dim3(32 * (LPB)), (size_t)(32 * (LPB)) * (4 * (STRIDE)) + 64, s, __VA_ARGS__)
#define TCONV_BPACK_DISPATCH1(G,STRIDE,BPACK,TIN,COUT,BB,...) do { if ((TIN) >= 32) TCONV_BP2(G, STRIDE, 16, TIN, COUT, BB, __VA_ARGS__); else if ((TIN) >= 16) TCONV_BP2(G, STRIDE, 8, TIN, COUT, BB, __VA_ARGS__); else TCONV_BPACK(G, STRIDE, BPACK, 16, TIN, COUT, BB, __VA_ARGS__); } while (0)
#define TCONV_BPACK_DISPATCH(G,STRIDE,BPACK,TIN,COUT,BB,...) do { if ((TIN) >= 32) TCONV_BP2(G, STRIDE, 16, TIN, COUT, BB, __VA_ARGS__); else if ((TIN) >= 16) TCONV_BP2(G, STRIDE, 8, TIN, COUT, BB, __VA_ARGS__); else TCONV_BPACK(G, STRIDE, BPACK, 16, TIN, COUT, BB, __VA_ARGS__); } while (0)
#define TCONV_BPACK_DISPATCH8(G,STRIDE,BPACK,TIN,COUT,BB,...) do { if ((TIN) >= 32) TCONV_BP2(G, STRIDE, 16, TIN, COUT, BB, __VA_ARGS__); else if ((TIN) >= 16) TCONV_BP2(G, STRIDE, 8, TIN, COUT, BB, __VA_ARGS__); else if ((TIN) <= 8) TCONV_BPACK(G, STRIDE, BPACK, 8, TIN, COUT, BB, __VA_ARGS__); else TCONV_BPACK(G, STRIDE, BPACK, 16, TIN, COUT, BB, __VA_ARGS__); } while (0)
__device__ __noinline__ void pack_latent_kernel(const __half *__restrict__ latent,
                                   const __half *__restrict__ state,
                                   __half *__restrict__ out, int Cin, int T,
                                   int L, int SC, int H, int B) {
    const int total = Cin * B * L;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const int j = idx % L, r = idx / L;
        const int b = r % B, ci = r / B;
        out[(size_t)ci * SC + b * L + j] =
            (j < H) ? state[((size_t)b * Cin + ci) * H + j]
                    : latent[((size_t)b * Cin + ci) * T + (j - H)];
    }
}
#define PC_VEC 8
__device__ __noinline__ void post_conv_kernel(
    const __half *__restrict__ in, const __half *__restrict__ w,
    const float *__restrict__ bias, __half *__restrict__ y,
    __half *__restrict__ ns, int Cin, int T, int L, int SC, int KT, int B,
    int N) {
    float *pw=reinterpret_cast<float*>(native_arena());
    for (int i = threadIdx.x; i < Cin * KT; i += 256) pw[i] = __half2float(w[i]);
    __syncthreads();
    const int H = KT - 1;
    const float bs = bias[0];
    for (int n = (blockIdx.x * 256 + threadIdx.x) * PC_VEC; n < N;
         n += gridDim.x * 256 * PC_VEC) {
        float v[PC_VEC];
#pragma unroll
        for (int i = 0; i < PC_VEC; ++i) v[i] = bs;
        for (int ci = 0; ci < Cin; ++ci) {
            const __half *xr = in + (size_t)ci * SC + n;
            const float *wr = pw + ci * KT;
            const int4 lo = *reinterpret_cast<const int4 *>(xr);
            const int4 hi = *reinterpret_cast<const int4 *>(xr + 8);
            unsigned u[8];
            u[0] = lo.x; u[1] = lo.y; u[2] = lo.z; u[3] = lo.w;
            u[4] = hi.x; u[5] = hi.y; u[6] = hi.z; u[7] = hi.w;
            float x[PC_VEC + 6];
#pragma unroll
            for (int j = 0; j < (PC_VEC + 6) / 2; ++j) {
                const float2 f = __half22float2(
                    *reinterpret_cast<const __half2 *>(&u[j]));
                x[2 * j] = f.x;
                x[2 * j + 1] = f.y;
            }
#pragma unroll
            for (int i = 0; i < PC_VEC; ++i)
                for (int k = 0; k < 7; ++k)
                    if (k < KT) v[i] = __fmaf_rn(wr[k], x[i + k], v[i]);
        }
        const int b0 = n / L, t0 = n - (n / L) * L;
#pragma unroll
        for (int i = 0; i < PC_VEC; ++i) {
            int t = t0 + i, b = b0;
            if (t >= L) { t -= L; b = b0 + 1; }
            if (t < T && b < B)
                y[(size_t)b * T + t] =
                    __float2half(fminf(fmaxf(v[i], -1.f), 1.f));
        }
    }
    if (ns != nullptr) {
        const int total = B * Cin * H;
        for (int idx = blockIdx.x * 256 + threadIdx.x; idx < total;
             idx += gridDim.x * 256) {
            const int j = idx % H, r = idx / H;
            const int ci = r % Cin, b = r / Cin;
            ns[((size_t)b * Cin + ci) * H + j] =
                in[(size_t)ci * SC + b * L + native_true_tokens(b,T) + j];
        }
    }
}
#define RP_JOBS 6
struct RepackSet {
    const __half *src[RP_JOBS];
    __half *dst[RP_JOBS];
    int Cout[RP_JOBS], CoutP[RP_JOBS], Cin[RP_JOBS];
    int base[RP_JOBS], nch[RP_JOBS], rs[RP_JOBS];
    int k3base;
    const __half *k3src[2];
    __half *k3dst[2];
};
__host__ __device__ inline int native_RP_COUT(int j) {const int v[6]={768, 384, 256, 128, 64, 32};return v[j];}
__host__ __device__ inline int native_RP_CIN(int j) {const int v[6]={384, 768, 128, 256, 32, 64};return v[j];}
__host__ __device__ inline int native_RP_COUTP(int j) {const int v[6]={768, 384, 256, 128, 128, 128};return v[j];}
template <int KT, int CIB>
__device__ __noinline__ void repack_kernel(RepackSet js) {
    const int bx = blockIdx.x;
    int job, job_bx;
    if (bx < 1152) {
        job = 0; job_bx = 0;
    } else if (bx < 2304) {
        job = 1; job_bx = 1152;
    } else if (bx < 2432) {
        job = 2; job_bx = 2304;
    } else if (bx < 2560) {
        job = 3; job_bx = 2432;
    } else if (bx < 2576) {
        job = 4; job_bx = 2560;
    } else {
        job = 5; job_bx = 2576;
    }
    const int Cout = js.Cout[job], CoutP = js.CoutP[job], Cin = js.Cin[job];
    const int idx = (bx - job_bx) * blockDim.x + threadIdx.x;
    const int co = idx / Cin, ci = idx - co * Cin;
    const int c = ci / CIB, cl = ci - c * CIB;
    __half *d = js.dst[job] + (size_t)c * (KT * CoutP * CIB) +
                (size_t)(co >> 3) * (CIB * 8) + (cl >> 3) * 64 + (co & 7) * 8 +
                (cl & 7);
    const __half *sp = js.src[job] + (size_t)(co * Cin + ci) * KT;
    const bool live = co < Cout;
#pragma unroll
    for (int k = 0; k < KT; ++k)
        d[(size_t)k * CoutP * CIB] = live ? sp[k] : __float2half(0.f);
}
template <int KT>
__device__ __noinline__ void repack_rs_kernel(
    const RepackSet &js, const __half *__restrict__ latent, const __half *__restrict__ latent_state,
    __half *__restrict__ latent_out, int latent_C, int latent_T, int latent_L,
    int latent_SC, int latent_H, int latent_B) {
    PDL_TRIGGER();
    constexpr int CO_TILE = 16;
    constexpr int CI_TILE = 64;
    constexpr int ROW_H = CI_TILE * KT + 8;
    constexpr int ROW4 = CI_TILE * KT / 8;
    __half *tile=reinterpret_cast<__half*>(native_arena());
    const int bx = blockIdx.x;
    {
        const int idx = bx + threadIdx.x * gridDim.x;
        const int total = latent_C * latent_B * latent_L;
        if (idx < total) {
            const int j = idx % latent_L, r = idx / latent_L;
            const int b = r % latent_B, ci = r / latent_B;
            latent_out[(size_t)ci * latent_SC + b * latent_L + j] =
                (j < latent_H)
                    ? latent_state[((size_t)b * latent_C + ci) * latent_H + j]
                    : latent[((size_t)b * latent_C + ci) * latent_T +
                             (j - latent_H)];
        }
    }
    if (bx >= js.k3base) {
        constexpr int K3 = 3;
        constexpr int R3_H = CI_TILE * K3 + 8;
        constexpr int R3_4 = CI_TILE * K3 / 8;
        const int r = bx - js.k3base;
        const int conv = r / 1152;
        const int loc = r - conv * 1152;
        const int Cout3 = conv ? 768 : 1536;
        const int Cin3 = conv ? 1536 : 768;
        const int nci3 = Cin3 / CI_TILE;
        const int cot3 = loc / nci3;
        if (cot3 * CO_TILE >= Cout3) return;
        const int co03 = cot3 * CO_TILE;
        const int ci03 = (loc - cot3 * nci3) * CI_TILE;
        const int NCH3 = Cin3 >> 4;
        const __half *const s3 = js.k3src[conv];
        __half *const d3 = js.k3dst[conv];
        const int t3 = threadIdx.x;
#pragma unroll
        for (int n = 0; n < CO_TILE * R3_4 / 128; ++n) {
            const int i = t3 + n * 128;
            const int co = i / R3_4, q = i - co * R3_4;
            reinterpret_cast<int4 *>(tile + co * R3_H)[q] =
                *reinterpret_cast<const int4 *>(
                    s3 + ((size_t)(co03 + co) * Cin3 + ci03) * K3 + q * 8);
        }
        __syncthreads();
        const int w3 = t3 >> 5, l3 = t3 & 31;
        const int rA3 = l3 >> 2, q43 = l3 & 3;
        const int cb3 = co03 >> 4;
        __half *const db3 = d3 + ((size_t)(cb3 >> 2) * NCH3 * K3) * 1024 +
                            ((cb3 & 3) * 32 + l3) * 8;
#pragma unroll
        for (int u = w3; u < (CI_TILE / 16) * K3; u += 4) {
            const int cl = u / K3, tap = u - cl * K3;
            const int cgi = (ci03 >> 4) + cl;
            const __half *sa = tile + rA3 * R3_H + (cl * 16 + 2 * q43) * K3 + tap;
            const __half *sb = sa + 8 * R3_H;
            uint32_t o[4];
            o[0] = __half_as_ushort(sa[0]) |
                   ((uint32_t)__half_as_ushort(sa[K3]) << 16);
            o[1] = __half_as_ushort(sb[0]) |
                   ((uint32_t)__half_as_ushort(sb[K3]) << 16);
            o[2] = __half_as_ushort(sa[8 * K3]) |
                   ((uint32_t)__half_as_ushort(sa[9 * K3]) << 16);
            o[3] = __half_as_ushort(sb[8 * K3]) |
                   ((uint32_t)__half_as_ushort(sb[9 * K3]) << 16);
            *reinterpret_cast<int4 *>(db3 + (size_t)(cgi * K3 + tap) * 1024) =
                make_int4(o[0], o[1], o[2], o[3]);
        }
        return;
    }
    int job = 0;
#pragma unroll
    for (int q = 1; q < RP_JOBS; ++q) if (bx >= js.base[q]) job = q;
    const int local = bx - js.base[job];
    const int Cout = js.Cout[job], Cin = js.Cin[job];
    const int nci = (Cin + CI_TILE - 1) / CI_TILE;
    const int cot = local / nci;
    const int co0 = cot * CO_TILE;
    const int ci0 = (local - cot * nci) * CI_TILE;
    const int NCH = js.nch[job];
    const int t = threadIdx.x;
    const int valid4 = (min(Cin - ci0, CI_TILE) * KT) >> 3;
    for (int i = t; i < CO_TILE * ROW4; i += 128) {
        const int co = i / ROW4, q = i - co * ROW4;
        int4 v = make_int4(0, 0, 0, 0);
        if (co0 + co < Cout && q < valid4)
            v = *reinterpret_cast<const int4 *>(
                js.src[job] + ((size_t)(co0 + co) * Cin + ci0) * KT + q * 8);
        reinterpret_cast<int4 *>(tile + co * ROW_H)[q] = v;
    }
    __syncthreads();
    const int warp = t >> 5, lane = t & 31;
    const int rA = lane >> 2, q4 = lane & 3;
    const int cb = co0 >> 4;
    const int mtile = cb >> 2, wsub = cb & 3;
    __half *const dbase = js.dst[job] +
                          ((size_t)mtile * NCH * KT) * 1024 + (wsub * 32 + lane) * 8;
    if (!js.rs[job]) {
        const int CoutP = js.CoutP[job];
        constexpr int UN2 = 16 * (CI_TILE / 8) * KT;
#pragma unroll
        for (int n = 0; n < UN2 / 128; ++n) {
            const int u = t + n * 128;
            const int col = u & 15, cig = (u >> 4) & (CI_TILE / 8 - 1);
            const int tap = u / (16 * (CI_TILE / 8));
            const int gci = ci0 + cig * 8;
            if (gci >= Cin) continue;
            const int gco = co0 + col;
            uint32_t o[4];
            const __half *sp = tile + col * ROW_H;
#pragma unroll
            for (int e = 0; e < 4; ++e)
                o[e] = __half_as_ushort(sp[(cig * 8 + 2 * e) * KT + tap]) |
                       ((uint32_t)__half_as_ushort(
                            sp[(cig * 8 + 2 * e + 1) * KT + tap]) << 16);
            __half *d = js.dst[job] +
                        (size_t)(gci >> 4) * ((size_t)KT * CoutP * 16) +
                        (size_t)tap * (CoutP * 16) + (size_t)(gco >> 3) * 128 +
                        ((gci >> 3) & 1) * 64 + (gco & 7) * 8;
            *reinterpret_cast<int4 *>(d) = make_int4(o[0], o[1], o[2], o[3]);
        }
        return;
    }
    constexpr int UNITS = (CI_TILE / 16) * KT;
#pragma unroll
    for (int u = warp; u < UNITS; u += 4) {
        const int cl = u / KT, tap = u - cl * KT;
        const int cgi = (ci0 >> 4) + cl;
        if (cgi >= NCH) continue;
        const __half *sa = tile + rA * ROW_H + (cl * 16 + 2 * q4) * KT + tap;
        const __half *sb = sa + 8 * ROW_H;
        uint32_t o[4];
        o[0] = __half_as_ushort(sa[0]) |
               ((uint32_t)__half_as_ushort(sa[KT]) << 16);
        o[1] = __half_as_ushort(sb[0]) |
               ((uint32_t)__half_as_ushort(sb[KT]) << 16);
        o[2] = __half_as_ushort(sa[8 * KT]) |
               ((uint32_t)__half_as_ushort(sa[9 * KT]) << 16);
        o[3] = __half_as_ushort(sb[8 * KT]) |
               ((uint32_t)__half_as_ushort(sb[9 * KT]) << 16);
        *reinterpret_cast<int4 *>(dbase + (size_t)(cgi * KT + tap) * 1024) =
            make_int4(o[0], o[1], o[2], o[3]);
    }
}
__device__ __noinline__ void repack3_rs_kernel(
    const __half *__restrict__ win, const __half *__restrict__ wsk,
    __half *__restrict__ din, __half *__restrict__ dsk,
    const __half *__restrict__ asrc, __half *__restrict__ adst, int SCA,
    int ncol, int ntile) {
    PDL_TRIGGER();
    if (blockIdx.y == 12) {
        if (adst == nullptr) return;
        for (int u = blockIdx.x; u < 96 * ntile; u += gridDim.x) {
            const int g = u / ntile;
            const int col = (u - g * ntile) * 128 + (int)threadIdx.x;
            if (col >= ncol) continue;
            __half v[8];
#pragma unroll
            for (int h = 0; h < 8; ++h)
                v[h] = asrc[(size_t)(g * 8 + h) * SCA + col];
            *reinterpret_cast<int4 *>(adst + (size_t)g * ((size_t)SCA * 8) +
                                      (size_t)col * 8) =
                *reinterpret_cast<const int4 *>(v);
        }
        return;
    }
    constexpr int KT = 3;
    constexpr int CO_TILE = 16;
    constexpr int CI_TILE = 64;
    constexpr int ROW_H = CI_TILE * KT + 8;
    constexpr int ROW4 = CI_TILE * KT / 8;
    __half *tile=reinterpret_cast<__half*>(native_arena());
    const int jy = blockIdx.y;
    const int conv = jy & 1, b = 5 - (jy >> 1);
    const int Cout = conv ? 768 : 1536;
    const int Cin = conv ? 1536 : 768;
    const __half *src = (conv ? wsk : win) + (size_t)(b + 1) * Cout * Cin * KT;
    __half *dst = (conv ? dsk : din) + (size_t)b * ((size_t)Cout * Cin * KT);
    const int nci = Cin / CI_TILE;
    const int cot = blockIdx.x / nci;
    const int co0 = cot * CO_TILE;
    const int ci0 = (blockIdx.x - cot * nci) * CI_TILE;
    const int NCH = Cin >> 4;
    const int t = threadIdx.x;
#pragma unroll
    for (int n = 0; n < CO_TILE * ROW4 / 128; ++n) {
        const int i = t + n * 128;
        const int co = i / ROW4, q = i - co * ROW4;
        reinterpret_cast<int4 *>(tile + co * ROW_H)[q] =
            *reinterpret_cast<const int4 *>(
                src + ((size_t)(co0 + co) * Cin + ci0) * KT + q * 8);
    }
    __syncthreads();
    const int warp = t >> 5, lane = t & 31;
    const int rA = lane >> 2, q4 = lane & 3;
    const int cb = co0 >> 4;
    __half *const dbase = dst + ((size_t)(cb >> 2) * NCH * KT) * 1024 +
                          ((cb & 3) * 32 + lane) * 8;
#pragma unroll
    for (int u = warp; u < (CI_TILE / 16) * KT; u += 4) {
        const int cl = u / KT, tap = u - cl * KT;
        const int cgi = (ci0 >> 4) + cl;
        const __half *sa = tile + rA * ROW_H + (cl * 16 + 2 * q4) * KT + tap;
        const __half *sb = sa + 8 * ROW_H;
        uint32_t o[4];
        o[0] = __half_as_ushort(sa[0]) |
               ((uint32_t)__half_as_ushort(sa[KT]) << 16);
        o[1] = __half_as_ushort(sb[0]) |
               ((uint32_t)__half_as_ushort(sb[KT]) << 16);
        o[2] = __half_as_ushort(sa[8 * KT]) |
               ((uint32_t)__half_as_ushort(sa[9 * KT]) << 16);
        o[3] = __half_as_ushort(sb[8 * KT]) |
               ((uint32_t)__half_as_ushort(sb[9 * KT]) << 16);
        *reinterpret_cast<int4 *>(dbase + (size_t)(cgi * KT + tap) * 1024) =
            make_int4(o[0], o[1], o[2], o[3]);
    }
}
__device__ __noinline__ void repack_activation_only(const __half *asrc,__half *adst,int SCA,int ncol,int ntile){
        if (adst == nullptr) return;
        for (int u = blockIdx.x; u < 96 * ntile; u += gridDim.x) {
            const int g = u / ntile;
            const int col = (u - g * ntile) * 128 + (int)threadIdx.x;
            if (col >= ncol) continue;
            __half v[8];
#pragma unroll
            for (int h = 0; h < 8; ++h)
                v[h] = asrc[(size_t)(g * 8 + h) * SCA + col];
            *reinterpret_cast<int4 *>(adst + (size_t)g * ((size_t)SCA * 8) +
                                      (size_t)col * 8) =
                *reinterpret_cast<const int4 *>(v);
        }
        return;
}
namespace {
struct Stack {
    int T, L, N, SC, H;
};
inline __host__ __device__ int tc_blk(int Tin, int qpt = 2) {
    int best = 256, bw = 1 << 30;
    for (int b = 256; b >= 32; b >>= 1) {
        const int per = qpt * b;
        const int waste = (Tin + per - 1) / per * per - Tin;
        if (waste < bw) { bw = waste; best = b; }
    }
    return best;
}
inline __host__ __device__ int roundup(int a, int b) { return (a + b - 1) / b * b; }
__host__ __device__ Stack mk(int B, int T, int K) {
    Stack s;
    s.T = T;
    s.H = K - 1;
    s.L = s.H + T;
    s.N = B * s.L;
    s.SC = roundup(s.N, 128) + 128;
    return s;
}
struct Plan {
    Stack s0, s1, s2, s3, s4;
    __half *LB, *X0, *M0, *Y0;
    __half *A, *Ac, *Bb, *M1;
    __half *P, *Q, *M2;
    __half *R, *S, *M3;
    __half *U;
    __half *W7[RP_JOBS];
    __half *W3i, *W3s;
    size_t total;
};
static constexpr int kTuneB = 32;
inline __host__ __device__ int nref(const Stack &st) { return kTuneB * st.L; }
inline __host__ __device__ bool use_rp3i(int, int F) { return (size_t)kTuneB * (4 * F + 2) > 400; }
inline __host__ __device__ bool use_rp3s(int, int F) { return (size_t)kTuneB * (4 * F + 2) > 400; }
__host__ __device__ Plan build_plan(int B, int F, __half *base) {
    Plan p;
    p.s0 = mk(B, 2 * F, 3);
    p.s1 = mk(B, 4 * F, 3);
    p.s2 = mk(B, 36 * F, 7);
    p.s3 = mk(B, 252 * F, 7);
    p.s4 = mk(B, 1764 * F, 7);
    size_t off = 0;
    auto alloc = [&](int ch, const Stack &s) -> __half * {
        __half *r = base ? base + off : nullptr;
        off += (size_t)ch * s.SC;
        return r;
    };
    p.LB = alloc(40, p.s0);
    p.X0 = alloc(768, p.s0);
    p.M0 = alloc(1536, p.s0);
    p.Y0 = alloc(768, p.s0);
    p.A = alloc(768, p.s1);
    p.Ac = use_rp3i(B, F) ? alloc(768, p.s1) : nullptr;
    p.Bb = alloc(768, p.s1);
    p.M1 = alloc(1536, p.s1);
    p.P = alloc(384, p.s2);
    p.Q = alloc(384, p.s2);
    p.M2 = alloc(768, p.s2);
    p.R = alloc(128, p.s3);
    p.S = alloc(128, p.s3);
    p.M3 = alloc(256, p.s3);
    p.U = alloc(32, p.s4);
    for (int j = 0; j < RP_JOBS; ++j) {
        p.W7[j] = base ? base + off : nullptr;
        off += (size_t)native_RP_COUTP(j) * native_RP_CIN(j) * 7;
    }
    p.W3i = nullptr;
    if (use_rp3i(B, F)) {
        p.W3i = base ? base + off + (size_t)3 * 1536 * 768 : nullptr;
        off += (size_t)7 * 3 * 1536 * 768;
    }
    p.W3s = nullptr;
    if (use_rp3s(B, F)) {
        p.W3s = base ? base + off + (size_t)3 * 768 * 1536 : nullptr;
        off += (size_t)7 * 3 * 768 * 1536;
    }
    p.total = off;
    return p;
}
__host__ __device__ const __half *H16(const void *p) { return static_cast<const __half *>(p); }
__host__ __device__ const float *F32(const void *p) { return static_cast<const float *>(p); }
}
size_t codec_scratch_bytes(int B, int F) {
    return build_plan(B, F, nullptr).total * sizeof(__half);
}
template <int KT, int BK, int NWG, int STAGES, int MINB, int WG_BN>
__device__ __forceinline__ void wg_launch(cudaStream_t s, const __half *in, const __half *w,
                      const float *bias, const float *alpha,
                      const __half *resid, __half *out, const __half *pad_src,
                      __half *ns, int Cin, int Cout, const Stack &st, int SCin,
                      int do_snake, int pad_off, int pad_len, bool prep = false) {
    constexpr int WG_BM = 64 * NWG, WG_NT = 128 * NWG, WG_CLD = WG_BN + 8;
    constexpr int KG = BK / 8;
    constexpr int A_E = (WG_BM / 8) * KG * WG_SLOT;
    constexpr int B_E = KG * (WG_BN / 8) * WG_SLOT;
    constexpr int GEMM_B = STAGES * (A_E + B_E) * 2;
    constexpr int EPI_B = WG_BM * WG_CLD * 4;
    constexpr int SH = GEMM_B > EPI_B ? GEMM_B : EPI_B;
    if (prep) {
        prepare_attribute(wg_conv_kernel<KT, BK, NWG, STAGES, MINB, WG_BN>, SH);
        return;
    }
    native_invoke(wg_conv_kernel<KT, BK, NWG, STAGES, MINB, WG_BN>,dim3(dim3(DIVUP(Cout, WG_BM), DIVUP(st.N, WG_BN))),dim3(WG_NT),SH,
            in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, st.T, st.L,
            SCin, st.SC, st.N, do_snake, pad_off, pad_len);
}
template <int KT, int CIB, int MSPLIT, int KSPLIT, int NSPLIT, int WG_BN,
          int SA, int SB, int MINB>
__device__ __forceinline__ void wg2_launch(cudaStream_t s, const __half *in, const __half *w,
                       const float *bias, const float *alpha,
                       const __half *resid, __half *out, const __half *pad_src,
                       __half *ns, int Cin, int Cout, const Stack &st, int SCin,
                       int do_snake, int pad_off, int pad_len,
                       bool prep = false) {
    constexpr int BK = CIB * KT;
    constexpr int WG_BM = 64 * MSPLIT;
    constexpr int WG_NT = 128 * MSPLIT * KSPLIT * NSPLIT;
    constexpr int WG_BNT = WG_BN * NSPLIT;
    constexpr int NCRAW = WG_BNT / 4;
    constexpr int EPN =
        (WG_NT % NCRAW == 0) ? WG_BNT : (((WG_BNT + 127) / 128) * 128);
    constexpr int WG_CLD = EPN + 8;
    constexpr int KG = BK / 8;
    constexpr int A_E = (WG_BM / 8) * KG * WG_SLOT;
    constexpr int B_E = KG * (WG_BN / 8) * WG_SLOT;
    constexpr int GEMM_B = KSPLIT * (SA * A_E + SB * NSPLIT * B_E) * 2;
    constexpr int EPI_B = KSPLIT * WG_BM * WG_CLD * 4;
    constexpr int SH = GEMM_B > EPI_B ? GEMM_B : EPI_B;
    if (prep) {
        prepare_attribute(
            wg2_conv_kernel<KT, CIB, MSPLIT, KSPLIT, NSPLIT, WG_BN, SA, SB,
                            MINB>,
            SH);
        return;
    }
    native_invoke_pf(wg2_conv_kernel<KT, CIB, MSPLIT, KSPLIT, NSPLIT, WG_BN, SA, SB, MINB>,dim3(dim3(DIVUP(Cout, WG_BM), DIVUP(st.N, WG_BNT))),dim3(WG_NT),SH,
            w, (size_t)KT * Cin * Cout * sizeof(__half),
            in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, st.T, st.L,
            SCin, st.SC, st.N, do_snake, pad_off, pad_len);
}
template <int KT, int BK, int NWG, int NSG, int MINB, int WG_BN, int STAGES>
__device__ __forceinline__ void wg3_launch(cudaStream_t s, const __half *in, const __half *w,
                       const float *bias, const float *alpha,
                       const __half *resid, __half *out, const __half *pad_src,
                       __half *ns, int Cin, int Cout, const Stack &st, int SCin,
                       int do_snake, int pad_off, int pad_len,
                       bool prep = false) {
    constexpr int WG_BM = 64 * NWG, WG_NT = 128 * NWG * NSG;
    constexpr int WG_BNT = WG_BN * NSG, WG_CLD = WG_BNT + 8;
    constexpr int KG = BK / 8;
    constexpr int A_E = (WG_BM / 8) * KG * WG_SLOT;
    constexpr int B_E = KG * (WG_BN / 8) * WG_SLOT;
    constexpr int GEMM_B = STAGES * (A_E + NSG * B_E) * 2;
    constexpr int EPI_B = WG_BM * WG_CLD * 4;
    constexpr int SH = GEMM_B > EPI_B ? GEMM_B : EPI_B;
    if (prep) {
        prepare_attribute(wg3_conv_kernel<KT, BK, NWG, NSG, MINB, WG_BN, STAGES>, SH);
        return;
    }
    native_invoke(wg3_conv_kernel<KT, BK, NWG, NSG, MINB, WG_BN, STAGES>,dim3(dim3(DIVUP(Cout, WG_BM), DIVUP(st.N, WG_BNT))),dim3(WG_NT),SH,
            in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, st.T, st.L,
            SCin, st.SC, st.N, do_snake, pad_off, pad_len);
}
template <int KT, int CIB, int MSPLIT, int NSPLIT, int WG_BN, int STAGES,
          int MINB, bool PREACT, int ABUF = 3, bool BC8 = false,
          bool OC8 = false, bool LATE_PDL = false>
__device__ __forceinline__ void wg4_launch(cudaStream_t s, const __half *in, const __half *w,
                       const float *bias, const float *alpha,
                       const __half *resid, __half *out, const __half *pad_src,
                       __half *ns, int Cin, int Cout, int CoutP,
                       const Stack &st, int SCin, int do_snake, int pad_off,
                       int pad_len, bool prep = false) {
    constexpr int NT = 128 * MSPLIT * NSPLIT;
    constexpr int BM = 64 * MSPLIT, BNT = WG_BN * NSPLIT;
    constexpr int NBR = BNT + KT - 1;
    constexpr int NBRP = BC8 ? ((NBR + 7) & ~7) : ((NBR + 31) & ~31);
    constexpr int NCRAW = BNT / 4;
    constexpr int EPN =
        (NT % NCRAW == 0) ? BNT : (((BNT + 127) / 128) * 128);
    constexpr int B_E = CIB * NBRP;
    constexpr int GEMM_B = STAGES * B_E * 2;
    constexpr int EPI_B = BM * (EPN + 8) * (PREACT ? 2 : 4);
    constexpr int SH = GEMM_B > EPI_B ? GEMM_B : EPI_B;
    if (prep) {
        prepare_attribute(
            wg4_conv_kernel<KT, CIB, MSPLIT, NSPLIT, WG_BN, STAGES, MINB,
                            PREACT, ABUF, BC8, OC8, true, LATE_PDL>,
            SH);
        return;
    }
    pdl_launch_pf(wg4_conv_kernel<KT, CIB, MSPLIT, NSPLIT, WG_BN, STAGES, MINB,
                               PREACT, ABUF, BC8, OC8, true, LATE_PDL>,
               dim3(DIVUP(Cout, BM), DIVUP(st.N, BNT)), dim3(NT), SH, s,
               w, (size_t)KT * CoutP * Cin * sizeof(__half),
               in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, CoutP,
               st.T, st.L, SCin, st.SC, st.N, do_snake, pad_off, pad_len);
}
template <int CIB, int MSPLIT, int KSPLIT, int WG_BN, int SA, int SB, int MINB,
          int WGD = 1, bool BC8 = false, bool OC8 = false, bool RC8 = false,
          int CIN = 0, bool PDL = false>
__device__ __forceinline__ void wg5_launch(cudaStream_t s, const __half *in, const __half *w,
                       const float *bias, const float *alpha,
                       const __half *resid, __half *out, const __half *pad_src,
                       __half *ns, int Cin, int Cout, const Stack &st, int SCin,
                       int do_snake, int pad_off, int pad_len,
                       bool prep = false) {
    constexpr int BM = 64 * MSPLIT;
    constexpr int NT = 128 * MSPLIT * KSPLIT;
    constexpr int NCRAW = WG_BN / 4;
    constexpr int EPN = (NT % NCRAW == 0) ? WG_BN : (((WG_BN + 127) / 128) * 128);
    constexpr int NBRP = BC8 ? (((WG_BN + 2) + 7) & ~7)
                             : (((WG_BN + 2) + 31) & ~31);
    constexpr int B_E = CIB * NBRP;
    constexpr int GEMM_B = KSPLIT * SB * B_E * 2;
    constexpr int EPI_COLS = OC8 ? WG_BN : EPN;
    constexpr int EPI_CLD_R = EPI_COLS + 8;
    constexpr int EPI_CLD = (EPI_CLD_R % 16 == 0) ? EPI_CLD_R + 4 : EPI_CLD_R;
    constexpr int EPI_B = KSPLIT * BM * EPI_CLD * (OC8 ? 2 : 4);
    constexpr int SH = GEMM_B > EPI_B ? GEMM_B : EPI_B;
    static_assert(SH <= 227 * 1024, "wg5 shared over budget");
    if (prep) {
        prepare_attribute(
            wg5_conv_kernel<CIB, MSPLIT, KSPLIT, WG_BN, SA, SB, MINB, WGD,
                            BC8, OC8, RC8, CIN, PDL>,
            SH);
        return;
    }
    if constexpr (PDL) {
        cudaLaunchConfig_t cfg = {};
        cudaLaunchAttribute attr[1] = {};
        native_config_grid(cfg) = dim3(DIVUP(Cout, BM), DIVUP(st.N, WG_BN));
        native_config_threads(cfg) = dim3(NT);
        cfg.dynamicSmemBytes = SH;
        cfg.stream = s;
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        native_cudaLaunchKernelEx_pf(
            &cfg,
            wg5_conv_kernel<CIB, MSPLIT, KSPLIT, WG_BN, SA, SB, MINB, WGD,
                            BC8, OC8, RC8, CIN, PDL>,
            w, (size_t)3 * Cin * Cout * sizeof(__half),
            in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, st.T, st.L,
            SCin, st.SC, st.N, do_snake, pad_off, pad_len);
    } else {
        native_invoke_pf(wg5_conv_kernel<CIB, MSPLIT, KSPLIT, WG_BN, SA, SB, MINB, WGD, BC8,
                        OC8, RC8, CIN, PDL>,dim3(dim3(DIVUP(Cout, BM), DIVUP(st.N, WG_BN))),dim3(NT),SH,
                w, (size_t)3 * Cin * Cout * sizeof(__half),
                in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, st.T,
                st.L, SCin, st.SC, st.N, do_snake, pad_off, pad_len);
    }
}
#ifndef D7W
#define D7W 96
#endif
#define D7_M2_C8I(BN,LATE) wg4_launch<7, 16, 2, 1, D7W, 4, 1, false, 2, true, false, LATE>( s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, CoutP, st, SCin, 1, pad_off, pad_len, prep)
#define D7_M2P_C8O(BN,LATE) wg4_launch<7, 16, 2, 1, D7W, 4, 1, true, 2, false, true, LATE>( s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, CoutP, st, SCin, 1, pad_off, pad_len, prep)
#define D7_M2N2(BN) wg4_launch<7, 16, 2, 2, BN, 4, 1, false, 2>( s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, CoutP, st, SCin, 1, pad_off, pad_len, prep)
#define D7_M1(BN) wg4_launch<7, 16, 1, 1, BN, 4, 2, false, 2>( s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, CoutP, st, SCin, 1, pad_off, pad_len, prep)
#define D7_C8I(W) \
    wg4_launch<7, 16, 2, 1, W, 4, 1, false, 2, true, false, false>( \
        s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, CoutP, st, \
        SCin, 1, pad_off, pad_len, prep)
#define D7_C8O(W) \
    wg4_launch<7, 16, 2, 1, W, 4, 1, true, 2, false, true, false>( \
        s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, CoutP, st, \
        SCin, 1, pad_off, pad_len, prep)
template <int BN>
__device__ __forceinline__ int d7_band_cost(int N, int mtiles, int resident) {
    const int v = mtiles * DIVUP(N, BN);
    const int sub = (DIVUP(v, resident) >= 2) ? 2 : 1;
    return DIVUP(DIVUP(v, sub), resident) * sub * (128 + BN + 6);
}
__device__ __forceinline__ int d7_band(int N, int mtiles, int resident) {
    int best = 64, bc = d7_band_cost<64>(N, mtiles, resident);
    const int c96 = d7_band_cost<96>(N, mtiles, resident);
    if (c96 < bc) { bc = c96; best = 96; }
    if (d7_band_cost<112>(N, mtiles, resident) < bc) best = 112;
    return best;
}
__device__ __forceinline__ void dense7(cudaStream_t s, int job, const __half *in, const __half *w,
                   const float *bias, const float *alpha, const __half *resid,
                   __half *out, const __half *pad_src, __half *ns,
                   const Stack &st, int SCin, int pad_off, int pad_len,
                   bool prep = false) {
    const int Cin = native_RP_CIN(job), Cout = native_RP_COUT(job), CoutP = native_RP_COUTP(job);
    const int b_ = d7_band(st.N, DIVUP(Cout, 128), native_resident());
    if ((job & 1) == 0) {
        if (b_ == 64) D7_C8O(64);
        else if (b_ == 96) D7_C8O(96);
        else D7_C8O(112);
    } else {
        if (b_ == 64) D7_C8I(64);
        else if (b_ == 96) D7_C8I(96);
        else D7_C8I(112);
    }
}
__device__ __forceinline__ int wg5_band_skip(int N, int mtiles, int resident) {
    const int wmin = DIVUP(mtiles * DIVUP(N, 104), resident);
    if (DIVUP(mtiles * DIVUP(N, 24), resident) == wmin) return 24;
    if (DIVUP(mtiles * DIVUP(N, 40), resident) == wmin) return 40;
    if (DIVUP(mtiles * DIVUP(N, 48), resident) == wmin) return 48;
    if (DIVUP(mtiles * DIVUP(N, 64), resident) == wmin) return 64;
    return 104;
}
__device__ __forceinline__ int wg5_band_main_long(int N, int mtiles,
                                                  int resident) {
    const int wmin = DIVUP(mtiles * DIVUP(N, 104), resident);
    if (DIVUP(mtiles * DIVUP(N, 64), resident) == wmin) return 64;
    return 104;
}
__device__ __forceinline__ int wg5_band_main_short(int N, int mtiles,
                                                   int resident) {
    const int wmin = DIVUP(mtiles * DIVUP(N, 96), resident);
    if (DIVUP(mtiles * DIVUP(N, 56), resident) == wmin) return 56;
    if (DIVUP(mtiles * DIVUP(N, 80), resident) == wmin) return 80;
    return 96;
}
#define DENSE_LAUNCH(KT,BM,BN,NW,WM,WN) native_invoke(dense_conv_kernel<KT, BM, BN, NW, WM, WN>,dim3(dim3(DIVUP(Cout, BM), DIVUP(st.N, BN))),dim3(NW * 32),0, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, st.T, st.L, SCin, st.SC, st.N, do_snake, pad_off, pad_len)
__device__ __forceinline__ void dense(cudaStream_t s, int KT, const __half *in, const __half *w,
                  const float *bias, const float *alpha, const __half *resid,
                  __half *out, const __half *pad_src, __half *ns, int Cin,
                  int Cout, const Stack &st, int SCin, int do_snake,
                  int pad_off, int pad_len, const __half *w3 = nullptr,
                  int layout_mode = 0, bool prep = false) {
    if (KT == 3 && w3 != nullptr) {
        if (Cin == 1536) {
#define WG5S(BN,O,R,PDL) wg5_launch<16, 1, 4, BN, 4, 3, 1, 1, true, O, R, 1536, PDL>( s, in, w3, bias, alpha, resid, out, pad_src, ns, Cin, Cout, st, SCin, do_snake, pad_off, pad_len, prep)
#define WG5SW(BN,O,R,PDL) wg5_launch<32, 1, 2, BN, 4, 3, 1, 1, true, O, R, 1536, PDL>( s, in, w3, bias, alpha, resid, out, pad_src, ns, Cin, Cout, st, SCin, do_snake, pad_off, pad_len, prep)
#define WG5S3(O,R) do { const int n_ = nref(st); if (n_ > 500 && n_ <= 640) { WG5SW(56, O, R, false); } else { const int b_ = wg5_band_skip(st.N, DIVUP(Cout, 64), native_resident()); if (b_ == 24) WG5S(24, O, R, true); else if (b_ == 40) WG5S(40, O, R, true); else if (b_ == 48) WG5S(48, O, R, true); else if (b_ == 64) WG5S(64, O, R, true); else WG5S(104, O, R, true); } } while (0)
            if (layout_mode == 1) WG5S3(true, false);
            else if (layout_mode == 2) WG5S3(true, true );
            else if (layout_mode == 3) WG5S3(false, true );
            else WG5S3(false, false);
#undef WG5S3
#undef WG5SW
#undef WG5S
        } else {
#define WG5I(BN,BC,MS,PDL) wg5_launch<16, MS, 2, BN, 4, 3, 1, 1, BC, true, false, 768, PDL>( s, in, w3, bias, alpha, resid, out, pad_src, ns, Cin, Cout, st, SCin, do_snake, pad_off, pad_len, prep)
#define WG5I4(BN,BC,PDL) wg5_launch<16, 1, 4, BN, 4, 3, 1, 1, BC, true, false, 768, PDL>( s, in, w3, bias, alpha, resid, out, pad_src, ns, Cin, Cout, st, SCin, do_snake, pad_off, pad_len, prep)
#define WG5I2W(BN,BC,PDL) wg5_launch<32, 1, 2, BN, 4, 3, 1, 1, BC, true, false, 768, PDL>( s, in, w3, bias, alpha, resid, out, pad_src, ns, Cin, Cout, st, SCin, do_snake, pad_off, pad_len, prep)
#define WG5I3(BC) do { const int n_ = nref(st); if (n_ > 640) { const int b_ = wg5_band_main_long(st.N, DIVUP(Cout, 128), native_resident()); if (b_ == 64) WG5I(64, BC, 2, true); else WG5I(104, BC, 2, true); } else if (n_ > 500) { WG5I2W(64, BC, false); } else { const int b_ = wg5_band_main_short(st.N, DIVUP(Cout, 64), native_resident()); if (b_ == 56) WG5I4(56, BC, true); else if (b_ == 80) WG5I4(80, BC, true); else WG5I4(96, BC, true); } } while (0)
            if (layout_mode == 1) WG5I3(true);
            else WG5I3(false);
#undef WG5I3
#undef WG5I2W
#undef WG5I4
#undef WG5I
        }
    } else if (KT == 3) {
        const int n_ = nref(st);
        if (Cin == 1536 && Cout == 768) {
            if (n_ <= 704)
                wg2_launch<3, 16, 1, 4, 1, 64, 2, 2, 1>(
                    s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout,
                    st, SCin, do_snake, pad_off, pad_len, prep);
            else
                wg2_launch<3, 16, 1, 4, 1, 128, 4, 2, 1>(
                    s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout,
                    st, SCin, do_snake, pad_off, pad_len, prep);
        } else if (Cin == 768 && Cout == 1536) {
            if (n_ > 1000)
                wg2_launch<3, 16, 1, 2, 1, 128, 4, 2, 1>(
                    s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout,
                    st, SCin, do_snake, pad_off, pad_len, prep);
            else if (n_ > 640 && n_ <= 768)
                wg2_launch<3, 16, 1, 2, 1, 64, 2, 2, 2>(
                    s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout,
                    st, SCin, do_snake, pad_off, pad_len, prep);
            else if (n_ <= 320)
                wg2_launch<3, 16, 1, 4, 1, 64, 2, 2, 1>(
                    s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout,
                    st, SCin, do_snake, pad_off, pad_len, prep);
            else if (n_ > 384 && n_ <= 512)
                wg2_launch<3, 16, 1, 4, 1, 112, 4, 2, 1>(
                    s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout,
                    st, SCin, do_snake, pad_off, pad_len, prep);
            else if (n_ > 512 && n_ <= 640)
                wg2_launch<3, 16, 1, 4, 1, 120, 4, 2, 1>(
                    s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout,
                    st, SCin, do_snake, pad_off, pad_len, prep);
            else
                wg2_launch<3, 16, 1, 4, 1, 128, 4, 2, 1>(
                    s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout,
                    st, SCin, do_snake, pad_off, pad_len, prep);
        } else {
            wg2_launch<3, 16, 1, 1, 1, 64, 3, 2, 2>(
                s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, st,
                SCin, do_snake, pad_off, pad_len, prep);
        }
    } else if (Cout >= 128 &&
               ((Cin >= 384 && st.T == 288) ||
                (Cin < 384 && st.T == 2016))) {
        wg3_launch<7, 112, 2, 2, 1, 128, 2>(
            s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, st,
            SCin, do_snake, pad_off, pad_len);
    } else {
        wg3_launch<7, 112, 1, 4, 1, 128, 1>(
            s, in, w, bias, alpha, resid, out, pad_src, ns, Cin, Cout, st,
            SCin, do_snake, pad_off, pad_len);
    }
}
__device__ void codec_program(const CodecArgs &a, const Plan &p, const RepackSet &js,
                              int repack_grid,
                              const __half *const *state_override = nullptr) {
    cudaStream_t s = nullptr;
    const int B = a.B;
    const void *const *W = a.inp;
    const __half *const *ST = state_override ? state_override :
        reinterpret_cast<const __half *const *>(a.st);
    __half *const *NS = reinterpret_cast<__half *const *>(a.ns);
    dense(s, 3, p.LB, H16(W[1]), F32(W[2]), nullptr, nullptr, p.X0, ST[1], NS[0],
          40, 768, p.s0, p.s0.SC, 0, 0, 2);
    const __half *k3iw = H16(W[3]);
    const float *k3ib = F32(W[4]);
    const float *k3ia = F32(W[5]);
    const __half *k3sw = H16(W[6]);
    const float *k3sb = F32(W[7]);
    const float *k3sa = F32(W[8]);
    const size_t IW = (size_t)1536 * 768 * 3, SW = (size_t)768 * 1536 * 3;
    dense(s, 3, p.X0, k3iw, k3ib, k3ia, nullptr, p.M0, ST[2], NS[1], 768, 1536,
          p.s0, p.s0.SC, 1, 0, 2, p.W3i ? p.W3i - IW : nullptr, 0);
    dense(s, 3, p.M0, k3sw, k3sb, k3sa, p.X0, p.Y0, ST[3], NS[2], 1536, 768,
          p.s0, p.s0.SC, 1, 1, 1, p.W3s ? p.W3s - SW : nullptr, 0);
    TCONV_BPACK_DISPATCH8(1, 2, 8, p.s0.T, 768, B,
        p.Y0, H16(W[9]), F32(W[10]), F32(W[11]), p.A, ST[4], NS[3], 768, 768,
        p.s0.T, p.s0.L, p.s0.SC, 1, p.s1.L, p.s1.SC, 2, 0, 2, B);
    if (p.W3i != nullptr)
        native_invoke(repack_activation_only,dim3(min(1152,96*DIVUP(p.s1.SC,128))),dim3(128),0,
            p.A, p.Ac, p.s1.SC, p.s1.SC, DIVUP(p.s1.SC,128));
    __half *cur = p.A, *nxt = p.Bb;
    for (int i = 0; i < 6; ++i) {
        const int bi = i + 1;
        const __half *ain = (i == 0 && p.Ac) ? p.Ac : cur;
        dense(s, 3, ain, k3iw + bi * IW, k3ib + bi * 1536, k3ia + bi * 768,
              nullptr, p.M1, ST[5 + 2 * i], NS[4 + 2 * i], 768, 1536, p.s1,
              p.s1.SC, 1, 0, 2, p.W3i ? p.W3i + (bi - 1) * IW : nullptr,
              (i > 0 || p.Ac) ? 1 : 0);
        const __half *ps = (i < 5) ? ST[6 + 2 * i] : ST[16];
        const int po = (i < 5) ? 0 : 1, pl = 1 + (i < 5);
        dense(s, 3, p.M1, k3sw + bi * SW, k3sb + bi * 768, k3sa + bi * 384, cur,
              nxt, ps, NS[5 + 2 * i], 1536, 768, p.s1, p.s1.SC, 1, po, pl,
              p.W3s ? p.W3s + (bi - 1) * SW : nullptr,
              i == 0 ? 1 : (i < 5 ? 2 : 3));
        __half *t = cur; cur = nxt; nxt = t;
    }
    TCONV_BPACK_DISPATCH1(2, 9, 16, p.s1.T, 384, B,
        cur, H16(W[12]), F32(W[13]), F32(W[14]), p.P, ST[17], NS[16], 768, 384,
        p.s1.T, p.s1.L, p.s1.SC, 1, p.s2.L, p.s2.SC, 6, 0, 6, B);
    dense7(s, 0, p.P, p.W7[0], F32(W[22]), F32(W[23]), nullptr, p.M2, ST[18],
           NS[17], p.s2, p.s2.SC, 0, 6);
    dense7(s, 1, p.M2, p.W7[1], F32(W[25]), F32(W[26]), p.P, p.Q, ST[19],
           NS[18], p.s2, p.s2.SC, 5, 1);
    TCONV_BPACK_DISPATCH(3, 7, 16, p.s2.T, 128, B,
        p.Q, H16(W[15]), F32(W[16]), F32(W[17]), p.R, ST[20], NS[19], 384, 128,
        p.s2.T, p.s2.L, p.s2.SC, 5, p.s3.L, p.s3.SC, 6, 0, 6, B);
    dense7(s, 2, p.R, p.W7[2], F32(W[28]), F32(W[29]), nullptr, p.M3, ST[21],
           NS[20], p.s3, p.s3.SC, 0, 6);
    dense7(s, 3, p.M3, p.W7[3], F32(W[31]), F32(W[32]), p.R, p.S, ST[22],
           NS[21], p.s3, p.s3.SC, 5, 1);
    if (p.s3.T >= 1500)
        pdl_launch(transpose_conv_kernel<4, 7, 8, true>, dim3(1, 32, B),
                   dim3(tc_blk(p.s3.T, 8)),
                   (size_t)tc_blk(p.s3.T, 8) * (8 * 7) * 2 + 64, s,
                   p.S, H16(W[18]), F32(W[19]), F32(W[20]), p.U, ST[23],
                   NS[22], 128, 32, p.s3.T, p.s3.L, p.s3.SC, 5, p.s4.L,
                   p.s4.SC, 6, 0, 6, B);
    else
        pdl_launch(transpose_conv_kernel<4, 7, 4, true>, dim3(1, 32, B),
                   dim3(tc_blk(p.s3.T, 4)),
                   (size_t)tc_blk(p.s3.T, 4) * (4 * 7) * 2 + 64, s,
                   p.S, H16(W[18]), F32(W[19]), F32(W[20]), p.U, ST[23],
                   NS[22], 128, 32, p.s3.T, p.s3.L, p.s3.SC, 5, p.s4.L,
                   p.s4.SC, 6, 0, 6, B);
    {
        const int ntile = DIVUP(p.s4.N, TL_BN);
        const int grid = ntile < 132 ? ntile : 132;
        pdl_launch_pf(tail_fused_kernel, dim3(grid), dim3(TL_NT), TL_SHARED, s,
                   p.U, (size_t)32 * p.s4.SC * sizeof(__half),
                   p.U, p.W7[4], p.W7[5], H16(W[39]), F32(W[34]), F32(W[35]),
                   F32(W[37]), F32(W[38]), F32(W[40]), ST[24], ST[25],
                   static_cast<__half *>(a.waveform), NS[23], NS[24], NS[25], B,
                   p.s4.T, p.s4.L, p.s4.SC, p.s4.N, ntile);
    }
}
