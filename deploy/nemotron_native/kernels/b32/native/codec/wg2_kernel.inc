__device__ __forceinline__ void cpa16(void *dsm, const void *src, bool pred) {
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(dsm));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a),
                 "l"(src), "r"(pred ? 16 : 0)
                 : "memory");
}
template <int OFF>
__device__ __forceinline__ void cpa16o(unsigned a, const void *src, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0+%3], [%1], 16, %2;\n" ::"r"(a),
                 "l"(src), "r"(pred ? 16 : 0), "n"(OFF)
                 : "memory");
}
template <int KT, int CIB, int MSPLIT, int KSPLIT, int NSPLIT, int WG_BN,
          int SA, int SB, int MINB>
__device__ __noinline__
void wg2_conv_kernel(
    const __half *__restrict__ in, const __half *__restrict__ w,
    const float *__restrict__ bias, const float *__restrict__ alpha,
    const __half *__restrict__ resid, __half *__restrict__ out,
    const __half *__restrict__ pad_src, __half *__restrict__ ns,
    int Cin, int Cout, int T, int L, int SCin, int SCout, int N,
    int do_snake, int pad_off, int pad_len) {
    constexpr int BK = CIB * KT;
    constexpr int WG_BM = 64 * MSPLIT;
    constexpr int WG_BNT = WG_BN * NSPLIT;
    constexpr int NTS = 128 * MSPLIT * NSPLIT;
    constexpr int WG_NT = NTS * KSPLIT;
    constexpr int NCRAW = WG_BNT / 4;
    constexpr int EPN =
        (WG_NT % NCRAW == 0) ? WG_BNT : (((WG_BNT + 127) / 128) * 128);
    constexpr int WG_CLD = EPN + 8;
    constexpr bool NB8_P2 = ((WG_BN / 8) & (WG_BN / 8 - 1)) == 0;
    constexpr int KG = BK / 8;
    constexpr int A_E = (WG_BM / 8) * KG * WG_SLOT;
    constexpr int B_E = KG * (WG_BN / 8) * WG_SLOT;
    constexpr int ST_E = SA * A_E + SB * NSPLIT * B_E;
    constexpr int SBO_AE = KG * WG_SLOT;
    constexpr int LBO_BE = (WG_BN / 8) * WG_SLOT;
    constexpr int NB8 = WG_BN / 8;
    constexpr bool ALDG = (WG_BN <= 64);
    char *arena=reinterpret_cast<char*>(native_arena());
    char *colok=reinterpret_cast<char*>(native_aux());
    float *row_alpha=reinterpret_cast<float*>(native_aux()+1024);
    float *row_ri=reinterpret_cast<float*>(native_aux()+2048);
    __half *smem = reinterpret_cast<__half *>(arena);
    float *Cs = reinterpret_cast<float *>(arena);
    const int tid = threadIdx.x;
    const int kgrp = tid / NTS;
    const int stid = tid - kgrp * NTS;
    const int wgl = stid >> 7;
    const int mg = wgl % MSPLIT;
    const int ngid = wgl / MSPLIT;
    const int warp = (stid & 127) >> 5, lane = tid & 31;
    const int m0 = blockIdx.x * WG_BM;
    const int nb = blockIdx.y * WG_BNT;
    const int n0 = nb + ngid * WG_BN;
    const int H = KT - 1;
    const int Kg = Cin * KT;
    const int half = Cout >> 1;
    const int NCH = (Kg + BK - 1) / BK;
    if (tid < WG_BM) {
        const int co = m0 + tid;
        if (do_snake && co < Cout && co < half) {
            const float a = alpha[co];
            row_alpha[tid] = a;
            row_ri[tid] = __frcp_rn(a + 1e-9f);
        }
    }
    for (int i = tid; i < EPN; i += WG_NT) {
        const int n = nb + i;
        colok[i] = (i < WG_BNT && n < N && (n % L) < T) ? 1 : 0;
    }
    const int er0 = mg * 64 + warp * 16 + (lane >> 2);
    const int eco0 = m0 + er0, eco1 = m0 + er0 + 8;
    const float bias0 = (kgrp == 0 && eco0 < Cout) ? bias[eco0] : 0.f;
    const float bias1 = (kgrp == 0 && eco1 < Cout) ? bias[eco1] : 0.f;
    float acc[WG_BN / 2];
#pragma unroll
    for (int i = 0; i < WG_BN / 2; ++i) acc[i] = ((i & 2) == 0) ? bias0 : bias1;
    constexpr int KGX = ALDG ? (KG <= 8 ? 8 : (KG <= 16 ? 16 : 32)) : KG;
    constexpr int ATOT = WG_BM * KGX;
    constexpr int AREG = (ATOT + NTS - 1) / NTS;
    constexpr int BTOT = NSPLIT * CIB * NB8;
    constexpr int BREG = (BTOT + NTS - 1) / NTS;
    const __half *aptr[AREG];
    int aoff[AREG];
    int akmax[AREG];
#pragma unroll
    for (int r = 0; r < AREG; ++r) {
        const int idx = stid + r * NTS;
        const int m = idx / KGX, v = idx - (idx / KGX) * KGX;
        const int co = m0 + m;
        const bool live = (ATOT % NTS == 0 || idx < ATOT) && (co < Cout) &&
                          v < KG;
        aoff[r] = (m >> 3) * SBO_AE + (v < KG ? v : 0) * WG_SLOT + (m & 7) * 8;
        akmax[r] = live ? (Kg - v * 8) : 0;
        aptr[r] = w + (size_t)(live ? co : 0) * Kg + v * 8;
    }
    const __half *bptr[BREG];
    int bcimax[BREG], bnc[BREG], bcil[BREG], bgrp[BREG];
#pragma unroll
    for (int r = 0; r < BREG; ++r) {
        const int idx = stid + r * NTS;
        const int gg = idx / (CIB * NB8);
        const int rr = idx - gg * (CIB * NB8);
        const int cil = rr / NB8, nc = rr - (rr / NB8) * NB8;
        bcil[r] = cil;
        bnc[r] = nc;
        bgrp[r] = gg;
        const bool live = (BTOT % NTS == 0 || idx < BTOT);
        bcimax[r] = live ? (Cin - cil) : 0;
        bptr[r] = in + (size_t)cil * SCin + nb + gg * WG_BN + nc * 8;
    }
    int4 br0[BREG], br1[BREG];
    int4 ar[ALDG ? AREG : 1];
    uint32_t edge[BREG];
    __half *const kaB = smem + kgrp * ST_E;
    __half *const kbB = kaB + SA * A_E;
    const unsigned barid = 1u + kgrp + (unsigned)(KSPLIT * native_subid());
#define WG2_BAR() asm volatile("barrier.sync %0, %1;" ::"r"(barid), "r"(NTS) : "memory")
#define WG2_LOADB(c) { const int _ci0 = (c) * CIB; _Pragma("unroll") for (int r = 0; r < BREG; ++r) { br0[r] = make_int4(0, 0, 0, 0); if constexpr (KT == 3) edge[r] = 0; else br1[r] = make_int4(0, 0, 0, 0); if (_ci0 < bcimax[r]) { const int4 *src = reinterpret_cast<const int4 *>( bptr[r] + (size_t)_ci0 * SCin); br0[r] = src[0]; if constexpr (KT == 3) { if (!NB8_P2 || bnc[r] == NB8 - 1) edge[r] = src[1].x; } else { br1[r] = src[1]; } } } }
#define WG2_LOADA(c) if constexpr (ALDG) { const int _kg0 = (c) * BK; _Pragma("unroll") for (int r = 0; r < AREG; ++r) { ar[r] = make_int4(0, 0, 0, 0); if (_kg0 < akmax[r]) ar[r] = *reinterpret_cast<const int4 *>(aptr[r] + _kg0); } }
#define WG2_PUTA(c,AD) { _Pragma("unroll") for (int r = 0; r < AREG; ++r) { if constexpr (ALDG) { if (akmax[r] > 0) *reinterpret_cast<int4 *>((AD) + aoff[r]) = ar[r]; } else { const int _kg0 = (c) * BK; if (akmax[r] > 0) cpa16((AD) + aoff[r], aptr[r] + _kg0, _kg0 < akmax[r]); } } if constexpr (!ALDG) asm volatile("cp.async.commit_group;\n" ::: "memory"); }
#define WG2_STSB(BD) { _Pragma("unroll") for (int r = 0; r < BREG; ++r) { const int cil = bcil[r], nc = bnc[r]; unsigned int u[8]; u[0] = br0[r].x; u[1] = br0[r].y; u[2] = br0[r].z; u[3] = br0[r].w; if constexpr (KT == 3) { if constexpr (NB8_P2) { u[4] = __shfl_down_sync(0xffffffffu, u[0], 1, NB8); if (nc == NB8 - 1) u[4] = edge[r]; } else { u[4] = edge[r]; } u[5] = u[6] = u[7] = 0; } else { u[4] = br1[r].x; u[5] = br1[r].y; u[6] = br1[r].z; u[7] = br1[r].w; } if (BTOT % NTS == 0 || (stid + r * NTS) < BTOT) { _Pragma("unroll") for (int k = 0; k < KT; ++k) { const int kh = k >> 1; int4 o; if (k & 1) { o.x = __funnelshift_r(u[kh + 0], u[kh + 1], 16); o.y = __funnelshift_r(u[kh + 1], u[kh + 2], 16); o.z = __funnelshift_r(u[kh + 2], u[kh + 3], 16); o.w = __funnelshift_r(u[kh + 3], u[kh + 4], 16); } else { o.x = u[kh + 0]; o.y = u[kh + 1]; o.z = u[kh + 2]; o.w = u[kh + 3]; } const int row = cil * KT + k; *reinterpret_cast<int4 *>((BD) + bgrp[r] * B_E + nc * WG_SLOT + (row >> 3) * LBO_BE + (row & 7) * 8) = o; } } } }
#define WG2_MMA(AD,BD) { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); const uint64_t da0 = wg_desc((AD) + mg * 8 * SBO_AE, WG_SLOT * 2, SBO_AE * 2); const uint64_t db0 = wg_desc((BD) + ngid * B_E, LBO_BE * 2, WG_SLOT * 2); _Pragma("unroll") for (int kk = 0; kk < BK; kk += 16) { const uint64_t da = da0 + (kk >> 3) * (WG_SLOT / 8); const uint64_t db = db0 + (kk >> 3) * (LBO_BE / 8); if constexpr (WG_BN == 64) { WGMMA_M64N64K16(acc, da, db); } else if constexpr (WG_BN == 256) { WGMMA_M64N256K16(acc, da, db); } else if constexpr (WG_BN == 104) { WGMMA_M64N104K16(acc, da, db); } else if constexpr (WG_BN == 112) { WGMMA_M64N112K16(acc, da, db); } else if constexpr (WG_BN == 120) { WGMMA_M64N120K16(acc, da, db); } else if constexpr (WG_BN == 224) { WGMMA_M64N224K16(acc, da, db); } else { static_assert(WG_BN == 128, "no wgmma macro for WG_BN"); WGMMA_M64N128K16(acc, da, db); } } asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
    constexpr int SMAX = SA > SB ? SA : SB;
    WG2_LOADA(kgrp)
    WG2_LOADB(kgrp)
#pragma unroll 1
    for (int s = 0; s < SMAX; ++s) {
        const int c = s * KSPLIT + kgrp;
        if (s < SA) {
            if (c < NCH) WG2_PUTA(c, kaB + s * A_E)
            if (c + KSPLIT < NCH) WG2_LOADA(c + KSPLIT)
        }
        if (s < SB) {
            if (c < NCH) WG2_STSB(kbB + s * NSPLIT * B_E)
            if (c + KSPLIT < NCH) WG2_LOADB(c + KSPLIT)
        }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    const int NIT = (NCH - kgrp + KSPLIT - 1) / KSPLIT;
    for (int i = 0; i < NIT; ++i) {
        if constexpr (!ALDG)
            asm volatile("cp.async.wait_group %0;\n" ::"n"(SA - 2) : "memory");
        WG2_BAR();
        WG2_MMA(kaB + (i % SA) * A_E, kbB + (i % SB) * NSPLIT * B_E)
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        if constexpr (NTS > 128) WG2_BAR();
        if (i >= 1) {
            const int ca = (i + SA - 1) * KSPLIT + kgrp;
            if (ca < NCH) WG2_PUTA(ca, kaB + ((i - 1) % SA) * A_E)
            if (ALDG && ca + KSPLIT < NCH) WG2_LOADA(ca + KSPLIT)
            const int cb = (i + SB - 1) * KSPLIT + kgrp;
            if (cb < NCH) {
                WG2_STSB(kbB + ((i - 1) % SB) * NSPLIT * B_E)
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                if (cb + KSPLIT < NCH) WG2_LOADB(cb + KSPLIT)
            } else if constexpr (ALDG) {
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            }
        }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#undef WG2_LOADB
#undef WG2_LOADA
#undef WG2_PUTA
#undef WG2_STSB
#undef WG2_MMA
#undef WG2_BAR
    __syncthreads();
    float *Ck = Cs + kgrp * WG_BM * WG_CLD;
    const int cb = (lane & 3) * 2;
#pragma unroll
    for (int j = 0; j < WG_BN / 8; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float2 v = make_float2(acc[j * 4 + h * 2], acc[j * 4 + h * 2 + 1]);
            *reinterpret_cast<float2 *>(Ck + (er0 + h * 8) * WG_CLD +
                                        ngid * WG_BN + j * 8 + cb) = v;
        }
    }
    __syncthreads();
    constexpr int NC = EPN / 4;
    constexpr int MSTEP = WG_NT / NC;
    constexpr int MITER = WG_BM / MSTEP;
    constexpr int UNR = (MITER % 4 == 0) ? 4 : ((MITER % 2 == 0) ? 2 : 1);
    static_assert(MITER % UNR == 0, "row batch must divide the row count");
    const int lc = (tid % NC) * 4;
    const int lr = tid / NC;
    const bool ok4 = colok[lc] && colok[lc + 1] && colok[lc + 2] && colok[lc + 3];
    const int n = nb + lc;
#pragma unroll 1
    for (int t0 = 0; t0 < MITER; t0 += UNR) {
        float4 v[UNR];
        __half2 r0[UNR], r1[UNR];
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            const int m = lr + (t0 + u) * MSTEP;
            v[u] = *reinterpret_cast<const float4 *>(Cs + m * WG_CLD + lc);
            if (resid != nullptr) {
                const int co = m0 + m;
                const int cc = co < Cout ? co : 0;
                const __half2 *rp = reinterpret_cast<const __half2 *>(
                    resid + (size_t)cc * SCout + n + H);
                r0[u] = rp[0];
                r1[u] = rp[1];
            }
        }
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            const int m = lr + (t0 + u) * MSTEP;
            const int co = m0 + m;
            if (co >= Cout) continue;
            float4 vv = v[u];
#pragma unroll
            for (int g = 1; g < KSPLIT; ++g) {
                float4 p = *reinterpret_cast<const float4 *>(
                    Cs + g * WG_BM * WG_CLD + m * WG_CLD + lc);
                vv.x += p.x; vv.y += p.y; vv.z += p.z; vv.w += p.w;
            }
            if (resid != nullptr) {
                vv.x += __half2float(r0[u].x); vv.y += __half2float(r0[u].y);
                vv.z += __half2float(r1[u].x); vv.w += __half2float(r1[u].y);
            }
            float *vp = reinterpret_cast<float *>(&vv);
            if (do_snake) {
                if (co < half) {
                    const float a = row_alpha[m];
                    const float ri = row_ri[m];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float sn = __sinf(a * vp[e]);
                        vp[e] = __fmaf_rn(sn * sn, ri, vp[e]);
                    }
                } else {
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        vp[e] = vp[e] > 0.f ? vp[e] : 0.01f * vp[e];
                }
            }
            __half *op = out + (size_t)co * SCout + n + H;
            if (ok4) {
                reinterpret_cast<__half2 *>(op)[0] =
                    __floats2half2_rn(vv.x, vv.y);
                reinterpret_cast<__half2 *>(op)[1] =
                    __floats2half2_rn(vv.z, vv.w);
            } else {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    if (colok[lc + e]) op[e] = __float2half(vp[e]);
            }
        }
    }
    {
        const int nbx = gridDim.x * gridDim.y;
        const int bxy = blockIdx.x + blockIdx.y * gridDim.x;
        const int B = N / L;
        if (ns != nullptr) {
            const int total = Cin * B * H;
            for (int idx = bxy * WG_NT + tid; idx < total;
                 idx += nbx * WG_NT) {
                const int j = idx % H, r = idx / H;
                const int b = r % B, ci = r / B;
                ns[((size_t)b * Cin + ci) * H + j] =
                    in[(size_t)ci * SCin + b * L + native_true_tokens(b,T) + j];
            }
        }
        if (pad_src != nullptr) {
            const int total = Cout * B * pad_len;
            for (int idx = bxy * WG_NT + tid; idx < total;
                 idx += nbx * WG_NT) {
                const int j = idx % pad_len, r = idx / pad_len;
                const int b = r % B, co = r / B;
                out[(size_t)co * SCout + b * L + pad_off + j] =
                    pad_src[((size_t)b * Cout + co) * pad_len + j];
            }
        }
    }
}
