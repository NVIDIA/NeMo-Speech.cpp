template <int CIB, int MSPLIT, int KSPLIT, int WG_BN, int SA, int SB, int MINB,
          int WGD, bool BC8 = false, bool OC8 = false, bool RC8 = false,
          int CIN = 0, bool PDL = false>
__device__ __noinline__
void wg5_conv_kernel(
    const __half *__restrict__ in, const __half *__restrict__ w,
    const float *__restrict__ bias, const float *__restrict__ alpha,
    const __half *__restrict__ resid, __half *__restrict__ out,
    const __half *__restrict__ pad_src, __half *__restrict__ ns,
    int Cin, int Cout, int T, int L, int SCin, int SCout, int N,
    int do_snake, int pad_off, int pad_len) {
    constexpr int KT = 3;
    constexpr int BM = 64 * MSPLIT;
    constexpr int BNT = WG_BN;
    constexpr int NTS = 128 * MSPLIT;
    constexpr int WG_NT = NTS * KSPLIT;
    constexpr int NCRAW = BNT / 4;
    constexpr int EPN =
        (WG_NT % NCRAW == 0) ? BNT : (((BNT + 127) / 128) * 128);
    constexpr int WG_CLD_R = (OC8 ? WG_BN : EPN) + 8;
    constexpr int WG_CLD = (WG_CLD_R % 16 == 0) ? WG_CLD_R + 4 : WG_CLD_R;
    constexpr int CG = CIB / 8;
    constexpr int CSUB = CIB / 16;
    constexpr int AK = KT * CSUB;
    constexpr int NBRP = BC8 ? (((BNT + KT - 1) + 7) & ~7)
                             : (((BNT + KT - 1) + 31) & ~31);
    constexpr int UCOL = NBRP / 32;
    constexpr int NUNIT = CG * UCOL;
    constexpr int NWARP = NTS / 32;
    constexpr int BITER = (NUNIT + NWARP - 1) / NWARP;
    constexpr int B_E = CIB * NBRP;
    constexpr int ST_E = SB * B_E;
    constexpr int LBO_B = NBRP * 8;
    constexpr int ABUF = 3;
    char *arena=reinterpret_cast<char*>(native_arena());
    char *colok=reinterpret_cast<char*>(native_aux());
    float *row_alpha=reinterpret_cast<float*>(native_aux()+1024);
    float *row_ri=reinterpret_cast<float*>(native_aux()+2048);
    __half *smem = reinterpret_cast<__half *>(arena);
    float *Cs = reinterpret_cast<float *>(arena);
    const int tid = threadIdx.x;
    const int kgrp = tid / NTS;
    const int stid = tid - kgrp * NTS;
    const int mg = stid >> 7;
    const int warp = (stid & 127) >> 5, lane = tid & 31;
    const int m0 = blockIdx.x * BM;
    const int nb = blockIdx.y * BNT;
    const int H = KT - 1;
    const int Kg = Cin * KT;
    const int half = Cout >> 1;
    constexpr int NCH = (CIN + CIB - 1) / CIB;
    static_assert(NCH % KSPLIT == 0, "wg5 K-split must divide the chunk count");
    bool active_tile = false;
    for (int row = nb / L; row <= min(N - 1, nb + BNT - 1) / L; ++row) {
        if (native_meta()->lengths[row] > 0) {
            active_tile = true;
            break;
        }
    }
    if (active_tile) {
    if (tid < BM) {
        const int co = m0 + tid;
        if (co < Cout && co < half) {
            const float a = alpha[co];
            row_alpha[tid] = a;
            row_ri[tid] = __frcp_rn(a + 1e-9f);
        }
    }
    for (int i = tid; i < EPN; i += WG_NT) {
        const int n = nb + i;
        colok[i] = (i < BNT && n < N && (n % L) < T) ? 1 : 0;
    }
    const int er0 = mg * 64 + warp * 16 + (lane >> 2);
    const int eco0 = m0 + er0, eco1 = m0 + er0 + 8;
    const float bias0 = (kgrp == 0 && eco0 < Cout) ? bias[eco0] : 0.f;
    const float bias1 = (kgrp == 0 && eco1 < Cout) ? bias[eco1] : 0.f;
    float acc[WG_BN / 2];
#pragma unroll
    for (int i = 0; i < WG_BN / 2; ++i) acc[i] = ((i & 2) == 0) ? bias0 : bias1;
    const int mtile = blockIdx.x * MSPLIT + mg;
    const __half *const apt = w + ((size_t)mtile * NCH * AK) * 1024 +
                              (size_t)(stid & 127) * 8;
    uint32_t areg[ABUF][AK][4];
    const int wi = stid >> 5;
    int bdst[BITER], boff[BITER];
    bool blive[BITER];
    const __half *const bbase = in + (size_t)(lane >> 2) * SCin + nb +
                                8 * (lane & 3);
#pragma unroll
    for (int t = 0; t < BITER; ++t) {
        const int u = wi + t * NWARP;
        const int gc = (u < NUNIT) ? u / UCOL : 0;
        const int cu = (u < NUNIT) ? u - gc * UCOL : 0;
        blive[t] = u < NUNIT;
        bdst[t] = gc * LBO_B + (cu * 32 + lane) * 8;
        boff[t] = gc * 8 * SCin + cu * 32;
    }
    const size_t SC8i = (size_t)SCin * 8;
    constexpr int BQ = B_E / 8;
    constexpr int BCREG = (BQ + NTS - 1) / NTS;
    int cdst[BCREG];
    const __half *csrc[BCREG];
    bool clive[BCREG];
#pragma unroll
    for (int r = 0; r < BCREG; ++r) {
        const int q = stid + r * NTS;
        const int gc = q / NBRP, rem = q - gc * NBRP;
        clive[r] = ((BQ % NTS == 0) || q < BQ) && gc < CG;
        cdst[r] = clive[r] ? gc * LBO_B + rem * 8 : 0;
        csrc[r] = in + (clive[r] ? (size_t)gc * SC8i +
                                  (size_t)nb * 8 + rem * 8 : 0);
    }
    uint32_t br[BITER][4];
    __half *const kbB = smem + kgrp * ST_E;
    const unsigned barid = 1u + kgrp + (unsigned)(KSPLIT * native_subid());
    uint64_t dbuf[ABUF];
#pragma unroll
    for (int s = 0; s < ABUF; ++s)
        dbuf[s] = wg_desc(kbB + s * B_E, LBO_B * 2u, 128u);
    uint32_t csm0[BCREG];
#pragma unroll
    for (int r = 0; r < BCREG; ++r)
        csm0[r] = static_cast<uint32_t>(
            __cvta_generic_to_shared(kbB + cdst[r]));
    const size_t CSTEP = (size_t)KSPLIT * CG * SC8i;
    constexpr int ASTEP = KSPLIT * AK * 1024;
    const __half *crol[BCREG];
#pragma unroll
    for (int r = 0; r < BCREG; ++r)
        crol[r] = csrc[r] + (size_t)(ABUF * KSPLIT + kgrp) * CG * SC8i;
    const __half *arol = apt + (size_t)(ABUF * KSPLIT + kgrp) * (AK * 1024);
    int crolc = ABUF * KSPLIT + kgrp;
#define WG5_BAR() asm volatile("bar.sync %0, %1;\n" ::"r"(barid), "r"(NTS))
#define WG5_CPB(c,BD) { const size_t _o = (size_t)(c) * CG * SC8i; _Pragma("unroll") for (int r = 0; r < BCREG; ++r) if (clive[r]) cpa16((BD) + cdst[r], csrc[r] + _o, true); if ((c) / KSPLIT == ABUF - 1) { const size_t _r = (size_t)(ABUF * KSPLIT + kgrp) * CG * SC8i; _Pragma("unroll") for (int r = 0; r < BCREG; ++r) csrc[r] += _r; } }
#define WG5_LOADA(BUF,c) { (void)(c); }
#define WG5_KEEPA(BUF) { _Pragma("unroll") for (int kk = 0; kk < AK; ++kk) asm volatile("" : "+r"(areg[BUF][kk][0]), "+r"(areg[BUF][kk][1]), "+r"(areg[BUF][kk][2]), "+r"(areg[BUF][kk][3])); }
#define WG5_LOADB(c) { const size_t _o = (size_t)(c) * CIB * SCin; _Pragma("unroll") for (int t = 0; t < BITER; ++t) { if (NUNIT == NWARP * BITER || blive[t]) { const int4 _v = *reinterpret_cast<const int4 *>( bbase + _o + boff[t]); br[t][0] = _v.x; br[t][1] = _v.y; br[t][2] = _v.z; br[t][3] = _v.w; } } }
#define WG5_BXPOSE(u) { const int _q = lane & 3; uint32_t _x, _y; _x = (_q & 1) ? u[0] : u[1]; _y = (_q & 1) ? u[2] : u[3]; _x = __shfl_xor_sync(0xffffffffu, _x, 1); _y = __shfl_xor_sync(0xffffffffu, _y, 1); if (_q & 1) { u[0] = _x; u[2] = _y; } else { u[1] = _x; u[3] = _y; } _x = (_q & 2) ? u[0] : u[2]; _y = (_q & 2) ? u[1] : u[3]; _x = __shfl_xor_sync(0xffffffffu, _x, 2); _y = __shfl_xor_sync(0xffffffffu, _y, 2); if (_q & 2) { u[0] = _x; u[1] = _y; } else { u[2] = _x; u[3] = _y; } }
#define WG5_STOREB(BD) { _Pragma("unroll") for (int t = 0; t < BITER; ++t) { if (NUNIT == NWARP * BITER || blive[t]) { WG5_BXPOSE(br[t]) const uint32_t _a = static_cast<uint32_t>( __cvta_generic_to_shared((BD) + bdst[t])); asm volatile( "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 " "[%0], {%1,%2,%3,%4};\n" ::"r"(_a), "r"(br[t][0]), "r"(br[t][1]), "r"(br[t][2]), "r"(br[t][3])); } } }
#define WG5_MMA(BUF) { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); const uint64_t db0 = dbuf[0] + (uint64_t)(BUF) * (B_E / 8); _Pragma("unroll") for (int h = 0; h < CSUB; ++h) { _Pragma("unroll") for (int kk = 0; kk < KT; ++kk) { WGMMA_RS_ANY_T(WG_BN, acc, areg[BUF][h * KT + kk], db0 + kk + h * (LBO_B / 4), 0); } } asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
#define WG5_CPB_R(SLOT) { _Pragma("unroll") for (int r = 0; r < BCREG; ++r) if (clive[r]) cpa16o<(SLOT) * B_E * 2>(csm0[r], csrc[r], true); }
#define WG5_LOADA_R(BUF) { }
#define WG5_ROLL() { crolc += KSPLIT; arol += ASTEP; if constexpr (BC8) { _Pragma("unroll") for (int r = 0; r < BCREG; ++r) csrc[r] += CSTEP; } }
    if constexpr (PDL) {
        (void)0;
        if (tid == 0)
            (void)0;
    }
    static_assert(SB == 3 && WGD == 1 && ABUF == 3, "wg5 pipeline shape");
    if constexpr (BC8) {
#pragma unroll 1
        for (int s = 0; s < SB; ++s) {
            const int c = s * KSPLIT + kgrp;
            if (c < NCH) WG5_CPB(c, kbB + s * B_E)
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }
    } else {
        WG5_LOADB(kgrp)
#pragma unroll 1
        for (int s = 0; s < SB; ++s) {
            const int c = s * KSPLIT + kgrp;
            if (c < NCH) WG5_STOREB(kbB + s * B_E)
            if (c + KSPLIT < NCH) WG5_LOADB(c + KSPLIT)
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    }
    WG5_LOADA(0, kgrp)
    if (KSPLIT + kgrp < NCH) WG5_LOADA(1, KSPLIT + kgrp)
    if (2 * KSPLIT + kgrp < NCH) WG5_LOADA(2, 2 * KSPLIT + kgrp)
    constexpr int NIT = NCH / KSPLIT;
    static_assert(NIT % ABUF == 0, "wg5 mainloop unroll shape");
#define WG5_BODY(BUF,i) { { const __half *_at = apt + (size_t)((i) * KSPLIT + kgrp) * (AK * 1024); _Pragma("unroll") for (int kk = 0; kk < AK; ++kk) { const int4 _v = *reinterpret_cast<const int4 *>(_at + kk * 1024); areg[BUF][kk][0] = _v.x; areg[BUF][kk][1] = _v.y; areg[BUF][kk][2] = _v.z; areg[BUF][kk][3] = _v.w; } } if constexpr (BC8) asm volatile("cp.async.wait_group %0;\n" : : "n"(SB - 2) : "memory"); WG5_BAR(); WG5_MMA(BUF) asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); WG5_KEEPA(BUF) if constexpr (!BC8) if ((i) > 1 && crolc < NCH) WG5_LOADB(crolc) if constexpr (NTS > 128) WG5_BAR(); if ((i) >= 1) { const int nb2 = ((BUF) + 2) % ABUF; if (crolc < NCH) { if constexpr (BC8) { WG5_CPB_R(nb2) } else { WG5_STOREB(kbB + nb2 * B_E) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); } WG5_LOADA_R(nb2) } WG5_ROLL() if constexpr (BC8) asm volatile("cp.async.commit_group;\n" ::: "memory"); } }
#pragma unroll
    for (int i = 0; i < NIT; i += 3) {
        WG5_BODY(0, i)
        WG5_BODY(1, i + 1)
        WG5_BODY(2, i + 2)
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    if constexpr (BC8)
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#undef WG5_BXPOSE
#undef WG5_CPB
#undef WG5_CPB_R
#undef WG5_LOADA_R
#undef WG5_ROLL
#undef WG5_LOADB
#undef WG5_STOREB
#undef WG5_MMA
#undef WG5_LOADA
#undef WG5_KEEPA
#undef WG5_BODY
#undef WG5_BAR
    __syncthreads();
    const int cb2 = (lane & 3) * 2;
    if constexpr (OC8) {
        __half *Ck = smem + kgrp * BM * WG_CLD;
#pragma unroll
        for (int j = 0; j < WG_BN / 8; ++j) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                *reinterpret_cast<__half2 *>(Ck + (er0 + h * 8) * WG_CLD +
                                             j * 8 + cb2) =
                    __floats2half2_rn(acc[j * 4 + h * 2],
                                      acc[j * 4 + h * 2 + 1]);
            }
        }
    } else {
        float *Ck = Cs + kgrp * BM * WG_CLD;
#pragma unroll
        for (int j = 0; j < WG_BN / 8; ++j) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float2 v = make_float2(acc[j * 4 + h * 2],
                                       acc[j * 4 + h * 2 + 1]);
                *reinterpret_cast<float2 *>(Ck + (er0 + h * 8) * WG_CLD +
                                            j * 8 + cb2) = v;
            }
        }
    }
    __syncthreads();
    if constexpr (OC8) {
        constexpr int GRP = BM / 8;
        constexpr int TOTG = GRP * WG_BN;
        constexpr int GITER = (TOTG + WG_NT - 1) / WG_NT;
        const size_t SC8o = (size_t)SCout * 8;
#pragma unroll
        for (int t = 0; t < GITER; ++t) {
            const int idx = tid + t * WG_NT;
            if (TOTG % WG_NT != 0 && idx >= TOTG) continue;
            const int j = idx - (idx / WG_BN) * WG_BN, g = idx / WG_BN;
            if (!colok[j]) continue;
            const int m0g = g * 8;
            if (m0 + m0g >= Cout) continue;
            const int n = nb + j;
            __half rr[8];
            if constexpr (CIN == 1536) {
                if constexpr (RC8) {
                    *reinterpret_cast<int4 *>(rr) =
                        *reinterpret_cast<const int4 *>(
                            resid + (size_t)((m0 + m0g) >> 3) * SC8o +
                            (size_t)(n + H) * 8);
                } else {
#pragma unroll
                    for (int h = 0; h < 8; ++h)
                        rr[h] = resid[(size_t)(m0 + m0g + h) * SCout + n + H];
                }
            }
            uint32_t pk[4];
            const bool snake = (m0 + m0g) < half;
#pragma unroll
            for (int h = 0; h < 8; h += 2) {
                float f[2];
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int m = m0g + h + e;
                    float a = __half2float(smem[m * WG_CLD + j]);
#pragma unroll
                    for (int kg = 1; kg < KSPLIT; ++kg)
                        a += __half2float(
                            smem[kg * BM * WG_CLD + m * WG_CLD + j]);
                    if constexpr (CIN == 1536)
                        a += __half2float(rr[h + e]);
                    if (snake) {
                        const float sn = __sinf(row_alpha[m] * a);
                        a = __fmaf_rn(sn * sn, row_ri[m], a);
                    } else {
                        a = a > 0.f ? a : 0.01f * a;
                    }
                    f[e] = a;
                }
                const __half2 p = __floats2half2_rn(f[0], f[1]);
                pk[h >> 1] = *reinterpret_cast<const uint32_t *>(&p);
            }
            *reinterpret_cast<int4 *>(
                out + (size_t)((m0 + m0g) >> 3) * SC8o +
                (size_t)(n + H) * 8) = make_int4(pk[0], pk[1], pk[2], pk[3]);
        }
    } else {
    constexpr int NC = EPN / 4;
    constexpr int MSTEP = WG_NT / NC;
    constexpr int MITER = BM / MSTEP;
    constexpr int UNR = (MITER % 4 == 0) ? 4 : ((MITER % 2 == 0) ? 2 : 1);
    static_assert(MITER % UNR == 0, "row batch must divide the row count");
    const int lc = (tid % NC) * 4;
    const int lr = tid / NC;
    const bool ok4 = colok[lc] && colok[lc + 1] && colok[lc + 2] && colok[lc + 3];
    const bool live4 = colok[lc] || colok[lc + 1] || colok[lc + 2] ||
                       colok[lc + 3];
    const int n = nb + lc;
#pragma unroll 1
    for (int t0 = 0; live4 && t0 < MITER; t0 += UNR) {
        float4 v[UNR];
        __half2 r0[UNR], r1[UNR];
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            const int m = lr + (t0 + u) * MSTEP;
            v[u] = *reinterpret_cast<const float4 *>(Cs + m * WG_CLD + lc);
            if constexpr (CIN == 1536) {
                const int co = m0 + m;
                const int cc = co < Cout ? co : 0;
                if constexpr (RC8) {
                    const __half *rp =
                        resid + (size_t)(cc >> 3) * ((size_t)SCout * 8) +
                        (size_t)(n + H) * 8 + (cc & 7);
                    r0[u].x = rp[0]; r0[u].y = rp[8];
                    r1[u].x = rp[16]; r1[u].y = rp[24];
                } else {
                    const __half2 *rp = reinterpret_cast<const __half2 *>(
                        resid + (size_t)cc * SCout + n + H);
                    r0[u] = rp[0];
                    r1[u] = rp[1];
                }
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
                    Cs + g * BM * WG_CLD + m * WG_CLD + lc);
                vv.x += p.x; vv.y += p.y; vv.z += p.z; vv.w += p.w;
            }
            if constexpr (CIN == 1536) {
                vv.x += __half2float(r0[u].x); vv.y += __half2float(r0[u].y);
                vv.z += __half2float(r1[u].x); vv.w += __half2float(r1[u].y);
            }
            float *vp = reinterpret_cast<float *>(&vv);
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
    }
    }
    {
        const int nbx = gridDim.x * gridDim.y;
        const int bxy = blockIdx.x + blockIdx.y * gridDim.x;
        const int B = N / L;
        static_assert(CIN == 768 || CIN == 1536);
        constexpr int COUT = (CIN == 768) ? 1536 : 768;
        if (ns != nullptr) {
            const int total = CIN * B;
            for (int idx = bxy * WG_NT + tid; idx < total;
                 idx += nbx * WG_NT) {
                const int b = idx / CIN, ci = idx - b * CIN;
                const int x = b * L + native_true_tokens(b,T);
                __half *const dst = ns + ((size_t)b * CIN + ci) * H;
                if constexpr (BC8) {
                    const __half *const src =
                        in + (size_t)(ci >> 3) * ((size_t)SCin * 8) +
                        (size_t)x * 8 + (ci & 7);
#pragma unroll
                    for (int j = 0; j < H; ++j) dst[j] = src[j * 8];
                } else {
                    const __half *const src = in + (size_t)ci * SCin + x;
#pragma unroll
                    for (int j = 0; j < H; j += 2)
                        *reinterpret_cast<uint32_t *>(dst + j) =
                            *reinterpret_cast<const uint32_t *>(src + j);
                }
            }
        }
        if (pad_src != nullptr) {
            const int total = COUT * B;
            for (int idx = bxy * WG_NT + tid; idx < total;
                 idx += nbx * WG_NT) {
                const int b = idx / COUT, co = idx - b * COUT;
                const int x = b * L + pad_off;
                const __half *const src =
                    pad_src + ((size_t)b * COUT + co) * pad_len;
                if constexpr (OC8) {
                    __half *const dst =
                        out + (size_t)(co >> 3) * ((size_t)SCout * 8) +
                        (size_t)x * 8 + (co & 7);
                    for (int j = 0; j < pad_len; ++j) dst[j * 8] = src[j];
                } else {
                    __half *const dst = out + (size_t)co * SCout + x;
                    for (int j = 0; j < pad_len; ++j) dst[j] = src[j];
                }
            }
        }
    }
}
