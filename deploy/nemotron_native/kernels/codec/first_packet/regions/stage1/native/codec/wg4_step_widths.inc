template <int KT, int CIB, int MSPLIT, int NSPLIT, int WG_BN, int STAGES,
          int MINB, bool PREACT, int ABUF_ = 3, bool BC8 = false,
          bool OC8 = false, bool PDL = true, bool LATE_PDL = false>
__device__ __noinline__ void wg4_conv_kernel(
    const __half *__restrict__ in, const __half *__restrict__ w,
    const float *__restrict__ bias, const float *__restrict__ alpha,
    const __half *__restrict__ resid, __half *__restrict__ out,
    const __half *__restrict__ pad_src, __half *__restrict__ ns,
    int Cin, int Cout, int CoutP, int T, int L, int SCin, int SCout, int N,
    int do_snake, int pad_off, int pad_len) {
    if constexpr (PDL && !LATE_PDL) PDL_ENTER();
    constexpr int NT = 128 * MSPLIT * NSPLIT;
    constexpr int BM = 64 * MSPLIT;
    constexpr int BNT = WG_BN * NSPLIT;
    constexpr int NCRAW = BNT / 4;
    constexpr int EPN = (NT % NCRAW == 0) ? BNT : (((BNT + 127) / 128) * 128);
    constexpr int WG_CLD = EPN + 8;
    constexpr int CG = CIB / 8;
    constexpr int NBR = BNT + KT - 1;
    constexpr int NBRP = BC8 ? ((NBR + 7) & ~7) : ((NBR + 31) & ~31);
    constexpr int UCOL = NBRP / 32;
    constexpr int NUNIT = CG * UCOL;
    constexpr int NWARP = NT / 32;
    constexpr int BITER = (NUNIT + NWARP - 1) / NWARP;
    constexpr int B_E = CIB * NBRP;
    constexpr int ST_E = B_E;
    constexpr int LBO_B = NBRP * 8;
    constexpr int ABUF = ABUF_;
    static_assert(!OC8 || PREACT,
                  "K=7 C8 output is only used for residual-free pair heads");
    char *arena=reinterpret_cast<char*>(native_arena());
    char *colok=reinterpret_cast<char*>(native_aux());
    float *row_alpha=reinterpret_cast<float*>(native_aux()+1024);
    float *row_ri=reinterpret_cast<float*>(native_aux()+2048);
    __half *smem = reinterpret_cast<__half *>(arena);
    __half *Hs = smem;
    float *Cs = reinterpret_cast<float *>(arena);
    const int tid = threadIdx.x;
    const int wgid = tid >> 7;
    const int mg = wgid % MSPLIT, ng = wgid / MSPLIT;
    const int warp = (tid & 127) >> 5, lane = tid & 31;
    const int m0 = blockIdx.x * BM;
    const int nb = blockIdx.y * BNT;
    const int H = KT - 1;
    bool active_tile = false;
    for (int row = nb / L; row <= min(N - 1, nb + BNT - 1) / L; ++row) {
        if (native_meta()->lengths[row] > 0) {
            active_tile = true;
            break;
        }
    }
    if (active_tile) {
    const int half = Cout >> 1;
    const int NCH = (Cin + CIB - 1) / CIB;
    for (int i = tid; i < BM; i += NT) {
        const int co = m0 + i;
        if (co < Cout && co < half) {
            const float a = alpha[co];
            row_alpha[i] = a;
            row_ri[i] = __frcp_rn(a + 1e-9f);
        }
    }
    for (int i = tid; i < EPN; i += NT) {
        const int n = nb + i;
        colok[i] = (i < BNT && n < N && (n % L) < T) ? 1 : 0;
    }
    const int er0 = mg * 64 + warp * 16 + (lane >> 2);
    const int eco0 = m0 + er0, eco1 = m0 + er0 + 8;
    const float bias0 = (eco0 < Cout) ? bias[eco0] : 0.f;
    const float bias1 = (eco1 < Cout) ? bias[eco1] : 0.f;
    float acc[WG_BN / 2];
#pragma unroll
    for (int i = 0; i < WG_BN / 2; ++i) acc[i] = ((i & 2) == 0) ? bias0 : bias1;
    const int mtile = blockIdx.x * MSPLIT + mg;
    const __half *const apt = w + ((size_t)mtile * NCH * KT) * 1024 +
                              (size_t)(tid & 127) * 8;
    uint32_t areg[ABUF][KT][4];
    const int wi = tid >> 5;
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
    constexpr int BCREG = (BQ + NT - 1) / NT;
    int cdst[BCREG];
    const __half *csrc[BCREG];
    bool clive[BCREG];
#pragma unroll
    for (int r = 0; r < BCREG; ++r) {
        const int q = tid + r * NT;
        const int gc = q / NBRP, rem = q - gc * NBRP;
        clive[r] = ((BQ % NT == 0) || q < BQ) && gc < CG;
        cdst[r] = clive[r] ? gc * LBO_B + rem * 8 : 0;
        csrc[r] = in + (clive[r] ? (size_t)gc * SC8i +
                                  (size_t)nb * 8 + rem * 8 : 0);
    }
    uint32_t br[BITER][4];
    constexpr int CPA_STAGES = 4;
    constexpr int BSTAGES = BC8 ? CPA_STAGES : STAGES;
    static_assert(STAGES >= CPA_STAGES, "C8 B-panel pipeline allocation");
    static_assert(ABUF == 2 && BSTAGES == 4, "wg4 unrolled pipeline shape");
    uint64_t dslot[BSTAGES];
#pragma unroll
    for (int s = 0; s < BSTAGES; ++s)
        dslot[s] = wg_desc(smem + s * ST_E + ng * WG_BN * 8, LBO_B * 2u, 128u);
    uint32_t csm0[BCREG];
#pragma unroll
    for (int r = 0; r < BCREG; ++r)
        csm0[r] = static_cast<uint32_t>(__cvta_generic_to_shared(smem + cdst[r]));
    uint32_t bsm0[BITER];
#pragma unroll
    for (int t = 0; t < BITER; ++t)
        bsm0[t] = static_cast<uint32_t>(__cvta_generic_to_shared(smem + bdst[t]));
    const size_t CSTEP = (size_t)CG * SC8i;
    const __half *crol[BCREG];
#pragma unroll
    for (int r = 0; r < BCREG; ++r) crol[r] = csrc[r] + CPA_STAGES * CSTEP;
    const __half *arol = apt + (size_t)ABUF * (KT * 1024);
#define WG4_CPB(c,BD) { const size_t _o = (size_t)(c) * CG * SC8i; _Pragma("unroll") for (int r = 0; r < BCREG; ++r) if (clive[r]) cpa16((BD) + cdst[r], csrc[r] + _o, true); if ((c) + 1 == CPA_STAGES) { _Pragma("unroll") for (int r = 0; r < BCREG; ++r) csrc[r] += (size_t)CPA_STAGES * CSTEP; } }
#define WG4_CPB_R(SLOT) { _Pragma("unroll") for (int r = 0; r < BCREG; ++r) if (clive[r]) cpa16o<(SLOT) * ST_E * 2>(csm0[r], csrc[r], true); _Pragma("unroll") for (int r = 0; r < BCREG; ++r) csrc[r] += CSTEP; }
#define WG4_LOADB(c) { const size_t _o = (size_t)(c) * CIB * SCin; _Pragma("unroll") for (int t = 0; t < BITER; ++t) { if (blive[t]) { const int4 _v = *reinterpret_cast<const int4 *>( bbase + _o + boff[t]); br[t][0] = _v.x; br[t][1] = _v.y; br[t][2] = _v.z; br[t][3] = _v.w; } } }
#define WG4_XPOSE(u) { const int _q = lane & 3; uint32_t _x, _y; _x = (_q & 1) ? u[0] : u[1]; _y = (_q & 1) ? u[2] : u[3]; _x = __shfl_xor_sync(0xffffffffu, _x, 1); _y = __shfl_xor_sync(0xffffffffu, _y, 1); if (_q & 1) { u[0] = _x; u[2] = _y; } else { u[1] = _x; u[3] = _y; } _x = (_q & 2) ? u[0] : u[2]; _y = (_q & 2) ? u[1] : u[3]; _x = __shfl_xor_sync(0xffffffffu, _x, 2); _y = __shfl_xor_sync(0xffffffffu, _y, 2); if (_q & 2) { u[0] = _x; u[1] = _y; } else { u[2] = _x; u[3] = _y; } }
#define WG4_STOREB(BD) { _Pragma("unroll") for (int t = 0; t < BITER; ++t) { if (blive[t]) { WG4_XPOSE(br[t]) const uint32_t _a = static_cast<uint32_t>( __cvta_generic_to_shared((BD) + bdst[t])); asm volatile( "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 " "[%0], {%1,%2,%3,%4};\n" ::"r"(_a), "r"(br[t][0]), "r"(br[t][1]), "r"(br[t][2]), "r"(br[t][3])); } } }
#define WG4_STOREB_R(SLOT) { _Pragma("unroll") for (int t = 0; t < BITER; ++t) { if (blive[t]) { WG4_XPOSE(br[t]) asm volatile( "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 " "[%0+%5], {%1,%2,%3,%4};\n" ::"r"(bsm0[t]), "r"(br[t][0]), "r"(br[t][1]), "r"(br[t][2]), "r"(br[t][3]), "n"((SLOT) * ST_E * 2)); } } }
#define WG4_LOADA_R(BUF) { _Pragma("unroll") for (int kk = 0; kk < 4; ++kk) { const int4 _v = *reinterpret_cast<const int4 *>(arol + kk * 1024); areg[0][kk][0] = _v.x; areg[0][kk][1] = _v.y; areg[0][kk][2] = _v.z; areg[0][kk][3] = _v.w; } }
#define WG4_MMA_R(BUF,SLOT) { const uint64_t db0 = dslot[0] + (uint64_t)(SLOT) * (ST_E / 8); asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); _Pragma("unroll") for (int kk = 0; kk < 4; ++kk) { WGMMA_RS_ANY_T(WG_BN, acc, areg[0][kk], db0 + kk, 0); } asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); _Pragma("unroll") for (int kk = 0; kk < 4; ++kk) asm volatile("" : "+r"(areg[0][kk][0]), "+r"(areg[0][kk][1]), "+r"(areg[0][kk][2]), "+r"(areg[0][kk][3])); _Pragma("unroll") for (int kk = 4; kk < KT; ++kk) { const int4 _v = *reinterpret_cast<const int4 *>(arol + kk * 1024); areg[0][kk - 4][0] = _v.x; areg[0][kk - 4][1] = _v.y; areg[0][kk - 4][2] = _v.z; areg[0][kk - 4][3] = _v.w; } asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); _Pragma("unroll") for (int kk = 4; kk < KT; ++kk) { WGMMA_RS_ANY_T(WG_BN, acc, areg[0][kk - 4], db0 + kk, 0); } asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); arol += KT * 1024; }
#define WG4_LOADA(BUF,c) { if constexpr ((BUF) == 0) { const __half *_p = apt + (size_t)(c) * (KT * 1024); _Pragma("unroll") for (int kk = 0; kk < 4; ++kk) { const int4 _v = *reinterpret_cast<const int4 *>(_p + kk * 1024); areg[0][kk][0] = _v.x; areg[0][kk][1] = _v.y; areg[0][kk][2] = _v.z; areg[0][kk][3] = _v.w; } arol = _p; } }
#define WG4_KEEPA(BUF) { _Pragma("unroll") for (int kk = 0; kk < KT - 4; ++kk) asm volatile("" : "+r"(areg[0][kk][0]), "+r"(areg[0][kk][1]), "+r"(areg[0][kk][2]), "+r"(areg[0][kk][3])); }
#define WG4_MMA(BUF,BD) { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); const uint64_t db0 = wg_desc((BD) + ng * WG_BN * 8, LBO_B * 2u, 128u); _Pragma("unroll") for (int kk = 0; kk < KT; ++kk) { WGMMA_RS_ANY_T(WG_BN, acc, areg[BUF][kk], db0 + kk, 0); } asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
    if constexpr (PDL && LATE_PDL) PDL_ENTER();
    constexpr int WGD = 1;
    if constexpr (BC8) {
#pragma unroll 1
        for (int st = 0; st < CPA_STAGES; ++st) {
            if (st < NCH) WG4_CPB(st, smem + st * ST_E)
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }
        WG4_LOADA(0, 0)
        if (ABUF > 1 && 1 < NCH) WG4_LOADA(1, 1)
        if (ABUF > 2 && 2 < NCH) WG4_LOADA(2, 2)
    } else {
        WG4_LOADB(0)
        WG4_LOADA(0, 0)
        if (ABUF > 1 && 1 < NCH) WG4_LOADA(1, 1)
        if (ABUF > 2 && 2 < NCH) WG4_LOADA(2, 2)
        WG4_STOREB(smem)
        if (1 < NCH) WG4_LOADB(1)
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
    }
#define WG4_BODY(BUF,i,SLOT) { if constexpr (BC8) { asm volatile("cp.async.wait_group %0;\n" : : "n"(CPA_STAGES - 3) : "memory"); __syncthreads(); if ((i) >= 2) { if ((i) + 2 < NCH) { WG4_CPB_R(((SLOT) + 2) % CPA_STAGES) } asm volatile("cp.async.commit_group;\n" ::: "memory"); } } else { if (i) __syncthreads(); } WG4_MMA_R(BUF, SLOT) if constexpr (NSPLIT > 1 || BC8) { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); WG4_KEEPA(0) if constexpr (NSPLIT > 1) __syncthreads(); } if constexpr (!BC8) { const int cb = (i) + 1; if (cb < NCH) { WG4_STOREB_R(((SLOT) + 1) % BSTAGES) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); if (cb + 1 < NCH) WG4_LOADB(cb + 1) } } if ((i) + 1 < NCH) { if constexpr (NSPLIT == 1 && !BC8) { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); WG4_KEEPA(0) } WG4_LOADA_R(0) } }
    for (int i = 0; i < NCH; i += 4) {
        WG4_BODY(0, i, 0)
        if (i + 1 < NCH) WG4_BODY(1, i + 1, 1)
        if (i + 2 < NCH) WG4_BODY(0, i + 2, 2)
        if (i + 3 < NCH) WG4_BODY(1, i + 3, 3)
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    if constexpr (BC8)
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#undef WG4_CPB
#undef WG4_CPB_R
#undef WG4_STOREB_R
#undef WG4_LOADA_R
#undef WG4_MMA_R
#undef WG4_LOADB
#undef WG4_XPOSE
#undef WG4_STOREB
#undef WG4_LOADA
#undef WG4_KEEPA
#undef WG4_MMA
#undef WG4_BODY
#define TL4_SW(c,r) ((c) * BM + (((r) & ~7) ^ (((c) & 7) << 3)) + ((r) & 7))
    __syncthreads();
    const int cb2 = (lane & 3) * 2;
    const bool snake = PREACT && (m0 + er0) < half;
    const float sn_a0 = snake ? row_alpha[er0] : 0.f;
    const float sn_r0 = snake ? row_ri[er0] : 0.f;
    const float sn_a1 = snake ? row_alpha[er0 + 8] : 0.f;
    const float sn_r1 = snake ? row_ri[er0 + 8] : 0.f;
#pragma unroll
    for (int j = 0; j < WG_BN / 8; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float2 v = make_float2(acc[j * 4 + h * 2], acc[j * 4 + h * 2 + 1]);
            const int row = er0 + h * 8;
            if constexpr (PREACT) {
                if (snake) {
                    const float a = h ? sn_a1 : sn_a0;
                    const float ri = h ? sn_r1 : sn_r0;
                    const float s0 = __sinf(a * v.x);
                    const float s1 = __sinf(a * v.y);
                    v.x = __fmaf_rn(s0 * s0, ri, v.x);
                    v.y = __fmaf_rn(s1 * s1, ri, v.y);
                } else {
                    v.x = v.x > 0.f ? v.x : 0.01f * v.x;
                    v.y = v.y > 0.f ? v.y : 0.01f * v.y;
                }
                if constexpr (OC8) {
                    const int c0 = ng * WG_BN + j * 8 + cb2;
                    Hs[TL4_SW(c0, row)] = __float2half(v.x);
                    Hs[TL4_SW(c0 + 1, row)] = __float2half(v.y);
                } else {
                    *reinterpret_cast<__half2 *>(
                        Hs + row * WG_CLD + ng * WG_BN + j * 8 + cb2) =
                        __floats2half2_rn(v.x, v.y);
                }
            } else {
                *reinterpret_cast<float2 *>(
                    Cs + row * WG_CLD + ng * WG_BN + j * 8 + cb2) = v;
            }
        }
    }
    __syncthreads();
    if constexpr (PREACT) {
        if constexpr (OC8) {
            constexpr int GRP = BM / 8;
            constexpr int TOTG = GRP * BNT;
            constexpr int GITER = (TOTG + NT - 1) / NT;
            const size_t SC8o = (size_t)SCout * 8;
#pragma unroll
            for (int t = 0; t < GITER; ++t) {
                const int idx = tid + t * NT;
                if (TOTG % NT != 0 && idx >= TOTG) continue;
                const int g = idx / BNT;
                const int j = idx - g * BNT;
                if (!colok[j]) continue;
                const int m0g = g * 8;
                if (m0 + m0g >= Cout) continue;
                *reinterpret_cast<int4 *>(
                    out + (size_t)((m0 + m0g) >> 3) * SC8o +
                    (size_t)(nb + j + H) * 8) =
                    *reinterpret_cast<const int4 *>(Hs + TL4_SW(j, m0g));
            }
        } else {
        static_assert((BNT & 3) == 0, "four-column preactivated epilogue");
        constexpr int NC = BNT / 4;
        constexpr int TOT = BM * NC;
        for (int q = tid; q < TOT; q += NT) {
            const int m = q / NC;
            const int lc = (q - m * NC) * 4;
            const int co = m0 + m;
            const bool any4 = (colok[lc] | colok[lc + 1] |
                               colok[lc + 2] | colok[lc + 3]) != 0;
            if (co >= Cout || !any4) continue;
            const __half2 *sp = reinterpret_cast<const __half2 *>(
                Hs + m * WG_CLD + lc);
            const __half2 v0 = sp[0], v1 = sp[1];
            __half *op = out + (size_t)co * SCout + nb + lc + H;
            const bool ok4 = (colok[lc] & colok[lc + 1] &
                              colok[lc + 2] & colok[lc + 3]) != 0;
            if (ok4) {
                reinterpret_cast<__half2 *>(op)[0] = v0;
                reinterpret_cast<__half2 *>(op)[1] = v1;
            } else {
                if (colok[lc + 0]) op[0] = v0.x;
                if (colok[lc + 1]) op[1] = v0.y;
                if (colok[lc + 2]) op[2] = v1.x;
                if (colok[lc + 3]) op[3] = v1.y;
            }
        }
        }
    } else {
    constexpr bool COMPACT_EPI =
        (WG_BN == 136 || WG_BN == 160 || WG_BN == 208);
    if constexpr (COMPACT_EPI) {
    static_assert((BNT & 3) == 0, "4-column epilogue groups");
    constexpr int NC = BNT / 4;
    constexpr int TOT = BM * NC;
    constexpr int MITER = (TOT + NT - 1) / NT;
    constexpr int UNR = (MITER >= 8) ? 8 : ((MITER >= 4) ? 4 :
                        ((MITER >= 2) ? 2 : 1));
#pragma unroll 1
    for (int t0 = 0; t0 < MITER; t0 += UNR) {
        float4 v[UNR];
        __half2 r0[UNR], r1[UNR];
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            const int q = tid + (t0 + u) * NT;
            if (q >= TOT) continue;
            const int m = q / NC;
            const int lc = (q - m * NC) * 4;
            const int co = m0 + m;
            const bool any4 = (colok[lc] | colok[lc + 1] |
                               colok[lc + 2] | colok[lc + 3]) != 0;
            if (co >= Cout || !any4) continue;
            v[u] = *reinterpret_cast<const float4 *>(Cs + m * WG_CLD + lc);
            if constexpr (!PREACT) {
                const int n = nb + lc;
                const __half2 *rp = reinterpret_cast<const __half2 *>(
                    resid + (size_t)co * SCout + n + H);
                r0[u] = rp[0];
                r1[u] = rp[1];
            }
        }
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            const int q = tid + (t0 + u) * NT;
            if (q >= TOT) continue;
            const int m = q / NC;
            const int lc = (q - m * NC) * 4;
            const int co = m0 + m;
            const bool any4 = (colok[lc] | colok[lc + 1] |
                               colok[lc + 2] | colok[lc + 3]) != 0;
            if (co >= Cout || !any4) continue;
            const int n = nb + lc;
            float4 vv = v[u];
            if constexpr (!PREACT) {
                vv.x += __half2float(r0[u].x); vv.y += __half2float(r0[u].y);
                vv.z += __half2float(r1[u].x); vv.w += __half2float(r1[u].y);
            }
            float *vp = reinterpret_cast<float *>(&vv);
            if (co < half) {
                const float a = row_alpha[m], ri = row_ri[m];
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
            const bool ok4 = (colok[lc] & colok[lc + 1] &
                              colok[lc + 2] & colok[lc + 3]) != 0;
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
    } else {
        constexpr int NC = EPN / 4;
        constexpr int MSTEP = NT / NC;
        constexpr int MITER = BM / MSTEP;
        constexpr int UNR = (MITER % 8 == 0) ? 8
                          : ((MITER % 4 == 0) ? 4
                                             : ((MITER % 2 == 0) ? 2 : 1));
        static_assert(MITER % UNR == 0, "row batch must divide row count");
        const int lc = (tid % NC) * 4;
        const int lr = tid / NC;
        const bool ok4 = colok[lc] && colok[lc + 1] &&
                         colok[lc + 2] && colok[lc + 3];
        const bool live4 = colok[lc] || colok[lc + 1] ||
                           colok[lc + 2] || colok[lc + 3];
        const int n = nb + lc;
#pragma unroll 1
        for (int t0 = 0; live4 && t0 < MITER; t0 += UNR) {
            float4 v[UNR];
            __half2 r0[UNR], r1[UNR];
#pragma unroll
            for (int u = 0; u < UNR; ++u) {
                const int m = lr + (t0 + u) * MSTEP;
                v[u] = *reinterpret_cast<const float4 *>(
                    Cs + m * WG_CLD + lc);
                if constexpr (!PREACT) {
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
                if constexpr (!PREACT) {
                    vv.x += __half2float(r0[u].x);
                    vv.y += __half2float(r0[u].y);
                    vv.z += __half2float(r1[u].x);
                    vv.w += __half2float(r1[u].y);
                }
                float *vp = reinterpret_cast<float *>(&vv);
                if (co < half) {
                    const float a = row_alpha[m], ri = row_ri[m];
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
    }
    {
        const int nbx = gridDim.x * gridDim.y;
        const int bxy = blockIdx.x + blockIdx.y * gridDim.x;
        const int B = N / L;
        if (ns != nullptr) {
            const int total = Cin * B;
            for (int idx = bxy * NT + tid; idx < total; idx += nbx * NT) {
                const int b = idx / Cin, ci = idx - b * Cin;
                const int x = b * L + native_true_tokens(b,T);
                __half *const dst = ns + ((size_t)b * Cin + ci) * H;
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
            const int total = Cout * B;
            for (int idx = bxy * NT + tid; idx < total; idx += nbx * NT) {
                const int b = idx / Cout, co = idx - b * Cout;
                const int x = b * L + pad_off;
                const __half *const src =
                    pad_src + ((size_t)b * Cout + co) * pad_len;
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
