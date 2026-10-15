#define TL_NT 512
#define TL_MW 512
#define TL_MLD TL_MW
#define TL_VW 504
#define TL_UW 520
#define TL_BN 496
#define TL_VLD 520
#define TL_LBO_U (TL_UW * 8)
#define TL_LBO_M (TL_MLD * 8)
#define TL_LBO_V (TL_VLD * 8)
#define TL_SBO_A 128
#define TL_PU_E (4 * TL_UW * 8)
#define TL_PM_E (8 * TL_MLD * 8)
#define TL_PV_E (32 * TL_VLD)
#define TL_AI_E (14 * 1024)
#define TL_AS_E (32 * 512 + 512)
#define TL_SHARED ((2 * TL_PU_E + TL_PM_E + TL_PV_E + TL_AI_E + TL_AS_E) * 2)
__device__ __forceinline__ void tl_stmatrix4(void *dst, const uint32_t r[4],
                                             bool trans) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    if (trans)
        asm volatile(
            "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(
                a),
            "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]));
    else
        asm volatile(
            "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(
                a),
            "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]));
}
__device__ __forceinline__ void tl_ldmatrix4t(uint32_t r[4], const void *src) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}
__device__ __forceinline__ void tl_stmatrix1t(void *dst, uint32_t r) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};\n" ::
            "r"(a), "r"(r));
}
template <int BW>
__device__ __forceinline__ void tl_skip_band(
    const __half *PU, __half *PM, const __half *AS,
    __half *PV, int c0, int warp, int lane, int er0, const float *sh_ask,
    const float *sh_ari_v, float b0v, float b1v) {
    constexpr int NB8 = BW / 8;
    float acc[BW / 2];
#pragma unroll
    for (int i = 0; i < BW / 2; ++i) acc[i] = ((i & 2) == 0) ? b0v : b1v;
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const uint64_t da0 =
            wg_desc(AS + c * 8 * 512, 128u, TL_SBO_A * 2u);
        const uint64_t db0 =
            wg_desc(PM + (2 * c) * TL_LBO_M + (c0 + 2) * 8,
                    TL_LBO_M * 2u, 128u);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint64_t da = da0 + 64 * (2 * p);
            const uint64_t db = db0 + 2 * p;
            if constexpr (BW == 120) {
                WGMMA_M64N120K16_T(acc, da, db, 0, 0);
            } else {
                WGMMA_M64N128K16_T(acc, da, db, 0, 0);
            }
        }
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    const int m_addr = lane >> 3, row = lane & 7;
    const int cg = (warp & 1) * 2;
    if (er0 >= 32) {
#pragma unroll
        for (int j = 0; j < NB8; j += 2) {
            uint32_t r[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int jj = j + (m >> 1), h = m & 1;
                if (jj < NB8) {
                    const __half2 hv = __floats2half2_rn(
                        acc[jj * 4 + h * 2 + 0], acc[jj * 4 + h * 2 + 1]);
                    r[m] = *reinterpret_cast<const uint32_t *>(&hv);
                } else {
                    r[m] = 0;
                }
            }
            const int jj = j + (m_addr >> 1), h = m_addr & 1;
            tl_stmatrix4(PV + (cg + h) * TL_LBO_V + (c0 + jj * 8 + row) * 8,
                         r, true);
        }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    if (er0 < 32) {
#pragma unroll
        for (int j = 0; j < NB8; j += 2) {
            uint32_t r[4];
            const int jja = j + (m_addr >> 1), ha = m_addr & 1;
            tl_ldmatrix4t(r, PV + (cg + ha) * TL_LBO_V +
                                 (c0 + jja * 8 + row + 1) * 8);
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int jj = j + (m >> 1), h = m & 1;
                if (jj < NB8) {
                    const __half2 hv = *reinterpret_cast<__half2 *>(&r[m]);
                    acc[jj * 4 + h * 2 + 0] += __half2float(hv.x);
                    acc[jj * 4 + h * 2 + 1] += __half2float(hv.y);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NB8; j += 2) {
            uint32_t r[4];
            const int jja = j + (m_addr >> 1), ha = m_addr & 1;
            tl_ldmatrix4t(r, PU + (cg + ha) * TL_LBO_U +
                                 (c0 + jja * 8 + row + 16) * 8);
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int jj = j + (m >> 1), h = m & 1;
                if (jj < NB8) {
                    const __half2 hv = *reinterpret_cast<__half2 *>(&r[m]);
                    acc[jj * 4 + h * 2 + 0] += __half2float(hv.x);
                    acc[jj * 4 + h * 2 + 1] += __half2float(hv.y);
                }
            }
        }
    }
    if (er0 >= 32) return;
    if (er0 < 16) {
        const float a0 = sh_ask[er0], r0 = sh_ari_v[er0];
        const float a1 = sh_ask[er0 + 8], r1 = sh_ari_v[er0 + 8];
#pragma unroll
        for (int i = 0; i < BW / 2; ++i) {
            const int h = (i & 2) >> 1;
            const float v = acc[i];
            const float sn = __sinf((h ? a1 : a0) * v);
            acc[i] = __fmaf_rn(sn * sn, h ? r1 : r0, v);
        }
    } else {
#pragma unroll
        for (int i = 0; i < BW / 2; ++i) {
            const float v = acc[i];
            acc[i] = v > 0.f ? v : 0.01f * v;
        }
    }
#pragma unroll
    for (int j = 0; j < NB8; j += 2) {
        uint32_t r[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int jj = j + (m >> 1), h = m & 1;
            if (jj < NB8) {
                const __half2 hv = __floats2half2_rn(
                    acc[jj * 4 + h * 2 + 0], acc[jj * 4 + h * 2 + 1]);
                r[m] = *reinterpret_cast<const uint32_t *>(&hv);
            } else {
                r[m] = 0;
            }
        }
        const int jj = j + (m_addr >> 1), h = m_addr & 1;
        tl_stmatrix4(PM + (cg + h) * TL_LBO_V +
                              (c0 + jj * 8 + row) * 8,
                     r, true);
    }
}
__device__ __noinline__ void tail_fused_kernel(
    const __half *__restrict__ U, const __half *__restrict__ win,
    const __half *__restrict__ wsk, const __half *__restrict__ wpo,
    const float *__restrict__ bin, const float *__restrict__ ain,
    const float *__restrict__ bsk, const float *__restrict__ ask,
    const float *__restrict__ bpo, const __half *__restrict__ st_m4,
    const __half *__restrict__ st_v, __half *__restrict__ wave,
    __half *__restrict__ ns_u, __half *__restrict__ ns_m4,
    __half *__restrict__ ns_v, int B, int T4, int L4, int SC4, int N4,
    int NTILE) {
    char *tl_arena=reinterpret_cast<char*>(native_arena());
    __half *const PU = reinterpret_cast<__half *>(tl_arena);
    __half *const PM = PU + 2 * TL_PU_E;
    __half *const PV = PM + TL_PM_E;
    __half *const AI = PV + TL_PV_E;
    __half *const AS = AI + TL_AI_E;
    __half *const PB = AS + 32 * 512;
    float *sh_ain=(float*)native_aux(), *sh_ari=sh_ain+32, *sh_ask=sh_ari+32, *sh_ari_v=sh_ask+16;
    const int tid = threadIdx.x;
    const int wg = tid >> 7;
    const int warp = (tid & 127) >> 5;
    const int lane = tid & 31;
    const int er0 = warp * 16 + (lane >> 2);
    const int cb2 = (lane & 3) * 2;
    for (int i = tid; i < 32; i += TL_NT) {
        const float a = ain[i];
        sh_ain[i] = a;
        sh_ari[i] = __frcp_rn(a + 1e-9f);
    }
    for (int i = tid; i < 16; i += TL_NT) {
        const float a = ask[i];
        sh_ask[i] = a;
        sh_ari_v[i] = __frcp_rn(a + 1e-9f);
    }
    for (int i = tid; i < TL_AI_E / 8; i += TL_NT) {
        const int blk = i / 128, off = (i - blk * 128) * 8;
        const int c = blk / 7, k = blk - c * 7;
        *reinterpret_cast<int4 *>(AI + blk * 1024 + off) =
            *reinterpret_cast<const int4 *>(win + (size_t)c * 14336 +
                                            (size_t)k * 2048 + off);
    }
    for (int i = tid; i < 32 * 64; i += TL_NT) {
        const int blk = i / 64, off = (i - blk * 64) * 8;
        const int c = blk >> 3, k = blk & 7;
        *reinterpret_cast<int4 *>(AS + blk * 512 + off) =
            (k < 7) ? *reinterpret_cast<const int4 *>(
                          wsk + (size_t)c * 14336 + (size_t)k * 2048 + off)
                    : make_int4(0, 0, 0, 0);
    }
    for (int i = tid; i < 4 * (TL_VLD - TL_VW) * 8; i += TL_NT) {
        const int g = i / ((TL_VLD - TL_VW) * 8);
        PV[g * TL_LBO_V + TL_VW * 8 + (i - g * ((TL_VLD - TL_VW) * 8))] =
            __float2half(0.f);
    }
    for (int i = tid; i < 512; i += TL_NT) PB[i] = __float2half(0.f);
    for (int i = tid; i < 2 * 16 * 8; i += TL_NT) {
        const int c = i / (16 * 8);
        const int r = i - c * 16 * 8;
        const int ki = r / 8, k = r & 7;
        PB[c * 128 + (ki >> 3) * 64 + (ki & 7) * 8 + k] =
            (k < 7) ? wpo[(c * 16 + ki) * 7 + k] : __float2half(0.f);
    }
    const float bias_i0 = bin[er0], bias_i1 = bin[er0 + 8];
    const float bias_s0 = (er0 < 32) ? bsk[er0] : 0.f;
    const float bias_s1 = (er0 + 8 < 32) ? bsk[er0 + 8] : 0.f;
    const float bpo0 = bpo[0];
    constexpr int UCOL = TL_UW / 32;
    constexpr int UUNIT = 4 * UCOL;
    constexpr int UWARP = TL_NT / 32;
    constexpr int UITER = (UUNIT + UWARP - 1) / UWARP;
    const int uwarp = tid >> 5;
    const int uq = lane & 3;
    int ugc[UITER], uoc[UITER];
#pragma unroll
    for (int t = 0; t < UITER; ++t) {
        const int bi = uwarp + t * UWARP;
        const int gc = bi / UCOL;
        ugc[t] = gc;
        uoc[t] = bi - gc * UCOL;
    }
    const __half *const urow =
        U + (size_t)(lane >> 2) * SC4 + 8 * uq;
    __syncthreads();
    PDL_ENTER();
#define TL_STAGE_U(PUD,UB) { _Pragma("unroll") for (int t2 = 0; t2 < UITER; t2 += 2) { uint32_t u[2][4]; _Pragma("unroll") for (int d = 0; d < 2; ++d) { const int t = t2 + d; const int n = (UB) + uoc[t] * 32; const int gn = n + 8 * uq; const bool ok = gn >= 0 && gn < N4; int4 v = make_int4(0, 0, 0, 0); if (ok) v = *reinterpret_cast<const int4 *>( urow + (size_t)ugc[t] * (8 * (size_t)SC4) + n); u[d][0] = v.x; u[d][1] = v.y; u[d][2] = v.z; u[d][3] = v.w; } _Pragma("unroll") for (int d = 0; d < 2; ++d) { const int t = t2 + d; uint32_t *q = u[d]; uint32_t x, y; x = (uq & 1) ? q[0] : q[1]; y = (uq & 1) ? q[2] : q[3]; x = __shfl_xor_sync(0xffffffffu, x, 1); y = __shfl_xor_sync(0xffffffffu, y, 1); if (uq & 1) { q[0] = x; q[2] = y; } else { q[1] = x; q[3] = y; } x = (uq & 2) ? q[0] : q[2]; y = (uq & 2) ? q[1] : q[3]; x = __shfl_xor_sync(0xffffffffu, x, 2); y = __shfl_xor_sync(0xffffffffu, y, 2); if (uq & 2) { q[0] = x; q[1] = y; } else { q[2] = x; q[3] = y; } tl_stmatrix4((PUD) + ugc[t] * TL_LBO_U + (uoc[t] * 32 + lane) * 8, q, true); } } if (uwarp < 4) { const int n = (UB) + 512; const int gn = n + 2 * (lane & 3); uint32_t q = 0; if (gn >= 0 && gn + 1 < N4) q = *reinterpret_cast<const uint32_t *>( U + (size_t)(uwarp * 8 + (lane >> 2)) * SC4 + gn); tl_stmatrix1t((PUD) + uwarp * TL_LBO_U + (512 + (lane & 7)) * 8, q); } }
    TL_STAGE_U(PU, (int)blockIdx.x * TL_BN - 16)
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    int ubuf = 0;
    for (int tile = blockIdx.x; tile < NTILE; tile += gridDim.x, ubuf ^= 1) {
        const int g0 = tile * TL_BN;
        bool active_tile = false;
        for (int row = max(0, g0 - 16) / L4;
             row <= min(N4 - 1, g0 - 16 + TL_UW - 1) / L4; ++row) {
            if (native_meta()->lengths[row] > 0) {
                active_tile = true;
                break;
            }
        }
        __half *const PUc = PU + ubuf * TL_PU_E;
        __half *const PUn = PU + (ubuf ^ 1) * TL_PU_E;
        {
            const int b0 = g0 / L4;
            for (int s = 0; s < 2; ++s) {
                const int sb = b0 + s;
                if (sb >= B) break;
                const int cv = (int)((long long)sb * L4 + native_true_tokens(sb,T4) - g0);
                if (cv < 0 || cv >= TL_BN) continue;
                for (int i = tid; i < 32 * 6; i += TL_NT) {
                    const int co = i / 6, h = i - co * 6;
                    ns_u[((size_t)sb * 32 + co) * 6 + h] =
                        PUc[(co >> 3) * TL_LBO_U +
                            (cv + h + 16) * 8 + (co & 7)];
                }
            }
        }
        if (active_tile) {
            float acc[64];
#pragma unroll
            for (int i = 0; i < 64; ++i)
                acc[i] = ((i & 2) == 0) ? bias_i0 : bias_i1;
            const int c0 = wg * 128;
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const uint64_t da0 =
                    wg_desc(AI + c * 7 * 1024, 128u, TL_SBO_A * 2u);
                const uint64_t db0 =
                    wg_desc(PUc + (2 * c) * TL_LBO_U + (c0 + 2) * 8,
                            TL_LBO_U * 2u, 128u);
#pragma unroll
                for (int k = 0; k < 7; ++k) {
                    const uint64_t da = da0 + 128 * k;
                    const uint64_t db = db0 + k;
                    WGMMA_M64N128K16_T(acc, da, db, 0, 0);
                }
            }
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
            if (tile + (int)gridDim.x < NTILE)
                TL_STAGE_U(PUn, (tile + (int)gridDim.x) * TL_BN - 16)
            asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
            if (er0 < 32) {
                const float a0 = sh_ain[er0], r0 = sh_ari[er0];
                const float a1 = sh_ain[er0 + 8], r1 = sh_ari[er0 + 8];
#pragma unroll
                for (int i = 0; i < 64; ++i) {
                    const int h = (i & 2) >> 1;
                    const float v = acc[i];
                    const float sn = __sinf((h ? a1 : a0) * v);
                    acc[i] = __fmaf_rn(sn * sn, h ? r1 : r0, v);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 64; ++i) {
                    const float v = acc[i];
                    acc[i] = v > 0.f ? v : 0.01f * v;
                }
            }
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                uint32_t r[4];
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const int jj = j + (m >> 1), h = m & 1;
                    const __half2 hv = __floats2half2_rn(
                        acc[jj * 4 + h * 2 + 0], acc[jj * 4 + h * 2 + 1]);
                    r[m] = *reinterpret_cast<const uint32_t *>(&hv);
                }
                const int m_addr = lane >> 3, row = lane & 7;
                const int jj = j + (m_addr >> 1), h = m_addr & 1;
                tl_stmatrix4(PM + (warp * 2 + h) * TL_LBO_M +
                                 (c0 + jj * 8 + row) * 8,
                             r, true);
            }
        } else if (tile + (int)gridDim.x < NTILE) {
            TL_STAGE_U(PUn, (tile + (int)gridDim.x) * TL_BN - 16)
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        {
            const int base = g0 - 8;
            const int b0 = (base >= 0) ? base / L4 : 0;
            const int cm0 = (int)((long long)b0 * L4 - base);
            const int cm1 = cm0 + L4;
            const bool hit0 = (b0 < B) && (cm0 + 6 > 0) && (cm0 < TL_MW);
            const bool hit1 = (b0 + 1 < B) && (cm1 + 6 > 0) && (cm1 < TL_MW);
            if (hit0 || hit1) {
                for (int s = 0; s < 2; ++s) {
                    if (!(s ? hit1 : hit0)) continue;
                    const int sb = b0 + s, cm = s ? cm1 : cm0;
                    for (int i = tid; i < 64 * 6; i += TL_NT) {
                        const int co = i / 6, h = i - co * 6;
                        const int c = cm + h;
                        if (c >= 0 && c < TL_MW)
                            PM[(co >> 3) * TL_LBO_M + c * 8 + (co & 7)] =
                                st_m4[((size_t)sb * 64 + co) * 6 + h];
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                __syncthreads();
            }
        }
        {
            const int b0 = g0 / L4;
            for (int s = 0; s < 2; ++s) {
                const int sb = b0 + s;
                if (sb >= B) break;
                const int cv = (int)((long long)sb * L4 + native_true_tokens(sb,T4) - g0);
                if (cv < 0 || cv >= TL_BN) continue;
                for (int i = tid; i < 64 * 6; i += TL_NT) {
                    const int co = i / 6, h = i - co * 6;
                    ns_m4[((size_t)sb * 64 + co) * 6 + h] =
                        PM[(co >> 3) * TL_LBO_M +
                           (cv + h + 8) * 8 + (co & 7)];
                }
            }
        }
        if (active_tile) {
            const int c0 = wg * 128;
            if (wg == 3)
                tl_skip_band<120>(PUc, PM, AS, PV, c0, warp, lane, er0, sh_ask,
                                  sh_ari_v, bias_s0, bias_s1);
            else
                tl_skip_band<128>(PUc, PM, AS, PV, c0, warp, lane, er0, sh_ask,
                                  sh_ari_v, bias_s0, bias_s1);
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        {
            const int b0 = g0 / L4;
            const int cv0 = (int)((long long)b0 * L4 - g0), cv1 = cv0 + L4;
            const bool hv0 = (b0 < B) && (cv0 + 6 > 0) && (cv0 < TL_VW);
            const bool hv1 = (b0 + 1 < B) && (cv1 + 6 > 0) && (cv1 < TL_VW);
            if (hv0 || hv1) {
                for (int s = 0; s < 2; ++s) {
                    if (!(s ? hv1 : hv0)) continue;
                    const int sb = b0 + s, cv = s ? cv1 : cv0;
                    for (int i = tid; i < 32 * 6; i += TL_NT) {
                        const int co = i / 6, h = i - co * 6;
                        const int c = cv + h;
                        if (c >= 0 && c < TL_VW)
                            PM[(co >> 3) * TL_LBO_V + c * 8 + (co & 7)] =
                                st_v[((size_t)sb * 32 + co) * 6 + h];
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                __syncthreads();
            }
        }
        {
            const int b0 = g0 / L4;
            for (int s = 0; s < 2; ++s) {
                const int sb = b0 + s;
                if (sb >= B) break;
                const int cv = (int)((long long)sb * L4 + native_true_tokens(sb,T4) - g0);
                if (cv < 0 || cv >= TL_BN) continue;
                for (int i = tid; i < 32 * 6; i += TL_NT) {
                    const int co = i / 6, h = i - co * 6;
                    ns_v[((size_t)sb * 32 + co) * 6 + h] =
                        PM[(co >> 3) * TL_LBO_V + (cv + h) * 8 + (co & 7)];
                }
            }
        }
        __syncthreads();
        constexpr int PC_LD = 10;
        if (active_tile) {
            float *const PC = reinterpret_cast<float *>(PV);
#pragma unroll
            for (int round = 0; round < 2; ++round) {
                float acc[4];
                const int p0 = round * 256 + wg * 64;
                asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
                const uint64_t da0 =
                    wg_desc(PM + p0 * 8, TL_LBO_V * 2u, 128u);
                const uint64_t db0 = wg_desc(PB, 128u, 128u);
                const uint64_t da1 =
                    wg_desc(PM + 2 * TL_LBO_V + p0 * 8,
                            TL_LBO_V * 2u, 128u);
                const uint64_t db1 = wg_desc(PB + 128, 128u, 128u);
                WGMMA_M64N8K16_ZERO_T(acc, da0, db0, 0, 1);
                WGMMA_M64N8K16_T(acc, da1, db1, 0, 1);
                asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
                asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
                const int row0 = p0 + warp * 16 + (lane >> 2);
                const int col = (lane & 3) * 2;
                *reinterpret_cast<float2 *>(PC + row0 * PC_LD + col) =
                    make_float2(acc[0], acc[1]);
                *reinterpret_cast<float2 *>(PC + (row0 + 8) * PC_LD + col) =
                    make_float2(acc[2], acc[3]);
            }
        }
        __syncthreads();
        if (active_tile) {
            const float *const PC = reinterpret_cast<const float *>(PV);
            const int p0 = tid;
            const bool live = p0 < TL_BN;
            const unsigned live_mask = __ballot_sync(0xffffffffu, live);
            if (live) {
                const float s0 = __fadd_rn(PC[(p0 + 0) * PC_LD + 0],
                                           PC[(p0 + 1) * PC_LD + 1]);
                const float s1 = __fadd_rn(PC[(p0 + 2) * PC_LD + 2],
                                           PC[(p0 + 3) * PC_LD + 3]);
                const float s2 = __fadd_rn(PC[(p0 + 4) * PC_LD + 4],
                                           PC[(p0 + 5) * PC_LD + 5]);
                const float s3 = PC[(p0 + 6) * PC_LD + 6];
                float y0 = __fadd_rn(__fadd_rn(s0, s1),
                                     __fadd_rn(s2, s3));
                y0 = fminf(fmaxf(y0 + bpo0, -1.f), 1.f);
                const float y1 = __shfl_down_sync(live_mask, y0, 1);
                if ((p0 & 1) == 0) {
                    const int b0 = g0 / L4;
                    const int rr = g0 - b0 * L4;
                    int t0 = rr + p0, bo0 = b0;
                    int t1 = t0 + 1, bo1 = bo0;
                    if (t0 >= L4) { t0 -= L4; ++bo0; }
                    if (t1 >= L4) { t1 -= L4; ++bo1; }
                    if (bo0 == bo1 && bo0 < B && t1 < T4) {
                        *reinterpret_cast<__half2 *>(
                            wave + (size_t)bo0 * T4 + t0) =
                            __floats2half2_rn(y0, y1);
                    } else {
                        if (bo0 < B && t0 < T4)
                            wave[(size_t)bo0 * T4 + t0] = __float2half(y0);
                        if (bo1 < B && t1 < T4)
                            wave[(size_t)bo1 * T4 + t1] = __float2half(y1);
                    }
                }
            }
        }
    }
#undef TL_STAGE_U
}
