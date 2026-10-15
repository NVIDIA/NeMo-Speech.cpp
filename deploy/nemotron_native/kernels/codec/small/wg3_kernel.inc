template <int KT, int BK, int NWG, int NSG, int MINB, int WG_BN, int STAGES>
__device__ __noinline__ void wg3_conv_kernel(
    const __half *__restrict__ in, const __half *__restrict__ w,
    const float *__restrict__ bias, const float *__restrict__ alpha,
    const __half *__restrict__ resid, __half *__restrict__ out,
    const __half *__restrict__ pad_src, __half *__restrict__ ns,
    int Cin, int Cout, int T, int L, int SCin, int SCout, int N,
    int do_snake, int pad_off, int pad_len) {
    constexpr int WG_BNT = WG_BN * NSG;
    constexpr int WG_CLD = WG_BNT + 8;
    constexpr int WG_NT = 128 * NWG * NSG;
    constexpr int WG_BM = 64 * NWG;
    constexpr int KG = BK / 8;
    constexpr int A_E = (WG_BM / 8) * KG * WG_SLOT;
    constexpr int B_E = KG * (WG_BN / 8) * WG_SLOT;
    constexpr int SBO_AE = KG * WG_SLOT;
    constexpr int LBO_BE = (WG_BN / 8) * WG_SLOT;
    char *arena=reinterpret_cast<char*>(native_arena());
    char *colok=reinterpret_cast<char*>(native_aux());
    float *row_alpha=reinterpret_cast<float*>(native_aux()+1024);
    float *row_ri=reinterpret_cast<float*>(native_aux()+2048);
    constexpr int ST_E = A_E + NSG * B_E;
    __half *As0 = reinterpret_cast<__half *>(arena);
    __half *Bs0 = As0 + A_E;
    __half *As1 = (STAGES == 2) ? As0 + ST_E : As0;
    __half *Bs1 = (STAGES == 2) ? Bs0 + ST_E : Bs0;
    float *Cs = reinterpret_cast<float *>(arena);
    const int tid = threadIdx.x;
    const int wgid = tid >> 7;
    const int mgid = wgid % NWG, ngid = wgid / NWG;
    const int warp = (tid & 127) >> 5, lane = tid & 31;
    const int m0 = blockIdx.x * WG_BM;
    constexpr int WGROWS = 8;
    const int nb = blockIdx.y * WG_BNT;
    const int n0 = nb + ngid * WG_BN;
    const int H = KT - 1;
    const int Kg = Cin * KT;
    const int half = Cout >> 1;
    if (tid < WG_BM) {
        const int co = m0 + tid;
        if (do_snake && co < Cout && co < half) {
            const float a = alpha[co];
            row_alpha[tid] = a;
            row_ri[tid] = __frcp_rn(a + 1e-9f);
        }
    }
    if (tid < WG_BNT) {
        const int n = nb + tid;
        colok[tid] = (n < N && (n % L) < T) ? 1 : 0;
    }
    const int er0 = mgid * 64 + warp * 16 + (lane >> 2);
    const int eco0 = m0 + er0, eco1 = m0 + er0 + 8;
    const float bias0 = (eco0 < Cout) ? bias[eco0] : 0.f;
    const float bias1 = (eco1 < Cout) ? bias[eco1] : 0.f;
    float acc[WG_BN / 2];
#pragma unroll
    for (int i = 0; i < WG_BN / 2; ++i) acc[i] = ((i & 2) == 0) ? bias0 : bias1;
    constexpr int ATOT = WG_BM * KG;
    constexpr int AREG = (ATOT + WG_NT - 1) / WG_NT;
    constexpr int CIB = BK / KT;
    constexpr int BTOT = NSG * CIB * (WG_BN / 8);
    constexpr int BREG = (BTOT + WG_NT - 1) / WG_NT;
    int4 ar[AREG];
    int4 br0[BREG], br1[BREG];
    uint32_t edge[BREG];
#define WG_LOADG(kg0) { const int _ci0 = (kg0) / KT; _Pragma("unroll") for (int r = 0; r < AREG; ++r) { const int idx = tid + r * WG_NT; ar[r] = make_int4(0, 0, 0, 0); if (ATOT % WG_NT == 0 || idx < ATOT) { const int m = idx / KG, v = idx - (idx / KG) * KG; const int co = m0 + m, kg = (kg0) + v * 8; if (co < Cout && kg < Kg) ar[r] = *reinterpret_cast<const int4 *>( w + (size_t)co * Kg + kg); } } _Pragma("unroll") for (int r = 0; r < BREG; ++r) { const int idx = tid + r * WG_NT; br0[r] = make_int4(0, 0, 0, 0); if constexpr (KT == 3) edge[r] = 0; else br1[r] = make_int4(0, 0, 0, 0); if (BTOT % WG_NT == 0 || idx < BTOT) { const int _g = idx / (CIB * (WG_BN / 8)); const int _r = idx - _g * (CIB * (WG_BN / 8)); const int ci = _ci0 + _r / (WG_BN / 8); const int nc = _r % (WG_BN / 8); if (ci < Cin) { const int4 *src = reinterpret_cast<const int4 *>( in + (size_t)ci * SCin + nb + _g * WG_BN + nc * 8); br0[r] = src[0]; if constexpr (KT == 3) { if (nc == WG_BN / 8 - 1) edge[r] = src[1].x; } else { br1[r] = src[1]; } } } } }
    WG_LOADG(0)
#define WG_STS(AD,BD) { _Pragma("unroll") for (int r = 0; r < AREG; ++r) { const int idx = tid + r * WG_NT; if (ATOT % WG_NT == 0 || idx < ATOT) { const int m = idx / KG, v = idx - (idx / KG) * KG; *reinterpret_cast<int4 *>( (AD) + (m >> 3) * SBO_AE + v * WG_SLOT + (m & 7) * 8) = ar[r]; } } _Pragma("unroll") for (int r = 0; r < BREG; ++r) { const int idx = tid + r * WG_NT; if (BTOT % WG_NT == 0 || idx < BTOT) { const int _g = idx / (CIB * (WG_BN / 8)); const int _r = idx - _g * (CIB * (WG_BN / 8)); const int cil = _r / (WG_BN / 8), nc = _r % (WG_BN / 8); unsigned int u[8]; u[0] = br0[r].x; u[1] = br0[r].y; u[2] = br0[r].z; u[3] = br0[r].w; if constexpr (KT == 3) { u[4] = __shfl_down_sync(0xffffffffu, u[0], 1, WG_BN / 8); if (nc == WG_BN / 8 - 1) u[4] = edge[r]; u[5] = u[6] = u[7] = 0; } else { u[4] = br1[r].x; u[5] = br1[r].y; u[6] = br1[r].z; u[7] = br1[r].w; } _Pragma("unroll") for (int k = 0; k < KT; ++k) { const int kh = k >> 1; int4 o; if (k & 1) { o.x = __funnelshift_r(u[kh + 0], u[kh + 1], 16); o.y = __funnelshift_r(u[kh + 1], u[kh + 2], 16); o.z = __funnelshift_r(u[kh + 2], u[kh + 3], 16); o.w = __funnelshift_r(u[kh + 3], u[kh + 4], 16); } else { o.x = u[kh + 0]; o.y = u[kh + 1]; o.z = u[kh + 2]; o.w = u[kh + 3]; } const int row = cil * KT + k; *reinterpret_cast<int4 *>((BD) + _g * B_E + nc * WG_SLOT + (row >> 3) * LBO_BE + (row & 7) * 8) = o; } } } if constexpr (WG_BN == 64) { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); } }
#define WG_MMA(AD,BD) { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); const uint64_t da0 = wg_desc((AD) + mgid * WGROWS * SBO_AE, WG_SLOT * 2, SBO_AE * 2); const uint64_t db0 = wg_desc((BD) + ngid * B_E, LBO_BE * 2, WG_SLOT * 2); _Pragma("unroll") for (int kk = 0; kk < BK; kk += 16) { const uint64_t da = da0 + (kk >> 3) * (WG_SLOT / 8); const uint64_t db = db0 + (kk >> 3) * (LBO_BE / 8); if constexpr (WG_BN == 64) { WGMMA_M64N64K16(acc, da, db); } else { WGMMA_M64N128K16(acc, da, db); } } asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
    WG_STS(As0, Bs0)
    if (BK < Kg) WG_LOADG(BK)
    __syncthreads();
    int cur = 0;
    for (int kg0 = 0; kg0 < Kg; kg0 += BK) {
        if (cur == 0) { WG_MMA(As0, Bs0) } else { WG_MMA(As1, Bs1) }
        if (kg0 + BK < Kg) {
            if (STAGES == 2)
                asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
            else
                asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
            __syncthreads();
            if (cur == 0) { WG_STS(As1, Bs1) } else { WG_STS(As0, Bs0) }
            if (kg0 + 2 * BK < Kg) WG_LOADG(kg0 + 2 * BK)
            __syncthreads();
        } else {
            asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        }
        if (STAGES == 2) cur ^= 1;
    }
#undef WG_LOADG
#undef WG_STS
#undef WG_MMA
    __syncthreads();
    const int r0 = er0;
    const int cb = (lane & 3) * 2;
#pragma unroll
    for (int j = 0; j < WG_BN / 8; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float2 v = make_float2(acc[j * 4 + h * 2], acc[j * 4 + h * 2 + 1]);
            *reinterpret_cast<float2 *>(Cs + (r0 + h * 8) * WG_CLD +
                                        ngid * WG_BN + j * 8 + cb) = v;
        }
    }
    __syncthreads();
    constexpr int NC = WG_BNT / 4;
    const int lc = (tid % NC) * 4;
    const int lr = tid / NC;
    const bool ok4 = colok[lc] && colok[lc + 1] && colok[lc + 2] && colok[lc + 3];
    for (int m = lr; m < WG_BM; m += WG_NT / NC) {
        const int co = m0 + m;
        if (co >= Cout) continue;
        const int n = nb + lc;
        float4 v = *reinterpret_cast<const float4 *>(Cs + m * WG_CLD + lc);
        __half *op = out + (size_t)co * SCout + n + H;
        if (resid != nullptr) {
            const __half2 *rp = reinterpret_cast<const __half2 *>(
                resid + (size_t)co * SCout + n + H);
            __half2 r0h = rp[0], r1h = rp[1];
            v.x += __half2float(r0h.x); v.y += __half2float(r0h.y);
            v.z += __half2float(r1h.x); v.w += __half2float(r1h.y);
        }
        if (do_snake) {
            if (co < half) {
                const float a = row_alpha[m];
                const float ri = row_ri[m];
                float *vp = reinterpret_cast<float *>(&v);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float sn = __sinf(a * vp[e]);
                    vp[e] = __fmaf_rn(sn * sn, ri, vp[e]);
                }
            } else {
                float *vp = reinterpret_cast<float *>(&v);
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    vp[e] = vp[e] > 0.f ? vp[e] : 0.01f * vp[e];
            }
        }
        if (ok4) {
            __half2 o0 = __floats2half2_rn(v.x, v.y);
            __half2 o1 = __floats2half2_rn(v.z, v.w);
            reinterpret_cast<__half2 *>(op)[0] = o0;
            reinterpret_cast<__half2 *>(op)[1] = o1;
        } else {
            const float *vp = reinterpret_cast<const float *>(&v);
#pragma unroll
            for (int e = 0; e < 4; ++e)
                if (colok[lc + e]) op[e] = __float2half(vp[e]);
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
