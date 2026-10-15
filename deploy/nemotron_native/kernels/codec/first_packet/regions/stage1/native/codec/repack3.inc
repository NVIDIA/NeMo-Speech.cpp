__device__ __noinline__ void repack3_kernel(
    const __half *__restrict__ win, const __half *__restrict__ wsk,
    __half *__restrict__ din, __half *__restrict__ dsk) {
    int4 (*sm)[49]=reinterpret_cast<int4(*)[49]>(native_arena());
    const int job = blockIdx.y;
    const int Cout = job ? 768 : 1536;
    const int Cin = job ? 1536 : 768;
    const __half *src = job ? wsk : win;
    __half *dst = job ? dsk : din;
    const int GPB = Cin >> 3;
    const int Kg = Cin * 3;
    const int GT = GPB >> 4;
    const int RT = Cout >> 5;
    const int t = threadIdx.x;
    for (int tile = blockIdx.x; tile < 6 * RT * GT; tile += gridDim.x) {
        const int gblk = tile % GT;
        const int rest = tile / GT;
        const int corow = rest % RT;
        const int b = rest / RT;
        const int co0 = corow * 32;
        const int g0 = gblk * 16;
        const __half *sbase = src + (size_t)(b + 1) * Cout * Kg +
                              (size_t)co0 * Kg + g0 * 24;
#pragma unroll
        for (int j = 0; j < 6; ++j) {
            const int idx = t + j * 256;
            const int r = idx / 48, c = idx - (idx / 48) * 48;
            sm[r][c] = *reinterpret_cast<const int4 *>(sbase + (size_t)r * Kg +
                                                       c * 8);
        }
        __syncthreads();
        __half *dbase = dst + (size_t)b * (3 * (size_t)Cout * Cin);
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int u = t + p * 256;
            const int col = (u & 7) + 8 * (u >> 7);
            const int g = g0 + ((u >> 3) & 15);
            const int co = co0 + col;
            const int c = g >> 1, v = g & 1;
            const int4 a = sm[col][((u >> 3) & 15) * 3];
            const int4 e = sm[col][((u >> 3) & 15) * 3 + 1];
            const int4 f = sm[col][((u >> 3) & 15) * 3 + 2];
            uint32_t h[12];
            h[0] = a.x; h[1] = a.y; h[2] = a.z; h[3] = a.w;
            h[4] = e.x; h[5] = e.y; h[6] = e.z; h[7] = e.w;
            h[8] = f.x; h[9] = f.y; h[10] = f.z; h[11] = f.w;
            __half *d = dbase + (size_t)c * (48 * Cout) +
                        (size_t)(co >> 3) * 128 + v * 64 + (co & 7) * 8;
#pragma unroll
            for (int tp = 0; tp < 3; ++tp) {
                uint32_t o[4];
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const uint32_t g0w = h[3 * jj], g1w = h[3 * jj + 1],
                                   g2w = h[3 * jj + 2];
                    o[jj] = (tp == 0) ? __byte_perm(g0w, g1w, 0x7610)
                          : (tp == 1) ? __byte_perm(g0w, g2w, 0x5432)
                                      : __byte_perm(g1w, g2w, 0x7610);
                }
                *reinterpret_cast<int4 *>(d + (size_t)tp * (16 * Cout)) =
                    make_int4(o[0], o[1], o[2], o[3]);
            }
        }
        __syncthreads();
    }
}
