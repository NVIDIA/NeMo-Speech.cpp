__device__ void sparse_moe_pack(const Backbone *bb) {
    const SparseMoe &s = *bb->sparse_moe;
    __shared__ int count;
    __shared__ int local_rows[BATCH];
    const int experts_per_wave = min(int(gridDim.x), BB_EXPERTS);
    const int first_expert = int(blockIdx.x) % experts_per_wave;
    const int rank = int(blockIdx.x) / experts_per_wave;
    const int team_size = (int(gridDim.x) + experts_per_wave - 1 - first_expert) /
                          experts_per_wave;
    for (int e = first_expert; e < BB_EXPERTS; e += experts_per_wave) {
        if (threadIdx.x == 0) {
            int n = 0;
            for (int row = 0; row < BATCH; ++row)
                if (bb->moe_w[row * BB_EXPERTS + e] != 0.f)
                    local_rows[n++] = row;
            count = n;
            if (rank == 0) {
                s.counts[e] = n;
                for (int i = 0; i < n; ++i)
                    s.rows[e * BATCH_PAD + i] = local_rows[i];
            }
        }
        __syncthreads();
        constexpr int chunks = D_MODEL / 8;
        const int padded = ((count + SPARSE_N - 1) / SPARSE_N) * SPARSE_N;
        for (int i = rank * THREADS + threadIdx.x; i < padded * chunks;
             i += team_size * THREADS) {
            const int row = i / chunks, col = i % chunks;
            const int packed = e * BATCH_PAD + row;
            auto *dst = reinterpret_cast<float4 *>(s.gathered +
                        (size_t)packed * D_MODEL + col * 8);
            if (row < count) {
                const int original = local_rows[row];
                *dst = *reinterpret_cast<const float4 *>(bb->normed +
                         (size_t)original * D_MODEL + col * 8);
            } else {
                *dst = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        __syncthreads();
    }
}
template<bool DOWN>
__device__ void sparse_moe_gemm(
    const weight_gl &w, const sparse_act_gl &a, const SparseMoe &s,
    bf16 *hidden, const float *routes, float *contributions,
    weight_tile (&w_s)[STAGES], act_tile (&a_s)[STAGES], out_tile &o_s,
    semaphore *filled, semaphore *drained, int &iter_io, int &tile_io) {
    constexpr int width = DOWN ? D_MODEL : BB_MOE_INTER;
    constexpr int k_total = DOWN ? BB_MOE_INTER : D_MODEL;
    constexpr int out_tiles = width / OUT_ATOM;
    constexpr int iters = k_total / KCHUNK;
    static_assert(BATCH_PAD >= SPARSE_N && BATCH_PAD % SPARSE_N == 0);
    static_assert(sizeof(sparse_act_tile) <= sizeof(act_tile));
    static_assert(sizeof(sparse_out_tile) <= sizeof(out_tile));
    __shared__ int prefix[BB_EXPERTS + 1];
    __shared__ int row_tiles[BB_EXPERTS];
    if (threadIdx.x == 0) {
        int total = 0;
        for (int e = 0; e < BB_EXPERTS; ++e) {
            prefix[e] = total;
            row_tiles[e] = (s.counts[e] + SPARSE_N - 1) / SPARSE_N;
            total += row_tiles[e] * out_tiles;
        }
        prefix[BB_EXPERTS] = total;
    }
    __syncthreads();
    const int n_tiles = prefix[BB_EXPERTS];
    int start = int(blockIdx.x) - tile_io % int(gridDim.x);
    if (start < 0) start += gridDim.x;
    tile_io += n_tiles;
    const int mine = start < n_tiles ? 1 + (n_tiles - 1 - start) / gridDim.x : 0;
    const int iter_base = iter_io;
    iter_io += mine * iters;
    if (warpgroup::groupid() == 1) {
        if (warpgroup::laneid() == 0) {
            int iter = iter_base;
            for (int f = start; f < n_tiles; f += gridDim.x) {
                int e = 0;
                while (prefix[e + 1] <= f) ++e;
                const int rem = f - prefix[e], nt = row_tiles[e];
                const int feature_tile = rem / nt;
                const int token_tile = rem - feature_tile * nt;
                const int weight_row = e * out_tiles + feature_tile;
                const int act_row = e * (BATCH_PAD / SPARSE_N) + token_tile;
                for (int k = 0; k < iters; ++k, ++iter) {
                    const int stage = iter % STAGES, phase = (iter / STAGES) & 1;
                    wait(drained[stage], phase);
                    tma::expect_bytes(filled[stage], sizeof(weight_tile) + sizeof(sparse_act_tile));
                    tma::load_async<2, cache_policy::EVICT_FIRST>(
                        w_s[stage], w, {weight_row, k}, filled[stage]);
                    tma::load_async<2, cache_policy::EVICT_LAST>(
                        *reinterpret_cast<sparse_act_tile *>(&a_s[stage]),
                        a, {act_row, k}, filled[stage]);
                }
            }
        }
    } else if (warpgroup::groupid() == 0) {
        int iter = iter_base;
        for (int f = start; f < n_tiles; f += gridDim.x) {
            int e = 0;
            while (prefix[e + 1] <= f) ++e;
            const int rem = f - prefix[e], nt = row_tiles[e];
            const int feature_tile = rem / nt;
            const int token_tile = rem - feature_tile * nt;
            rt_fl<16, SPARSE_N> accum;
            warp::zero(accum);
            for (int k = 0; k < iters; ++k, ++iter) {
                const int stage = iter % STAGES, phase = (iter / STAGES) & 1;
                wait(filled[stage], phase);
                warpgroup::mma_ABt(accum, w_s[stage],
                    *reinterpret_cast<sparse_act_tile *>(&a_s[stage]));
                warpgroup::mma_async_wait();
                if (warp::laneid() == 0) arrive(drained[stage]);
            }
            sparse_out_tile &out = *reinterpret_cast<sparse_out_tile *>(&o_s);
            warpgroup::store(out, accum);
            warpgroup::sync(1);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            for (int i = warpgroup::laneid(); i < OUT_ATOM * SPARSE_N; i += 128) {
                const int row = i / OUT_ATOM, col = i % OUT_ATOM;
                const int local = token_tile * SPARSE_N + row;
                const int packed = e * BATCH_PAD + local;
                const int output = feature_tile * OUT_ATOM + col;
                const float value = float(out[{col, row}]);
                if constexpr (!DOWN) {
                    hidden[(size_t)packed * BB_MOE_INTER + output] =
                        bf16(value / (1.f + __expf(-value)));
                } else if (local < s.counts[e]) {
                    const int original = s.rows[packed];
                    const float weight = routes[original * BB_EXPERTS + e];
                    contributions[((size_t)original * (BB_EXPERTS + 1) + e) *
                                  D_MODEL + output] = __fmul_rn(weight, value);
                }
            }
            warpgroup::sync(1);
        }
    }
}
