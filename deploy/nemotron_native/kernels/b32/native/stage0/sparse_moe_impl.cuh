struct SparseMoeSchedule {
    static constexpr int UP_TILES = BB_MOE_INTER / OUT_ATOM;
    static constexpr int MAX_TILES =
        BB_EXPERTS * (BATCH_PAD / SPARSE_N) * UP_TILES;
    int base[BB_EXPERTS + 1];
    unsigned char row_tiles[BB_EXPERTS];
    unsigned char narrow[BB_EXPERTS];
    unsigned int route_mask[BB_EXPERTS];
    unsigned char expert[MAX_TILES];
    int plan_tile_io;
    int plan_first;
    int plan_step;
    int plan_mine;
    int plan_first2;
    int plan_step2;
    int plan_mine2;
};
template<bool DOWN>
__device__ __forceinline__ void sparse_moe_tile(
        const SparseMoeSchedule &schedule, int f,
        int &expert, int &feature_tile, int &token_tile) {
    const int lookup = DOWN ? (f >> 1) : f;
    expert = schedule.expert[lookup];
    const int local = lookup - schedule.base[expert];
    const bool two_rows = schedule.row_tiles[expert] == 2;
    if constexpr (DOWN) {
        feature_tile = two_rows ? local : ((local << 1) | (f & 1));
        token_tile = two_rows ? (f & 1) : 0;
    } else {
        feature_tile = two_rows ? (local >> 1) : local;
        token_tile = two_rows ? (local & 1) : 0;
    }
}
__device__ __forceinline__ void sparse_moe_build_schedule(
        const Backbone *bb, SparseMoeSchedule &schedule) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int e = warp; e < BB_EXPERTS; e += WARPS) {
        const bool hit = lane < BATCH &&
                         bb->moe_w[lane * BB_EXPERTS + e] != 0.f;
        const unsigned ballot = __ballot_sync(0xffffffffu, hit);
        if (lane == 0) schedule.route_mask[e] = ballot;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const bool valid = lane < BB_EXPERTS;
        const int count = valid ? __popc(schedule.route_mask[lane]) : 0;
        const int rt = (count + SPARSE_N - 1) / SPARSE_N;
        const int contrib = rt * SparseMoeSchedule::UP_TILES;
        int scan = contrib;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int up = __shfl_up_sync(0xffffffffu, scan, off);
            if (lane >= off) scan += up;
        }
        if (valid) {
            const int begin = scan - contrib;
            schedule.base[lane] = begin;
            schedule.row_tiles[lane] = (unsigned char)rt;
            schedule.narrow[lane] = (unsigned char)(count <= 8);
            #pragma unroll
            for (int i = 0; i < 2 * SparseMoeSchedule::UP_TILES; ++i)
                if (i < contrib)
                    schedule.expert[begin + i] = (unsigned char)lane;
        }
        if (lane == 31) schedule.base[BB_EXPERTS] = scan;
    }
    __syncthreads();
}
__device__ void sparse_moe_pack(const Backbone *bb,
                                SparseMoeSchedule &schedule) {
    const SparseMoe &s = *bb->sparse_moe;
    __shared__ int count;
    __shared__ int local_rows[BATCH];
    sparse_moe_build_schedule(bb, schedule);
    const int experts_per_wave = min(int(gridDim.x), BB_EXPERTS);
    const int first_expert = int(blockIdx.x) % experts_per_wave;
    const int rank = int(blockIdx.x) / experts_per_wave;
    const int team_size = (int(gridDim.x) + experts_per_wave - 1 - first_expert) /
                          experts_per_wave;
    for (int e = first_expert; e < BB_EXPERTS; e += experts_per_wave) {
        static_assert(BATCH <= 32, "the compaction below is one warp wide");
        if (threadIdx.x < 32) {
            const int row = threadIdx.x;
            const unsigned int ballot = schedule.route_mask[e];
            const bool hit = row < BATCH && ((ballot >> row) & 1u);
            if (hit) local_rows[__popc(ballot & ((1u << row) - 1u))] = row;
            if (row == 0) count = __popc(ballot);
        }
        __syncthreads();
        if (rank == 0) {
            if (threadIdx.x == 0) s.counts[e] = count;
            for (int i = threadIdx.x; i < count; i += THREADS)
                s.rows[e * BATCH_PAD + i] = local_rows[i];
        }
        constexpr int chunks = D_MODEL / 8;
        const int padded = count <= 8
            ? 8 : ((count + SPARSE_N - 1) / SPARSE_N) * SPARSE_N;
        {
            const int first = rank * THREADS + threadIdx.x;
            const int step = team_size * THREADS;
            const int limit = padded * chunks;
            constexpr int MAX_STRIPES = 2;
            float4 staged[MAX_STRIPES];
            for (int base_i = first; base_i < limit; base_i += step * MAX_STRIPES) {
                #pragma unroll
                for (int u = 0; u < MAX_STRIPES; ++u) {
                    const int i = base_i + u * step;
                    if (i >= limit) continue;
                    const int row = i / chunks, col = i % chunks;
                    staged[u] = row < count
                        ? *reinterpret_cast<const float4 *>(
                              bb->normed + (size_t)local_rows[row] * D_MODEL + col * 8)
                        : make_float4(0.f, 0.f, 0.f, 0.f);
                }
                #pragma unroll
                for (int u = 0; u < MAX_STRIPES; ++u) {
                    const int i = base_i + u * step;
                    if (i >= limit) continue;
                    const int row = i / chunks, col = i % chunks;
                    *reinterpret_cast<float4 *>(s.gathered +
                        (size_t)(e * BATCH_PAD + row) * D_MODEL + col * 8) = staged[u];
                }
            }
        }
        __syncthreads();
    }
}
template<bool DOWN>
__device__ __forceinline__ void sparse_moe_plan(
        int &tile_io, int n_tiles, int &first, int &step, int &mine,
        int &first2, int &step2, int &mine2) {
    const int start = advance_tile_cursor(tile_io, n_tiles);
    first = start; step = (int)gridDim.x;
    mine = start < n_tiles ? 1 + (n_tiles - 1 - start) / gridDim.x : 0;
    first2 = 0; step2 = 1; mine2 = 0;
    if constexpr (DOWN) {
        const int G = (int)gridDim.x;
        constexpr int follow_tiles = D_MODEL / OUT_ATOM;
        if (G > follow_tiles && n_tiles >= G) {
            const int longc = G - follow_tiles;
            const int waves = n_tiles / G - 1;
            const int head = waves * G;
            const int tail = n_tiles - head;
            int d = (int)blockIdx.x - tile_io;
            if (d < 0) d += G;
            mine = waves;
            if (d >= follow_tiles) {
                const int i = d - follow_tiles;
                first2 = head + i;
                step2 = longc;
                mine2 = i < tail ? 1 + (tail - 1 - i) / longc : 0;
            }
        }
    }
}
__device__ __forceinline__ void stage_next_sparse_down(
        weight_tile (&slots)[STAGES], const weight_gl &w,
        SparseMoeSchedule &schedule, int tile_io, int iter_io,
        semaphore *filled, semaphore *drained, int pf) {
    if (threadIdx.x != 0) return;
    constexpr int iters = BB_MOE_INTER / KCHUNK;
    constexpr int out_tiles = D_MODEL / OUT_ATOM;
    const int n_tiles = schedule.base[BB_EXPERTS] * 2;
    int first, step, mine, first2, step2, mine2;
    int cursor = tile_io;
    sparse_moe_plan<true>(cursor, n_tiles, first, step, mine, first2, step2,
                          mine2);
    schedule.plan_tile_io = cursor;
    schedule.plan_first = first;
    schedule.plan_step = step;
    schedule.plan_mine = mine;
    schedule.plan_first2 = first2;
    schedule.plan_step2 = step2;
    schedule.plan_mine2 = mine2;
    if (mine + mine2 == 0) return;
    const int f = mine ? first : first2;
    int e, feature_tile, token_tile;
    sparse_moe_tile<true>(schedule, f, e, feature_tile, token_tile);
    const int weight_row = e * out_tiles + feature_tile;
    const int n = pf < iters ? pf : iters;
    for (int k = 0; k < n; ++k) {
        const int iter = iter_io + k;
        const int stage = iter % STAGES, phase = (iter / STAGES) & 1;
        wait(drained[stage], phase);
        tma::expect_bytes(filled[stage],
                          sizeof(weight_tile) + sizeof(sparse_act_tile));
        tma::load_async<2, cache_policy::EVICT_FIRST>(
            slots[stage], w, {weight_row, k}, filled[stage]);
    }
}
__device__ __forceinline__ void stage_next_sparse_up(
        weight_tile (&slots)[STAGES], const weight_gl &w,
        SparseMoeSchedule &schedule, int tile_io, int iter_io,
        semaphore *filled, semaphore *drained, int pf) {
    if (threadIdx.x != 0) return;
    constexpr int iters = D_MODEL / KCHUNK;
    constexpr int out_tiles = BB_MOE_INTER / OUT_ATOM;
    const int n_tiles = schedule.base[BB_EXPERTS];
    int first, step, mine, first2, step2, mine2;
    int cursor = tile_io;
    sparse_moe_plan<false>(cursor, n_tiles, first, step, mine, first2, step2,
                           mine2);
    schedule.plan_tile_io = cursor;
    schedule.plan_first = first;
    schedule.plan_step = step;
    schedule.plan_mine = mine;
    schedule.plan_first2 = first2;
    schedule.plan_step2 = step2;
    schedule.plan_mine2 = mine2;
    if (mine == 0) return;
    int e, feature_tile, token_tile;
    sparse_moe_tile<false>(schedule, first, e, feature_tile, token_tile);
    const int weight_row = e * out_tiles + feature_tile;
    const int n = pf < iters ? pf : iters;
    for (int k = 0; k < n; ++k) {
        const int iter = iter_io + k;
        const int stage = iter % STAGES, phase = (iter / STAGES) & 1;
        wait(drained[stage], phase);
        tma::expect_bytes(filled[stage],
                          sizeof(weight_tile) + sizeof(sparse_act_tile));
        tma::load_async<2, cache_policy::EVICT_FIRST>(
            slots[stage], w, {weight_row, k}, filled[stage]);
    }
}
template<bool DOWN>
__device__ void sparse_moe_gemm(
    const weight_gl &w, const sparse_act_gl &a, const SparseMoe &s,
    bf16 *hidden, const float *routes, float *contributions,
    weight_tile (&w_s)[STAGES], act_tile (&a_s)[STAGES], out_tile &o_s,
    semaphore *filled, semaphore *drained, int &iter_io, int &tile_io,
    SparseMoeSchedule &schedule, int prestage = 0) {
    constexpr int width = DOWN ? D_MODEL : BB_MOE_INTER;
    constexpr int k_total = DOWN ? BB_MOE_INTER : D_MODEL;
    constexpr int out_tiles = width / OUT_ATOM;
    constexpr int iters = k_total / KCHUNK;
    static_assert(BATCH_PAD >= SPARSE_N && BATCH_PAD % SPARSE_N == 0);
    static_assert(BATCH <= 2 * SPARSE_N,
                  "sparse tile div/rem assumes one or two token tiles");
    static_assert(sizeof(sparse_act_tile) <= sizeof(act_tile));
    static_assert(sizeof(sparse_out_tile) <= sizeof(out_tile));
    tile_io = schedule.plan_tile_io;
    const int first = schedule.plan_first;
    const int step = schedule.plan_step;
    const int mine = schedule.plan_mine;
    const int first2 = schedule.plan_first2;
    const int step2 = schedule.plan_step2;
    const int mine2 = schedule.plan_mine2;
    const int total = mine + mine2;
    const int iter_base = iter_io;
    iter_io += total * iters;
    if (warpgroup::groupid() == 1) {
        if (warpgroup::laneid() == 0) {
            int iter = iter_base;
            int staged = total ? (prestage < iters ? prestage : iters) : 0;
            for (int n = 0, f = mine ? first : first2; n < total; ++n,
                 f = n < mine ? first + n * step
                              : first2 + (n - mine) * step2) {
                int e, feature_tile, token_tile;
                sparse_moe_tile<DOWN>(schedule, f, e, feature_tile, token_tile);
                const int weight_row = e * out_tiles + feature_tile;
                const int act_row = e * (BATCH_PAD / SPARSE_N) + token_tile;
                for (int k = 0; k < iters; ++k, ++iter) {
                    const int stage = iter % STAGES, phase = (iter / STAGES) & 1;
                    if (staged) { --staged; } else {
                    wait(drained[stage], phase);
                    tma::expect_bytes(filled[stage], sizeof(weight_tile) + sizeof(sparse_act_tile));
                    tma::load_async<2, cache_policy::EVICT_FIRST>(
                        w_s[stage], w, {weight_row, k}, filled[stage]); }
                    tma::load_async<2, cache_policy::EVICT_LAST>(
                        *reinterpret_cast<sparse_act_tile *>(&a_s[stage]),
                        a, {act_row, k}, filled[stage]);
                }
            }
        }
    } else if (warpgroup::groupid() == 0) {
        int iter = iter_base;
        for (int n = 0, f = mine ? first : first2; n < total; ++n,
             f = n < mine ? first + n * step
                          : first2 + (n - mine) * step2) {
            int e, feature_tile, token_tile;
            sparse_moe_tile<DOWN>(schedule, f, e, feature_tile, token_tile);
            const bool narrow = schedule.narrow[e] != 0;
            const int lane0 = warpgroup::laneid();
            constexpr int NROW0 = SPARSE_N / 2;
            int original0[NROW0];
            float weight0[NROW0];
            if constexpr (DOWN) {
                const int count0 = s.counts[e];
                const int first_row0 = lane0 >> 6;
                if (narrow) {
                    #pragma unroll
                    for (int j = 0; j < NROW0 / 2; ++j) {
                        const int local = first_row0 + 2 * j;
                        original0[j] = local < count0
                            ? s.rows[e * BATCH_PAD + local] : -1;
                    }
                    #pragma unroll
                    for (int j = 0; j < NROW0 / 2; ++j)
                        weight0[j] = original0[j] >= 0
                            ? routes[original0[j] * BB_EXPERTS + e] : 0.f;
                } else {
                    #pragma unroll
                    for (int j = 0; j < NROW0; ++j) {
                        const int local = token_tile * SPARSE_N + first_row0 + 2 * j;
                        original0[j] = local < count0
                            ? s.rows[e * BATCH_PAD + local] : -1;
                    }
                    #pragma unroll
                    for (int j = 0; j < NROW0; ++j)
                        weight0[j] = original0[j] >= 0
                            ? routes[original0[j] * BB_EXPERTS + e] : 0.f;
                }
            }
            rt_fl<16, SPARSE_N> accum;
            warp::zero(accum);
            int prior_stage = -1;
            for (int k = 0; k < iters; ++k, ++iter) {
                const int stage = iter % STAGES, phase = (iter / STAGES) & 1;
                wait(filled[stage], phase);
                if (narrow)
                    bf16_mma_ABt_n8(accum, w_s[stage],
                        *reinterpret_cast<act16_tile *>(&a_s[stage]), 0);
                else
                    warpgroup::mma_ABt(accum, w_s[stage],
                        *reinterpret_cast<sparse_act_tile *>(&a_s[stage]));
                if (prior_stage >= 0) {
                    warpgroup::mma_async_wait<1>();
                    if (warp::laneid() == 0) arrive(drained[prior_stage]);
                }
                prior_stage = stage;
            }
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(drained[prior_stage]);
            sparse_out_tile &out = *reinterpret_cast<sparse_out_tile *>(&o_s);
            warpgroup::store(out, accum);
            warpgroup::sync(1);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            const int lane = warpgroup::laneid();
            const int col = lane & (OUT_ATOM - 1);
            const int first_row = lane >> 6;
            if constexpr (DOWN) {
                constexpr int NROW = NROW0;
                const int *original = original0;
                const float *weight = weight0;
                const int output = feature_tile * OUT_ATOM + col;
                if (narrow) {
                    #pragma unroll
                    for (int j = 0; j < NROW / 2; ++j) {
                        if (original[j] < 0) continue;
                        const float value = float(out[{col, first_row + 2 * j}]);
                        contributions[((size_t)original[j] * (BB_EXPERTS + 1) + e) *
                                      D_MODEL + output] = __fmul_rn(weight[j], value);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < NROW; ++j) {
                        if (original[j] < 0) continue;
                        const float value = float(out[{col, first_row + 2 * j}]);
                        contributions[((size_t)original[j] * (BB_EXPERTS + 1) + e) *
                                      D_MODEL + output] = __fmul_rn(weight[j], value);
                    }
                }
            } else {
            if (narrow) {
                #pragma unroll
                for (int row = first_row; row < 8; row += 2) {
                    const int packed = e * BATCH_PAD + row;
                    const int output = feature_tile * OUT_ATOM + col;
                    const float value = float(out[{col, row}]);
                    hidden[(size_t)packed * BB_MOE_INTER + output] =
                        bf16(value / (1.f + __expf(-value)));
                }
            } else {
                #pragma unroll
                for (int row = first_row; row < SPARSE_N; row += 2) {
                    const int local = token_tile * SPARSE_N + row;
                    const int packed = e * BATCH_PAD + local;
                    const int output = feature_tile * OUT_ATOM + col;
                    const float value = float(out[{col, row}]);
                    hidden[(size_t)packed * BB_MOE_INTER + output] =
                        bf16(value / (1.f + __expf(-value)));
                }
            }
            }
            warpgroup::sync(1);
        }
    }
}
