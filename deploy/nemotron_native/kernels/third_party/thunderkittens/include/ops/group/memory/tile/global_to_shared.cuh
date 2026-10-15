













template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();

    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::rows*ST::cols + GROUP_THREADS*elem_per_memcpy-1) / (GROUP_THREADS*elem_per_memcpy);

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    int laneid = threadIdx.x % GROUP_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * GROUP_THREADS + laneid;

        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % ST::cols;

        if constexpr (assume_aligned) {
            float4 tmp;
            move<float4>::ldg(tmp, (float4*)&src_ptr[row*row_stride + col]);
            move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
        }
        else {
            if (row + unit_coord.template dim<axis>() < src.template shape<axis>()) {
                float4 tmp;
                move<float4>::ldg(tmp, (float4*)&src_ptr[row*row_stride + col]);
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
            }
            else {
                float4 zeros = {0.f,0.f,0.f,0.f};
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), zeros);
            }
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    load<2, false, ST, GL, COORD>(dst, src, idx);
}









template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    using T = typename ST::dtype;
    const int row_stride = dst.template stride<axis>();

    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::rows*ST::cols + GROUP_THREADS*elem_per_memcpy-1) / (GROUP_THREADS*elem_per_memcpy);

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst[unit_coord];
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    int laneid = threadIdx.x % GROUP_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * GROUP_THREADS + laneid;

        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % ST::cols;

        if constexpr (assume_aligned) {
            float4 tmp;
            move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
            move<float4>::stg((float4*)&dst_ptr[row*row_stride + col], tmp);
        }
        else {
            if (row + unit_coord.template dim<axis>() < dst.template shape<axis>()) {
                float4 tmp;
                move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
                move<float4>::stg((float4*)&dst_ptr[row*row_stride + col], tmp);
            }
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    store<2, false, ST, GL, COORD>(dst, src, idx);
}










template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx) {
    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();

    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::rows*ST::cols + GROUP_THREADS*elem_per_memcpy-1) / (GROUP_THREADS*elem_per_memcpy);

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    int laneid = threadIdx.x % GROUP_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * GROUP_THREADS + laneid;

        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % ST::cols;

        if constexpr (assume_aligned) {
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                :: "r"(dst.idx(dst_ptr, {row, col})), "l"(&src_ptr[row*row_stride + col])
                : "memory"
            );
        }
        else {
            if (row + unit_coord.template dim<axis>() < src.template shape<axis>()) {
                asm volatile(
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                    :: "r"(dst.idx(dst_ptr, {row, col})), "l"(&src_ptr[row*row_stride + col])
                    : "memory"
                );
            }
            else {

                float4 zeros = {0.f,0.f,0.f,0.f};
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), zeros);
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx) {
    load_async<2, false, ST, GL, COORD>(dst, src, idx);
}
