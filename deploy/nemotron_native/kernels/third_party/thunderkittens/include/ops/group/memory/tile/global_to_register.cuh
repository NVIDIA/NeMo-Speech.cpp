












template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;

    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    using U2 = base_types::packing<U>::packed_type;
    constexpr int pack_size = base_types::packing<U2>::num();
    constexpr int half_col = RT::tile_size_col / 2;
    int warp_laneid = threadIdx.x % WARP_THREADS;
    int local_warpid;
    if constexpr(GROUP_WARPS % 4 == 0) local_warpid = (warpid()/4+(warpid()%4)*(GROUP_WARPS/4));
    else local_warpid = warpid();
    const int row_offset = RT::rows*local_warpid;
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        int row = row_offset + i*RT::tile_size_row + (warp_laneid / 4);
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            int col = j*RT::tile_size_col + pack_size*(warp_laneid % 4);
            dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row+0)*row_stride + (col+0)]));
            dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row+0)*row_stride + (col+half_col)]));
        }
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            int col = j*RT::tile_size_col + pack_size*(warp_laneid % 4);
            dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row+8)*row_stride + (col+0)]));
            dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row+8)*row_stride + (col+half_col)]));
        }
    }
}








template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T = typename RT::T;
    using U = typename GL::dtype;

    #if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    static_assert(!std::is_same_v<T, fp8e4m3> && !std::is_same_v<T, fp8e5m2>, "Unsupported type for load/store");
    #endif

    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    int warp_laneid = threadIdx.x % WARP_THREADS;
    int local_warpid;
    if constexpr(GROUP_WARPS % 4 == 0) local_warpid = (warpid()/4+(warpid()%4)*(GROUP_WARPS/4));
    else local_warpid = warpid();
    const int row_offset = RT::rows*local_warpid;
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        int row = row_offset + i*RT::tile_size_row + 2*(warp_laneid % 4);
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            int col = j*RT::tile_size_col + (warp_laneid / 4);
            dst.tiles[i][j].data[0].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + (col+0)]);
            dst.tiles[i][j].data[1].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            int col = j*RT::tile_size_col + (warp_laneid / 4);
            dst.tiles[i][j].data[0].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + (col+0)]);
            dst.tiles[i][j].data[1].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            int col = j*RT::tile_size_col + (warp_laneid / 4);
            dst.tiles[i][j].data[2].x = base_types::convertor<T, U>::convert(src_ptr[(row+8)*row_stride + (col+0)]);
            dst.tiles[i][j].data[3].x = base_types::convertor<T, U>::convert(src_ptr[(row+8)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            int col = j*RT::tile_size_col + (warp_laneid / 4);
            dst.tiles[i][j].data[2].y = base_types::convertor<T, U>::convert(src_ptr[(row+9)*row_stride + (col+0)]);
            dst.tiles[i][j].data[3].y = base_types::convertor<T, U>::convert(src_ptr[(row+9)*row_stride + (col+8)]);
        }
    }
}
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    load<2>(dst, src, idx);
}








template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    using U2 = base_types::packing<U>::packed_type;
    constexpr int pack_size = base_types::packing<U2>::num();
    constexpr int half_col = RT::tile_size_col / 2;
    int warp_laneid = threadIdx.x % WARP_THREADS;
    int local_warpid;
    if constexpr(GROUP_WARPS % 4 == 0) local_warpid = (warpid()/4+(warpid()%4)*(GROUP_WARPS/4));
    else local_warpid = warpid();
    const int row_offset = RT::rows*local_warpid;
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        int row = row_offset + i*RT::tile_size_row + (warp_laneid / 4);
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            int col = j*RT::tile_size_col + pack_size*(warp_laneid % 4);
            *(U2*)(&dst_ptr[(row+0)*row_stride + (col+0)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
            *(U2*)(&dst_ptr[(row+0)*row_stride + (col+half_col)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
        }
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            int col = j*RT::tile_size_col + pack_size*(warp_laneid % 4);
            *(U2*)(&dst_ptr[(row+8)*row_stride + (col+0)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            *(U2*)(&dst_ptr[(row+8)*row_stride + (col+half_col)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
        }
    }
}








template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;

    #if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    static_assert(!std::is_same_v<T, fp8e4m3_4> && !std::is_same_v<T, fp8e5m2_4>, "Unsupported type for load/store");
    #endif

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    int warp_laneid = threadIdx.x % WARP_THREADS;
    int local_warpid;
    if constexpr(GROUP_WARPS % 4 == 0) local_warpid = (warpid()/4+(warpid()%4)*(GROUP_WARPS/4));
    else local_warpid = warpid();
    const int row_offset = RT::rows*local_warpid;
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        int row = row_offset + i*RT::tile_size_row + 2*(warp_laneid % 4);
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            int col = j*RT::tile_size_col + (warp_laneid / 4);
            dst_ptr[(row+0)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
            dst_ptr[(row+0)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
        }
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            int col = j*RT::tile_size_col + (warp_laneid / 4);
            dst_ptr[(row+1)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
            dst_ptr[(row+1)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
        }
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            int col = j*RT::tile_size_col + (warp_laneid / 4);
            dst_ptr[(row+8)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].x);
            dst_ptr[(row+8)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].x);
        }
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            int col = j*RT::tile_size_col + (warp_laneid / 4);
            dst_ptr[(row+9)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].y);
            dst_ptr[(row+9)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].y);
        }
    }
}
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    store<2>(dst, src, idx);
}
