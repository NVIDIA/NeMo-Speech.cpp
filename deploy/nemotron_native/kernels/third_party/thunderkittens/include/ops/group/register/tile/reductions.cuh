


















template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(V::outer_dim == T::height);

    using dtype = V::dtype;

    #pragma unroll
    for(int i = 0; i < T::height; i++) {
        dtype accum_top_row    = op::template op<dtype>(src.tiles[i][0].data[0], src.tiles[i][0].data[2]);
        dtype accum_bottom_row = op::template op<dtype>(src.tiles[i][0].data[1], src.tiles[i][0].data[3]);
        #pragma unroll
        for(int j = 1; j < T::width; j++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile; k+=2) {
                accum_top_row    = op::template op<dtype>(accum_top_row,    src.tiles[i][j].data[k+0]);
                accum_bottom_row = op::template op<dtype>(accum_bottom_row, src.tiles[i][j].data[k+1]);
            }
        }
        dtype accum_packed;
        accum_packed.x = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_top_row.x,    accum_top_row.y);
        accum_packed.y = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_bottom_row.x, accum_bottom_row.y);



        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_xor_sync(MASK_ALL, accum_packed, 2));
        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_xor_sync(MASK_ALL, accum_packed, 1));

        if(reset) {
            row_accum[i][0] = accum_packed;
        }
        else {
            row_accum[i][0] = op::template op<dtype>(src_accum[i][0], accum_packed);
        }
    }
}














template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(V::outer_dim == T::height);

    using dtype = V::dtype;

    #pragma unroll
    for(int i = 0; i < T::height; i++) {
        dtype accum_top_rows    = op::template op<dtype>(src.tiles[i][0].data[0], src.tiles[i][0].data[1]);
        dtype accum_bottom_rows = op::template op<dtype>(src.tiles[i][0].data[2], src.tiles[i][0].data[3]);
        #pragma unroll
        for(int j = 1; j < T::width; j++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile/2; k++) {
                accum_top_rows    = op::template op<dtype>(accum_top_rows,    src.tiles[i][j].data[k+0]);
                accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, src.tiles[i][j].data[k+2]);
            }
        }



        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_xor_sync(MASK_ALL, accum_top_rows, 16));
        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_xor_sync(MASK_ALL, accum_top_rows, 8));
        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_xor_sync(MASK_ALL, accum_top_rows, 4));

        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_xor_sync(MASK_ALL, accum_bottom_rows, 16));
        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_xor_sync(MASK_ALL, accum_bottom_rows, 8));
        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_xor_sync(MASK_ALL, accum_bottom_rows, 4));

        if(reset) {
            row_accum[i][0] = accum_top_rows;
            row_accum[i][1] = accum_bottom_rows;
        }
        else {
            row_accum[i][0] = op::template op<dtype>(src_accum[i][0], accum_top_rows);
            row_accum[i][1] = op::template op<dtype>(src_accum[i][1], accum_bottom_rows);
        }
    }
}
















template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {

    KITTENS_CHECK_WARP
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>);
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(V::outer_dim == T::width);

    using dtype = V::dtype;

    #pragma unroll
    for(int j = 0; j < T::width; j++) {
        dtype accum_left_cols  = op::template op<dtype>(src.tiles[0][j].data[0], src.tiles[0][j].data[1]);
        dtype accum_right_cols = op::template op<dtype>(src.tiles[0][j].data[2], src.tiles[0][j].data[3]);
        #pragma unroll
        for(int i = 1; i < T::height; i++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile/2; k++) {
                accum_left_cols  = op::template op<dtype>(accum_left_cols,  src.tiles[i][j].data[k+0]);
                accum_right_cols = op::template op<dtype>(accum_right_cols, src.tiles[i][j].data[k+2]);
            }
        }



        accum_left_cols = op::template op<dtype>(accum_left_cols, packed_shfl_xor_sync(MASK_ALL, accum_left_cols, 16));
        accum_left_cols = op::template op<dtype>(accum_left_cols, packed_shfl_xor_sync(MASK_ALL, accum_left_cols, 8));
        accum_left_cols = op::template op<dtype>(accum_left_cols, packed_shfl_xor_sync(MASK_ALL, accum_left_cols, 4));

        accum_right_cols = op::template op<dtype>(accum_right_cols, packed_shfl_xor_sync(MASK_ALL, accum_right_cols, 16));
        accum_right_cols = op::template op<dtype>(accum_right_cols, packed_shfl_xor_sync(MASK_ALL, accum_right_cols, 8));
        accum_right_cols = op::template op<dtype>(accum_right_cols, packed_shfl_xor_sync(MASK_ALL, accum_right_cols, 4));

        if(reset) {
            col_accum[j][0] = accum_left_cols;
            col_accum[j][1] = accum_right_cols;
        }
        else {
            col_accum[j][0] = op::template op<dtype>(src_accum[j][0], accum_left_cols);
            col_accum[j][1] = op::template op<dtype>(src_accum[j][1], accum_right_cols);
        }
    }
}














template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {

    KITTENS_CHECK_WARP
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>);
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(V::outer_dim == T::width);

    using dtype = V::dtype;
    #pragma unroll
    for(int j = 0; j < T::width; j++) {
        dtype accum_left_col  = op::template op<dtype>(src.tiles[0][j].data[0], src.tiles[0][j].data[2]);
        dtype accum_right_col = op::template op<dtype>(src.tiles[0][j].data[1], src.tiles[0][j].data[3]);
        #pragma unroll
        for(int i = 1; i < T::height; i++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile; k+=2) {
                accum_left_col  = op::template op<dtype>(accum_left_col,  src.tiles[i][j].data[k+0]);
                accum_right_col = op::template op<dtype>(accum_right_col, src.tiles[i][j].data[k+1]);
            }
        }
        dtype accum_packed;
        accum_packed.x = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_left_col.x,  accum_left_col.y);
        accum_packed.y = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_right_col.x, accum_right_col.y);



        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_xor_sync(MASK_ALL, accum_packed, 2));
        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_xor_sync(MASK_ALL, accum_packed, 1));

        if(reset) {
            col_accum[j][0] = accum_packed;
        }
        else {
            col_accum[j][0] = op::template op<dtype>(src_accum[j][0], accum_packed);
        }
    }
}













template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_max(V &row_accum, const T &src)  {
    row_reduce<base_ops::max, V, T, true>(row_accum, src, row_accum);
}








template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_min(V &row_accum, const T &src)  {
    row_reduce<base_ops::min, V, T, true>(row_accum, src, row_accum);
}








template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_sum(V &row_accum, const T &src)  {
    row_reduce<base_ops::sum, V, T, true>(row_accum, src, row_accum);
}








template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_prod(V &row_accum, const T &src) {
    row_reduce<base_ops::mul, V, T, true>(row_accum, src, row_accum);
}










template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_max(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::max, V, T, false>(row_accum, src, src_accum);
}









template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_min(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::min, V, T, false>(row_accum, src, src_accum);
}









template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_sum(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::sum, V, T, false>(row_accum, src, src_accum);
}









template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_prod(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::mul, V, T, false>(row_accum, src, src_accum);
}











template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_max(V &col_accum, const T &src)  {
    col_reduce<base_ops::max, V, T, true>(col_accum, src, col_accum);
}








template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_min(V &col_accum, const T &src)  {
    col_reduce<base_ops::min, V, T, true>(col_accum, src, col_accum);
}








template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_sum(V &col_accum, const T &src)  {
    col_reduce<base_ops::sum, V, T, true>(col_accum, src, col_accum);
}








template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_prod(V &col_accum, const T &src) {
    col_reduce<base_ops::mul, V, T, true>(col_accum, src, col_accum);
}










template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_max(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::max, V, T, false>(col_accum, src, src_accum);
}









template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_min(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::min, V, T, false>(col_accum, src, src_accum);
}









template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_sum(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::sum, V, T, false>(col_accum, src, src_accum);
}









template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_prod(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::mul, V, T, false>(col_accum, src, src_accum);
}



template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void max(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_max(dst, src, src_accum);
    else col_max(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline auto max(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_max(dst, src, src_accum);
    else col_max(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void max(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_max(dst, src);
    else col_max(dst, src);
}
template<int ax, ducks::rt::all T>
__device__ static inline auto max(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_max(dst, src);
    else col_max(dst, src);
    return dst;
}

template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void min(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_min(dst, src, src_accum);
    else col_min(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline auto min(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_min(dst, src, src_accum);
    else col_min(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void min(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_min(dst, src);
    else col_min(dst, src);
}
template<int ax, ducks::rt::all T>
__device__ static inline auto min(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_min(dst, src);
    else col_min(dst, src);
    return dst;
}

template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void sum(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_sum(dst, src, src_accum);
    else col_sum(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline auto sum(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_sum(dst, src, src_accum);
    else col_sum(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void sum(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_sum(dst, src);
    else col_sum(dst, src);
}
template<int ax, ducks::rt::all T>
__device__ static inline auto sum(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_sum(dst, src);
    else col_sum(dst, src);
    return dst;
}

template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void prod(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_prod(dst, src, src_accum);
    else col_prod(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline auto prod(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_prod(dst, src, src_accum);
    else col_prod(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void prod(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_prod(dst, src);
    else col_prod(dst, src);
}
template<int ax, ducks::rt::all T>
__device__ static inline auto prod(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_prod(dst, src);
    else col_prod(dst, src);
    return dst;
}
