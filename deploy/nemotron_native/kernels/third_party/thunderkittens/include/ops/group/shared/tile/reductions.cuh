















template<typename op, ducks::sv::all V, ducks::st::all T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    using dtype = typename V::dtype;
    for (int row = laneid(); row < T::rows; row += GROUP_THREADS) {
        dtype accum = src[{row, 0}];
        #pragma unroll
        for (int col = 1; col < T::cols; col++) {
            accum = op::template op<dtype>(accum, src[{row, col}]);
        }
        if (reset) {
            row_accum[row] = accum;
        } else {
            row_accum[row] = op::template op<dtype>(src_accum[row], accum);
        }
    }
}












template<typename op, ducks::sv::all V, ducks::st::all T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    using dtype = typename V::dtype;
    for (int col = laneid(); col < T::cols; col += GROUP_THREADS) {
        dtype accum = src[{0, col}];
        #pragma unroll
        for (int row = 1; row < T::rows; row++) {
            accum = op::template op<dtype>(accum, src[{row, col}]);
        }
        if (reset) {
            col_accum[col] = accum;
        } else {
            col_accum[col] = op::template op<dtype>(src_accum[col], accum);
        }
    }
}











template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_max(V &row_accum, const T &src)  {
    row_reduce<base_ops::max, V, T, true>(row_accum, src, row_accum);
}








template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_min(V &row_accum, const T &src)  {
    row_reduce<base_ops::min, V, T, true>(row_accum, src, row_accum);
}








template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_sum(V &row_accum, const T &src)  {
    row_reduce<base_ops::sum, V, T, true>(row_accum, src, row_accum);
}








template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_prod(V &row_accum, const T &src) {
    row_reduce<base_ops::mul, V, T, true>(row_accum, src, row_accum);
}










template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_max(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::max, V, T, false>(row_accum, src, src_accum);
}









template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_min(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::min, V, T, false>(row_accum, src, src_accum);
}









template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_sum(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::sum, V, T, false>(row_accum, src, src_accum);
}









template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_prod(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::mul, V, T, false>(row_accum, src, src_accum);
}









template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_max(V &col_accum, const T &src)  {
    col_reduce<base_ops::max, V, T, true>(col_accum, src, col_accum);
}








template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_min(V &col_accum, const T &src)  {
    col_reduce<base_ops::min, V, T, true>(col_accum, src, col_accum);
}








template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_sum(V &col_accum, const T &src)  {
    col_reduce<base_ops::sum, V, T, true>(col_accum, src, col_accum);
}








template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_prod(V &col_accum, const T &src) {
    col_reduce<base_ops::mul, V, T, true>(col_accum, src, col_accum);
}










template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_max(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::max, V, T, false>(col_accum, src, src_accum);
}









template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_min(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::min, V, T, false>(col_accum, src, src_accum);
}









template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_sum(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::sum, V, T, false>(col_accum, src, src_accum);
}









template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_prod(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::mul, V, T, false>(col_accum, src, src_accum);
}



template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void max(V &dst, const T &src, const V &src_accum) {
    if constexpr (ax == axis::COL) row_max(dst, src, src_accum);
    else col_max(dst, src, src_accum);
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline auto max(const T &src, const V &src_accum) {
    V dst;
    if constexpr (ax == axis::COL) row_max(dst, src, src_accum);
    else col_max(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void max(V &dst, const T &src) {
    if constexpr (ax == axis::COL) row_max(dst, src);
    else col_max(dst, src);
}
template<int ax, ducks::st::all T>
__device__ static inline auto max(const T &src) {
    using V = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    V dst;
    if constexpr (ax == axis::COL) row_max(dst, src);
    else col_max(dst, src);
    return dst;
}

template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void min(V &dst, const T &src, const V &src_accum) {
    if constexpr (ax == axis::COL) row_min(dst, src, src_accum);
    else col_min(dst, src, src_accum);
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline auto min(const T &src, const V &src_accum) {
    V dst;
    if constexpr (ax == axis::COL) row_min(dst, src, src_accum);
    else col_min(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void min(V &dst, const T &src) {
    if constexpr (ax == axis::COL) row_min(dst, src);
    else col_min(dst, src);
}
template<int ax, ducks::st::all T>
__device__ static inline auto min(const T &src) {
    using V = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    V dst;
    if constexpr (ax == axis::COL) row_min(dst, src);
    else col_min(dst, src);
    return dst;
}

template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void sum(V &dst, const T &src, const V &src_accum) {
    if constexpr (ax == axis::COL) row_sum(dst, src, src_accum);
    else col_sum(dst, src, src_accum);
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline auto sum(const T &src, const V &src_accum) {
    V dst;
    if constexpr (ax == axis::COL) row_sum(dst, src, src_accum);
    else col_sum(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void sum(V &dst, const T &src) {
    if constexpr (ax == axis::COL) row_sum(dst, src);
    else col_sum(dst, src);
}
template<int ax, ducks::st::all T>
__device__ static inline auto sum(const T &src) {
    using V = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    V dst;
    if constexpr (ax == axis::COL) row_sum(dst, src);
    else col_sum(dst, src);
    return dst;
}

template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void prod(V &dst, const T &src, const V &src_accum) {
    if constexpr (ax == axis::COL) row_prod(dst, src, src_accum);
    else col_prod(dst, src, src_accum);
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline auto prod(const T &src, const V &src_accum) {
    V dst;
    if constexpr (ax == axis::COL) row_prod(dst, src, src_accum);
    else col_prod(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void prod(V &dst, const T &src) {
    if constexpr (ax == axis::COL) row_prod(dst, src);
    else col_prod(dst, src);
}
template<int ax, ducks::st::all T>
__device__ static inline auto prod(const T &src) {
    using V = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    V dst;
    if constexpr (ax == axis::COL) row_prod(dst, src);
    else col_prod(dst, src);
    return dst;
}
