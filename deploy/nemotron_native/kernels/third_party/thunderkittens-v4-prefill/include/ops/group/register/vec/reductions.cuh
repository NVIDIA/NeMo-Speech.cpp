




















template<typename op, ducks::rv::all RV, bool reset>
__device__ static inline void reduce(
        typename base_types::packing<typename RV::dtype>::unpacked_type &dst_accum,
        const RV &src,
        const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    KITTENS_CHECK_WARP
    using T = base_types::packing<typename RV::dtype>::unpacked_type;
    int laneid = kittens::laneid();
    if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        T accum = op::template op<T>(src[0][0].x, src[0][0].y);
        #pragma unroll
        for(int i = 1; i < RV::outer_dim; i++) {
            accum = op::template op<T>(accum, src[i][0].x);
            accum = op::template op<T>(accum, src[i][0].y);
        }

        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 16));
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 8));
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 4));

        if constexpr (!reset) accum = op::template op<T>(accum, src_accum);

        dst_accum = packed_shfl_sync(kittens::MASK_ALL, accum, 0);
    }
    else if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        T accum = op::template op<T>(src[0][0].x, src[0][0].y);
        accum = op::template op<T>(accum,       src[0][1].x);
        accum = op::template op<T>(accum,       src[0][1].y);
        #pragma unroll
        for(int i = 1; i < RV::outer_dim; i++) {

            accum = op::template op<T>(accum, src[i][0].x);
            accum = op::template op<T>(accum, src[i][0].y);
            accum = op::template op<T>(accum, src[i][1].x);
            accum = op::template op<T>(accum, src[i][1].y);
        }

        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 2));
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 1));

        if constexpr (!reset) accum = op::template op<T>(accum, src_accum);

        dst_accum = packed_shfl_sync(kittens::MASK_ALL, accum, 0);
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        T accum = src[0][0];
        #pragma unroll
        for(int i = 1; i < RV::outer_dim; i++) {
            if (i < RV::outer_dim-1 || i*kittens::TILE_ROW_DIM<T>*2 + laneid < RV::length) {
                accum = op::template op<T>(accum, src[i][0]);
            }
        }
        if(RV::length > 16) accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 16));
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 8));
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 4));
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 2));
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 1));
        if constexpr (!reset) accum = op::template op<T>(accum, src_accum);
        dst_accum = packed_shfl_sync(kittens::MASK_ALL, accum, 0);
    }
}









template<ducks::rv::all RV>
__device__ static inline void max(typename base_types::packing<typename RV::dtype>::unpacked_type &max_val, const RV &src) {
    reduce<base_ops::max, RV, true>(max_val, src, max_val);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type max(const RV &src) {
    typename base_types::packing<typename RV::dtype>::unpacked_type max_val;
    reduce<base_ops::max, RV, true>(max_val, src, max_val);
    return max_val;
}








template<ducks::rv::all RV>
__device__ static inline void min(typename base_types::packing<typename RV::dtype>::unpacked_type &min_val, const RV &src) {
    reduce<base_ops::min, RV, true>(min_val, src, min_val);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type min(const RV &src) {
    typename base_types::packing<typename RV::dtype>::unpacked_type min_val;
    reduce<base_ops::min, RV, true>(min_val, src, min_val);
    return min_val;
}








template<ducks::rv::all RV>
__device__ static inline void sum(typename base_types::packing<typename RV::dtype>::unpacked_type &sum_val, const RV &src) {
    reduce<base_ops::sum, RV, true>(sum_val, src, sum_val);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type sum(const RV &src) {
    typename base_types::packing<typename RV::dtype>::unpacked_type sum_val;
    reduce<base_ops::sum, RV, true>(sum_val, src, sum_val);
    return sum_val;
}








template<ducks::rv::all RV>
__device__ static inline void prod(typename base_types::packing<typename RV::dtype>::unpacked_type &prod_val, const RV &src) {
    reduce<base_ops::mul, RV, true>(prod_val, src, prod_val);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type prod(const RV &src) {
    typename base_types::packing<typename RV::dtype>::unpacked_type prod_val;
    reduce<base_ops::mul, RV, true>(prod_val, src, prod_val);
    return prod_val;
}











template<ducks::rv::all RV>
__device__ static inline void max(typename base_types::packing<typename RV::dtype>::unpacked_type &max_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::max, RV, false>(max_val, src, src_accum);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type max(const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    typename base_types::packing<typename RV::dtype>::unpacked_type max_val;
    reduce<base_ops::max, RV, false>(max_val, src, src_accum);
    return max_val;
}









template<ducks::rv::all RV>
__device__ static inline void min(typename base_types::packing<typename RV::dtype>::unpacked_type &min_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::min, RV, false>(min_val, src, src_accum);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type min(const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    typename base_types::packing<typename RV::dtype>::unpacked_type min_val;
    reduce<base_ops::min, RV, false>(min_val, src, src_accum);
    return min_val;
}









template<ducks::rv::all RV>
__device__ static inline void sum(typename base_types::packing<typename RV::dtype>::unpacked_type &sum_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::sum, RV, false>(sum_val, src, src_accum);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type sum(const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    typename base_types::packing<typename RV::dtype>::unpacked_type sum_val;
    reduce<base_ops::sum, RV, false>(sum_val, src, src_accum);
    return sum_val;
}









template<ducks::rv::all RV>
__device__ static inline void prod(typename base_types::packing<typename RV::dtype>::unpacked_type &prod_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::mul, RV, false>(prod_val, src, src_accum);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type prod(const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    typename base_types::packing<typename RV::dtype>::unpacked_type prod_val;
    reduce<base_ops::mul, RV, false>(prod_val, src, src_accum);
    return prod_val;
}
