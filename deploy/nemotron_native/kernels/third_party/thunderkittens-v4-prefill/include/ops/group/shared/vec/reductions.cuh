





















template<typename op, ducks::sv::all SV, bool reset>
__device__ static inline void reduce(typename SV::dtype &dst_accum, const SV &src, const typename SV::dtype &src_accum) {
    if constexpr (GROUP_WARPS == 1) {
        using T = SV::dtype;
        int lane = laneid();
        T accum;
        if(lane < SV::length) accum = src[lane];
        __syncwarp();
        for(int i = lane+kittens::WARP_THREADS; i < SV::length; i+=kittens::WARP_THREADS) {
            accum = op::template op<T>(accum, src[i]);
        }
        __syncwarp();

        if constexpr (SV::length > 16) {
            accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 16));
            __syncwarp();
        }
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 8));
        __syncwarp();
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 4));
        __syncwarp();
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 2));
        __syncwarp();
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 1));
        __syncwarp();
        if constexpr (!reset) accum = op::template op<T>(accum, src_accum);

        dst_accum = packed_shfl_sync(kittens::MASK_ALL, accum, 0);
    }
    else {
        ::kittens::group<1>::reduce<op, SV, reset>(dst_accum, src, src_accum);
    }
}










template<ducks::sv::all SV>
__device__ static inline void max(typename SV::dtype &max_val, const SV &src) {
    reduce<base_ops::max, SV, true>(max_val, src, max_val);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype max(const SV &src) {
    typename SV::dtype max_val;
    reduce<base_ops::max, SV, true>(max_val, src, max_val);
    return max_val;
}








template<ducks::sv::all SV>
__device__ static inline void min(typename SV::dtype &min_val, const SV &src) {
    reduce<base_ops::min, SV, true>(min_val, src, min_val);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype min(const SV &src) {
    typename SV::dtype min_val;
    reduce<base_ops::min, SV, true>(min_val, src, min_val);
    return min_val;
}








template<ducks::sv::all SV>
__device__ static inline void sum(typename SV::dtype &sum_val, const SV &src) {
    reduce<base_ops::sum, SV, true>(sum_val, src, sum_val);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype sum(const SV &src) {
    typename SV::dtype sum_val;
    reduce<base_ops::sum, SV, true>(sum_val, src, sum_val);
    return sum_val;
}








template<ducks::sv::all SV>
__device__ static inline void prod(typename SV::dtype &prod_val, const SV &src) {
    reduce<base_ops::mul, SV, true>(prod_val, src, prod_val);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype prod(const SV &src) {
    typename SV::dtype prod_val;
    reduce<base_ops::mul, SV, true>(prod_val, src, prod_val);
    return prod_val;
}











template<ducks::sv::all SV>
__device__ static inline void max(typename SV::dtype &max_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::max, SV, false>(max_val, src, src_accum);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype max(const SV &src, const typename SV::dtype &src_accum) {
    typename SV::dtype max_val;
    reduce<base_ops::max, SV, false>(max_val, src, src_accum);
    return max_val;
}









template<ducks::sv::all SV>
__device__ static inline void min(typename SV::dtype &min_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::min, SV, false>(min_val, src, src_accum);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype min(const SV &src, const typename SV::dtype &src_accum) {
    typename SV::dtype min_val;
    reduce<base_ops::min, SV, false>(min_val, src, src_accum);
    return min_val;
}









template<ducks::sv::all SV>
__device__ static inline void sum(typename SV::dtype &sum_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::sum, SV, false>(sum_val, src, src_accum);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype sum(const SV &src, const typename SV::dtype &src_accum) {
    typename SV::dtype sum_val;
    reduce<base_ops::sum, SV, false>(sum_val, src, src_accum);
    return sum_val;
}









template<ducks::sv::all SV>
__device__ static inline void prod(typename SV::dtype &prod_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::mul, SV, false>(prod_val, src, src_accum);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype prod(const SV &src, const typename SV::dtype &src_accum) {
    typename SV::dtype prod_val;
    reduce<base_ops::mul, SV, false>(prod_val, src, src_accum);
    return prod_val;
}
