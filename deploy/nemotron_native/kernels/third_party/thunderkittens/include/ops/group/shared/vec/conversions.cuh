


















template<ducks::sv::all SV1, ducks::sv::all SV2>
__device__ static inline void copy(SV1 &dst, const SV2 &src) {
    static_assert(SV1::length == SV2::length, "Source and destination vectors must have the same length.");
    #pragma unroll
    for(int i = laneid(); i < SV1::length; i+=GROUP_THREADS) {
        dst[i] = base_types::convertor<typename SV1::dtype, typename SV2::dtype>::convert(src[i]);
    }
}
