















template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load(SV &dst, const GL &src, const COORD &idx) {
    constexpr uint32_t elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr uint32_t total_calls = SV::length / elem_per_transfer;
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    #pragma unroll
    for(uint32_t i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        if(i * elem_per_transfer < SV::length) {
            float4 tmp;
            move<float4>::ldg(tmp, (float4*)&src_ptr[i*elem_per_transfer]);
            move<float4>::sts(dst_ptr + sizeof(typename SV::dtype)*i*elem_per_transfer, tmp);
        }
    }
}












template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store(GL &dst, const SV &src, const COORD &idx) {
    constexpr uint32_t elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr uint32_t total_calls = SV::length / elem_per_transfer;
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst[(idx.template unit_coord<-1, 3>())];
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    #pragma unroll
    for(uint32_t i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        if(i * elem_per_transfer < SV::length) {
            float4 tmp;
            move<float4>::lds(tmp, src_ptr + sizeof(typename SV::dtype)*i*elem_per_transfer);
            move<float4>::stg((float4*)&dst_ptr[i*elem_per_transfer], tmp);
        }
    }
}









template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx) {
    constexpr uint32_t elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr uint32_t total_calls = SV::length / elem_per_transfer;
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    #pragma unroll
    for(uint32_t i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        if(i * elem_per_transfer < SV::length) {
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                :: "r"(dst_ptr + (uint32_t)sizeof(typename SV::dtype)*i*elem_per_transfer), "l"((uint64_t)&src_ptr[i*elem_per_transfer])
                : "memory"
            );
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}










template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async_small(SV &dst, const GL &src, const COORD &idx) {
    constexpr uint32_t elem_per_transfer = sizeof(float) / sizeof(typename SV::dtype);
    constexpr uint32_t total_calls = SV::length / elem_per_transfer;
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    #pragma unroll
    for(uint32_t i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        if(i * elem_per_transfer < SV::length) {
            asm volatile(
                "cp.async.ca.shared.global.L2::128B [%0], [%1], 4;\n"
                :: "r"(dst_ptr + (uint32_t)sizeof(typename SV::dtype)*i*elem_per_transfer), "l"((uint64_t)&src_ptr[i*elem_per_transfer])
                : "memory"
            );
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
