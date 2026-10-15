














template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void prefetch(SV &dst, const GL &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        ::kittens::detail::tma::vec_prefetch_tma_internal<policy>(tma_ptr, tma_coord);
    }
}
__KITTENS_TMA_DEFINE_DEFAULT_LOAD_CACHE_VEC__(prefetch)














template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_async(const GL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_async)

template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_async(const PGL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_async)












template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_add_async(const GL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_add_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_add_async)

template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_add_async(const PGL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_add_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_add_async)












template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_min_async(const GL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_min_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_min_async)

template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_min_async(const PGL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_min_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_min_async)











template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_max_async(const GL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_max_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_max_async)

template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_max_async(const PGL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_max_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_max_async)












template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore& bar) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t dst_i_ptr = dst_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_load_async_tma_internal<policy>(tma_ptr, dst_i_ptr, mbar_ptr, tma_coord);
    }
}
__KITTENS_TMA_DEFINE_SEMAPHORE_CACHE_VEC__(load_async)
