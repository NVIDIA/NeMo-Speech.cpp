




















template<ducks::rt::row_layout D>
__device__ static inline void mma_fence(D &dst) {
    KITTENS_CHECK_WARPGROUP
    #pragma unroll
    for(int i = 0; i < D::height; i++) {
        #pragma unroll
        for(int j = 0; j < D::width; j++) {
            #pragma unroll
            for(int k = 0; k < D::packed_per_tile; k++) {
                if constexpr(std::is_same_v<typename D::T, float>) {
                    asm volatile("" : "+f"(dst.tiles[i][j].data[k].x) :: "memory");
                    asm volatile("" : "+f"(dst.tiles[i][j].data[k].y) :: "memory");
                } else if constexpr(std::is_same_v<typename D::T, int>) {
                    asm volatile("" : "+r"(dst.tiles[i][j].data[k].x) :: "memory");
                    asm volatile("" : "+r"(dst.tiles[i][j].data[k].y) :: "memory");
                } else {
                    asm volatile("" : "+r"(*(uint32_t*)&dst.tiles[i][j].data[k]) :: "memory");
                }
            }
        }
    }
    asm volatile ("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
}
template<ducks::crt::row_layout D>
__device__ static inline void mma_fence(D &dst) {
    KITTENS_CHECK_WARPGROUP
    #pragma unroll
    for(int i = 0; i < D::height; i++) {
        #pragma unroll
        for(int j = 0; j < D::width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.real.packed_per_tile; k++) {
                if constexpr(std::is_same_v<typename D::T, float>) {
                    asm volatile("" : "+f"(dst.real.tiles[i][j].data[k].x) :: "memory");
                    asm volatile("" : "+f"(dst.real.tiles[i][j].data[k].y) :: "memory");
                    asm volatile("" : "+f"(dst.imag.tiles[i][j].data[k].x) :: "memory");
                    asm volatile("" : "+f"(dst.imag.tiles[i][j].data[k].y) :: "memory");
                } else if constexpr(std::is_same_v<typename D::T, int>) {
                    asm volatile("" : "+r"(dst.real.tiles[i][j].data[k].x) :: "memory");
                    asm volatile("" : "+r"(dst.real.tiles[i][j].data[k].y) :: "memory");
                    asm volatile("" : "+r"(dst.imag.tiles[i][j].data[k].x) :: "memory");
                    asm volatile("" : "+r"(dst.imag.tiles[i][j].data[k].y) :: "memory");
                } else {
                    asm volatile("" : "+r"(*(uint32_t*)&dst.real.tiles[i][j].data[k]) :: "memory");
                    asm volatile("" : "+r"(*(uint32_t*)&dst.imag.tiles[i][j].data[k]) :: "memory");
                }
            }
        }
    }
    asm volatile ("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
}
template<typename T=kittens::ducks::default_type>
__device__ static inline void mma_fence() {
    KITTENS_CHECK_WARPGROUP
    asm volatile ("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
}




template<typename T=kittens::ducks::default_type>
__device__ static inline void mma_commit_group() {
    KITTENS_CHECK_WARPGROUP
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}








template<int N=0>
__device__ static inline void mma_async_wait() {
    KITTENS_CHECK_WARPGROUP
    asm volatile ("wgmma.wait_group.sync.aligned %0;" : : "n"(N) : "memory");
}












































template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b) {

    KITTENS_CHECK_WARPGROUP
    constexpr int M_DIV_4 = A::height;
    static_assert(D::height == M_DIV_4);
    constexpr int N = B::cols / kittens::TILE_COL_DIM<typename B::T>;
    static_assert(D::width == N);
    constexpr int K = A::width;
    static_assert(B::rows/kittens::TILE_ROW_DIM<typename B::T> == K);
    static_assert(std::is_same_v<typename A::T, typename B::T>);


    using T_AB = A::T;
    using T_D  = D::T;
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_AB, int8> && !std::is_same_v<T_AB, uint8>, "Transpose is not supported for integer WGMMA; use mma_ABt / mm_ABt.");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 1>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc(b);

    if constexpr (fence) { mma_fence(d); }


    #pragma unroll
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_D>, TILE_COL_DIM<T_D>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d, m);
        base::rt_st(
            d_ref,
            a.tiles[m][0],
            b_desc.chunk_descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.tiles[m][k],
                b_desc.chunk_descriptor(k),
                1
            );
        }
    }
    mma_commit_group();
}
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d,
                              const A &a,
                              const B &b) {
    mma_AB<D, A, B, 1, 0>(d, a, b);
}

template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b) {

    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::rows / kittens::TILE_ROW_DIM<typename A::T>;
    static_assert(M == 4);
    static_assert(D::height == 1);
    constexpr int N = B::cols / kittens::TILE_COL_DIM<typename B::T>;
    constexpr int K = A::cols / kittens::TILE_COL_DIM<typename A::T>;
    static_assert(B::rows / kittens::TILE_ROW_DIM<typename B::T> == K);
    static_assert(std::is_same_v<typename A::T, typename B::T>);


    using T_AB = A::T;
    using T_D  = D::T;
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_AB, int8> && !std::is_same_v<T_AB, uint8>, "Transpose is not supported for integer WGMMA; use mma_ABt / mm_ABt.");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 0, 1>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc(b);

    if constexpr (fence) { mma_fence(d); }


    base::st_st(
        d,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            a_desc.chunk_descriptor(k),
            b_desc.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group();
}
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d,
                              const A &a,
                              const B &b) {
    mma_AB<D, A, B, 1, 0>(d, a, b);
}
















template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {

    KITTENS_CHECK_WARPGROUP
    constexpr int M_DIV_4 = A::height;
    static_assert(D::height == M_DIV_4);
    constexpr int N = B::rows / kittens::TILE_ROW_DIM<typename B::T>;
    constexpr int K = A::width;
    static_assert(B::cols/kittens::TILE_COL_DIM<typename B::T> == K);
    static_assert(std::is_same_v<typename A::T, typename B::T>);


    using T_AB = A::T;
    using T_D  = D::T;
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 0>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc(b);

    if constexpr (fence) { mma_fence(d); }


    #pragma unroll
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_D>, TILE_COL_DIM<T_D>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d, m);
        base::rt_st(
            d_ref,
            a.tiles[m][0],
            b_desc.chunk_descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.tiles[m][k],
                b_desc.chunk_descriptor(k),
                1
            );
        }
    }
    mma_commit_group();
}
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d,
                               const A &a,
                               const B &b) {
    mma_ABt<D, A, B, 1, 0>(d, a, b);
}
















template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {

    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::rows / kittens::TILE_ROW_DIM<typename A::T>;
    static_assert(M == 4);
    static_assert(D::height == 1);
    constexpr int N = B::rows / kittens::TILE_ROW_DIM<typename B::T>;
    constexpr int K = A::cols / kittens::TILE_COL_DIM<typename A::T>;
    static_assert(B::cols/kittens::TILE_COL_DIM<typename B::T> == K);
    static_assert(std::is_same_v<typename A::T, typename B::T>);


    using T_AB = A::T;
    using T_D  = D::T;
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 0>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc(b);

    if constexpr (fence) { mma_fence(d); }


    base::st_st(
        d,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            a_desc.chunk_descriptor(k),
            b_desc.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group();
}
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d,
                               const A &a,
                               const B &b) {
    mma_ABt<D, A, B, 1, 0>(d, a, b);
}
















template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b) {

    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::cols / kittens::TILE_COL_DIM<typename A::T>;
    static_assert(M == 4);
    static_assert(D::height == 1);
    constexpr int N = B::cols / kittens::TILE_COL_DIM<typename B::T>;
    constexpr int K = A::rows / kittens::TILE_ROW_DIM<typename A::T>;
    static_assert(B::rows/kittens::TILE_ROW_DIM<typename B::T> == K);
    static_assert(std::is_same_v<typename A::T, typename B::T>);


    using T_AB = A::T;
    using T_D  = D::T;
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_AB, int8> && !std::is_same_v<T_AB, uint8>, "Transpose is not supported for integer WGMMA; use mma_ABt / mm_ABt.");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 1, 1>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc(b);

    if constexpr (fence) { mma_fence(d); }


    base::st_st(
        d,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            a_desc.chunk_descriptor(k),
            b_desc.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group();
}
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtB(D &d,
                               const A &a,
                               const B &b) {
    mma_AtB<D, A, B, 1, 0>(d, a, b);
}












template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b) {

    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::cols / kittens::TILE_COL_DIM<typename A::T>;
    static_assert(M == 4);
    static_assert(D::height == 1);
    constexpr int N = B::rows / kittens::TILE_ROW_DIM<typename B::T>;
    constexpr int K = A::rows / kittens::TILE_ROW_DIM<typename A::T>;
    static_assert(B::cols/kittens::TILE_COL_DIM<typename B::T> == K);
    static_assert(std::is_same_v<typename A::T, typename B::T>);


    using T_AB = A::T;
    using T_D  = D::T;
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_AB, int8> && !std::is_same_v<T_AB, uint8>, "Transpose is not supported for integer WGMMA; use mma_ABt / mm_ABt.");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 1, 0>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc(b);

    if constexpr (fence) { mma_fence(d); }


    base::st_st(
        d,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            a_desc.chunk_descriptor(k),
            b_desc.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group();
}
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtBt(D &d,
                                const A &a,
                                const B &b) {
    mma_AtBt<D, A, B, 1, 0>(d, a, b);
}















































template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b) {

    KITTENS_CHECK_WARPGROUP
    constexpr int M_DIV_4 = A::height;
    static_assert(D::height == M_DIV_4);
    constexpr int N = B::cols / kittens::TILE_COL_DIM<typename B::T>;
    constexpr int K = A::width;
    static_assert(B::rows/kittens::TILE_ROW_DIM<typename B::T> == K);
    static_assert(std::is_same_v<typename A::T, typename B::T>);


    using T_AB = A::T;
    using T_D  = D::T;
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_AB, int8> && !std::is_same_v<T_AB, uint8>, "Transpose is not supported for integer WGMMA; use mma_ABt / mm_ABt.");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 1>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_real(b.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_imag(b.imag);

    if constexpr (fence) { mma_fence(d); }


    #pragma unroll
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_D>, TILE_COL_DIM<T_D>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d.real, m);
        base::rt_st(
            d_ref,
            a.real.tiles[m][0],
            b_desc_real.chunk_descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.real.tiles[m][k],
                b_desc_real.chunk_descriptor(k),
                1
            );
        }
        #pragma unroll
        for(int k = 0; k < K; k++) {
            base::rt_st<-1>(
                d_ref,
                a.imag.tiles[m][k],
                b_desc_imag.chunk_descriptor(k),
                1
            );
        }
    }
    #pragma unroll
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_AB>, TILE_COL_DIM<T_AB>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d.imag, m);
        base::rt_st(
            d_ref,
            a.real.tiles[m][0],
            b_desc_imag.chunk_descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.real.tiles[m][k],
                b_desc_imag.chunk_descriptor(k),
                1
            );
        }
        #pragma unroll
        for(int k = 0; k < K; k++) {
            base::rt_st(
                d_ref,
                a.imag.tiles[m][k],
                b_desc_real.chunk_descriptor(k),
                1
            );
        }
    }
    mma_commit_group();
}
template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_AB(D &d,
                              const A &a,
                              const B &b) {
    mma_AB<D, A, B, 1, 0>(d, a, b);
}

template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b) {

    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::rows / kittens::TILE_ROW_DIM<typename A::T>;
    static_assert(M == 4);
    static_assert(D::height == 1);
    constexpr int N = B::cols / kittens::TILE_COL_DIM<typename B::T>;
    constexpr int K = A::cols / kittens::TILE_COL_DIM<typename A::T>;
    static_assert(B::rows/kittens::TILE_ROW_DIM<typename B::T> == K);
    static_assert(std::is_same_v<typename A::T, typename B::T>);


    using T_AB = A::T;
    using T_D  = D::T;
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_AB, int8> && !std::is_same_v<T_AB, uint8>, "Transpose is not supported for integer WGMMA; use mma_ABt / mm_ABt.");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 0, 1>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc_real(a.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc_imag(a.imag);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_real(b.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_imag(b.imag);

    if constexpr (fence) { mma_fence(d); }


    base::st_st(
        d.real,
        a_desc_real.chunk_descriptor(0),
        b_desc_real.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.real,
            a_desc_real.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st<-1>(
            d.real,
            a_desc_imag.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    base::st_st(
        d.imag,
        a_desc_real.chunk_descriptor(0),
        b_desc_imag.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_real.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_imag.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group();
}
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_AB(D &d,
                              const A &a,
                              const B &b) {
    mma_AB<D, A, B, 1, 0>(d, a, b);
}
















template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {

    KITTENS_CHECK_WARPGROUP
    constexpr int M_DIV_4 = A::height;
    static_assert(D::height == M_DIV_4);
    constexpr int N = B::rows / kittens::TILE_ROW_DIM<typename B::T>;
    constexpr int K = A::width;
    static_assert(B::cols/kittens::TILE_COL_DIM<typename B::T> == K);
    static_assert(std::is_same_v<typename A::T, typename B::T>);


    using T_AB = A::T;
    using T_D  = D::T;
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 0>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_real(b.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_imag(b.imag);

    if constexpr (fence) { mma_fence(d); }


    #pragma unroll
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_D>, TILE_ROW_DIM<T_D>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d.real, m);
        base::rt_st(
            d_ref,
            a.real.tiles[m][0],
            b_desc_real.chunk_descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.real.tiles[m][k],
                b_desc_real.chunk_descriptor(k),
                1
            );
        }
        #pragma unroll
        for(int k = 0; k < K; k++) {
            base::rt_st<-1>(
                d_ref,
                a.imag.tiles[m][k],
                b_desc_imag.chunk_descriptor(k),
                1
            );
        }
    }
    #pragma unroll
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_AB>, TILE_ROW_DIM<T_AB>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d.imag, m);
        base::rt_st(
            d_ref,
            a.real.tiles[m][0],
            b_desc_imag.chunk_descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.real.tiles[m][k],
                b_desc_imag.chunk_descriptor(k),
                1
            );
        }
        #pragma unroll
        for(int k = 0; k < K; k++) {
            base::rt_st(
                d_ref,
                a.imag.tiles[m][k],
                b_desc_real.chunk_descriptor(k),
                1
            );
        }
    }
    mma_commit_group();
}
template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_ABt(D &d,
                               const A &a,
                               const B &b) {
    mma_ABt<D, A, B, 1, 0>(d, a, b);
}
















template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {

    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::rows / kittens::TILE_ROW_DIM<typename A::T>;
    static_assert(M == 4);
    static_assert(D::height == 1);
    constexpr int N = B::rows / kittens::TILE_ROW_DIM<typename B::T>;
    constexpr int K = A::cols / kittens::TILE_COL_DIM<typename A::T>;
    static_assert(B::cols/kittens::TILE_COL_DIM<typename B::T> == K);
    static_assert(std::is_same_v<typename A::T, typename B::T>);


    using T_AB = A::T;
    using T_D  = D::T;
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 0, 0>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc_real(a.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc_imag(a.imag);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_real(b.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_imag(b.imag);

    if constexpr (fence) { mma_fence(d); }


    base::st_st(
        d.real,
        a_desc_real.chunk_descriptor(0),
        b_desc_real.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.real,
            a_desc_real.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st<-1>(
            d.real,
            a_desc_imag.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    base::st_st(
        d.imag,
        a_desc_real.chunk_descriptor(0),
        b_desc_imag.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_real.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_imag.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group();
}
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_ABt(D &d,
                               const A &a,
                               const B &b) {
    mma_ABt<D, A, B, 1, 0>(d, a, b);
}
















template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b) {

    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::cols / kittens::TILE_COL_DIM<typename A::T>;
    static_assert(M == 4);
    static_assert(D::height == 1);
    constexpr int N = B::cols / kittens::TILE_COL_DIM<typename B::T>;
    constexpr int K = A::rows / kittens::TILE_ROW_DIM<typename A::T>;
    static_assert(B::rows/kittens::TILE_ROW_DIM<typename B::T> == K);
    static_assert(std::is_same_v<typename A::T, typename B::T>);


    using T_AB = A::T;
    using T_D  = D::T;
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_AB, int8> && !std::is_same_v<T_AB, uint8>, "Transpose is not supported for integer WGMMA; use mma_ABt / mm_ABt.");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 1, 1>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc_real(a.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc_imag(a.imag);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_real(b.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_imag(b.imag);

    if constexpr (fence) { mma_fence(d); }


    base::st_st(
        d.real,
        a_desc_real.chunk_descriptor(0),
        b_desc_real.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.real,
            a_desc_real.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st<-1>(
            d.real,
            a_desc_imag.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    base::st_st(
        d.imag,
        a_desc_real.chunk_descriptor(0),
        b_desc_imag.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_real.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_imag.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group();
}
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_AtB(D &d,
                               const A &a,
                               const B &b) {
    mma_AtB<D, A, B, 1, 0>(d, a, b);
}












template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b) {

    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::cols / kittens::TILE_COL_DIM<typename A::T>;
    static_assert(M == 4);
    static_assert(D::height == 1);
    constexpr int N = B::rows / kittens::TILE_ROW_DIM<typename B::T>;
    constexpr int K = A::rows / kittens::TILE_ROW_DIM<typename A::T>;
    static_assert(B::cols/kittens::TILE_COL_DIM<typename B::T> == K);
    static_assert(std::is_same_v<typename A::T, typename B::T>);


    using T_AB = A::T;
    using T_D  = D::T;
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_AB, int8> && !std::is_same_v<T_AB, uint8>, "Transpose is not supported for integer WGMMA; use mma_ABt / mm_ABt.");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 1, 0>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc_real(a.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc_imag(a.imag);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_real(b.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_imag(b.imag);

    if constexpr (fence) { mma_fence(d); }


    base::st_st(
        d.real,
        a_desc_real.chunk_descriptor(0),
        b_desc_real.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.real,
            a_desc_real.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st<-1>(
            d.real,
            a_desc_imag.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    base::st_st(
        d.imag,
        a_desc_real.chunk_descriptor(0),
        b_desc_imag.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_real.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_imag.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group();
}
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_AtBt(D &d,
                                const A &a,
                                const B &b) {
    mma_AtBt<D, A, B, 1, 0>(d, a, b);
}



template<int trans_A, int trans_B, typename D, typename A, typename B>
__device__ static inline void mma(D &d,
                                  const A &a,
                                  const B &b) {
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mma_AtBt(d, a, b);
        } else {
            mma_AtB(d, a, b);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mma_ABt(d, a, b);
        } else {
            mma_AB(d, a, b);
        }
    }
}
template<int trans_A, int trans_B, typename D, typename A, typename B>
__device__ static inline void mm(D &d,
                                  const A &a,
                                  const B &b) {
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mm_AtBt(d, a, b);
        } else {
            mm_AtB(d, a, b);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mm_ABt(d, a, b);
        } else {
            mm_AB(d, a, b);
        }
    }
}
