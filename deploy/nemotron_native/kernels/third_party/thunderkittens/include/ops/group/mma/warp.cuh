





















__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                        const bf16_2 &a0, const bf16_2 &a1, const bf16_2 &a2, const bf16_2 &a3,
                                        const bf16_2 &b0, const bf16_2 &b1,
                                        const float2 &c0, const float2 &c1                                    ) {
    asm volatile(

        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{%0, %1, %2, %3}, " \
        "{%4, %5, %6, %7}, " \
        "{%8, %9}, " \
        "{%10, %11, %12, %13};"


    :   "+f"(d0.x), "+f"(d0.y),
        "+f"(d1.x), "+f"(d1.y)


    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),


        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),


        "f"(c0.x), "f"(c0.y),
        "f"(c1.x), "f"(c1.y)
    );
}

















__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                        const half_2 &a0, const half_2 &a1, const half_2 &a2, const half_2 &a3,
                                        const half_2 &b0, const half_2 &b1,
                                        const float2 &c0, const float2 &c1                                    ) {
    asm volatile(

        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
        "{%0, %1, %2, %3}, " \
        "{%4, %5, %6, %7}, " \
        "{%8, %9}, " \
        "{%10, %11, %12, %13};"


    :   "+f"(d0.x), "+f"(d0.y),
        "+f"(d1.x), "+f"(d1.y)


    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),


        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),


        "f"(c0.x), "f"(c0.y),
        "f"(c1.x), "f"(c1.y)
    );
}

















__device__ static inline void hmma16816(      half_2 &d0,       half_2 &d1,
                                        const half_2 &a0, const half_2 &a1, const half_2 &a2, const half_2 &a3,
                                        const half_2 &b0, const half_2 &b1,
                                        const half_2 &c0, const half_2 &c1                                    ) {
    asm volatile(

        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 " \
        "{%0, %1}, " \
        "{%2, %3, %4, %5}, " \
        "{%6, %7}, " \
        "{%8, %9};"


    :   "=r"(*(uint32_t*)(&d0)), "=r"(*(uint32_t*)(&d1))


    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),


        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),


        "r"(*(uint32_t*)(&c0)), "r"(*(uint32_t*)(&c1))
    );
}

#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)


















__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                       const fp8e4m3_4 &a0, const fp8e4m3_4 &a1,
                                       const fp8e4m3_4 &a2, const fp8e4m3_4 &a3,
                                       const fp8e4m3_4 &b0, const fp8e4m3_4 &b1,
                                       const float2 &c0, const float2 &c1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"


        : "+f"(d0.x), "+f"(d0.y),
          "+f"(d1.x), "+f"(d1.y)


        : "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
          "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),


        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),


        "f"(c0.x), "f"(c0.y),
        "f"(c1.x), "f"(c1.y)
    );
}
#endif






__device__ static inline void imma16832(int2 &d0, int2 &d1,
                                        const int8_4 &a0, const int8_4 &a1, const int8_4 &a2, const int8_4 &a3,
                                        const int8_4 &b0, const int8_4 &b1,
                                        const int2 &c0, const int2 &c1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
    :   "=r"(d0.x), "=r"(d0.y),
        "=r"(d1.x), "=r"(d1.y)
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),
        "r"(c0.x), "r"(c0.y),
        "r"(c1.x), "r"(c1.y)
    );
}
__device__ static inline void imma16832(int2 &d0, int2 &d1,
                                        const int8_4 &a0, const int8_4 &a1, const int8_4 &a2, const int8_4 &a3,
                                        const uint8_4 &b0, const uint8_4 &b1,
                                        const int2 &c0, const int2 &c1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.u8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
    :   "=r"(d0.x), "=r"(d0.y),
        "=r"(d1.x), "=r"(d1.y)
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),
        "r"(c0.x), "r"(c0.y),
        "r"(c1.x), "r"(c1.y)
    );
}
__device__ static inline void imma16832(int2 &d0, int2 &d1,
                                        const uint8_4 &a0, const uint8_4 &a1, const uint8_4 &a2, const uint8_4 &a3,
                                        const int8_4 &b0, const int8_4 &b1,
                                        const int2 &c0, const int2 &c1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
    :   "=r"(d0.x), "=r"(d0.y),
        "=r"(d1.x), "=r"(d1.y)
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),
        "r"(c0.x), "r"(c0.y),
        "r"(c1.x), "r"(c1.y)
    );
}
__device__ static inline void imma16832(int2 &d0, int2 &d1,
                                        const uint8_4 &a0, const uint8_4 &a1, const uint8_4 &a2, const uint8_4 &a3,
                                        const uint8_4 &b0, const uint8_4 &b1,
                                        const int2 &c0, const int2 &c1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
    :   "=r"(d0.x), "=r"(d0.y),
        "=r"(d1.x), "=r"(d1.y)
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),
        "r"(c0.x), "r"(c0.y),
        "r"(c1.x), "r"(c1.y)
    );
}












__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::row> &d,
                                    const rt_base<bf16,  ducks::rt_layout::row> &a,
                                    const rt_base<bf16,  ducks::rt_layout::col> &b,
                                    const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}












__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::row> &d,
                                    const rt_base<half,  ducks::rt_layout::row> &a,
                                    const rt_base<half,  ducks::rt_layout::col> &b,
                                    const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)











__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::row> &d,
                                    const rt_base<fp8e4m3,  ducks::rt_layout::row> &a,
                                    const rt_base<fp8e4m3,  ducks::rt_layout::col> &b,
                                    const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif











__device__ static inline void mma_AB_base(rt_base<half, ducks::rt_layout::row> &d,
                                    const rt_base<half, ducks::rt_layout::row> &a,
                                    const rt_base<half, ducks::rt_layout::col> &b,
                                    const rt_base<half, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}











__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<bf16,  ducks::rt_layout::row> &a,
                                     const rt_base<bf16,  ducks::rt_layout::row> &b,
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}












__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<half,  ducks::rt_layout::row> &a,
                                     const rt_base<half,  ducks::rt_layout::row> &b,
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)











__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::row> &a,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::row> &b,
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif
template<typename A, typename B>
__device__ static inline void mma_ABt_base(rt_base<int, ducks::rt_layout::row> &d,
                                     const rt_base<A,   ducks::rt_layout::row> &a,
                                     const rt_base<B,   ducks::rt_layout::row> &b,
                                     const rt_base<int, ducks::rt_layout::row> &c) {
    static_assert((std::is_same_v<A, int8> || std::is_same_v<A, uint8>) &&
                  (std::is_same_v<B, int8> || std::is_same_v<B, uint8>));
    imma16832(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    imma16832(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}













__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<bf16,  ducks::rt_layout::col> &a,
                                     const rt_base<bf16,  ducks::rt_layout::col> &b,
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}












__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<half,  ducks::rt_layout::col> &a,
                                     const rt_base<half,  ducks::rt_layout::col> &b,
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)











__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::col> &a,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::col> &b,
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif












__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::row> &d,
                                      const rt_base<bf16,  ducks::rt_layout::col> &a,
                                      const rt_base<bf16,  ducks::rt_layout::row> &b,
                                      const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}












__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::row> &d,
                                      const rt_base<half,  ducks::rt_layout::col> &a,
                                      const rt_base<half,  ducks::rt_layout::row> &b,
                                      const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)











__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::row> &d,
                                      const rt_base<fp8e4m3,  ducks::rt_layout::col> &a,
                                      const rt_base<fp8e4m3,  ducks::rt_layout::row> &b,
                                      const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif















template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b,
                               const C &c) {
    KITTENS_CHECK_WARP
    static_assert(D::rows == A::rows && D::cols == B::cols);
    static_assert(A::cols == B::rows);
    static_assert(D::rows == C::rows && D::cols == C::cols);
    #if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)   ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AB_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_AB_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}














template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    KITTENS_CHECK_WARP
    static_assert(D::rows == A::rows && D::cols == B::rows);
    static_assert(A::cols == B::cols);
    static_assert(D::rows == C::rows && D::cols == C::cols);
    #if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)  ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, int> &&
            (std::is_same_v<typename A::T, int8> || std::is_same_v<typename A::T, uint8>) &&
            (std::is_same_v<typename B::T, int8> || std::is_same_v<typename B::T, uint8>) &&
            std::is_same_v<typename C::T, int>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>) ||
        (std::is_same_v<typename D::T, int> &&
            (std::is_same_v<typename A::T, int8> || std::is_same_v<typename A::T, uint8>) &&
            (std::is_same_v<typename B::T, int8> || std::is_same_v<typename B::T, uint8>) &&
            std::is_same_v<typename C::T, int>)
    );
    #endif
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_ABt_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_ABt_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}














template<ducks::rt::row_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    KITTENS_CHECK_WARP
    static_assert(D::rows == A::cols && D::cols == B::cols);
    static_assert(A::rows == B::rows);
    static_assert(D::rows == C::rows && D::cols == C::cols);
    #if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)   ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtB_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtB_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}














template<ducks::rt::row_layout D, ducks::rt::col_layout A, ducks::rt::row_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b,
                                 const C &c) {
    KITTENS_CHECK_WARP
    static_assert(D::rows == A::cols && D::cols == B::rows);
    static_assert(A::rows == B::cols);
    static_assert(D::rows == C::rows && D::cols == C::cols);
    #if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)   ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtBt_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtBt_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}

template<int trans_A, int trans_B, ducks::rt::all D, ducks::rt::all A, ducks::rt::all B, ducks::rt::all C>
__device__ static inline void mma(D &d,
                                  const A &a,
                                  const B &b,
                                  const C &c) {
    KITTENS_CHECK_WARP
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mma_AtBt(d, a, b, c);
        } else {
            mma_AtB(d, a, b, c);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mma_ABt(d, a, b, c);
        } else {
            mma_AB(d, a, b, c);
        }
    }
}
template<int trans_A, int trans_B, ducks::rt::all A, ducks::rt::all B, ducks::rt::all C>
__device__ static inline C mma(const A &a,
                               const B &b,
                               const C &c) {
    KITTENS_CHECK_WARP
    C d;
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mma_AtBt(d, a, b, c);
        } else {
            mma_AtB(d, a, b, c);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mma_ABt(d, a, b, c);
        } else {
            mma_AB(d, a, b, c);
        }
    }
    return d;
}























template<int N, int K, int M>
__device__ static inline void mma_AB(crt_hf<N, M, ducks::rt_layout::row> &d,
                               const crt_hf<N, K, ducks::rt_layout::row> &a,
                               const crt_hf<K, M, ducks::rt_layout::col> &b,
                               const crt_hf<N, M, ducks::rt_layout::row> &c) {
    KITTENS_CHECK_WARP


    ::kittens::group<1>::copy(d.real, c.real);
    ::kittens::group<1>::copy(d.imag, c.imag);


    rt_hf<N, K, ducks::rt_layout::row> tmp;

    constexpr half factor = std::bit_cast<__half>(uint16_t(0xFB80));
    ::kittens::group<1>::mul(tmp, a.imag, factor);
    mma_AB(d.real, a.real, b.real, d.real);
    mma_AB(d.real, tmp, b.imag, d.real);

    mma_AB(d.imag, a.real, b.imag, d.imag);
    mma_AB(d.imag, a.imag, b.real, d.imag);
}














template<int N, int K, int M>
__device__ static inline void mma_AB(crt_fl<N, M, ducks::rt_layout::row> &d,
                               const crt_bf<N, K, ducks::rt_layout::row> &a,
                               const crt_bf<K, M, ducks::rt_layout::col> &b,
                               const crt_fl<N, M, ducks::rt_layout::row> &c) {
    KITTENS_CHECK_WARP


    ::kittens::group<1>::copy(d.real, c.real);
    ::kittens::group<1>::copy(d.imag, c.imag);


    kittens::rt_bf<N, K, ducks::rt_layout::row> tmp;

    constexpr bf16 factor = std::bit_cast<__nv_bfloat16>(uint16_t(0xBF80));
    ::kittens::group<1>::mul(tmp, a.imag, factor);
    mma_AB(d.real, a.real, b.real, d.real);
    mma_AB(d.real, tmp, b.imag, d.real);

    mma_AB(d.imag, a.real, b.imag, d.imag);
    mma_AB(d.imag, a.imag, b.real, d.imag);
}
