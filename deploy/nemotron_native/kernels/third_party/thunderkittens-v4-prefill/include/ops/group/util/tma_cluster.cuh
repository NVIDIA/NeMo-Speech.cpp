










__device__ static inline void wait(semaphore& bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

__device__ static inline bool try_wait(semaphore &bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint32_t success;

    asm volatile(
        "{\n"
        ".reg .pred P1; \n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%1], %2; \n"
        "selp.b32 %0, 1, 0, P1; \n"
        "}\n"
        : "=r"(success)
        : "r"(mbar_ptr), "r"(kPhaseBit)
        : "memory"
    );

    return static_cast<bool>(success);
}















__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::expect_bytes(bar, bytes);
    }
}
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes, int dst_cta) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::expect_bytes(bar, bytes, dst_cta);
    }
}















template<typename T, typename... args>
__device__ static inline void expect(semaphore& bar, const T& _1, const args&... _2) {
    expect_bytes(bar, size_bytes<T, args...>);
}










__device__ static inline void arrive(semaphore& bar, int dst_cta, uint32_t count=1) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::arrive(bar, dst_cta, count);
    }
}



__device__ static inline void load_async(void *dst, void *src, uint32_t size_bytes, semaphore& bar, uint16_t cta_mask) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::load_async(dst, src, size_bytes, bar, cta_mask);
    }
}
template<typename T>
__device__ static inline void load_async(T &dst, T &src, uint32_t size_bytes, semaphore& bar, uint16_t cta_mask) {
    load_async(reinterpret_cast<void*>(&dst), reinterpret_cast<void*>(&src), size_bytes, bar, cta_mask);
}
