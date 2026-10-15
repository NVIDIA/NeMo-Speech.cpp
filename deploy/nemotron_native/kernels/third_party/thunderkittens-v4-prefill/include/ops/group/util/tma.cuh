
















__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes) {
    if(laneid() == 0) {
        ::kittens::tma::expect_bytes(bar, bytes);
    }
}





template<typename T, typename... args>
__device__ static inline void expect(semaphore& bar, const T& _1, const args&... _2) {
    expect_bytes(bar, size_bytes<T, args...>);
}






__device__ static inline void store_commit_group() {
    asm volatile("cp.async.bulk.commit_group;");
}





template <int N=0>
__device__ static inline void store_async_wait() {
    asm volatile (
        "cp.async.bulk.wait_group %0;"
        :
        : "n"(N)
        : "memory"
    );
}





template <int N=0>
__device__ static inline void store_async_read_wait() {
    asm volatile (
        "cp.async.bulk.wait_group.read %0;"
        :
        : "n"(N)
        : "memory"
    );
}



__device__ static inline void load_async(void *dst, void *src, uint32_t size_bytes, semaphore& bar) {
    if(laneid() == 0) {
        ::kittens::tma::load_async(dst, src, size_bytes, bar);
    }
}
template<typename T>
__device__ static inline void load_async(T &dst, T &src, uint32_t size_bytes, semaphore& bar) {
    load_async(reinterpret_cast<void*>(&dst), reinterpret_cast<void*>(&src), size_bytes, bar);
}

__device__ static inline void store_async(void *dst, void *src, uint32_t size_bytes) {
    if(laneid() == 0) {
        ::kittens::tma::store_async(dst, src, size_bytes);
    }
}
template<typename T>
__device__ static inline void store_async(T &dst, T &src, uint32_t size_bytes) {
    store_async(reinterpret_cast<void*>(&dst), reinterpret_cast<void*>(&src), size_bytes);
}
