




#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"
#include "sync.cuh"

namespace kittens {



namespace tma {













__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes) {
    void const* const ptr = &bar;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_ptr), "r"(bytes));
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
    asm volatile (
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
           "l"(src), "r"(size_bytes),
           "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&bar)))
        : "memory"
    );
}
template<typename T>
__device__ static inline void load_async(T &dst, T &src, uint32_t size_bytes, semaphore& bar) {
    load_async(reinterpret_cast<void*>(&dst), reinterpret_cast<void*>(&src), size_bytes, bar);
}

__device__ static inline void store_async(void *dst, void *src, uint32_t size_bytes) {
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile (
        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
        :: "l"(dst), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src))), "r"(size_bytes)
        : "memory"
    );
    store_commit_group();
}
template<typename T>
__device__ static inline void store_async(T &dst, T &src, uint32_t size_bytes) {
    store_async(reinterpret_cast<void*>(&dst), reinterpret_cast<void*>(&src), size_bytes);
}



namespace cluster {







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

__device__ static inline void careful_wait(semaphore& bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    asm volatile (
        "{\n"
        ".reg .b64                 start_clock, current_clock;\n"
        "mov.b64                   start_clock, %clock64;\n"
        ".reg .pred                P_CLOCK;\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "mov.b64                   current_clock, %clock64;\n"
        "sub.u64                   current_clock, current_clock, start_clock;\n"
        "setp.ge.u64               P_CLOCK, current_clock, 1000000;\n"
        "@P_CLOCK                  trap;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}















__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile ("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
        :: "r"(mbar_addr), "r"(bytes));
}
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes, int dst_cta) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t neighbor_mbar_addr;
    asm volatile (
        "mapa.shared::cluster.u32  %0, %1, %2;\n"
        : "=r"(neighbor_mbar_addr)
        : "r"(mbar_addr), "r"(dst_cta)
    );
    asm volatile ("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
        :: "r"(neighbor_mbar_addr), "r"(bytes));
}















template<typename T, typename... args>
__device__ static inline void expect(semaphore& bar, const T& _1, const args&... _2) {
    expect_bytes(bar, size_bytes<T, args...>);
}










__device__ static inline void arrive(semaphore& bar, int dst_cta, uint32_t count=1) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t neighbor_mbar_addr;
    asm volatile (
        "mapa.shared::cluster.u32  %0, %1, %2;\n"
        : "=r"(neighbor_mbar_addr)
        : "r"(mbar_addr), "r"(dst_cta)
    );
    asm volatile (
        "mbarrier.arrive.shared::cluster.b64 _, [%0], %1;\n"
        :
        : "r"(neighbor_mbar_addr), "r" (count)
        : "memory"
    );
}



__device__ static inline void load_async(void *dst, void *src, uint32_t size_bytes, semaphore& bar, uint16_t cta_mask) {
    asm volatile (
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
           "l"(src), "r"(size_bytes),
           "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&bar))),
           "h"(cta_mask)
        : "memory"
    );
}
template<typename T>
__device__ static inline void load_async(T &dst, T &src, uint32_t size_bytes, semaphore& bar, uint16_t cta_mask) {
    load_async(reinterpret_cast<void*>(&dst), reinterpret_cast<void*>(&src), size_bytes, bar, cta_mask);
}

}
}
}
