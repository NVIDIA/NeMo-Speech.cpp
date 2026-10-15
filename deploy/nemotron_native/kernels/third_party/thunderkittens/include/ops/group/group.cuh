




#pragma once

#include "../../common/common.cuh"
#include "../../types/types.cuh"
#include "../thread/thread.cuh"

#define KITTENS_CHECK_WARP static_assert(GROUP_WARPS==1, "Warp (GROUP_WARPS=1) function called from a non-warp group.");

#define KITTENS_CHECK_WARPGROUP static_assert(GROUP_WARPS==4, "Warpgroup (GROUP_WARPS=4) function called from a non-warpgroup group.");


#if defined(KITTENS_SM90)
#include "mma/base/base.cuh"
#endif

namespace kittens {



template<int _GROUP_WARPS>
struct group {
static constexpr int GROUP_WARPS = _GROUP_WARPS;
static constexpr int GROUP_THREADS = GROUP_WARPS * kittens::WARP_THREADS;
__device__ static inline int laneid() { return threadIdx.x % GROUP_THREADS; }
__device__ static inline int warpid() { return laneid() / kittens::WARP_THREADS; }
__device__ static inline int groupid() { return threadIdx.x / GROUP_THREADS; }

__device__ static inline void sync(int id) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "n"(GROUP_THREADS));
}
template<uint32_t MASK=0xFFFFFFFF> __device__ static inline void sync() {
    static_assert(GROUP_WARPS==1, "barrier-less sync() can only be called by a single warp!");
    asm volatile("bar.warp.sync %0;\n" :: "n"(MASK));
}
__device__ static inline void arrive(int id) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "n"(GROUP_THREADS));
}

#include "memory/memory.cuh"
#include "shared/shared.cuh"
#include "register/register.cuh"
#include "mma/mma.cuh"
#include "util/util.cuh"

#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)

template<int n_reg> __device__ static inline void increase_registers() {
    static_assert(n_reg % 8 == 0, "n_reg must be a multiple of 8");
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(n_reg));
}
template<int n_reg> __device__ static inline void decrease_registers() {
    static_assert(n_reg % 8 == 0, "n_reg must be a multiple of 8");
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(n_reg));
}
__device__ static inline void producer_registers() { decrease_registers<24>(); }
template<int NCWG> __device__ static inline void consumer_registers() { increase_registers<480/NCWG - 8*(NCWG>3) - 224*(NCWG==1)>(); }






struct tma {
#include "memory/tile/tma.cuh"
#include "memory/vec/tma.cuh"
#include "util/tma.cuh"
struct cluster {
#include "memory/tile/tma_cluster.cuh"
#include "memory/vec/tma_cluster.cuh"
#include "util/tma_cluster.cuh"
};
};

#endif

};

namespace everyone {


__device__ static inline void sync(int id) {
    asm volatile("bar.sync %0;\n" :: "r"(id));
}


namespace tma {
namespace cluster {
__device__ static inline void arrive() {
    asm volatile ("barrier.cluster.arrive.release;\n");
}
__device__ static inline void arrive_aligned() {
    asm volatile ("barrier.cluster.arrive.release.aligned;\n");
}
__device__ static inline void wait() {
    asm volatile ("barrier.cluster.wait.acquire;\n");
}
__device__ static inline void wait_aligned() {
    asm volatile ("barrier.cluster.wait.acquire.aligned;\n");
}
__device__ static inline void sync() {
    arrive_aligned();
    wait_aligned();
}
}
}

};

using warp = group<1>;
using warpgroup = group<4>;

}
