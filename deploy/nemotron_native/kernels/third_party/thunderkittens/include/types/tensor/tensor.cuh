




#pragma once

#include "tt.cuh"


namespace kittens {
namespace ducks {





namespace tensor_allocator {



struct identifier {};







template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;
}
}

template<int _nblocks_per_sm, int _ncta, bool _managed = true> struct tensor_allocator {
    static_assert(_nblocks_per_sm == 1 || _nblocks_per_sm == 2, "nblocks_per_sm must be 1 or 2");
    static_assert(_ncta == 1 || _ncta == 2, "ncta must be 1 or 2");

    using identifier = ducks::tensor_allocator::identifier;

    static constexpr int nblocks_per_sm = _nblocks_per_sm;
    static constexpr int cols =((MAX_TENSOR_COLS/nblocks_per_sm) / 32) * 32;
    static_assert(cols>0 && cols%32==0, "cols must be a multiple of 32");
    static constexpr int ncta = _ncta;
    static constexpr bool managed = _managed;

    uint32_t addr;

    template<ducks::tt::all TT, int col_offset> __device__ inline void check_bounds() {
        static_assert(col_offset >= 0 && col_offset + TT::cols <= cols, "Tile allocation extends out of bounds of the tensor allocator!");
    }

    __device__ inline void set_addr(uint32_t _addr) {
        addr = _addr;
    }

    __device__ inline uint32_t get_addr(int superlane, int col_offset) const {
        return addr + ((superlane*16) << 16) + col_offset;
    }

    __device__ inline uint32_t get_addr(int col_offset) const {
        return addr + col_offset;
    }

    __device__ inline void provision(uint32_t &shared_addr) {




        if constexpr (ncta == 1) {
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32  [%0], %1;\n"
            ::  "l"(reinterpret_cast<uint64_t>(&shared_addr)), "n"(cols)
            );
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
        }
        else {
            asm volatile(
                "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32  [%0], %1;\n"
            ::  "l"(reinterpret_cast<uint64_t>(&shared_addr)), "n"(cols)
            );
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n");
        }
    }

    __device__ inline tensor_allocator() {
        if constexpr (managed) {
            __shared__ uint32_t shared_addr;
            if (warpid() == 0) provision(shared_addr);
            asm volatile("tcgen05.fence::before_thread_sync;\n");
            asm volatile("bar.sync 0;\n");
            asm volatile("tcgen05.fence::after_thread_sync;\n");
            set_addr(shared_addr);
        }
    }

    template<ducks::tt::half TT> __device__ inline auto allocate(int superlane, int col_offset) {
#ifndef NDEBUG
        int allocate_cols = std::is_same_v<typename TT::dtype, fp8e8m0> ? TT::cols/4 : TT::cols;
        if(col_offset + allocate_cols > cols) {
            if(laneid() == 0) printf("Tile allocation extends out of bounds of the tensor allocator! col_offset: %d, TT::cols: %d, allocator cols: %d\n", col_offset, TT::cols, cols);
            asm volatile("trap;");
        }
        if(superlane < 0 || superlane > 1) {
            printf("Superlane must be 0 or 1! superlane: %d\n", superlane);
            asm volatile("trap;");
        }
#endif
        return TT(get_addr(superlane, col_offset));
    }

    template<ducks::tt::full TT> __device__ inline auto allocate(int col_offset) {
#ifndef NDEBUG
        int allocate_cols = std::is_same_v<typename TT::dtype, fp8e8m0> ? TT::cols/4 : TT::cols;
        if(col_offset + allocate_cols > cols) {
            if(laneid() == 0) printf("Tile allocation extends out of bounds of the tensor allocator! col_offset: %d, TT::cols: %d, allocator cols: %d\n", col_offset, TT::cols, cols);
            asm volatile("trap;");
        }
#endif
        return TT(get_addr(0, col_offset));
    }

    __device__ inline void deprovision() {



        if constexpr (ncta == 1) {
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32  %0, %1;\n"
            ::  "r"(addr), "n"(cols)
            );
        } else {
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32  %0, %1;\n"
            ::  "r"(addr), "n"(cols)
            );
        }
    }

    __device__ inline ~tensor_allocator() {
        if constexpr (managed) {
            if (warpid() == 0) {
                if (ncta == 2) {
                    asm volatile ("barrier.cluster.arrive.release.aligned;\n");
                    asm volatile ("barrier.cluster.wait.acquire.aligned;\n");
                }
                deprovision();
            }
        }
    }
};

}
