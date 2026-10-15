#pragma once

#include <kittens.cuh>

#ifdef KITTENS_SM10X

#include <fstream>
#include <string>
#include <vector>



































template <int _TIMING_WIDTH=128>
struct __align__(1024) TKProfiler {
    static constexpr int TIMING_WIDTH = _TIMING_WIDTH;
    int timings[TIMING_WIDTH];
    uint64_t start_time;
    using timing_t = kittens::gl<int, 1, -1, -1, TIMING_WIDTH>;





    __device__ inline void init() {
        if (threadIdx.x < TIMING_WIDTH)
            timings[threadIdx.x] = 0;
        if (threadIdx.x == 0) {
            uint64_t start_time_reg;
            asm volatile("{mov.u64 %0, %%globaltimer;}" : "=l"(start_time_reg));
            uint32_t start_time_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&start_time));
            asm volatile("{st.shared.b64 [%0], %1;}" : : "r"(start_time_addr), "l"(start_time_reg));
        }
        __syncthreads();
    }





    __device__ inline void record(int index) {
        uint64_t current_time;
        asm volatile("{mov.u64 %0, %%globaltimer;}" : "=l"(current_time));
        timings[index] = static_cast<int>(current_time - start_time);
    }







    __device__ inline void store_and_reset(const timing_t &timings_global, int i, int j) {
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(timings));
        uint64_t dst_ptr = reinterpret_cast<uint64_t>(&timings_global[{i, j, 0}]);
        asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory");
        asm volatile("{cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;}"
            :: "l"(dst_ptr), "r"(src_ptr), "n"(TIMING_WIDTH*sizeof(int)) : "memory");
        kittens::tma::store_commit_group();
        kittens::tma::store_async_read_wait();
        asm volatile("{st.bulk.weak [%0], %1, 0;}" :: "r"(src_ptr), "n"(TIMING_WIDTH*sizeof(int)) : "memory");
    }







    __host__ static inline timing_t allocate(int N, int M) {
        const size_t timings_size = static_cast<size_t>(N)*M*TIMING_WIDTH;
        int *d_timings;
        CUDACHECK(cudaMalloc(&d_timings, timings_size*sizeof(int)));
        CUDACHECK(cudaMemset(d_timings, 0, timings_size*sizeof(int)));
        return timing_t{d_timings, nullptr, static_cast<size_t>(N), static_cast<size_t>(M), nullptr};
    }





    __host__ static inline void deallocate(timing_t &timings_global) {
        CUDACHECK(cudaFree(timings_global.raw_ptr));
    }






    __host__ static inline void save(timing_t &timings_global, const std::string &filename) {
        const int M = static_cast<int>(timings_global.depth());
        const int N = static_cast<int>(timings_global.rows());
        const size_t timings_size = static_cast<size_t>(M)*N*TIMING_WIDTH;

        std::vector<int> h_timings(timings_size);
        CUDACHECK(cudaMemcpy(h_timings.data(), timings_global.raw_ptr, timings_size*sizeof(int), cudaMemcpyDeviceToHost));

        std::ofstream outfile(filename);
        if (outfile.is_open()) {
            outfile << "# shape: (" << M << ", " << N << ", " << TIMING_WIDTH << ")\n";
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < TIMING_WIDTH; k++) {
                        int idx = (i * N + j) * TIMING_WIDTH + k;
                        outfile << h_timings[idx];
                        if (k < TIMING_WIDTH - 1) outfile << " ";
                    }
                    outfile << "\n";
                }
            }
            outfile.close();
        }
    }
};

#endif
