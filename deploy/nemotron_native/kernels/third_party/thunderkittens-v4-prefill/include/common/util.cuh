




#pragma once






namespace kittens {






constexpr int BASE_TILE_DIM = 16;
template<typename T> constexpr int TILE_COL_DIM = sizeof(T) == 1 ? BASE_TILE_DIM * 2 : BASE_TILE_DIM;
template<typename T> constexpr int TILE_ROW_DIM = BASE_TILE_DIM;



template<typename T> constexpr int TILE_ELEMENTS{TILE_COL_DIM<T>*TILE_ROW_DIM<T>};



constexpr int WARP_THREADS{32};



constexpr int WARPGROUP_THREADS{128};




constexpr int WARPGROUP_WARPS{4};





__device__ __forceinline__ int warpid() { return threadIdx.x >> 5; }




__device__ __forceinline__ int warpgroupid() { return threadIdx.x >> 7; }




__device__ __forceinline__ int laneid() { return threadIdx.x & 0x1f; }




__device__ __forceinline__ int smid() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
    return static_cast<int>(r);
}

#if defined(KITTENS_SM80)
constexpr int MAX_SHARED_MEMORY = 163 * 1024;
#elif defined(KITTENS_SM90) || defined(KITTENS_SM10X)
constexpr int MAX_SHARED_MEMORY = 227 * 1024;
#elif defined(KITTENS_SM120)
constexpr int MAX_SHARED_MEMORY = 99 * 1024;
#endif

struct transpose {
    static constexpr int N = 0;
    static constexpr int T = 1;
};
struct axis {
    static constexpr int ROW = 0;
    static constexpr int COL = 1;
};











namespace ducks {




struct default_type {};


#define typeof(A) typename std::remove_const<typename std::remove_reference<decltype(A)>::type>::type

}






static constexpr uint32_t MASK_ALL = 0xFFFFFFFF;









template<typename T>
__device__ static inline T packed_shfl_down_sync(uint32_t mask, const T &f, int delta) {
    return __shfl_down_sync(mask, f, delta);
}
template<>
__device__ inline float2 packed_shfl_down_sync<float2>(uint32_t mask, const float2 &f, int delta) {
    float2 r;
    r.x = __shfl_down_sync(mask, f.x, delta);
    r.y = __shfl_down_sync(mask, f.y, delta);
    return r;
}








template<typename T>
__device__ static inline T packed_shfl_xor_sync(uint32_t mask, const T &f, int lane_mask) {
    return __shfl_xor_sync(mask, f, lane_mask);
}
template<>
__device__ inline float2 packed_shfl_xor_sync<float2>(uint32_t mask, const float2 &f, int lane_mask) {
    float2 r;
    r.x = __shfl_xor_sync(mask, f.x, lane_mask);
    r.y = __shfl_xor_sync(mask, f.y, lane_mask);
    return r;
}








template<typename T>
__device__ static inline T packed_shfl_sync(uint32_t mask, const T &f, int src) {
    return __shfl_sync(mask, f, src);
}
template<>
__device__ inline float2 packed_shfl_sync<float2>(uint32_t mask, const float2 &f, int src) {
    float2 r;
    r.x = __shfl_sync(mask, f.x, src);
    r.y = __shfl_sync(mask, f.y, src);
    return r;
}
























#if defined(__CUDACC__)
#define KITTENS_ALIGN_AS(n) __align__(n)
#else
#define KITTENS_ALIGN_AS(n) alignas(n)
#endif

#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
#define KITTENS_DEFAULT_ALIGN KITTENS_ALIGN_AS(128)
#else
#define KITTENS_DEFAULT_ALIGN KITTENS_ALIGN_AS(16)
#endif








template <typename T>
__host__ __device__ constexpr T cdiv(T a, T b) {
    return (a + b - 1) / b;
}




struct KITTENS_DEFAULT_ALIGN alignment_dummy { int dummy; };




#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
template<int default_alignment=1024>
#else
template<int default_alignment=16>
#endif
struct shared_allocator {
    int *ptr;

private:

    template<typename A, size_t... dims>
    struct variadic_array;
    template<typename A, size_t first_dim, size_t... rest_dims>
    struct variadic_array<A, first_dim, rest_dims...> {
        using type = typename variadic_array<A, rest_dims...>::type[first_dim];
    };
    template<typename A>
    struct variadic_array<A> {
        using type = A;
    };
    template<typename A, size_t... dims>
    using variadic_array_t = typename variadic_array<A, dims...>::type;

    template<int alignment>
    __device__ inline void align_ptr() {
        if constexpr (alignment > 0) {
            uint64_t p = reinterpret_cast<uint64_t>(ptr);
            if(p % alignment != 0) {
                ptr = (int*)(p + (alignment-(p%alignment)));
            }
        }
    }

public:




    __device__ shared_allocator(int *_ptr): ptr(_ptr) {}






    template<typename A, size_t... dims>
    __device__ inline variadic_array_t<A, dims...>& allocate() {

        align_ptr<default_alignment>();
        using at = variadic_array_t<A, dims...>;
        at*p = reinterpret_cast<at*>(ptr);
        ptr += sizeof(at)/sizeof(int);
        return *p;
    }







    template<int alignment, typename A, size_t... dims>
    __device__ inline variadic_array_t<A, dims...>& allocate() {

        align_ptr<alignment>();
        using at = variadic_array_t<A, dims...>;
        at*p = reinterpret_cast<at*>(ptr);
        ptr += sizeof(at)/sizeof(int);
        return *p;
    }
};

#if (defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120))



using tma_allocator = shared_allocator<1024>;
using tma_swizzle_allocator = tma_allocator;




__device__ static inline int3 clusterIdx() {
    int3 cluster_idx;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(cluster_idx.x));
    asm volatile("mov.u32 %0, %clusterid.y;\n" : "=r"(cluster_idx.y));
    asm volatile("mov.u32 %0, %clusterid.z;\n" : "=r"(cluster_idx.z));
    return cluster_idx;
}


__device__ static inline int3 clusterid() {
    int3 clusterid;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(clusterid.x));
    asm volatile("mov.u32 %0, %clusterid.y;\n" : "=r"(clusterid.y));
    asm volatile("mov.u32 %0, %clusterid.z;\n" : "=r"(clusterid.z));
    return clusterid;
}


__device__ static inline int3 nclusterid() {
    int3 nclusterid;
    asm volatile("mov.u32 %0, %nclusterid.x;\n" : "=r"(nclusterid.x));
    asm volatile("mov.u32 %0, %nclusterid.y;\n" : "=r"(nclusterid.y));
    asm volatile("mov.u32 %0, %nclusterid.z;\n" : "=r"(nclusterid.z));
    return nclusterid;
}


__device__ static inline int cluster_ctarank() {
    uint32_t ctarank;
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(ctarank));
    return ctarank;
}


__device__ static inline int cluster_nctarank() {
    uint32_t nctarank;
    asm volatile("mov.u32 %0, %cluster_nctarank;\n" : "=r"(nctarank));
    return nctarank;
}

#endif


















template <int SUPERGROUP_SIZE, bool ROW_MAJOR = true>
__device__ static inline int2 get_swizzled_2d_idx(const int num_rows, const int num_cols, const int linear_idx) {
    static_assert(SUPERGROUP_SIZE > 0, "SUPERGROUP_SIZE must be greater than 0");
    if constexpr (ROW_MAJOR) {
        const int supergroup_numel = num_rows*SUPERGROUP_SIZE;
        const int supergroup_idx = linear_idx/supergroup_numel;
        const int supersection_cols = (num_cols/SUPERGROUP_SIZE)*SUPERGROUP_SIZE;
        const int supersection_numel = num_rows*supersection_cols;
        const int finalsection_cols = num_cols-supersection_cols;
        int row_idx, col_idx;
        if (linear_idx < supersection_numel) {
            row_idx = (linear_idx%supergroup_numel)/SUPERGROUP_SIZE;
            col_idx = supergroup_idx*SUPERGROUP_SIZE + linear_idx%SUPERGROUP_SIZE;
        } else {
            const int remainder_task_id = linear_idx - supersection_numel;
            row_idx = remainder_task_id/finalsection_cols;
            col_idx = supersection_cols + remainder_task_id%finalsection_cols;
        }
        return { (supergroup_idx&1) ? num_rows-row_idx-1 : row_idx, col_idx };
    } else {
        const int supergroup_numel = num_cols*SUPERGROUP_SIZE;
        const int supergroup_idx = linear_idx/supergroup_numel;
        const int supersection_rows = (num_rows/SUPERGROUP_SIZE)*SUPERGROUP_SIZE;
        const int supersection_numel = num_cols*supersection_rows;
        const int finalsection_rows = num_rows-supersection_rows;
        int row_idx, col_idx;
        if (linear_idx < supersection_numel) {
            row_idx = supergroup_idx*SUPERGROUP_SIZE + linear_idx%SUPERGROUP_SIZE;
            col_idx = (linear_idx%supergroup_numel)/SUPERGROUP_SIZE;
        } else {
            const int remainder_task_id = linear_idx - supersection_numel;
            row_idx = supersection_rows + remainder_task_id%finalsection_rows;
            col_idx = remainder_task_id/finalsection_rows;
        }
        return { row_idx, (supergroup_idx&1) ? num_cols-col_idx-1 : col_idx };
    }
}

template<int half>
__device__ static inline int get_phasebit(uint32_t bitfield, int ring_id) {
    if constexpr (half == 0)
        return (bitfield >> (ring_id)) & 0b1;
    else if constexpr (half == 1)
        return (bitfield >> (ring_id + 16)) & 0b1;
    else
        asm volatile ("brkpt;\n");
    return -1;
}

template<int half>
__device__ static inline void update_phasebit(uint32_t &bitfield, int ring_id) {
    if constexpr (half == 0)
        bitfield ^= (1 << (ring_id));
    else if constexpr (half == 1)
        bitfield ^= (1 << (ring_id + 16));
    else
        asm volatile ("brkpt;\n");
}

template<int N> __device__ static inline int ring_advance(int ring, int distance=1) { return (ring + distance) % N; }
template<int N> __device__ static inline int ring_retreat(int ring, int distance=1) { return (ring + 16*N - distance) % N; }



#ifndef KITTENS_NO_HOST


#define CUCHECK(cmd) do {                                  \
    CUresult err = cmd;                                    \
    if (err != CUDA_SUCCESS) {                             \
        const char *errStr;                                \
        cuGetErrorString(err, &errStr);                    \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n", \
            __FILE__, __LINE__, errStr);                   \
        exit(EXIT_FAILURE);                                \
    }                                                      \
} while(0)


#define CUDACHECK(cmd) do {                                \
    cudaError_t err = cmd;                                 \
    if (err != cudaSuccess) {                              \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n", \
            __FILE__, __LINE__, cudaGetErrorString(err));  \
        exit(EXIT_FAILURE);                                \
    }                                                      \
} while(0)


#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T> __host__ void check(
    T err, char const* const func, char const* const file, int const line
) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}






 __host__ inline int num_sms(int device_id = -1) {
    if (device_id < 0) CUDACHECK(cudaGetDevice(&device_id));
    int sm_count;
    CUDACHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
    return sm_count;
}

template <bool CLUSTER = false, bool PDL = false>
struct LaunchConfig {
    static constexpr unsigned int num_attributes = CLUSTER ? (PDL ? 3 : 2) : 1;

    cudaLaunchAttribute attributes[num_attributes];
    cudaLaunchConfig_t config = {0};

    __host__ inline LaunchConfig(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                                 cudaStream_t stream) noexcept requires(!CLUSTER) {
        if constexpr (PDL) {
            attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attributes[0].val.programmaticStreamSerializationAllowed = 1;
            config.attrs = attributes;
            config.numAttrs = num_attributes;
        } else {
            config.numAttrs = 0;
        }
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes= dynamic_shared_memory;
        config.stream = stream;
    }

    __host__ inline LaunchConfig(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                                 cudaStream_t stream, dim3 cluster_preferred, dim3 cluster_minimum) noexcept requires(CLUSTER) {
        attributes[0].id = cudaLaunchAttributePreferredClusterDimension;
        attributes[0].val.preferredClusterDim.x = cluster_preferred.x;
        attributes[0].val.preferredClusterDim.y = cluster_preferred.y;
        attributes[0].val.preferredClusterDim.z = cluster_preferred.z;
        attributes[1].id = cudaLaunchAttributeClusterDimension;
        attributes[1].val.clusterDim.x = cluster_minimum.x;
        attributes[1].val.clusterDim.y = cluster_minimum.y;
        attributes[1].val.clusterDim.z = cluster_minimum.z;
        if constexpr (PDL) {
            attributes[2].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attributes[2].val.programmaticStreamSerializationAllowed = 1;
        }
        config.attrs = attributes;
        config.numAttrs = num_attributes;
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes= dynamic_shared_memory;
        config.stream = stream;
    }

    __host__ inline LaunchConfig(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                                 cudaStream_t stream, dim3 cluster) noexcept requires(CLUSTER)
        : LaunchConfig(grid, block, dynamic_shared_memory, stream, cluster, cluster) { }

    __host__ inline LaunchConfig(const LaunchConfig& other) noexcept : config(other.config) {
        std::copy_n(other.attributes, num_attributes, attributes);
        config.attrs = attributes;
    }

    __host__ inline LaunchConfig& operator=(const LaunchConfig& other) noexcept {
        if (this != &other) {
            std::copy_n(other.attributes, num_attributes, attributes);
            config = other.config;
            config.attrs = attributes;
        }
        return *this;
    }

    __host__ LaunchConfig(LaunchConfig&&) = delete;
    __host__ LaunchConfig& operator=(LaunchConfig&&) = delete;

    __host__ inline operator cudaLaunchConfig_t*() noexcept { return &config; }
    __host__ inline operator const cudaLaunchConfig_t*() const noexcept { return &config; }
};

#endif

}
