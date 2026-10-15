




#pragma once

#include "../../common/common.cuh"
#include "rv_layout.cuh"

namespace kittens {




namespace ducks {





namespace rv {






struct identifier {};







template<typename T>
concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;

template<typename T> concept naive_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::naive>;
template<typename T> concept align_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::align>;
template<typename T> concept ortho_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::ortho>;
template<typename T> concept tile_layout  = align_layout<T> || ortho_layout<T>;
}
}













template<typename _T, size_t _length, ducks::rv_layout::all _layout=ducks::rv_layout::naive>
struct rv {
    static_assert(!std::is_same_v<_T, char>, "Bare 'char' is not permitted because its signedness is ISA-dependent.");
    using identifier = ducks::rv::identifier;
    static_assert(kittens::ducks::base_types::T1<_T>);
    using layout = _layout;
    static constexpr bool is_naive = std::is_same_v<layout, ducks::rv_layout::naive>;
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = std::conditional_t<is_naive, T, T2>;

    static constexpr int length = _length;
    static_assert(length % kittens::TILE_ROW_DIM<T> == 0, "Length must be divisible by the tile dimension");
    static constexpr int tiles  = _length / kittens::TILE_ROW_DIM<T>;
    static constexpr int inner_dim = layout::inner_dim;
    static constexpr int outer_dim = is_naive ? (tiles+1)/2 : tiles;
    #if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for fp8");
    #endif

    dtype data[outer_dim][inner_dim];

    __device__ inline       dtype* operator[](size_t idx)       { return &data[idx][0]; }
    __device__ inline const dtype* operator[](size_t idx) const { return &data[idx][0]; }
    __device__ inline       dtype& operator[](int2 outin)       { return data[outin.x][outin.y]; }
    __device__ inline const dtype& operator[](int2 outin) const { return data[outin.x][outin.y]; }

    __device__ inline void operator=(const T &value) {
        dtype value2;
        if constexpr(is_naive) {
            value2 = value;
        } else {
            value2 = base_types::packing<T>::pack(value);
        }
        #pragma unroll
        for(int i = 0; i < outer_dim; i++) {
            #pragma unroll
            for(int j = 0; j < inner_dim; j++) {
                data[i][j] = value2;
            }
        }
    }
    template<typename U>
    __device__ inline void operator=(const rv<U, length, layout> &other) {
        using U2 = base_types::packing<U>::packed_type;
        #pragma unroll
        for(int i = 0; i < outer_dim; i++) {
            #pragma unroll
            for(int j = 0; j < inner_dim; j++) {
                data[i][j] = base_types::convertor<T2, U2>::convert(other.data[i][j]);
            }
        }
    }
};

template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_fl = rv<float, _l, layout>;
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_bf = rv<bf16,  _l, layout>;
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_hf = rv<half,  _l, layout>;











template<ducks::rv::all RV>
__device__ void print(const RV &vec) {
    if (laneid() == 0) {
        printf("Block %d, Warp %d: Register Vector %d (Type: %s, Layout: %s) - Distributed View:\n",
               blockIdx.x, threadIdx.x / WARP_THREADS, RV::length,
               std::is_same_v<typename RV::T, float> ? "float" :
               std::is_same_v<typename RV::T, bf16> ? "bf16" :
               std::is_same_v<typename RV::T, half> ? "half" :
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
               std::is_same_v<typename RV::T, fp8e8m0> ? "fp8e8m0" :
#endif
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
               std::is_same_v<typename RV::T, fp8e4m3> ? "fp8e4m3" :
               std::is_same_v<typename RV::T, fp8e5m2> ? "fp8e5m2" : "unknown",
#endif
               RV::is_naive ? "naive" : "tile");
        printf("Each thread holds %dx%d elements\n", RV::outer_dim, RV::inner_dim);
        printf("\n");
    }
    __syncwarp();


    for (int tid = 0; tid < WARP_THREADS; tid++) {
        if (laneid() == tid) {
            printf("Thread %2d: ", tid);


            for (int i = 0; i < RV::outer_dim; i++) {
                printf("Outer[%d]: ", i);
                for (int j = 0; j < RV::inner_dim; j++) {
                    auto value = vec.data[i][j];

                    if constexpr (std::is_same_v<typename RV::dtype, typename RV::T>) {

                        if constexpr (std::is_same_v<typename RV::T, float>) {
                            printf("%.3f ", value);
                        } else if constexpr (std::is_same_v<typename RV::T, half>) {
                            printf("%.3f ", __half2float(value));
                        } else if constexpr (std::is_same_v<typename RV::T, bf16>) {
                            printf("%.3f ", __bfloat162float(value));
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e8m0>) {
                            printf("%.3f ", (float)value);
#endif
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e4m3>) {
                            printf("%.3f ", (float)value);
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e5m2>) {
                            printf("%.3f ", (float)value);
#endif
                        } else {
                            printf("%.3f ", (float)value);
                        }
                    } else {

                        if constexpr (std::is_same_v<typename RV::T, float>) {
                            printf("[%.3f, %.3f] ", value.x, value.y);
                        } else if constexpr (std::is_same_v<typename RV::T, bf16>) {

                            printf("[%.3f, %.3f] ", __bfloat162float(value.x), __bfloat162float(value.y));
                        } else if constexpr (std::is_same_v<typename RV::T, half>) {

                            printf("[%.3f, %.3f] ", __half2float(value.x), __half2float(value.y));
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e8m0>) {

                            __nv_fp8_e8m0 *vals = reinterpret_cast<__nv_fp8_e8m0*>(const_cast<fp8e8m0_4*>(&value));
                            printf("[%.3f,%.3f,%.3f,%.3f] ",
                                   (float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
#endif
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e4m3>) {

                            __nv_fp8_e4m3 *vals = reinterpret_cast<__nv_fp8_e4m3*>(const_cast<fp8e4m3_4*>(&value));
                            printf("[%.3f,%.3f,%.3f,%.3f] ",
                                   (float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e5m2>) {

                            __nv_fp8_e5m2 *vals = reinterpret_cast<__nv_fp8_e5m2*>(const_cast<fp8e5m2_4*>(&value));
                            printf("[%.3f,%.3f,%.3f,%.3f] ",
                                   (float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
#endif
                        } else {

                            printf("0x%x ", *(uint32_t*)&value);
                        }
                    }
                }
                printf(" ");
            }
            printf("\n");
        }
        __syncwarp();
    }

    if (laneid() == 0) {
        printf("\n");
    }
}

}
