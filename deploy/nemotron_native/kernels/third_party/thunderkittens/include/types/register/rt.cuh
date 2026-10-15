




#pragma once

#include "../../common/common.cuh"

#include "rt_layout.cuh"
#include "rt_base.cuh"
#include "rv.cuh"

namespace kittens {




namespace ducks {





namespace rt {






struct identifier {};







template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;








template<typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;








template<typename T>
concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::col>;
}
}
















template<typename _T, int _rows, int _cols, ducks::rt_layout::all _layout=ducks::rt_layout::row>
struct rt {
    static_assert(!std::is_same_v<_T, char>, "Bare 'char' is not permitted because its signedness is ISA-dependent.");
    using identifier = ducks::rt::identifier;
    using layout = _layout;
    static_assert(kittens::ducks::base_types::T1<_T>);
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = T2;

    static constexpr int rows                = _rows;
    static_assert(rows % rt_base<T, layout>::tile_size_row == 0, "Rows must be divisible by the tile size");
    static constexpr int cols                = _cols;
    static_assert(cols % rt_base<T, layout>::tile_size_col == 0, "Columns must be divisible by the tile size");
    static constexpr int height              = rows / rt_base<T, layout>::tile_size_row;
    static constexpr int width               = cols / rt_base<T, layout>::tile_size_col;
    static constexpr int tile_size_row        = rt_base<T, layout>::tile_size_row;
    static constexpr int tile_size_col        = rt_base<T, layout>::tile_size_col;
    static constexpr int num_elements        = rt_base<T, layout>::num_elements        * width * height;
    static constexpr int elements_per_thread = rt_base<T, layout>::elements_per_thread * width * height;
    static constexpr int packed_per_thread   = rt_base<T, layout>::packed_per_thread   * width * height;
    static constexpr int packed_per_tile     = rt_base<T, layout>::packed_per_thread;

    rt_base<T, layout> tiles[height][width];

    using row_vec = rv<T, cols, typename rt_base<T, layout>::row_vec_layout>;
    using col_vec = rv<T, rows, typename rt_base<T, layout>::col_vec_layout>;

    __device__ inline void operator=(const T &value) {
        T2 value2 = base_types::packing<T>::pack(value);
        #pragma unroll
        for(int i = 0; i < height; i++) {
            #pragma unroll
            for(int j = 0; j < width; j++) {
                #pragma unroll
                for(int k = 0; k < packed_per_tile; k++) {
                    tiles[i][j].data[k] = value2;
                }
            }
        }
    }
    template<typename U>
    __device__ inline void operator=(const rt<U, rows, cols, layout> &other) {
        using U2 = base_types::packing<U>::packed_type;
        #pragma unroll
        for(int i = 0; i < height; i++) {
            #pragma unroll
            for(int j = 0; j < width; j++) {
                #pragma unroll
                for(int k = 0; k < packed_per_tile; k++) {
                    tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(other.tiles[i][j].data[k]);
                }
            }
        }
    }
};






template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl = rt<float, _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf = rt<bf16,  _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_hf = rt<half,  _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_int8 = rt<int8,  _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_uint8 = rt<uint8,  _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_int = rt<int, _r, _c, layout>;
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp8e4m3 = rt<fp8e4m3,  _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp8e5m2 = rt<fp8e5m2,  _r, _c, layout>;
#endif
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp8e8m0 = rt<fp8e8m0,  _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp4e2m1_2 = rt<fp4e2m1_2,  _r, _c, layout>;
#endif






template<typename T, int rows, int cols>
__device__ constexpr const char* get_rt_type_name() {
    if constexpr (std::is_same_v<T, float>) {
        return "rt_fl";
    } else if constexpr (std::is_same_v<T, half>) {
        return "rt_hf";
    } else if constexpr (std::is_same_v<T, bf16>) {
        return "rt_bf";
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    } else if constexpr (std::is_same_v<T, fp4e2m1_2>) {
        return "rt_fp4_e2m1_2";
    } else if constexpr (std::is_same_v<T, fp8e8m0>) {
        return "rt_fp8_e8m0";
#endif
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    } else if constexpr (std::is_same_v<T, fp8e4m3>) {
        return "rt_fp8_e4m3";
    } else if constexpr (std::is_same_v<T, fp8e5m2>) {
        return "rt_fp8_e5m2";
#endif
    } else {
        return "rt_unknown";
    }
}










template<ducks::rt::all RT>
__device__ inline void print(const RT& tile) {
    if (laneid() == 0) {
        printf("Block %d, Warp %d: Register Tile %dx%d (Type: %s<%d,%d>) - Distributed View:\n",
               blockIdx.x, threadIdx.x / WARP_THREADS, RT::rows, RT::cols,
               get_rt_type_name<typename RT::T, RT::rows, RT::cols>(), RT::rows, RT::cols);
        printf("Each thread holds %d elements (%d packed)\n",
               RT::elements_per_thread, RT::packed_per_thread);
        printf("\n");
    }
    __syncwarp();


    for (int tid = 0; tid < WARP_THREADS; tid++) {
        if (laneid() == tid) {
            printf("Thread %2d: ", tid);


            for (int i = 0; i < RT::height; i++) {
                for (int j = 0; j < RT::width; j++) {
                    printf("Subtile[%d][%d]: ", i, j);
                    for (int k = 0; k < RT::packed_per_tile && k < 4; k++) {
                        auto packed_val = tile.tiles[i][j].data[k];

                        if constexpr (std::is_same_v<typename RT::dtype, typename RT::T>) {

                            if constexpr (std::is_same_v<typename RT::T, float>) {
                                printf("%.3f ", packed_val);
                            } else if constexpr (std::is_same_v<typename RT::T, half>) {
                                printf("%.3f ", __half2float(packed_val));
                            } else if constexpr (std::is_same_v<typename RT::T, bf16>) {
                                printf("%.3f ", __bfloat162float(packed_val));
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
                            } else if constexpr (std::is_same_v<typename RT::T, fp4e2m1>) {
                                printf("%.3f ", (float)packed_val);
#endif
                            } else {
                                printf("%.3f ", (float)packed_val);
                            }
                        } else {

                            if constexpr (std::is_same_v<typename RT::T, float>) {
                                printf("[%.3f, %.3f] ", packed_val.x, packed_val.y);
                            } else if constexpr (std::is_same_v<typename RT::T, bf16>) {

                                printf("[%.3f, %.3f] ", __bfloat162float(packed_val.x), __bfloat162float(packed_val.y));
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
                            } else if constexpr (std::is_same_v<typename RT::T, fp8e8m0>) {

                                __nv_fp8_e8m0 *vals = reinterpret_cast<__nv_fp8_e8m0*>(const_cast<fp8e8m0_4*>(&packed_val));
                                printf("[%.3f,%.3f,%.3f,%.3f] ",
                                       (float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
                            } else if constexpr (std::is_same_v<typename RT::T, fp4e2m1>) {

                                uint8_t *vals = reinterpret_cast<uint8_t*>(const_cast<fp4e2m1_4*>(&packed_val));
                                printf("[%.3f,%.3f,%.3f,%.3f] ", (float)fp4e2m1(vals[0] & 0xF), (float)fp4e2m1((vals[0] >> 4) & 0xF), (float)fp4e2m1(vals[1] & 0xF), (float)fp4e2m1((vals[1] >> 4) & 0xF));
#endif
                            } else {

                                printf("0x%x ", *(uint32_t*)&packed_val);
                            }
                        }
                    }
                    if (RT::packed_per_tile > 4) printf("... ");
                }
            }
            printf("\n");
        }
        __syncwarp();
    }

    if (laneid() == 0) {
        printf("\n");
    }
    __syncwarp();
}

}
