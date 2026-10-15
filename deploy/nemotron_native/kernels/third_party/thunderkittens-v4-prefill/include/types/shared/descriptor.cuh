




#pragma once

#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)

#include "../../common/common.cuh"
#include "st.cuh"
#include "cst.cuh"

namespace kittens {
namespace ducks {
namespace st_descriptor {
struct identifier {};
}
}

namespace detail {


__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

template <typename T>
__device__ static inline uint64_t matrix_descriptor_raw(
    T *addr,
    uint32_t leading_dim_offset,
    uint32_t stride_dim_offset,
    uint32_t swizzle_mode
) {
#ifdef KITTENS_SM10X

    return matrix_descriptor_encode(reinterpret_cast<uint64_t>(addr)) |
           (1llu << 46) |
#else
    return matrix_descriptor_encode(reinterpret_cast<uint64_t>(addr)) |
#endif
           matrix_descriptor_encode((uint64_t)leading_dim_offset) << 16 |
           matrix_descriptor_encode((uint64_t)stride_dim_offset) << 32 |
           (uint64_t)swizzle_mode << 62;
}

}

template<kittens::ducks::st::all _ST, int MN_major>
struct st_descriptor {
    using identifier = ducks::st_descriptor::identifier;
    using ST = _ST;
    using T = typename ST::T;
    static constexpr int rows = ST::rows;
    static constexpr int cols = ST::cols;
    static constexpr bool swizzle = ST::swizzle;
    static_assert(swizzle, "Non-swizzled descriptor is not supported yet.");
    uint64_t base_desc;
    __device__ inline st_descriptor(const ST &tile) {

        if constexpr (MN_major) {
            if constexpr (ST::swizzle_bytes == 128)
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 2048*ST::rows/TILE_ROW_DIM<T>, 1024, 1);
            else if constexpr (ST::swizzle_bytes == 64)
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 1024*ST::rows/TILE_ROW_DIM<T>, 512, 2);
            else
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 512*ST::rows/TILE_ROW_DIM<T>, 256, 3);
        }
        else {
            if constexpr (ST::swizzle_bytes == 128)
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 16                      , 1024, 1);
            else if constexpr (ST::swizzle_bytes == 64)
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 16                      , 512, 2);
            else
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 16                      , 256, 3);
        }
    }
    __device__ inline st_descriptor(const st_descriptor<ST, MN_major> &other) : base_desc(other.base_desc) {}
    __device__ inline uint64_t chunk_descriptor(int chunk_idx) {






        if constexpr (MN_major) {
            if constexpr (ST::swizzle_bytes == 128) {
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*2048);
            }
            else if constexpr (ST::swizzle_bytes == 64) {
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*1024);
            }
            else {
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*512);
            }
        }
        else {
            if constexpr (ST::swizzle_bytes == 128) {

                return base_desc + detail::matrix_descriptor_encode((chunk_idx%4)*32 + (chunk_idx/4)*(ST::rows/TILE_ROW_DIM<T>)*2048);
            }
            else if constexpr (ST::swizzle_bytes == 64) {

                return base_desc + detail::matrix_descriptor_encode((chunk_idx%2)*32 + (chunk_idx/2)*(ST::rows/TILE_ROW_DIM<T>)*1024);
            }
            else {

                return base_desc + detail::matrix_descriptor_encode(chunk_idx*(ST::rows/TILE_ROW_DIM<T>)*512);
            }
        }
    }
#ifdef KITTENS_SM103
    __device__ inline uint64_t chunk_descriptor_k96(int chunk_idx) {
        static_assert(!MN_major, "K96 chunk descriptors are only supported for K-major tcgen05 operands.");
        static_assert(std::is_same_v<T, fp4e2m1_2>, "K96 chunk descriptors are only supported for packed FP4.");
        static_assert(ST::swizzle_bytes == 128, "K96 chunk descriptors require 128B swizzle mode.");
        const int start_byte = chunk_idx*48;
        const int edge_gap = 128 - start_byte%128;
        const int cont_byte = start_byte + (edge_gap < 48 ? edge_gap : 32);

        const uint64_t start_desc = base_desc + detail::matrix_descriptor_encode(start_byte%128 + (start_byte/128)*(ST::rows/TILE_ROW_DIM<T>)*2048);
        const uint64_t cont_desc  = base_desc + detail::matrix_descriptor_encode(cont_byte%128 + (cont_byte/128)*(ST::rows/TILE_ROW_DIM<T>)*2048);
        return (start_desc & ~(0x3FFFull << 16)) | ((cont_desc & 0x3FFFull) << 16) | (1ull << 52);
    }
#endif
};

namespace ducks {
namespace st_descriptor {

template<typename T> concept input = ducks::st::all<T> || (requires {typename T::identifier;} && std::is_same_v<typename T::identifier, ducks::st_descriptor::identifier>);
template<typename T> concept complex_input = ducks::cst::all<T>;
namespace detail {
template<typename T> struct st_getter { using type = typename T::ST; };
template<ducks::st::all T> struct st_getter<T> { using type = T; };
template<ducks::cst::all T> struct st_getter<T> { using type = T::component; };
template<typename T> using get_st = typename st_getter<T>::type;
}
}
}

}

#endif
