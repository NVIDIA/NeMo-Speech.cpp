




#pragma once

#include "../../common/common.cuh"
#include "rt_layout.cuh"
#include "rv_layout.cuh"

namespace kittens {



namespace ducks {





namespace rt_base {






struct identifier {};
}
}












template<typename _T, ducks::rt_layout::all _layout> struct rt_base {
    using identifier = ducks::rt_base::identifier;
    using layout = _layout;
    static_assert(kittens::ducks::base_types::T1<_T>);
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = T2;

#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    static_assert(
        std::is_same_v<dtype, half_2> || std::is_same_v<dtype, bf16_2> || std::is_same_v<dtype, float2> ||
        std::is_same_v<dtype, int8_4> || std::is_same_v<dtype, uint8_4> || std::is_same_v<dtype, int2> ||
        std::is_same_v<dtype, fp8e4m3_4> || std::is_same_v<dtype, fp8e5m2_4>
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
        || std::is_same_v<dtype, fp8e8m0_4> || std::is_same_v<dtype, fp4e2m1_4>
#endif
        ,
        "rt_base was provided an unsupported type."
    );
#else
    static_assert(
        std::is_same_v<dtype, bf16_2> || std::is_same_v<dtype, float2> || std::is_same_v<dtype, half_2> ||
        std::is_same_v<dtype, int8_4> || std::is_same_v<dtype, uint8_4> || std::is_same_v<dtype, int2>,
        "rt_base was provided an unsupported type."
    );
#endif

    static constexpr int tile_size_row        = kittens::TILE_ROW_DIM<T>;
    static constexpr int tile_size_col        = kittens::TILE_COL_DIM<T>;
    static constexpr int rows                 = tile_size_row;
    static constexpr int cols                 = tile_size_col;
    static constexpr int num_elements         = rows*cols;
    static constexpr int elements_per_thread  = num_elements / 32;

    static constexpr int packed_per_thread    = (elements_per_thread / base_types::packing<dtype>::num()) ;
    static constexpr int registers_per_thread = packed_per_thread * sizeof(dtype) / 4;

    using row_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::align, ducks::rv_layout::ortho>;
    using col_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::ortho, ducks::rv_layout::align>;

    dtype data[packed_per_thread];
};







namespace ducks {
namespace rt_base {







template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;
}
}



template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fl = rt_base<float, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_bf = rt_base<bf16, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_hf = rt_base<half, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_int8 = rt_base<int8, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_uint8 = rt_base<uint8, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_int = rt_base<int, L>;
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fp8e4m3 = rt_base<fp8e4m3, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fp8e5m2 = rt_base<fp8e5m2, L>;
#endif
#ifdef KITTENS_SM10X
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fp8e8m0 = rt_base<fp8e8m0, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fp4e2m1_2 = rt_base<fp4e2m1_2, L>;
#endif
}
