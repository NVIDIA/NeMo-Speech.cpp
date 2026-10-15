




#pragma once

#include "rt.cuh"
#include "crv.cuh"

namespace kittens {

namespace ducks {
namespace crt {






struct identifier {};








template <typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier> && ducks::rt::all<typename T::component>;






template<typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;








template<typename T>
concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::col>;
}
}














template<typename _T, int _rows, int _cols, ducks::rt_layout::all _layout=ducks::rt_layout::row>
struct crt {
    using identifier = ducks::crt::identifier;
    using component  = rt<_T, _rows, _cols, _layout>;
    using layout     = component::layout;
    using T          = component::T;
    using T2         = component::T2;
    using dtype      = component::dtype;

    static constexpr int rows       = component::rows;
    static constexpr int cols       = component::cols;
    static constexpr int height     = component::height;
    static constexpr int width      = component::width;


    component real;
    component imag;

    using row_vec = crv<T, cols, typename rt_base<T, layout>::row_vec_layout>;
    using col_vec = crv<T, rows, typename rt_base<T, layout>::col_vec_layout>;
};

template<int _rows, int _cols, ducks::rt_layout::all layout=ducks::rt_layout::row> using crt_fl = crt<float, _rows, _cols, layout>;
template<int _rows, int _cols, ducks::rt_layout::all layout=ducks::rt_layout::row> using crt_bf = crt<bf16, _rows, _cols, layout>;
template<int _rows, int _cols, ducks::rt_layout::all layout=ducks::rt_layout::row> using crt_hf = crt<half, _rows, _cols, layout>;



}
