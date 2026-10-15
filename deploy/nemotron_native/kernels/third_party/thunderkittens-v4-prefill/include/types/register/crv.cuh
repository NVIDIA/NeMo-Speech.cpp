




#pragma once

#include "../../common/common.cuh"
#include "rv.cuh"

namespace kittens {




namespace ducks {





namespace crv {






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
struct crv {
    using identifier = ducks::crv::identifier;
    using component  = rv<_T, _length, _layout>;
    using layout     = component::layout;

    using T          = component::T;
    using T2         = component::T2;
    using dtype      = component::dtype;

    static constexpr int length     = component::length;
    static constexpr int tiles      = component::tiles;


    component real;
    component imag;
};


template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using crv_fl = crv<float, _l, layout>;
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using crv_bf = crv<bf16,  _l, layout>;
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using crv_hf = crv<half,  _l, layout>;

}
