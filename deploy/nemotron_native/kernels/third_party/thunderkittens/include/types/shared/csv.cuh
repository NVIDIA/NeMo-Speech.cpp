




#pragma once

#include "sv.cuh"

namespace kittens {

namespace ducks {
namespace csv {






struct identifier {};








template <typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier> && ducks::sv::all<typename T::component>;

}
}












template<typename _T, int _length>
struct csv {
    using identifier = ducks::csv::identifier;
    using component  = sv<_T, _length>;
    using T          = component::T;
    using T2         = component::T2;
    using dtype      = component::dtype;

    static constexpr int length     = component::length;
    static constexpr int tiles      = component::tiles;




    component real;
    component imag;
};




template<int _length> using csv_bf = csv<bf16,  _length>;
template<int _length> using csv_hf = csv<half,  _length>;
template<int _length> using csv_fl = csv<float, _length>;

}
