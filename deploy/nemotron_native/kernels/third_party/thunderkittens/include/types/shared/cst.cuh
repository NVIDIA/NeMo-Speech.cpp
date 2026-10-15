




#pragma once

#include "st.cuh"
#include "csv.cuh"

namespace kittens {

namespace ducks {
namespace cst {






struct identifier {};









template <typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier> && ducks::st::all<typename T::component>;

}
}













template<typename _T, int _rows, int _cols>
struct cst {
    using identifier = ducks::cst::identifier;
    using component  = st<_T, _rows, _cols>;
    using T          = component::T;
    using T2         = component::T2;
    using dtype      = component::dtype;

    static constexpr int rows       = component::rows;
    static constexpr int cols       = component::cols;




    component real;
    component imag;


    using col_vec = csv<dtype, rows>;
    using row_vec = csv<dtype, cols>;
};



template<int _rows, int _cols> using cst_bf = cst<bf16,  _rows, _cols>;
template<int _rows, int _cols> using cst_hf = cst<half,  _rows, _cols>;
template<int _rows, int _cols> using cst_fl = cst<float, _rows, _cols>;

}
