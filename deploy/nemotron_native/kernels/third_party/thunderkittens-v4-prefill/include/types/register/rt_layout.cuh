




#pragma once

namespace kittens {
namespace ducks {





namespace rt_layout {




struct row {};



struct col {};




template<typename T>
concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;




template<all L> struct transpose      { using type = col; };
template<>      struct transpose<col> { using type = row; };

}
}
}
