




#pragma once

namespace kittens {
namespace ducks {





namespace rv_layout {




struct align { constexpr static int inner_dim = 2; };



struct ortho { constexpr static int inner_dim = 1; };



struct naive { constexpr static int inner_dim = 1; };




template<typename T>
concept all = std::is_same_v<T, align> || std::is_same_v<T, ortho> || std::is_same_v<T, naive>;

}
}
}
