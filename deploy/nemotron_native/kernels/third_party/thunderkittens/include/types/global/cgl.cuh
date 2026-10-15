




#pragma once

#include "gl.cuh"

namespace kittens {



namespace ducks {
namespace cgl {
struct identifier {};
}
}






template<kittens::ducks::gl::all _GL>
struct cgl {
    using identifier = ducks::cgl::identifier;
    using component  = _GL;
    using T          = component::T;
    using T2         = component::T2;
    using dtype      = component::dtype;
    component real, imag;
};

namespace ducks {
namespace cgl {







template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;
}
}

}
