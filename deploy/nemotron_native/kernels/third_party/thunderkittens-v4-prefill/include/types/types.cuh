




#pragma once

#include "register/register.cuh"
#include "shared/shared.cuh"
#include "global/global.cuh"
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
#include "system/system.cuh"
#endif
#ifdef KITTENS_SM10X
#include "tensor/tensor.cuh"
#endif



namespace kittens {















template<typename T>
using row_vec = T::row_vec;















template<typename T>
using col_vec = T::col_vec;




using row_l = ducks::rt_layout::row;
using col_l = ducks::rt_layout::col;


using align_l = ducks::rv_layout::align;
using ortho_l = ducks::rv_layout::ortho;
using naive_l = ducks::rv_layout::naive;

}
