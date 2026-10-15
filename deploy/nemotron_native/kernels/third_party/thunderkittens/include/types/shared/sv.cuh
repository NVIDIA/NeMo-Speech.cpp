




#pragma once

#include "../../common/common.cuh"

namespace kittens {



namespace ducks {





namespace sv {






struct identifier {};







template<typename T>
concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;
}
}











template<typename _T, size_t _length>
struct KITTENS_DEFAULT_ALIGN sv {
    static_assert(!std::is_same_v<_T, char>, "Bare 'char' is not permitted because its signedness is ISA-dependent.");
    using identifier = ducks::sv::identifier;
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T;

    static constexpr int length = _length;
    static_assert(length % TILE_ROW_DIM<T> == 0, "Length must be divisible by the tile dimension");
    static constexpr int tiles  = length / TILE_ROW_DIM<T>;

#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    static constexpr int num_alloc_elements = ((length * sizeof(dtype) + 127) / 128) * (128 / sizeof(dtype));
#else
    static constexpr int num_alloc_elements = length;
#endif
    dtype data[num_alloc_elements];

    __device__ static inline T* idx(T *ptr, int idx) {
        return ptr[idx];
    }

    __device__ inline       dtype& operator[](size_t idx)       { return data[idx]; }
    __device__ inline const dtype& operator[](size_t idx) const { return data[idx]; }

    template<int sub_length> __device__ inline sv<_T, sub_length> &subvec(int idx) {
        return *(sv<dtype, sub_length>*)&data[idx * sub_length];
    }
    template<int sub_length> __device__ inline const sv<_T, sub_length> &subvec(int idx) const {
        return *(sv<dtype, sub_length>*)&data[idx * sub_length];
    }

    __device__ inline void operator=(const dtype &value) {
        #pragma unroll
        for(int i = kittens::laneid(); i < length; i += WARP_THREADS) {
            data[i] = value;
        }
    }
};

#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
namespace detail {
namespace tma {



template<typename SV, int D=16> struct find_vector_divider {
    static constexpr int value = (SV::length % (16*D) == 0 && (SV::length < 256 || ((16*D)*sizeof(typename SV::dtype)) % 128 == 0)) ?
        16*D : find_vector_divider<SV, D-1>::value;
};
template<typename SV> struct find_vector_divider<SV, 1> { static constexpr int value = 16; };
template<typename SV> constexpr int sv_tma_dim1 = find_vector_divider<SV>::value;
template<typename SV> constexpr int sv_tma_dim2 = (SV::length / sv_tma_dim1<SV>);
}
}
#endif




template<size_t _length> using sv_bf = sv<bf16,  _length>;
template<size_t _length> using sv_hf = sv<half,  _length>;
template<size_t _length> using sv_fl = sv<float, _length>;
template<size_t _length> using sv_int8 = sv<int8, _length>;
template<size_t _length> using sv_uint8 = sv<uint8, _length>;
template<size_t _length> using sv_int = sv<int, _length>;
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
template<int _length> using sv_fp8e4m3 = sv<fp8e4m3, _length>;
template<int _length> using sv_fp8e5m2 = sv<fp8e5m2, _length>;
#endif
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
template<int _length> using sv_fp8e8m0 = sv<fp8e8m0, _length>;
template<int _length> using sv_fp4e2m1_2 = sv<fp4e2m1_2, _length>;
#endif



template<ducks::sv::all SV>
__device__ inline void print(const SV& sv) {
    printf("Shared Vector %d:\n", SV::length);
    for(int i = 0; i < SV::length; i++) {
        if constexpr (std::is_same_v<typename SV::dtype, bf16>) {
            printf("%f ", __bfloat162float(sv[i]));
        } else if constexpr (std::is_same_v<typename SV::dtype, half>) {
            printf("%f ", __half2float(sv[i]));
        } else if constexpr (std::is_same_v<typename SV::dtype, float>) {
            printf("%f ", sv[i]);
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
        } else if constexpr (std::is_same_v<typename SV::dtype, fp8e4m3>) {
            printf("%f ", static_cast<float>(sv[i]));
#endif
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
        } else if constexpr (std::is_same_v<typename SV::dtype, fp8e8m0>) {
            printf("%f ", static_cast<float>(sv[i]));
#endif
        } else {
            printf("%d ", (int)(sv[i]));
        }
    }
    printf("\n");
}

}
