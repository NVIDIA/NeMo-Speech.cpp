




#include "conversions.cuh"
#include "maps.cuh"
#include "reductions.cuh"

template<ducks::sv::all SV>
__device__ static inline bool hasnan(const SV &src) {
    KITTENS_CHECK_WARP
    bool nan_detected = false;
    #pragma unroll
    for(int i = laneid(); i < SV::length; i+=GROUP_THREADS) {
        if constexpr (std::is_same_v<typename SV::T, float>) {
            if(isnan(src[i])) {
                nan_detected = true;
            }
        }
        else if constexpr (std::is_same_v<typename SV::T, bf16>) {
            if(isnan(__bfloat162float(src[i]))) {
                nan_detected = true;
            }
        }
        else if constexpr (std::is_same_v<typename SV::T, half>) {
            if(isnan(__half2float(src[i]))) {
                nan_detected = true;
            }
        }
        else {
            static_assert(sizeof(typename SV::T) == 999, "Unsupported dtype");
        }
    }

    return (__ballot_sync(0xffffffff, nan_detected) != 0);
}
