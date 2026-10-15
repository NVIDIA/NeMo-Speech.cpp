#pragma once
__device__ __forceinline__ bool bb_router_select_parallel(
    const float *score, const bf16 *gate_bias, float *out) {
    static_assert(BB_EXPERTS == 24 && BB_TOP_K == 4);
    constexpr unsigned full = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const float unbiased = lane < BB_EXPERTS ? score[lane] : 0.f;
    float remaining = lane < BB_EXPERTS
        ? unbiased + __bfloat162float(gate_bias[lane]) : -CUDART_INF_F;
    const bool finite = lane >= BB_EXPERTS ||
        (((__float_as_uint(unbiased) & 0x7f800000u) != 0x7f800000u) &&
         ((__float_as_uint(remaining) & 0x7f800000u) != 0x7f800000u));
    if (!__all_sync(full, finite)) return false;
    bool selected = false;
    float total = 0.f;
    #pragma unroll
    for (int k = 0; k < BB_TOP_K; ++k) {
        float best_score = remaining;
        int best_id = lane < BB_EXPERTS ? lane : 0x7fffffff;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float other_score = __shfl_xor_sync(full, best_score, off);
            const int other_id = __shfl_xor_sync(full, best_id, off);
            if (other_score > best_score ||
                (other_score == best_score && other_id < best_id)) {
                best_score = other_score;
                best_id = other_id;
            }
        }
        total += __shfl_sync(full, unbiased, best_id);
        if (lane == best_id) {
            selected = true;
            remaining = -CUDART_INF_F;
        }
    }
    const float norm = BB_ROUTED_SCALE / (total + 1e-20f);
    if (lane < BB_EXPERTS) out[lane] = selected ? unbiased * norm : 0.f;
    return true;
}
