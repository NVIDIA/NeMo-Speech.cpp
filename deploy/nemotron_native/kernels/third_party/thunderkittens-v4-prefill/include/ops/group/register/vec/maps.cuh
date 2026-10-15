














template<typename op, ducks::rv::all T>
__device__ static inline void unary_op(T &dst, const T &src) {
    #pragma unroll
    for(int i = 0; i < T::outer_dim; i++) {
        #pragma unroll
        for(int j = 0; j < T::inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(src[i][j]);
        }
    }
}









template<typename op, ducks::rv::all T>
__device__ static inline void bin_op(T &dst, const T &lhs, const T &rhs) {
    #pragma unroll
    for(int i = 0; i < T::outer_dim; i++) {
        #pragma unroll
        for(int j = 0; j < T::inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(lhs[i][j], rhs[i][j]);
        }
    }
}









template<typename op, ducks::rv::all T>
__device__ static inline void bin_op(T &dst, const T &src, const typename T::dtype &param) {
    #pragma unroll
    for(int i = 0; i < T::outer_dim; i++) {
        #pragma unroll
        for(int j = 0; j < T::inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(src[i][j], param);
        }
    }
}









template<typename op, ducks::rv::tile_layout T>
__device__ static inline void bin_op(T &dst, const T &src, const typename base_types::packing<typename T::dtype>::unpacked_type &param) {
    bin_op<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}


template<ducks::rv::all RV, typename Lambda>
__device__ static inline void apply(RV &dst, const RV &src, Lambda &&lambda) {
    int group_offset = 0;
    if constexpr(GROUP_WARPS > 1) {
        group_offset = warpid()*RV::length;
    }
    static_assert(sizeof(RV::T) != 1, "Cannot apply lambda to 8-bit types");
    if constexpr (ducks::rv::ortho_layout<RV>) {
        #pragma unroll
        for(int i = 0; i < RV::outer_dim; i++) {
            int base_idx = group_offset + i*16 + ::kittens::laneid()/4;
            dst[i][0].x = lambda(base_idx+0, src[i][0].x);
            dst[i][0].y = lambda(base_idx+8, src[i][0].y);
        }
    }
    else if constexpr (ducks::rv::align_layout<RV>) {
        #pragma unroll
        for(int i = 0; i < RV::outer_dim; i++) {
            int base_idx = group_offset + i*16 + 2*(::kittens::laneid()%4);
            dst[i][0].x = lambda(base_idx+0, src[i][0].x);
            dst[i][0].y = lambda(base_idx+1, src[i][0].y);
            dst[i][1].x = lambda(base_idx+8, src[i][1].x);
            dst[i][1].y = lambda(base_idx+9, src[i][1].y);
        }
    }
    else {
        #pragma unroll
        for(int i = 0; i < RV::outer_dim; i++) {
            int base_idx = group_offset + i*32 + ::kittens::laneid();
            if (i < RV::outer_dim-1 || RV::length%32 == 0 || ::kittens::laneid()<16) {
                dst[i][0] = lambda(base_idx, src[i][0]);
            }
        }
    }
}
template<ducks::rv::all RV, typename Lambda>
__device__ static inline RV apply(const RV &src, Lambda &&lambda) {
    RV dst;
    apply<RV, Lambda>(dst, src, std::forward<Lambda>(lambda));
    return dst;
}











template<ducks::rv::all T>
__device__ static inline void zero(T &dst) {
    unary_op<base_ops::zero, T>(dst, dst);
}






template<ducks::rv::all T>
__device__ static inline void one(T &dst) {
    unary_op<base_ops::one, T>(dst, dst);
}






template<ducks::rv::all T>
__device__ static inline void pos_infty(T &dst) {
    unary_op<base_ops::pos_infty, T>(dst, dst);
}






template<ducks::rv::all T>
__device__ static inline void neg_infty(T &dst) {
    unary_op<base_ops::neg_infty, T>(dst, dst);
}











template<ducks::rv::all T, typename U>
__device__ static inline void copy(T &dst, const U &src) {
    bin_op<base_ops::copy2, T>(dst, dst, src);
}







template<ducks::rv::all T>
__device__ static inline void exp(T &dst, const T &src) {
    unary_op<base_ops::exp, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T exp(const T &src) {
    T dst;
    exp(dst, src);
    return dst;
}







template<ducks::rv::all T>
__device__ static inline void exp2(T &dst, const T &src) {
    unary_op<base_ops::exp2, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T exp2(const T &src) {
    T dst;
    exp2(dst, src);
    return dst;
}







template<ducks::rv::all T>
__device__ static inline void log(T &dst, const T &src) {
    unary_op<base_ops::log, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T log(const T &src) {
    T dst;
    log(dst, src);
    return dst;
}







template<ducks::rv::all T>
__device__ static inline void log2(T &dst, const T &src) {
    unary_op<base_ops::log2, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T log2(const T &src) {
    T dst;
    log2(dst, src);
    return dst;
}







template<ducks::rv::all T>
__device__ static inline void abs(T &dst, const T &src) {
    unary_op<base_ops::abs, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T abs(const T &src) {
    T dst;
    abs(dst, src);
    return dst;
}







template<ducks::rv::all T>
__device__ static inline void relu(T &dst, const T &src) {
    unary_op<base_ops::relu, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T relu(const T &src) {
    T dst;
    relu(dst, src);
    return dst;
}












template<ducks::rv::all T, typename U>
__device__ static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::max, T>(dst, lhs, rhs);
}
template<ducks::rv::all T, typename U>
__device__ static inline T max(const T &lhs, const U &rhs) {
    T dst;
    max(dst, lhs, rhs);
    return dst;
}









template<ducks::rv::all T, typename U>
__device__ static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::min, T>(dst, lhs, rhs);
}
template<ducks::rv::all T, typename U>
__device__ static inline T min(const T &lhs, const U &rhs) {
    T dst;
    min(dst, lhs, rhs);
    return dst;
}









template<ducks::rv::all T, typename U>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sum, T>(dst, lhs, rhs);
}









template<ducks::rv::all T, typename U>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sub, T>(dst, lhs, rhs);
}









template<ducks::rv::all T, typename U>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::mul, T>(dst, lhs, rhs);
}









template<ducks::rv::all T, typename U>
__device__ static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::div, T>(dst, lhs, rhs);
}
