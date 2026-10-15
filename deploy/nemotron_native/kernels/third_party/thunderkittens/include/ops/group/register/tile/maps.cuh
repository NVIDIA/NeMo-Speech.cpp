














template<typename op, ducks::rt::all T>
__device__ static inline void unary_map(T &dst, const T &src) {
    #pragma unroll
    for(int i = 0; i < T::height; i++) {
        #pragma unroll
        for(int j = 0; j < T::width; j++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k]);
            }
        }
    }
}










template<typename op, ducks::rt::all T>
__device__ static inline void bin_map(T &dst, const T &src, const typename T::dtype &param) {
    #pragma unroll
    for(int i = 0; i < T::height; i++) {
        #pragma unroll
        for(int j = 0; j < T::width; j++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k], param);
            }
        }
    }
}









template<typename op, ducks::rt::all T>
__device__ static inline void bin_map(T &dst, const T &src, const typename base_types::packing<typename T::dtype>::unpacked_type &param) {

    bin_map<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}









template<typename op, ducks::rt::all T>
__device__ static inline void bin_map(T &dst, const T &lhs, const T &rhs) {
    #pragma unroll
    for(int i = 0; i < T::height; i++) {
        #pragma unroll
        for(int j = 0; j < T::width; j++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(lhs.tiles[i][j].data[k], rhs.tiles[i][j].data[k]);
            }
        }
    }
}

template<ducks::rt::all RT, typename Lambda>
__device__ static inline void apply(RT &dst, const RT &src, Lambda &&lambda) {
    int row_offset = 0;
    if constexpr(GROUP_WARPS > 1) {
        row_offset = warpid()*RT::height;
    }
    static_assert(sizeof(RT::T) != 1, "Cannot apply lambda to 8-bit types");
    if constexpr (ducks::rt::row_layout<RT>) {
        #pragma unroll
        for(int i = 0; i < RT::height; i++) {
            #pragma unroll
            for(int j = 0; j < RT::width; j++) {
                #pragma unroll
                for(int k = 0; k < RT::packed_per_tile; k++) {
                    int row = row_offset + i*TILE_ROW_DIM<typename RT::T> + (k%2) * (TILE_ROW_DIM<typename RT::T>/2) + ::kittens::laneid()/4;
                    int col = j*TILE_COL_DIM<typename RT::T> + (k/2) * (TILE_COL_DIM<typename RT::T>/2) + (::kittens::laneid()%4)*2;
                    dst.tiles[i][j].data[k].x = lambda(row, col+0, src.tiles[i][j].data[k].x);
                    dst.tiles[i][j].data[k].y = lambda(row, col+1, src.tiles[i][j].data[k].y);
                }
            }
        }
    }
    else {
        #pragma unroll
        for(int i = 0; i < RT::height; i++) {
            #pragma unroll
            for(int j = 0; j < RT::width; j++) {
                #pragma unroll
                for(int k = 0; k < RT::packed_per_tile; k++) {
                    int row = row_offset + i*TILE_ROW_DIM<typename RT::T> + (k/2) * (TILE_ROW_DIM<typename RT::T>/2) + (::kittens::laneid()%4)*2;
                    int col = j*TILE_COL_DIM<typename RT::T> + (k%2) * (TILE_COL_DIM<typename RT::T>/2) + ::kittens::laneid()/4;
                    dst.tiles[i][j].data[k].x = lambda(row+0, col, src.tiles[i][j].data[k].x);
                    dst.tiles[i][j].data[k].y = lambda(row+1, col, src.tiles[i][j].data[k].y);
                }
            }
        }
    }
}
template<ducks::rt::all RT, typename Lambda>
__device__ static inline RT apply(const RT &src, Lambda &&lambda) {
    RT dst;
    apply<RT, Lambda>(dst, src, std::forward<Lambda>(lambda));
    return dst;
}













template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ static inline void row_map(T &dst, const T &src, const V &row_values) {

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(V::outer_dim == T::height);

    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < T::height; i++) {
        dtype packed_top_row    = base_types::packing<dtype>::pack(row_values[i][0].x);
        dtype packed_bottom_row = base_types::packing<dtype>::pack(row_values[i][0].y);
        #pragma unroll
        for(int j = 0; j < T::width; j++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile; k+=2) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], packed_top_row);
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(src.tiles[i][j].data[k+1], packed_bottom_row);
            }
        }
    }
}










template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ static inline void row_map(T &dst, const T &src, const V &row_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    static_assert(V::outer_dim == T::height);

    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < T::height; i++) {
        #pragma unroll
        for(int j = 0; j < T::width; j++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile/2; k++) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], row_values[i][0]);
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(src.tiles[i][j].data[k+2], row_values[i][1]);
            }
        }
    }
}















template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ static inline void row_map(T &dst, const T &a, const T &b, const V &row_values) {

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(V::outer_dim == T::height);

    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < T::height; i++) {
        dtype packed_top_row    = base_types::packing<dtype>::pack(row_values[i][0].x);
        dtype packed_bottom_row = base_types::packing<dtype>::pack(row_values[i][0].y);
        #pragma unroll
        for(int j = 0; j < T::width; j++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile; k+=2) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], packed_top_row);
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(a.tiles[i][j].data[k+1], b.tiles[i][j].data[k+1], packed_bottom_row);
            }
        }
    }
}











template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ static inline void row_map(T &dst, const T &a, const T &b, const V &row_values) {

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(V::outer_dim == T::height);

    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < T::height; i++) {
        #pragma unroll
        for(int j = 0; j < T::width; j++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile/2; k++) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], row_values[i][0]);
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(a.tiles[i][j].data[k+2], b.tiles[i][j].data[k+2], row_values[i][1]);
            }
        }
    }
}













template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ static inline void col_map(T &dst, const T &src, const V &col_values) {
    KITTENS_CHECK_WARP

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>);
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(V::outer_dim == T::width);

    using dtype = T::dtype;

    #pragma unroll
    for(int j = 0; j < T::width; j++) {
        #pragma unroll
        for(int i = 0; i < T::height; i++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile/2; k++) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], col_values[j][0]);
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(src.tiles[i][j].data[k+2], col_values[j][1]);
            }
        }
    }
}










template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ static inline void col_map(T &dst, const T &src, const V &col_values) {
    KITTENS_CHECK_WARP

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>);
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(V::outer_dim == T::width);

    using dtype = T::dtype;

    #pragma unroll
    for(int j = 0; j < T::width; j++) {
        dtype packed_left_col  = base_types::packing<dtype>::pack(col_values[j][0].x);
        dtype packed_right_col = base_types::packing<dtype>::pack(col_values[j][0].y);
        #pragma unroll
        for(int i = 0; i < T::height; i++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile; k+=2) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], packed_left_col);
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(src.tiles[i][j].data[k+1], packed_right_col);
            }
        }
    }
}













template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ static inline void col_map(T &dst, const T &a, const T &b, const V &col_values) {
    KITTENS_CHECK_WARP

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>);
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(V::outer_dim == T::width);

    using dtype = T::dtype;

    #pragma unroll
    for(int j = 0; j < T::width; j++) {
        #pragma unroll
        for(int i = 0; i < T::height; i++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile/2; k++) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], col_values[j][0]);
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(a.tiles[i][j].data[k+2], b.tiles[i][j].data[k+2], col_values[j][1]);
            }
        }
    }
}











template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ static inline void col_map(T &dst, const T &a, const T &b, const V &col_values) {
    KITTENS_CHECK_WARP

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>);
    static_assert(V::outer_dim == T::width);

    using dtype = T::dtype;
    #pragma unroll
    for(int j = 0; j < T::width; j++) {
        dtype packed_left_col  = base_types::packing<dtype>::pack(col_values[j][0].x);
        dtype packed_right_col = base_types::packing<dtype>::pack(col_values[j][0].y);
        #pragma unroll
        for(int i = 0; i < T::height; i++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile; k+=2) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], packed_left_col);
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(a.tiles[i][j].data[k+1], b.tiles[i][j].data[k+1], packed_right_col);
            }
        }
    }
}













template<ducks::rt::all T>
__device__ static inline void zero(T &dst) {
    unary_map<base_ops::zero, T>(dst, dst);
}






template<ducks::rt::all T>
__device__ static inline void one(T &dst) {
    unary_map<base_ops::one, T>(dst, dst);
}






template<ducks::rt::all T>
__device__ static inline void pos_infty(T &dst) {
    unary_map<base_ops::pos_infty, T>(dst, dst);
}






template<ducks::rt::all T>
__device__ static inline void neg_infty(T &dst) {
    unary_map<base_ops::neg_infty, T>(dst, dst);
}








template<ducks::rt::all T>
__device__ static inline void exp(T &dst, const T &src) {
    unary_map<base_ops::exp, T>(dst, src);
}
template<ducks::rt::all T>
__device__ static inline T exp(const T &src) {
    T dst;
    exp(dst, src);
    return dst;
}








template<ducks::rt::all T>
__device__ static inline void exp2(T &dst, const T &src) {
    unary_map<base_ops::exp2, T>(dst, src);
}
template<ducks::rt::all T>
__device__ static inline T exp2(const T &src) {
    T dst;
    exp2(dst, src);
    return dst;
}








template<ducks::rt::all T>
__device__ static inline void log(T &dst, const T &src) {
    unary_map<base_ops::log, T>(dst, src);
}
template<ducks::rt::all T>
__device__ static inline T log(const T &src) {
    T dst;
    log(dst, src);
    return dst;
}








template<ducks::rt::all T>
__device__ static inline void log2(T &dst, const T &src) {
    unary_map<base_ops::log2, T>(dst, src);
}
template<ducks::rt::all T>
__device__ static inline T log2(const T &src) {
    T dst;
    log2(dst, src);
    return dst;
}








template<ducks::rt::all T>
__device__ static inline void abs(T &dst, const T &src) {
    unary_map<base_ops::abs, T>(dst, src);
}
template<ducks::rt::all T>
__device__ static inline T abs(const T &src) {
    T dst;
    abs(dst, src);
    return dst;
}








template<ducks::rt::all T>
__device__ static inline void relu(T &dst, const T &src) {
    unary_map<base_ops::relu, T>(dst, src);
}
template<ducks::rt::all T>
__device__ static inline T relu(const T &src) {
    T dst;
    relu(dst, src);
    return dst;
}








template<ducks::rt::all T>
__device__ static inline void neg(T &dst, const T &src) {
    unary_map<base_ops::neg, T>(dst, src);
}
template<ducks::rt::all T>
__device__ static inline T neg(const T &src) {
    T dst;
    neg(dst, src);
    return dst;
}








template<ducks::rt::all T>
__device__ static inline void sqrt(T &dst, const T &src) {
    unary_map<base_ops::sqrt, T>(dst, src);
}
template<ducks::rt::all T>
__device__ static inline T sqrt(const T &src) {
    T dst;
    sqrt(dst, src);
    return dst;
}








template<ducks::rt::all T>
__device__ static inline void rsqrt(T &dst, const T &src) {
    unary_map<base_ops::rsqrt, T>(dst, src);
}
template<ducks::rt::all T>
__device__ static inline T rsqrt(const T &src) {
    T dst;
    rsqrt(dst, src);
    return dst;
}









template<ducks::rt::all T, typename U>
__device__ static inline void copy(T &dst, const U &src) {
    bin_map<base_ops::copy2, T>(dst, src);
}










template<ducks::rt::all T, typename U>
__device__ static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::max, T>(dst, lhs, rhs);
}
template<ducks::rt::all T, typename U>
__device__ static inline T max(const T &lhs, const U &rhs) {
    T dst;
    max(dst, lhs, rhs);
    return dst;
}










template<ducks::rt::all T, typename U>
__device__ static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::min, T>(dst, lhs, rhs);
}
template<ducks::rt::all T, typename U>
__device__ static inline T min(const T &lhs, const U &rhs) {
    T dst;
    min(dst, lhs, rhs);
    return dst;
}










template<ducks::rt::all T, typename U>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sum, T>(dst, lhs, rhs);
}










template<ducks::rt::all T, typename U>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sub, T>(dst, lhs, rhs);
}









template<ducks::rt::all T, typename U>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::mul, T>(dst, lhs, rhs);
}










template<ducks::rt::all T, typename U>
__device__ static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::div, T>(dst, lhs, rhs);
}










template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void add_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sum, T, V>(dst, src, row_values);
}










template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void sub_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sub, T, V>(dst, src, row_values);
}










template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void mul_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::mul, T, V>(dst, src, row_values);
}










template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void div_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::div, T, V>(dst, src, row_values);
}









template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void broadcast_row(T &dst, const V &row_values) {
    row_map<base_ops::copy2, T, V>(dst, dst, row_values);
}
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline T broadcast_row(const V &row_values) {
    T dst;
    broadcast_row(dst, row_values);
    return dst;
}












template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void add_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sum, T, V>(dst, src, col_values);
}










template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void sub_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sub, T, V>(dst, src, col_values);
}










template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void mul_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::mul, T, V>(dst, src, col_values);
}










template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void div_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::div, T, V>(dst, src, col_values);
}









template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void broadcast_col(T &dst, const V &col_values) {
    col_map<base_ops::copy2, T, V>(dst, dst, col_values);
}
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline T broadcast_col(const V &col_values) {
    T dst;
    broadcast_col(dst, col_values);
    return dst;
}


template<ducks::rt::all RT>
__device__ static inline void tril(RT &dst, const RT &src, int diagonal=0, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    apply(dst, src, [val, diagonal]__device__(int row, int col, auto &src_val) {
        return col <= row + diagonal ? src_val : val;
    });
}
template<ducks::rt::all RT>
__device__ static inline void triu(RT &dst, const RT &src, int diagonal=0, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    apply(dst, src, [val, diagonal]__device__(int row, int col, auto &src_val) {
        return col >= row + diagonal ? src_val : val;
    });
}
