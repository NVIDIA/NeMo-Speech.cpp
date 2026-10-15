
















__device__ static inline void swap_layout_8(bf16_2 &dst, const bf16_2 &src) {
    KITTENS_CHECK_WARP
    asm volatile (
        "movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
    :   "+r"(*(uint32_t*)(&dst))
    :   "r"(*(uint32_t*)(&src))
    );
}











template<typename T, ducks::rt_layout::all layout>
__device__ static inline void swap_layout(rt_base<T, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt_base<T, layout> &src) {
    swap_layout_8(dst.data[0], src.data[0]);


    typename rt_base<T, layout>::T2 data1_cache = src.data[1];
    swap_layout_8(dst.data[1], src.data[2]);
    swap_layout_8(dst.data[2], data1_cache);
    swap_layout_8(dst.data[3], src.data[3]);
}













template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void swap_layout(rt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt<T2, _height, _width, layout> &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            swap_layout(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}












template<typename T2, ducks::rt_layout::all layout>
__device__ static inline rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(const rt_base<T2, layout> &src) {
    rt_base<T2, typename ducks::rt_layout::transpose<layout>::type> &dst = *(rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>*)(&src);
    swap_layout(dst, src);
    return dst;
}













template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(rt<T2, _rows, _cols, layout> &tile) {
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < tile.width; j++) {
            swap_layout_inplace(tile.tiles[i][j]);
        }
    }
    return *(rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>*)(&tile);
}











template<typename T, ducks::rt_layout::all layout>
__device__ static inline void transpose(rt_base<T, layout> &dst, const rt_base<T, layout> &src) {
    swap_layout_8(dst.data[0], src.data[0]);


    typename rt_base<T, layout>::T2 data1_cache = src.data[1];
    swap_layout_8(dst.data[1], src.data[2]);
    swap_layout_8(dst.data[2], data1_cache);
    swap_layout_8(dst.data[3], src.data[3]);
}













template<ducks::rt::all RT>
__device__ static inline void transpose_sep(RT &dst, const rt<typename RT::T, RT::cols, RT::rows, typename RT::layout> &src) {
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            transpose(dst.tiles[i][j], src.tiles[j][i]);
        }
    }
}









template<typename T2, ducks::rt_layout::all layout>
__device__ static inline rt_base<T2, layout>& transpose_inplace(rt_base<T2, layout> &src) {
    transpose(src, src);
    return src;
}










template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _rows, _cols, layout>& transpose_inplace(rt<T2, _rows, _cols, layout> &tile) {
    static_assert(_cols == _rows, "in-place register tile transpose is only allowed for square tiles.");
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < i; j++) {
            rt_base<T2, layout> tmp;
            copy(tmp, tile.tiles[i][j]);
            transpose(tile.tiles[i][j], tile.tiles[j][i]);
            transpose(tile.tiles[j][i], tmp);
        }
        transpose_inplace(tile.tiles[i][i]);
    }
    return tile;
}












template<typename T, typename U, ducks::rt_layout::all layout>
__device__ static inline void copy(rt_base<T, layout> &dst, const rt_base<U, layout> &src) {
    using T2 = typename base_types::packing<T>::packed_type;
    using U2 = typename base_types::packing<U>::packed_type;
    #pragma unroll
    for(int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_types::convertor<T2, U2>::convert(src.data[k]);
    }
}
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)











template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void copy(rt<T2, _height, _width, layout> &dst, const rt<U2, _height, _width, layout> &src) {

    if constexpr (
        (std::is_same_v<U2, float> && std::is_same_v<T2, fp8e4m3>) ||
        (std::is_same_v<U2, float> && std::is_same_v<T2, fp8e5m2>) ||
        (std::is_same_v<U2, kittens::bf16> && std::is_same_v<T2, fp8e4m3>) ||
        (std::is_same_v<U2, kittens::bf16> && std::is_same_v<T2, fp8e5m2>) ||
        (std::is_same_v<U2, half> && std::is_same_v<T2, fp8e4m3>) ||
        (std::is_same_v<U2, half> && std::is_same_v<T2, fp8e5m2>)
    ) {

        int laneid = threadIdx.x % 32;

        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                #pragma unroll
                for(int k = 0; k < dst.tiles[0][0].packed_per_thread; k++) {


                    using src_t = std::conditional_t<std::is_same_v<U2, float>, float2, std::conditional_t<std::is_same_v<U2, kittens::bf16>, bf16_2, half2>>;
                    src_t val1, val2;


                    if (laneid % 2 == 0) {

                        val1 = src.tiles[i][2*j + k/2].data[(k%2)+0];
                        val2 = src.tiles[i][2*j + k/2].data[(k%2)+2];
                    } else {

                        val1 = src.tiles[i][2*j + k/2].data[(k%2)+2];
                        val2 = src.tiles[i][2*j + k/2].data[(k%2)+0];
                    }


                    int row_mask = 4 * ( laneid / 4 );
                    int row_offset = row_mask + ( (laneid-row_mask) / 2 ) + ( laneid % 2 );
                    int src_offset = (laneid % 2 == 0 ) ? row_offset + 0 : ( row_offset + 1 );
                    src_t val01 = packed_shfl_sync(MASK_ALL, val1, src_offset);

                    int src_offset2 = (laneid % 4 < 2 ) ? src_offset + 1 : (src_offset - 1);
                    src_t val23 = packed_shfl_sync(MASK_ALL, val2, src_offset2);


                    float4 f4;
                    using fp8_4_t = std::conditional_t<std::is_same_v<T2, fp8e4m3>, fp8e4m3_4, fp8e5m2_4>;
                    fp8_4_t f4_fp8;
                    if ( laneid % 4 < 2 ) {
                        f4.x = val01.x;
                        f4.y = val01.y;
                        f4.z = val23.x;
                        f4.w = val23.y;
                        f4_fp8 = base_types::convertor<fp8_4_t, float4>::convert(f4);
                        dst.tiles[i][j].data[k] = f4_fp8;
                    } else {
                        f4.x = val23.x;
                        f4.y = val23.y;
                        f4.z = val01.x;
                        f4.w = val01.y;
                        f4_fp8 = base_types::convertor<fp8_4_t, float4>::convert(f4);
                        dst.tiles[i][j].data[k] = f4_fp8;
                    }
                }
            }
        }
    }
    else if constexpr (
        (std::is_same_v<U2, fp8e4m3> && std::is_same_v<T2, float>) ||
        (std::is_same_v<U2, fp8e5m2> && std::is_same_v<T2, float>) ||
        (std::is_same_v<U2, fp8e4m3> && std::is_same_v<T2, kittens::bf16>) ||
        (std::is_same_v<U2, fp8e5m2> && std::is_same_v<T2, kittens::bf16>) ||
        (std::is_same_v<U2, fp8e4m3> && std::is_same_v<T2, half>) ||
        (std::is_same_v<U2, fp8e5m2> && std::is_same_v<T2, half>)
    ) {

        int laneid = threadIdx.x % 32;

        #pragma unroll
        for(int i = 0; i < src.height; i++) {
            #pragma unroll
            for(int j = 0; j < src.width; j++) {
                #pragma unroll
                for(int k = 0; k < src.tiles[0][0].packed_per_thread; k++) {
                    int dst_j = 2*j + k/2;


                    using fp8_4_t = std::conditional_t<std::is_same_v<U2, fp8e4m3>, fp8e4m3_4, fp8e5m2_4>;
                    fp8_4_t val = src.tiles[i][j].data[k];
                    float4 f4 = base_types::convertor<float4, fp8_4_t>::convert(val);
                    float2 f2_0, f2_1;
                    if ( laneid % 4 < 2 ) {
                        f2_0 = make_float2(f4.x, f4.y);
                        f2_1 = make_float2(f4.z, f4.w);
                    }
                    else {
                        f2_0 = make_float2(f4.z, f4.w);
                        f2_1 = make_float2(f4.x, f4.y);
                    }

                    int row_offset = 4 * (laneid/4) + (laneid%2) * 2 + (laneid%4) / 2;
                    float2 f2_0_shfl = packed_shfl_sync(MASK_ALL, f2_0, row_offset);
                    float2 f2_1_shfl = packed_shfl_sync(MASK_ALL, f2_1, row_offset^2);


                    using dst_t = std::conditional_t<std::is_same_v<T2, float>, float2, std::conditional_t<std::is_same_v<T2, kittens::bf16>, bf16_2, half2>>;
                    if constexpr (!(std::is_same_v<T2, float>)) {
                        dst_t f2_0_shfl_t = base_types::convertor<dst_t, float2>::convert(f2_0_shfl);
                        dst_t f2_1_shfl_t = base_types::convertor<dst_t, float2>::convert(f2_1_shfl);
                        if (laneid % 2 == 0) {
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_0_shfl_t;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_1_shfl_t;
                        } else {
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_1_shfl_t;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_0_shfl_t;
                        }
                    } else {
                        if (laneid % 2 == 0) {
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_0_shfl;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_1_shfl;
                        } else {
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_1_shfl;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_0_shfl;
                        }
                    }
                }
            }
        }
    }

    else {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                copy(dst.tiles[i][j], src.tiles[i][j]);
            }
        }
    }
}
#else











template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void copy(rt<T2, _height, _width, layout> &dst, const rt<U2, _height, _width, layout> &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            copy(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}
#endif















template<ducks::rt::row_layout RT>
__device__ static inline void make_causal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    static_assert(!std::is_same_v<typename RT::dtype, fp8e4m3_4> && !std::is_same_v<typename RT::dtype, fp8e5m2_4>, "Unsupported type for make_causal");
    #endif
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            if(j < i) {
                #pragma unroll
                for(int k = 0; k < RT::packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j > i) {
                #pragma unroll
                for(int k = 0; k < RT::packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else {
                constexpr uint32_t MASK_X = 0xFF773311, MASK_Y = 0xF7733110;
                dst.tiles[i][j].data[1] = src.tiles[i][j].data[1];
                dst.tiles[i][j].data[2] = packed_val;
                if((MASK_X >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x;
                    dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x;
                }
                else {
                    dst.tiles[i][j].data[0].x = val;
                    dst.tiles[i][j].data[3].x = val;
                }
                if((MASK_Y >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y;
                    dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y;
                }
                else {
                    dst.tiles[i][j].data[0].y = val;
                    dst.tiles[i][j].data[3].y = val;
                }
            }
            __syncwarp();
        }
    }
}













template<ducks::rt::row_layout RT>
__device__ static inline void make_causal_t(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    static_assert(!std::is_same_v<typename RT::dtype, fp8e4m3_4> && !std::is_same_v<typename RT::dtype, fp8e5m2_4>, "Unsupported type for make_causal");
    #endif
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            if(j > i) {
                #pragma unroll
                for(int k = 0; k < RT::packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j < i) {
                #pragma unroll
                for(int k = 0; k < RT::packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else {
                constexpr uint32_t MASK_X = 0x88CCEEF;
                constexpr uint32_t MASK_Y = 0x88CCEEFF;

                dst.tiles[i][j].data[1] = packed_val;
                dst.tiles[i][j].data[2] = src.tiles[i][j].data[2];


                if((MASK_X >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x;
                    dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x;
                }

                else {
                    dst.tiles[i][j].data[0].x = val;
                    dst.tiles[i][j].data[3].x = val;
                }


                if((MASK_Y >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y;
                    dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y;
                }

                else {
                    dst.tiles[i][j].data[0].y = val;
                    dst.tiles[i][j].data[3].y = val;
                }

            }
            __syncwarp();
        }
    }
}












template<ducks::rt::row_layout RT>
__device__ static inline void tril(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                const int global_row_idx   = (i * RT::tile_size_row) + ((k % 2) * 8) + (laneid() / 4);
                const int global_col_idx_x = (j * RT::tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int global_col_idx_y = (j * RT::tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;

                if (global_col_idx_x <= global_row_idx + diagonal) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else                                               { dst.tiles[i][j].data[k].x = val; }

                if (global_col_idx_y <= global_row_idx + diagonal) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else                                               { dst.tiles[i][j].data[k].y = val; }
            }
        }
        __syncwarp();
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void tril(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                const int global_row_idx_x = (i * RT::tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int global_row_idx_y = (i * RT::tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                const int global_col_idx   = (j * RT::tile_size_col) + ((k % 2) * 8) + (laneid() / 4);

                if (global_col_idx <= global_row_idx_x + diagonal) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else                                               { dst.tiles[i][j].data[k].x = val; }
                if (global_col_idx <= global_row_idx_y + diagonal) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else                                               { dst.tiles[i][j].data[k].y = val; }
            }
        }
        __syncwarp();
    }
}










template<ducks::rt::row_layout RT>
__device__ static inline void triu(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                const int global_row_idx   = (i * RT::tile_size_row) + ((k % 2) * 8) + (laneid() / 4);
                const int global_col_idx_x = (j * RT::tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int global_col_idx_y = (j * RT::tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;

                if (global_col_idx_x >= global_row_idx + diagonal) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else                                               { dst.tiles[i][j].data[k].x = val; }

                if (global_col_idx_y >= global_row_idx + diagonal) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else                                               { dst.tiles[i][j].data[k].y = val; }
            }
        }
        __syncwarp();
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void triu(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                const int global_row_idx_x = (i * RT::tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int global_row_idx_y = (i * RT::tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                const int global_col_idx   = (j * RT::tile_size_col) + ((k % 2) * 8) + (laneid() / 4);

                if (global_col_idx >= global_row_idx_x + diagonal) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else                                               { dst.tiles[i][j].data[k].x = val; }

                if (global_col_idx >= global_row_idx_y + diagonal) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else                                               { dst.tiles[i][j].data[k].y = val; }
            }
        }
        __syncwarp();
    }
}












template<ducks::rt::row_layout RT>
__device__ static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    if(col_idx >= RT::cols) return;
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                const int col_idx_x = (j * RT::tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int col_idx_y = (j * RT::tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                if (col_idx_x >= col_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                       { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (col_idx_y >= col_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                       { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                const int t_col_idx = (j * RT::tile_size_col) + ((k % 2) * 8) + (laneid() / 4);
                if (t_col_idx >= col_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
        __syncwarp();
    }
}










template<ducks::rt::row_layout RT>
__device__ static inline void left_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    if(col_idx <= 0) return;
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                const int col_idx_x = (j * RT::tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int col_idx_y = (j * RT::tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                if (col_idx_x < col_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                      { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (col_idx_y < col_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                      { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void left_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                const int thread_col = (j * RT::tile_size_col) + ((k % 2) * 8) + ((laneid() / 4));
                if (thread_col < col_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
        __syncwarp();
    }
}










template<ducks::rt::row_layout RT>
__device__ static inline void upper_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    if(row_idx <= 0) return;
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                const int thread_row = (i * RT::tile_size_row) + ((k % 2) * 8) + ((laneid() / 4));
                if (thread_row < row_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void upper_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                const int row_idx_x = (i * RT::tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int row_idx_y = (i * RT::tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                if (row_idx_x < row_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                      { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (row_idx_y < row_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                      { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}










template<ducks::rt::row_layout RT>
__device__ static inline void lower_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    if(row_idx >= RT::rows) return;
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                const int thread_row = (i * RT::tile_size_row) + ((k % 2) * 8) + ((laneid() / 4));
                if (thread_row >= row_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                        { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void lower_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                const int row_idx_x = (i * RT::tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int row_idx_y = (i * RT::tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                if (row_idx_x >= row_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                       { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (row_idx_y >= row_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                       { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}













template<ducks::rt::row_layout RT>
__device__ static inline void upper_right_fill(RT &dst, const RT &src, const int row_idx, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    if(col_idx >= RT::cols || row_idx < 0) return;
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                const int row_idx_xy = (i * RT::tile_size_row) + ((k % 2) * 8) + ((laneid() / 4));
                const int col_idx_x = (j * RT::tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int col_idx_y = (j * RT::tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                if (col_idx_x >= col_idx && row_idx_xy < row_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                       { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (col_idx_y >= col_idx && row_idx_xy < row_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                       { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
        __syncwarp();
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void upper_right_fill(RT &dst, const RT &src, const int row_idx, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    if(col_idx >= RT::cols || row_idx < 0) return;
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                const int row_idx_x = (i * RT::tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int row_idx_y = (i * RT::tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                const int col_idx_xy = (j * RT::tile_size_col) + ((k % 2) * 8) + (laneid() / 4);
                if (row_idx_x < row_idx && col_idx_xy < col_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                      { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (row_idx_y < row_idx && col_idx_xy < col_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                      { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
        __syncwarp();
    }
}













template<int subtile_rows, ducks::rt::all RT>
__device__ static inline rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout> &subtile_inplace(RT & src, int idx) {
    KITTENS_CHECK_WARP
    using T = typename RT::T;
    static_assert(RT::height % (subtile_rows / TILE_ROW_DIM<T>) == 0, "subtile height should evenly divide tile height.");
    return reinterpret_cast<rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout>&>(
        src.tiles[idx*(subtile_rows / TILE_ROW_DIM<T>)]
    );
}
