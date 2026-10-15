












template<ducks::rv::all RV, ducks::gl::all GL>
__device__ inline static void load(RV &dst, const GL &src, const coord<rv<typename RV::T, GROUP_WARPS*RV::length, typename RV::layout>> &idx) {
    if constexpr (GROUP_WARPS == 1) {
        using T2 = RV::dtype;
        using U = typename GL::dtype;
        using U2 = base_types::packing<U>::packed_type;
        using T = base_types::packing<T2>::unpacked_type;

        U *src_ptr = (U*)&src[(idx.template unit_coord<-1, 3>())];
        int laneid = ::kittens::laneid();

        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            #pragma unroll
            for(auto w = 0; w < (RV::outer_dim+3)/4; w++) {
                int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
                int o_dim = w*4 + (laneid/4) / 2;
                int i_dim = (laneid/4) % 2;

                if(idx < RV::outer_dim*16)
                    dst[o_dim][i_dim] = base_types::convertor<T2, U2>::convert(*(U2*)&src_ptr[idx]);
            }

            #pragma unroll
            for(auto w = 0; w < RV::outer_dim; w++) {
                int leader = 8*(w%4) + (laneid%4);
                dst[w][0] = packed_shfl_sync(MASK_ALL, dst[w][0], leader);
                dst[w][1] = packed_shfl_sync(MASK_ALL, dst[w][1], leader+4);
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {


            #pragma unroll
            for(auto w = 0; w < (RV::outer_dim+1)/2; w++) {
                int idx = w*32 + (laneid%4)*8 + (laneid/4);
                int o_dim = w*2 + (laneid%4) / 2;

                if(idx < RV::outer_dim*16) {
                    T tmp = base_types::convertor<T, U>::convert(src_ptr[idx]);
                    if(laneid%2==0) dst[o_dim][0].x = tmp;
                    else dst[o_dim][0].y = tmp;
                }
            }

            #pragma unroll
            for(auto w = 0; w < RV::outer_dim; w++) {
                int leader = (laneid/4)*4 + 2*(w%2);
                dst[w][0].x = __shfl_sync(MASK_ALL, dst[w][0].x, leader);
                dst[w][0].y = __shfl_sync(MASK_ALL, dst[w][0].y, leader+1);
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
            #pragma unroll
            for(auto w = 0; w < RV::outer_dim; w++) {
                if(w < RV::outer_dim-1 || RV::length%32 == 0 || laneid<16) {
                    dst[w][0] = base_types::convertor<T, U>::convert(src_ptr[w*32 + laneid]);
                }
            }
        }
    }
    else {

        ::kittens::group<1>::load(dst, src, coord<RV>(idx.b, idx.d, idx.r, idx.c*GROUP_WARPS+warpid()));
    }
}








template<ducks::rv::all RV, ducks::gl::all GL>
__device__ inline static void store(GL &dst, const RV &src, const coord<rv<typename RV::T, GROUP_WARPS*RV::length, typename RV::layout>> &idx) {
    if constexpr (GROUP_WARPS == 1) {
        using T2 = RV::dtype;
        using U = typename GL::dtype;
        using U2 = base_types::packing<U>::packed_type;
        using T = base_types::packing<T2>::unpacked_type;

        U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
        int laneid = ::kittens::laneid();

        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            #pragma unroll
            for(auto w = 0; w < (RV::outer_dim+3)/4; w++) {
                int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
                int o_dim = w*4 + (laneid/4) / 2;
                int i_dim = (laneid/4) % 2;

                if(idx < RV::outer_dim*16)
                    *(U2*)&dst_ptr[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {


            #pragma unroll
            for(auto w = 0; w < (RV::outer_dim+1)/2; w++) {
                int idx = w*32 + (laneid%4)*8 + (laneid/4);
                int o_dim = w*2 + (laneid%4) / 2;

                if(idx < RV::outer_dim*16) {
                    U tmp;
                    if(laneid%2==0) tmp = base_types::convertor<U, T>::convert(src[o_dim][0].x);
                    else tmp = base_types::convertor<U, T>::convert(src[o_dim][0].y);
                    dst_ptr[idx] = tmp;
                }
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
            #pragma unroll
            for(auto w = 0; w < RV::outer_dim; w++) {
                if(w < RV::outer_dim-1 || RV::length%32 == 0 || laneid<16) {
                    dst_ptr[w*32 + laneid] = base_types::convertor<U, T>::convert(src[w][0]);
                }
            }
        }
    }
    else {

        ::kittens::group<1>::store(dst, src, coord<RV>(idx.b, idx.d, idx.r, idx.c*GROUP_WARPS+warpid()));
    }
}
