




#pragma once

#include "../../common/common.cuh"
#include "sv.cuh"




namespace kittens {
namespace ducks {





namespace st {







struct identifier {};







template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;
}
}


template<
    typename ST,
    int _subtile_height,
    int _subtile_width
>
struct st_subtile;








template<typename _T, int _rows, int _cols, bool _swizzle=true, int _swizzle_bytes=0>
struct KITTENS_DEFAULT_ALIGN st {
    static_assert(!std::is_same_v<_T, char>, "Bare 'char' is not permitted because its signedness is ISA-dependent.");
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    static_assert(!std::is_same_v<_T, fp4e2m1>, "For FP4 types, you must use a packed type (i.e., fp4e2m1_2 or fp4e2m1_4).");
#endif
    using identifier = ducks::st::identifier;
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T;

    static constexpr bool swizzle = _swizzle;


    static constexpr int underlying_rows          = _rows;
    static constexpr int underlying_cols          = _cols;
    static constexpr int underlying_num_elements  = underlying_rows * underlying_cols;

    static constexpr int rows                = _rows;
    static constexpr int cols                = _cols;
    static constexpr int num_elements        = rows * cols;

    static_assert(!swizzle || (rows % kittens::TILE_ROW_DIM<T> == 0), "Rows must be divisible by the tile dimension");
    static_assert((swizzle && (cols % kittens::TILE_COL_DIM<T> == 0)) || (!swizzle && (cols % kittens::BASE_TILE_DIM == 0)), "Cols must be divisible by the tile dimension");

#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)

    static_assert(base_types::packing<dtype>::num() == 1 || std::is_same_v<dtype, fp4e2m1_2>);
#else
    static_assert(base_types::packing<dtype>::num() == 1);
#endif


    static_assert(_swizzle_bytes == 0 || _swizzle_bytes == 32 || _swizzle_bytes == 64 || _swizzle_bytes == 128);
    static constexpr int swizzle_bytes = _swizzle_bytes > 0 ? _swizzle_bytes : (
        sizeof(dtype) == 1 ? (
            (cols/kittens::TILE_COL_DIM<T>)%4 == 0 ? 128 :
            (cols/kittens::TILE_COL_DIM<T>)%2 == 0 ?  64 : 32
        ) :
        sizeof(dtype) == 2 ? (
            (cols/kittens::TILE_COL_DIM<T>)%4 == 0 ? 128 :
            (cols/kittens::TILE_COL_DIM<T>)%2 == 0 ?  64 : 32
        ) :
        sizeof(dtype) == 4 ? (
            (cols/kittens::TILE_COL_DIM<T>)%2 == 0 ? 128 : 64
        ) : -1
    );

    dtype data[rows*cols];

    __device__ static inline T* idx(T *ptr, int2 coord) {
        int r = coord.x, c = coord.y;
        if constexpr (swizzle) {
            static constexpr int swizzle_repeat = swizzle_bytes * 8;
            static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
            const int outer_idx = c/subtile_cols;
            const uint64_t addr = (uint64_t)(&ptr[outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
            return (T*)(addr ^ swizzle);
        } else {
            return &ptr[r*cols + c];
        }
    }
    __device__ static inline uint32_t idx(uint32_t ptr, int2 coord) {
        int r = coord.x, c = coord.y;
        if constexpr (swizzle) {
            static constexpr int swizzle_repeat = swizzle_bytes * 8;
            static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
            const int outer_idx = c/subtile_cols;
            const uint32_t addr = ptr + sizeof(T)*(outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
            return (addr ^ swizzle);
        } else {
            return ptr + sizeof(T)*(r*cols + c);
        }
    }






    __device__ inline       dtype& operator[](const int2 &rowcol)       {
        return *idx(data, rowcol);
    }
    __device__ inline const dtype& operator[](const int2 &rowcol) const {
        return *(const dtype*)idx((dtype*)data, rowcol);
    }
    __device__ inline       dtype& operator[](int idx)       {
        return data[idx];
    }
    __device__ inline const dtype& operator[](int idx) const {
        return data[idx];
    }

    template<int subtile_rows, int subtile_cols>
    __device__ inline st_subtile<st<_T, _rows, _cols, _swizzle, _swizzle_bytes>, subtile_rows, subtile_cols> subtile(int2 rowcol);





    template<int subtile_cols>
    __device__ inline st<_T, _rows, subtile_cols, _swizzle, swizzle_bytes                                > &subtile(int idx) {
        static_assert(swizzle_bytes > 0, "Parent shared tile must have an explicit swizzle_bytes.");
        constexpr int swizzle_elements = swizzle_bytes / sizeof(T);
        static_assert(subtile_cols >= 0 && subtile_cols % swizzle_elements == 0);
        return *reinterpret_cast<st<_T, _rows, subtile_cols, _swizzle, swizzle_bytes> *>(
            &data[rows*swizzle_elements*(subtile_cols/swizzle_elements)*idx]
        );
    }


    using col_vec = sv<dtype, rows>;
    using row_vec = sv<dtype, cols>;
};













template<
    typename _ST,
    int _subtile_rows,
    int _subtile_cols
>
struct st_subtile {
    using identifier = ducks::st::identifier;
    using ST = _ST;
    using T = ST::T;
    using T2 = ST::T2;
    using dtype = T;

    static constexpr bool swizzle = ST::swizzle;

    static constexpr int underlying_rows          = ST::underlying_rows;
    static_assert(underlying_rows % kittens::TILE_ROW_DIM<T> == 0, "Underlying rows must be divisible by the tile dimension");
    static constexpr int underlying_cols          = ST::underlying_cols;
    static_assert(underlying_cols % kittens::TILE_COL_DIM<T> == 0, "Underlying cols must be divisible by the tile dimension");
    static constexpr int underlying_num_elements  = ST::underlying_num_elements;

    static constexpr int rows                = _subtile_rows;
    static_assert(rows % kittens::TILE_ROW_DIM<T> == 0, "Rows must be divisible by the tile dimension");
    static constexpr int cols                = _subtile_cols;
    static_assert(cols % kittens::TILE_COL_DIM<T> == 0, "Cols must be divisible by the tile dimension");
    static constexpr int num_elements        = rows * cols;

    static constexpr int swizzle_bytes = ST::swizzle_bytes;

    dtype *data;
    int row_offset, col_offset;

    __device__ st_subtile(ST &src, int2 rowcol) {
        data = &src.data[0];
        row_offset = rowcol.x * rows;
        col_offset = rowcol.y * cols;
    }

    __device__ inline T* idx(T *ptr, const int2 coord) const {
        int r = coord.x+row_offset, c = coord.y+col_offset;
        if constexpr (swizzle) {
            static constexpr int swizzle_repeat = swizzle_bytes * 8;
            static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
            const int outer_idx = c/subtile_cols;
            const uint64_t addr = (uint64_t)(&ptr[outer_idx*underlying_rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
            return (T*)(addr ^ swizzle);
        } else {
            return &ptr[r*cols + c];
        }
    }
    __device__ inline uint32_t idx(uint32_t ptr, const int2 coord) const {
        int r = coord.x+row_offset, c = coord.y+col_offset;
        if constexpr(swizzle) {
            static constexpr int swizzle_repeat = swizzle_bytes * 8;
            static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
            const int outer_idx = c/subtile_cols;
            const uint32_t addr = ptr + sizeof(T)*(outer_idx*underlying_rows*subtile_cols + r*subtile_cols + c%subtile_cols);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
            return (addr ^ swizzle);
        } else {
            return ptr + sizeof(T)*(r*cols + c);
        }
    }






    __device__ inline       dtype& operator[](const int2 &rowcol)       {
        return *idx(data, rowcol);
    }
    __device__ inline const dtype& operator[](const int2 &rowcol) const {
        return *(const dtype*)idx((dtype*)data, rowcol);
    }





    using col_vec = sv<dtype, rows>;
    using row_vec = sv<dtype, cols>;

    __device__ inline void operator=(const dtype &value) {
        #pragma unroll
        for(int i = kittens::laneid(); i < num_elements; i += WARP_THREADS) {
            data[i] = value;
        }
    }
};

template <typename _T, int _rows, int _cols, bool _swizzle, int _swizzle_bytes>
template <int subtile_rows, int subtile_cols>
__device__ inline st_subtile<st<_T, _rows, _cols, _swizzle, _swizzle_bytes>, subtile_rows, subtile_cols>
st<_T, _rows, _cols, _swizzle, _swizzle_bytes>::subtile(int2 rowcol)
{

    using ST_t = st<_T, _rows, _cols>;
    using dtype = typename ST_t::dtype;


    static_assert(subtile_rows > 0 && subtile_cols > 0, "Subtile dimensions must be positive.");
    static_assert(subtile_rows % kittens::TILE_ROW_DIM<dtype> == 0,
        "Subtile rows must be divisible by the base tile row dimension.");
    static_assert(subtile_cols % kittens::TILE_COL_DIM<dtype> == 0,
        "Subtile cols must be divisible by the base tile col dimension.");


    static_assert(ST_t::rows % subtile_rows == 0,
        "Parent tile rows must be divisible by subtile rows.");
    static_assert(ST_t::cols % subtile_cols == 0,
        "Parent tile cols must be divisible by subtile cols.");



    static_assert(ST_t::rows == ST_t::underlying_rows && ST_t::cols == ST_t::underlying_cols,
        "Cannot create a subtile from an object that appears to be a subtile view (rows/cols mismatch underlying).");




    return st_subtile<ST_t, subtile_rows, subtile_cols>(*this, rowcol);
}



template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0>
using st_bf = st<bf16,  _height, _width, _swizzle, _swizzle_bytes>;
template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0>
using st_hf = st<half,  _height, _width, _swizzle, _swizzle_bytes>;
template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0>
using st_fl = st<float, _height, _width, _swizzle, _swizzle_bytes>;
template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0>
using st_int8 = st<int8, _height, _width, _swizzle, _swizzle_bytes>;
template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0>
using st_uint8 = st<uint8, _height, _width, _swizzle, _swizzle_bytes>;
template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0>
using st_int = st<int, _height, _width, _swizzle, _swizzle_bytes>;
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0>
using st_fp8e4m3 = st<fp8e4m3, _height, _width, _swizzle, _swizzle_bytes>;
template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0>
using st_fp8e5m2 = st<fp8e5m2, _height, _width, _swizzle, _swizzle_bytes>;
#endif
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0>
using st_fp8e8m0 = st<fp8e8m0, _height, _width, _swizzle, _swizzle_bytes>;
template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0>
using st_fp4e2m1_2 = st<fp4e2m1_2, _height, _width, _swizzle, _swizzle_bytes>;
#endif






template<typename T, int rows, int cols>
__device__ constexpr const char* get_tile_type_name() {
    if constexpr (std::is_same_v<T, float>) {
        return "st_fl";
    } else if constexpr (std::is_same_v<T, half>) {
        return "st_hf";
    } else if constexpr (std::is_same_v<T, bf16>) {
        return "st_bf";
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    } else if constexpr (std::is_same_v<T, fp4e2m1_2>) {
        return "st_fp4_e2m1_2";
    } else if constexpr (std::is_same_v<T, fp8e8m0>) {
        return "st_fp8_e8m0";
#endif
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    } else if constexpr (std::is_same_v<T, fp8e4m3>) {
        return "st_fp8_e4m3";
    } else if constexpr (std::is_same_v<T, fp8e5m2>) {
        return "st_fp8_e5m2";
#endif
    } else {
        return "st_unknown";
    }
}

#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)








template<ducks::st::all ST>
__device__ inline void print_fp4(const ST& tile) {
    if (std::is_same_v<typename ST::dtype, fp4e2m1>) {

        constexpr int cols = ST::cols * 2;
        printf("Block %d: Shared Tile %dx%d (Type: %s<%d,%d>):\n", blockIdx.x, ST::rows, cols, get_tile_type_name<typename ST::dtype, ST::rows, cols>(), ST::rows, cols);


        printf("     ");
        for (int c = 0; c < cols; c++) {
            printf("%8d ", c);
        }
        printf("\n");


        printf("     ");
        for (int c = 0; c < cols; c++) {
            printf("--------+");
        }
        printf("\n");


        for (int r = 0; r < ST::rows; r++) {
            printf("%3d |", r);
            for (int c = 0; c < cols; c += 2) {
                uint8_t *vals = reinterpret_cast<uint8_t*>(const_cast<fp4e2m1*>(&tile[{r,c/2}]));


                float f1 = static_cast<float>(fp4e2m1(vals[0] & 0xF));
                float f2 = static_cast<float>(fp4e2m1((vals[0] >> 4) & 0xF));

                printf("%8.3f %8.3f ", f1, f2);

            }
            printf("\n");
        }
        printf("\n");
    } else {
        printf("Type must be FP4 in this function\n");
    }
}
#endif









template<ducks::st::all ST>
__device__ inline void print(const ST& tile) {
    printf("Block %d: Shared Tile %dx%d (Type: %s<%d,%d>):\n", blockIdx.x, ST::rows, ST::cols, get_tile_type_name<typename ST::dtype, ST::rows, ST::cols>(), ST::rows, ST::cols);


    printf("     ");
    for (int c = 0; c < ST::cols; c++) {
        printf("%8d ", c);
    }
    printf("\n");


    printf("     ");
    for (int c = 0; c < ST::cols; c++) {
        printf("--------+");
    }
    printf("\n");


    for (int r = 0; r < ST::rows; r++) {
        printf("%3d |", r);
        for (int c = 0; c < ST::cols; c++) {
            if constexpr (std::is_same_v<typename ST::dtype, float>) {
                printf("%8.3f ", tile[{r,c}]);
            } else if constexpr (std::is_same_v<typename ST::dtype, __nv_bfloat16>) {
                printf("%8.3f ", __bfloat162float(tile[{r,c}]));
            } else if constexpr (std::is_integral_v<typename ST::dtype>) {
                printf("%8d ", (int)tile[{r,c}]);
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
            } else if constexpr (std::is_same_v<typename ST::dtype, fp8e4m3>) {
                printf("%8.3f ", static_cast<float>(tile[{r,c}]));
            } else if constexpr (std::is_same_v<typename ST::dtype, fp8e5m2>) {
                printf("%8.3f ", static_cast<float>(tile[{r,c}]));
#endif
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
            } else if constexpr (std::is_same_v<typename ST::dtype, fp8e8m0>) {
                printf("%8.3f ", static_cast<float>(tile[{r,c}]));
#endif
            } else {
                printf("%8.3f ", (float)tile[{r,c}]);
            }
        }
        printf("\n");
    }
    printf("\n");
}









template<typename ST, int subtile_rows, int subtile_cols>
__device__ inline void print(const st_subtile<ST, subtile_rows, subtile_cols>& subtile) {
    printf("Block %d: Shared Subtile %dx%d (offset: [%d,%d], Type: %s<%d,%d> from %s<%d,%d>):\n",
            blockIdx.x, subtile.rows, subtile.cols,
            subtile.row_offset, subtile.col_offset,
            get_tile_type_name<typename ST::dtype, subtile.rows, subtile.cols>(), subtile.rows, subtile.cols,
            get_tile_type_name<typename ST::dtype, ST::rows, ST::cols>(), ST::rows, ST::cols);


    printf("     ");
    for (int c = 0; c < subtile.cols; c++) {
        printf("%8d ", c);
    }
    printf("\n");


    printf("     ");
    for (int c = 0; c < subtile.cols; c++) {
        printf("--------+");
    }
    printf("\n");


    for (int r = 0; r < subtile.rows; r++) {
        printf("%3d |", r);
        for (int c = 0; c < subtile.cols; c++) {
            if constexpr (std::is_same_v<typename ST::dtype, float>) {
                printf("%8.3f ", subtile[{r,c}]);
            } else if constexpr (std::is_same_v<typename ST::dtype, __nv_bfloat16>) {
                printf("%8.3f ", __bfloat162float(subtile[{r,c}]));
            } else if constexpr (std::is_integral_v<typename ST::dtype>) {
                printf("%8d ", (int)subtile[{r,c}]);
            } else {
                printf("%8.3f ", (float)subtile[{r,c}]);
            }
        }
        printf("\n");
    }
    printf("\n");
}









template<ducks::st::all ST>
__device__ inline void fill_value(ST& tile, float value) {
    printf("Filling Tile %dx%d with %f:\n", ST::rows, ST::cols, value);


    for (int r = 0; r < ST::rows; r++) {
        for (int c = 0; c < ST::cols; c++) {
            if constexpr (std::is_same_v<typename ST::dtype, float>) {
                tile[{r,c}] = value;
            } else if constexpr (std::is_same_v<typename ST::dtype, __nv_bfloat16>) {
                tile[{r,c}] = __float2bfloat16(value);
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
            } else if constexpr (std::is_same_v<typename ST::dtype, fp4e2m1>) {
                tile.data[r*ST::cols + c] = fp4e2m1(value);
            } else if constexpr (std::is_same_v<typename ST::dtype, fp8e8m0>) {
                tile.data[r*ST::cols + c] = fp8e8m0(value);
#endif
            } else {
                tile[{r,c}] = value;
            }
        }
    }
}









template<ducks::st::all ST>
__device__ inline void fill_identity(ST& tile) {
    printf("Filling Tile %dx%d with identity:\n", ST::rows, ST::cols);


    for (int r = 0; r < ST::rows; r++) {
        for (int c = 0; c < ST::cols; c++) {
            if constexpr (std::is_same_v<typename ST::dtype, float>) {

                tile[{r,c}] = 1.0f;
            } else if constexpr (std::is_same_v<typename ST::dtype, __nv_bfloat16>) {

                tile[{r,c}] = __float2bfloat16(1.0f);
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
            } else if constexpr (std::is_same_v<typename ST::dtype, fp4e2m1>) {
                if(r == c){

                    tile.data[r*ST::cols + c] = std::bit_cast<fp4e2m1>(uint8_t(0xFF));
                } else {
                    tile.data[r*ST::cols + c] = std::bit_cast<fp4e2m1>(uint8_t(0x00));
                }
#endif
            } else {

            }
        }
    }
}










template<ducks::st::all ST>
__device__ inline void print_bits(const ST& tile, bool unswizzle = false) {
    printf("Block %d: Shared Tile %dx%d (Type: %s<%d,%d>):\n", blockIdx.x, ST::rows, ST::cols, get_tile_type_name<typename ST::dtype, ST::rows, ST::cols>(), ST::rows, ST::cols);


    printf(" ");
    for (int c = 0; c < ST::cols; c++) {
        printf("%11d ", c);
    }
    printf("\n");


    printf("     ");
    for (int c = 0; c < ST::cols; c++) {
        printf("-----------+");
    }
    printf("\n");


    for (int r = 0; r < ST::rows; r++) {
        printf("%3d |", r);
        for (int c = 0; c < ST::cols; c++) {
            if constexpr (std::is_same_v<typename ST::dtype, float>) {
                printf("%8.3f ", tile[{r,c}]);
            } else if constexpr (std::is_same_v<typename ST::dtype, __nv_bfloat16>) {
                printf("%8.3f ", __bfloat162float(tile[{r,c}]));


#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
            } else if constexpr (std::is_same_v<typename ST::dtype, fp4e2m1> || std::is_same_v<typename ST::dtype, fp8e8m0>) {


                uint8_t bits;
                if(unswizzle){
                    bits = *reinterpret_cast<const uint8_t*>(&tile.data[r*ST::cols + c]);
                } else {
                    bits = *reinterpret_cast<const uint8_t*>(&tile[{r,c}]);
                }

                printf("0b");

                for (int bit = 7; bit >= 0; bit--) {
                    printf("%d", (bits >> bit) & 0x1);
                    if (bit % 4 == 0 && bit > 0) printf("_");
                }
                printf(" ");
#endif
            } else {
                printf("%8.3f ", (float)tile[{r,c}]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

}
