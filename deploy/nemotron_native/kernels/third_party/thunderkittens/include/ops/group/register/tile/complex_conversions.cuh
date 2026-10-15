



















template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void swap_layout(crt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type> &dst, const crt<T2, _height, _width, layout> &src) {
    swap_layout(dst.real, src.real);
    swap_layout(dst.real, src.real);
}










template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline crt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(crt<T2, _height, _width, layout> &tile) {
    tile.real = swap_layout_inplace(tile.real);
    tile.imag = swap_layout_inplace(tile.imag);
    return tile;
}
















template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void transpose_sep(crt<T2, _width, _height, layout> &dst, const crt<T2, _height, _width, layout> &src) {
    transpose_sep(dst.real, src.real);
    transpose_sep(dst.imag, src.imag);
}










template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline crt<T2, _height, _width, layout>& transpose_inplace(crt<T2, _height, _width, layout> &tile) {
    tile.real = transpose_inplace(tile.real);
    tile.imag = transpose_inplace(tile.imag);

    return tile;
}














template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void copy(crt<T2, _height, _width, layout> &dst, const crt<U2, _height, _width, layout> &src) {
    copy(dst.real, src.real);
    copy(dst.imag, src.imag);
}
