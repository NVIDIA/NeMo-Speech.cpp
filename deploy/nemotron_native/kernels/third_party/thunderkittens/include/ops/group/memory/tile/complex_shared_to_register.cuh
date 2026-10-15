












template<ducks::crt::all CRT, ducks::cst::all CST>
__device__ inline static void load(CRT &dst, const CST &src) {
    load(dst.real, src.real);
    load(dst.imag, src.imag);
}









template<ducks::crt::all CRT, ducks::cst::all CST>
__device__ inline static void store(CST &dst, const CRT &src) {
    store(dst.real, src.real);
    store(dst.imag, src.imag);
}
