













template<int axis, ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void load(CRT &dst, const CGL &src, const COORD &idx) {
    load<axis, CRT::component, CGL::component, COORD>(dst.real, src.real, idx);
    load<axis, CRT::component, CGL::component, COORD>(dst.imag, src.imag, idx);
}
template<ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void load(CRT &dst, const CGL &src, const COORD &idx) {
    load<2, CRT, CGL>(dst, src, idx);
}










template<int axis, ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void store(CGL &dst, const CRT &src, const COORD &idx) {
    store<axis, typename CRT::component, typename CGL::component>(dst.real, src.real, idx);
    store<axis, typename CRT::component, typename CGL::component>(dst.imag, src.imag, idx);
}
template<ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void store(CGL &dst, const CRT &src, const COORD &idx) {
    store<2, CRT, CGL>(dst, src, idx);
}
