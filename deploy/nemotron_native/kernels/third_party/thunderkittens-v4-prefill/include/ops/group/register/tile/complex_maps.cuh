










template<ducks::crt::all T>
__device__ static inline void zero(T &dst) {
    zero(dst.real);
    zero(dst.imag);
}







template<ducks::crt::all T>
__device__ static inline void exp(T &dst, const T &src) {
    using dtype = T::dtype;
    dtype tmp;

    dtype rdst;
    dtype idst;


    exp(rdst, src.real);
    copy(idst, rdst);

    cos(tmp, src.imag);
    mul(rdst, rdst, tmp);
    sin(tmp, src.imag);
    mul(idst, idst, tmp);

    copy(dst.real, rdst);
    copy(dst.imag, idst);
}








template<ducks::crt::all T>
__device__ static inline void add(T &dst, const T &lhs, const T &rhs) {
    add(dst.real, lhs.real, rhs.real);
    add(dst.imag, lhs.imag, rhs.imag);
}









template<ducks::crt::all T>
__device__ static inline void sub(T &dst, const T &lhs, const T &rhs) {
    sub(dst.real, lhs.real, rhs.real);
    sub(dst.imag, lhs.imag, rhs.imag);
}








template<ducks::crt::all T>
__device__ static inline void mul(T &dst, const T &lhs, const T &rhs) {
    using dtype = T::component;
    dtype tmp;

    dtype rdst;
    dtype idst;



    mul(rdst, lhs.real, rhs.real);
    mul(tmp, lhs.imag, rhs.imag);
    sub(rdst, rdst, tmp);


    mul(idst, lhs.imag, rhs.real);
    mul(tmp, lhs.real, rhs.imag);
    add(idst, idst, tmp);

    copy(dst.real, rdst);
    copy(dst.imag, idst);
}








template<ducks::crt::all T>
__device__ static inline void div(T &dst, const T &lhs, const T &rhs) {
    using dtype = T::dtype;
    dtype tmp;
    dtype denom;

    dtype rdst;
    dtype idst;


    mul(tmp, rhs.real, rhs.real);
    mul(denom, rhs.imag, rhs.imag);
    add(denom, tmp, denom);

    mul(rdst, lhs.real, rhs.real);
    mul(tmp, lhs.imag, rhs.imag);
    add(rdst, rdst, tmp);

    mul(dst.imag, lhs.imag, rhs.real);
    mul(tmp, lhs.real, rhs.imag);
    sub(idst, idst, tmp);

    div(rdst, rdst, denom);
    div(idst, idst, denom);
    copy(dst.real, rdst);
    copy(dst.imag, idst);
}


