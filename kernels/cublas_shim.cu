// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Drop-in libcublas.so.13 replacement for nemo-speech.
//
// ggml-cuda calls a small set of cuBLAS GEMM entry points for the matmuls that
// are not quantized (FastConformer attention scores/context, the subsampling
// convs, and the CTC head). The real cuBLAS pulls in libcublasLt (514 MB). This
// shim implements exactly the symbols ggml imports, backed by CUDA kernels
// specialized for the shapes used here (including WMMA tensor-core paths, but
// no cuBLASLt), so the runtime needs neither real cuBLAS nor cuBLASLt.
//
// Built as `libcublas.so.13` (SONAME + symbols versioned `libcublas.so.13`);
// it is the image's only libcublas. ggml is not modified.
//
// cublas_v2.h is deliberately not included: it tags these functions
// __host__ __device__ under nvcc. The cuBLAS enums/handles are passed at fixed
// ABI values, so plain int/void* declarations stay call-compatible.
//
// Semantics match cuBLAS: C(m x n, col-major, ldc) = alpha*op(A)*op(B) + beta*C,
// op = N|T, with strided- or pointer-array batching. Accumulate in fp32.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

// cuBLAS ABI constants (from cublas_api.h / library_types.h)
enum { OP_N = 0, OP_T = 1 };
enum { R_32F = 0, R_16F = 2, R_16BF = 14 };
enum { COMPUTE_16F = 64, COMPUTE_32F = 68 };
enum { STATUS_SUCCESS = 0 };

// cudaDataType is already defined by library_types.h (via cuda_runtime.h).
typedef void* cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;
typedef int cublasComputeType_t;
typedef int cublasGemmAlgo_t;
typedef int cublasMath_t;
typedef int cublasSideMode_t;
typedef int cublasFillMode_t;
typedef int cublasDiagType_t;

namespace {
struct ShimHandle {
    struct SplitKWorkspace {
        void* ptr = nullptr;
        bool async = false;
        uint64_t last_used = 0;
    };

    int device = -1;
    cudaStream_t stream = nullptr;
    std::mutex mutex;
    std::unordered_map<cudaStream_t, SplitKWorkspace> splitk_workspaces;
    uint64_t splitk_workspace_clock = 0;
};

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kGemvWarpsPerBlock = 4;
constexpr int kBlockWmmaM = 64;
constexpr int kBlockWmmaN = 32;
constexpr int kBlockWmmaN2 = 64;
constexpr int kBlockWmmaWarps = 8;
constexpr int kSmallNMax = 8;
constexpr int kSmallMWmmaN = 64;
constexpr int kSmallMWmmaWarps = 4;
constexpr int kSmallMSplitK = 32;
// The small-M split kernel tiles M in 16-row blocks.
constexpr int kSmallMSplitMaxM = 32;
constexpr int kSmallMSplitMaxN = 1024;
constexpr int kBlockSplitK = 8;
constexpr size_t kMaxSplitKWorkspaces = 8;
// Eight FP32 partitions keep the per-stream workspace bounded to 12 MiB.
constexpr int kBlockSplitMaxElems = 384 * 1024;
constexpr size_t kSmallMSplitWorkspaceFloats =
    (size_t)kSmallMSplitK * kSmallMSplitMaxM * kSmallMSplitMaxN;
constexpr size_t kBlockSplitWorkspaceFloats = (size_t)kBlockSplitK * kBlockSplitMaxElems;
constexpr size_t kShimWorkspaceFloats = kSmallMSplitWorkspaceFloats > kBlockSplitWorkspaceFloats
                                            ? kSmallMSplitWorkspaceFloats
                                            : kBlockSplitWorkspaceFloats;
constexpr size_t kShimWorkspaceBytes = kShimWorkspaceFloats * sizeof(float);

void
free_splitk_workspace(ShimHandle::SplitKWorkspace& workspace) {
    if (workspace.ptr != nullptr) {
        cudaFree(workspace.ptr);
        cudaGetLastError();
        workspace.ptr = nullptr;
    }
}

void*
splitk_workspace_for_stream_locked(ShimHandle* sh, cudaStream_t stream) {
    auto it = sh->splitk_workspaces.find(stream);
    if (it != sh->splitk_workspaces.end()) {
        it->second.last_used = ++sh->splitk_workspace_clock;
        return it->second.ptr;
    }

    if (sh->splitk_workspaces.size() >= kMaxSplitKWorkspaces) {
        auto victim = std::min_element(
            sh->splitk_workspaces.begin(), sh->splitk_workspaces.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.second.last_used < rhs.second.last_used;
            });
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        free_splitk_workspace(victim->second);
        sh->splitk_workspaces.erase(victim);
    }

    ShimHandle::SplitKWorkspace workspace;
    if (cudaMallocAsync(&workspace.ptr, kShimWorkspaceBytes, stream) == cudaSuccess) {
        workspace.async = true;
    } else {
        workspace.ptr = nullptr;
        cudaGetLastError();
        if (cudaMalloc(&workspace.ptr, kShimWorkspaceBytes) != cudaSuccess) {
            workspace.ptr = nullptr;
            cudaGetLastError();
            return nullptr;
        }
    }

    workspace.last_used = ++sh->splitk_workspace_clock;
    void* ptr = workspace.ptr;
    sh->splitk_workspaces.emplace(stream, workspace);
    return ptr;
}

cudaStream_t
stream_for_handle(ShimHandle* sh) {
    std::lock_guard<std::mutex> lock(sh->mutex);
    return sh->stream;
}

__device__ __forceinline__ float
ld(const void* p, size_t idx, int dt) {
    if (dt == R_32F)
        return ((const float*)p)[idx];
    if (dt == R_16F)
        return __half2float(((const __half*)p)[idx]);
    return __bfloat162float(((const __nv_bfloat16*)p)[idx]);
}
__device__ __forceinline__ void
st(void* p, size_t idx, int dt, float v) {
    if (dt == R_32F)
        ((float*)p)[idx] = v;
    else if (dt == R_16F)
        ((__half*)p)[idx] = __float2half(v);
    else
        ((__nv_bfloat16*)p)[idx] = __float2bfloat16(v);
}
__device__ __forceinline__ void
gemm_one(
    int i, int j, int k, int opA, int opB, const void* A, int lda, int ta, const void* B, int ldb,
    int tb, void* C, int ldc, int tc, float alpha, float beta) {
    float acc = 0.f;
    for (int l = 0; l < k; l++) {
        size_t ai = (opA == OP_N) ? ((size_t)i + (size_t)l * lda) : ((size_t)l + (size_t)i * lda);
        size_t bi = (opB == OP_N) ? ((size_t)l + (size_t)j * ldb) : ((size_t)j + (size_t)l * ldb);
        acc += ld(A, ai, ta) * ld(B, bi, tb);
    }
    size_t ci = (size_t)i + (size_t)j * ldc;
    float out = alpha * acc + (beta != 0.f ? beta * ld(C, ci, tc) : 0.f);
    st(C, ci, tc, out);
}
__global__ void
k_strided(
    int m, int n, int k, int opA, int opB, const void* A, int lda, long long sa, int ta,
    const void* B, int ldb, long long sb, int tb, void* C, int ldc, long long sc, int tc,
    float alpha, float beta, size_t esa, size_t esb, size_t esc, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y,
        b = blockIdx.z;
    if (i >= m || j >= n || b >= batch)
        return;
    gemm_one(
        i, j, k, opA, opB, (const char*)A + (size_t)b * sa * esa, lda, ta,
        (const char*)B + (size_t)b * sb * esb, ldb, tb, (char*)C + (size_t)b * sc * esc, ldc, tc,
        alpha, beta);
}
__global__ void
k_ptrs(
    int m, int n, int k, int opA, int opB, const void* const* Ap, int lda, int ta,
    const void* const* Bp, int ldb, int tb, void* const* Cp, int ldc, int tc, float alpha,
    float beta, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y,
        b = blockIdx.z;
    if (i >= m || j >= n || b >= batch)
        return;
    gemm_one(i, j, k, opA, opB, Ap[b], lda, ta, Bp[b], ldb, tb, Cp[b], ldc, tc, alpha, beta);
}

__global__ void
k_hgemm_tn_wmma(
    int m, int n, int k, const void* A, int lda, long long sa, const void* B, int ldb, long long sb,
    void* C, int ldc, long long sc, int tc, float alpha, float beta, size_t esc, int batch) {
#if __CUDA_ARCH__ >= 700
    const int tile_m = blockIdx.x * kWmmaM;
    const int tile_n = blockIdx.y * kWmmaN;
    const int b = blockIdx.z;
    if (b >= batch) {
        return;
    }

    const __half* Ab = (const __half*)((const char*)A + (size_t)b * sa * sizeof(__half));
    const __half* Bb = (const __half*)((const char*)B + (size_t)b * sb * sizeof(__half));
    void* Cb = (char*)C + (size_t)b * sc * esc;

    __shared__ __half As[kWmmaM * kWmmaK];
    __shared__ __half Bs[kWmmaK * kWmmaN];
    __shared__ float Cs[kWmmaM * kWmmaN];

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::col_major>
        b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int kk = 0; kk < k; kk += kWmmaK) {
        for (int idx = threadIdx.x; idx < kWmmaM * kWmmaK; idx += blockDim.x) {
            const int r = idx / kWmmaK;
            const int c = idx - r * kWmmaK;
            const int gr = tile_m + r;
            const int gc = kk + c;
            As[idx] = (gr < m && gc < k) ? Ab[(size_t)gc + (size_t)gr * lda] : __float2half(0.0f);
        }
        for (int idx = threadIdx.x; idx < kWmmaK * kWmmaN; idx += blockDim.x) {
            const int r = idx % kWmmaK;
            const int c = idx / kWmmaK;
            const int gr = kk + r;
            const int gc = tile_n + c;
            Bs[idx] = (gr < k && gc < n) ? Bb[(size_t)gr + (size_t)gc * ldb] : __float2half(0.0f);
        }
        __syncthreads();

        nvcuda::wmma::load_matrix_sync(a_frag, As, kWmmaK);
        nvcuda::wmma::load_matrix_sync(b_frag, Bs, kWmmaK);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(Cs, c_frag, kWmmaM, nvcuda::wmma::mem_col_major);
    __syncthreads();

    for (int idx = threadIdx.x; idx < kWmmaM * kWmmaN; idx += blockDim.x) {
        const int r = idx % kWmmaM;
        const int c = idx / kWmmaM;
        const int gr = tile_m + r;
        const int gc = tile_n + c;
        if (gr >= m || gc >= n) {
            continue;
        }
        const size_t ci = (size_t)gr + (size_t)gc * ldc;
        float out = alpha * Cs[idx] + (beta != 0.0f ? beta * ld(Cb, ci, tc) : 0.0f);
        st(Cb, ci, tc, out);
    }
#else
    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)sa;
    (void)B;
    (void)ldb;
    (void)sb;
    (void)C;
    (void)ldc;
    (void)sc;
    (void)tc;
    (void)alpha;
    (void)beta;
    (void)esc;
    (void)batch;
#endif
}

__global__ void
k_hgemm_tn_wmma_facc(
    int m, int n, int k, const void* A, int lda, long long sa, const void* B, int ldb, long long sb,
    void* C, int ldc, long long sc, int batch) {
#if __CUDA_ARCH__ >= 700
    const int tile_m = blockIdx.x * kWmmaM;
    const int tile_n = blockIdx.y * kWmmaN;
    const int b = blockIdx.z;
    if (b >= batch) {
        return;
    }

    const __half* Ab = (const __half*)((const char*)A + (size_t)b * sa * sizeof(__half));
    const __half* Bb = (const __half*)((const char*)B + (size_t)b * sb * sizeof(__half));
    __half* Cb = (__half*)((char*)C + (size_t)b * sc * sizeof(__half));

    // Half inputs and outputs still need a float accumulator: valid GEMMs can
    // have large partial sums that cancel to a representable final value.
    __shared__ __half As[kWmmaM * kWmmaK];
    __shared__ __half Bs[kWmmaK * kWmmaN];
    __shared__ float Cs[kWmmaM * kWmmaN];

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::col_major>
        b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    const bool full_tile = tile_m + kWmmaM <= m && tile_n + kWmmaN <= n && (k % kWmmaK) == 0;
    for (int kk = 0; kk < k; kk += kWmmaK) {
        if (full_tile) {
            nvcuda::wmma::load_matrix_sync(a_frag, Ab + (size_t)tile_m * lda + kk, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, Bb + kk + (size_t)tile_n * ldb, ldb);
        } else {
            for (int idx = threadIdx.x; idx < kWmmaM * kWmmaK; idx += blockDim.x) {
                const int r = idx / kWmmaK;
                const int c = idx - r * kWmmaK;
                const int gr = tile_m + r;
                const int gc = kk + c;
                As[idx] =
                    (gr < m && gc < k) ? Ab[(size_t)gc + (size_t)gr * lda] : __float2half(0.0f);
            }
            for (int idx = threadIdx.x; idx < kWmmaK * kWmmaN; idx += blockDim.x) {
                const int r = idx % kWmmaK;
                const int c = idx / kWmmaK;
                const int gr = kk + r;
                const int gc = tile_n + c;
                Bs[idx] =
                    (gr < k && gc < n) ? Bb[(size_t)gr + (size_t)gc * ldb] : __float2half(0.0f);
            }
            __syncthreads();
            nvcuda::wmma::load_matrix_sync(a_frag, As, kWmmaK);
            nvcuda::wmma::load_matrix_sync(b_frag, Bs, kWmmaK);
        }
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        if (!full_tile) {
            __syncthreads();
        }
    }

    nvcuda::wmma::store_matrix_sync(Cs, c_frag, kWmmaM, nvcuda::wmma::mem_col_major);
    __syncthreads();
    for (int idx = threadIdx.x; idx < kWmmaM * kWmmaN; idx += blockDim.x) {
        const int r = idx % kWmmaM;
        const int c = idx / kWmmaM;
        const int gr = tile_m + r;
        const int gc = tile_n + c;
        if (gr < m && gc < n) {
            Cb[(size_t)gr + (size_t)gc * ldc] = __float2half(Cs[idx]);
        }
    }
#else
    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)sa;
    (void)B;
    (void)ldb;
    (void)sb;
    (void)C;
    (void)ldc;
    (void)sc;
    (void)batch;
#endif
}

__global__ void
k_hgemm_tn_wmma64x32_facc(
    int m, int n, int k, const void* A, int lda, long long sa, const void* B, int ldb, long long sb,
    void* C, int ldc, long long sc, int batch) {
#if __CUDA_ARCH__ >= 700
    const int warp = threadIdx.x >> 5;
    const int warp_m = warp & 3;
    const int warp_n = warp >> 2;
    const int tile_m = blockIdx.x * kBlockWmmaM;
    const int tile_n = blockIdx.y * kBlockWmmaN;
    const int b = blockIdx.z;
    if (b >= batch) {
        return;
    }

    const __half* Ab = (const __half*)((const char*)A + (size_t)b * sa * sizeof(__half));
    const __half* Bb = (const __half*)((const char*)B + (size_t)b * sb * sizeof(__half));
    __half* Cb = (__half*)((char*)C + (size_t)b * sc * sizeof(__half));

    __shared__ __half As[kBlockWmmaM * kWmmaK];
    __shared__ __half Bs[kWmmaK * kBlockWmmaN];
    __shared__ float Cs[kBlockWmmaM * kBlockWmmaN];

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::col_major>
        b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int kk = 0; kk < k; kk += kWmmaK) {
        for (int idx = threadIdx.x; idx < kBlockWmmaM * kWmmaK; idx += blockDim.x) {
            const int r = idx / kWmmaK;
            const int c = idx - r * kWmmaK;
            const int gr = tile_m + r;
            const int gc = kk + c;
            As[idx] = (gr < m && gc < k) ? Ab[(size_t)gc + (size_t)gr * lda] : __float2half(0.0f);
        }
        for (int idx = threadIdx.x; idx < kWmmaK * kBlockWmmaN; idx += blockDim.x) {
            const int r = idx % kWmmaK;
            const int c = idx / kWmmaK;
            const int gr = kk + r;
            const int gc = tile_n + c;
            Bs[idx] = (gr < k && gc < n) ? Bb[(size_t)gr + (size_t)gc * ldb] : __float2half(0.0f);
        }
        __syncthreads();

        nvcuda::wmma::load_matrix_sync(a_frag, As + warp_m * kWmmaM * kWmmaK, kWmmaK);
        nvcuda::wmma::load_matrix_sync(b_frag, Bs + warp_n * kWmmaN * kWmmaK, kWmmaK);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(
        Cs + warp_m * kWmmaM + (size_t)warp_n * kWmmaN * kBlockWmmaM, c_frag, kBlockWmmaM,
        nvcuda::wmma::mem_col_major);
    __syncthreads();

    for (int idx = threadIdx.x; idx < kBlockWmmaM * kBlockWmmaN; idx += blockDim.x) {
        const int r = idx % kBlockWmmaM;
        const int c = idx / kBlockWmmaM;
        const int gr = tile_m + r;
        const int gc = tile_n + c;
        if (gr < m && gc < n) {
            Cb[(size_t)gr + (size_t)gc * ldc] = __float2half(Cs[idx]);
        }
    }
#else
    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)sa;
    (void)B;
    (void)ldb;
    (void)sb;
    (void)C;
    (void)ldc;
    (void)sc;
    (void)batch;
#endif
}

__global__ void
k_hgemm_tn_wmma64x64_facc(
    int m, int n, int k, const void* A, int lda, long long sa, const void* B, int ldb, long long sb,
    void* C, int ldc, long long sc, int batch) {
#if __CUDA_ARCH__ >= 700
    const int warp = threadIdx.x >> 5;
    const int warp_m = warp & 3;
    const int warp_n_pair = warp >> 2;
    const int tile_m = blockIdx.x * kBlockWmmaM;
    const int tile_n = blockIdx.y * kBlockWmmaN2;
    const int b = blockIdx.z;
    if (b >= batch) {
        return;
    }

    const __half* Ab = (const __half*)((const char*)A + (size_t)b * sa * sizeof(__half));
    const __half* Bb = (const __half*)((const char*)B + (size_t)b * sb * sizeof(__half));
    __half* Cb = (__half*)((char*)C + (size_t)b * sc * sizeof(__half));

    __shared__ __half As[kBlockWmmaM * kWmmaK];
    __shared__ __half Bs[kWmmaK * kBlockWmmaN2];
    __shared__ float Cs[kBlockWmmaM * kBlockWmmaN2];

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::col_major>
        b_frag0;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::col_major>
        b_frag1;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag0;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag1;
    nvcuda::wmma::fill_fragment(c_frag0, 0.0f);
    nvcuda::wmma::fill_fragment(c_frag1, 0.0f);

    for (int kk = 0; kk < k; kk += kWmmaK) {
        for (int idx = threadIdx.x; idx < kBlockWmmaM * kWmmaK; idx += blockDim.x) {
            const int r = idx / kWmmaK;
            const int c = idx - r * kWmmaK;
            const int gr = tile_m + r;
            const int gc = kk + c;
            As[idx] = (gr < m && gc < k) ? Ab[(size_t)gc + (size_t)gr * lda] : __float2half(0.0f);
        }
        for (int idx = threadIdx.x; idx < kWmmaK * kBlockWmmaN2; idx += blockDim.x) {
            const int r = idx % kWmmaK;
            const int c = idx / kWmmaK;
            const int gr = kk + r;
            const int gc = tile_n + c;
            Bs[idx] = (gr < k && gc < n) ? Bb[(size_t)gr + (size_t)gc * ldb] : __float2half(0.0f);
        }
        __syncthreads();

        const int b_col0 = warp_n_pair * 2 * kWmmaN;
        nvcuda::wmma::load_matrix_sync(a_frag, As + warp_m * kWmmaM * kWmmaK, kWmmaK);
        nvcuda::wmma::load_matrix_sync(b_frag0, Bs + b_col0 * kWmmaK, kWmmaK);
        nvcuda::wmma::load_matrix_sync(b_frag1, Bs + (b_col0 + kWmmaN) * kWmmaK, kWmmaK);
        nvcuda::wmma::mma_sync(c_frag0, a_frag, b_frag0, c_frag0);
        nvcuda::wmma::mma_sync(c_frag1, a_frag, b_frag1, c_frag1);
        __syncthreads();
    }

    const int c_col0 = warp_n_pair * 2 * kWmmaN;
    nvcuda::wmma::store_matrix_sync(
        Cs + warp_m * kWmmaM + (size_t)c_col0 * kBlockWmmaM, c_frag0, kBlockWmmaM,
        nvcuda::wmma::mem_col_major);
    nvcuda::wmma::store_matrix_sync(
        Cs + warp_m * kWmmaM + (size_t)(c_col0 + kWmmaN) * kBlockWmmaM, c_frag1, kBlockWmmaM,
        nvcuda::wmma::mem_col_major);
    __syncthreads();

    for (int idx = threadIdx.x; idx < kBlockWmmaM * kBlockWmmaN2; idx += blockDim.x) {
        const int r = idx % kBlockWmmaM;
        const int c = idx / kBlockWmmaM;
        const int gr = tile_m + r;
        const int gc = tile_n + c;
        if (gr < m && gc < n) {
            Cb[(size_t)gr + (size_t)gc * ldc] = __float2half(Cs[idx]);
        }
    }
#else
    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)sa;
    (void)B;
    (void)ldb;
    (void)sb;
    (void)C;
    (void)ldc;
    (void)sc;
    (void)batch;
#endif
}

__global__ void
k_hgemm_tn_wmma16x64_facc(
    int m, int n, int k, const void* A, int lda, long long sa, const void* B, int ldb, long long sb,
    void* C, int ldc, long long sc, int batch) {
#if __CUDA_ARCH__ >= 700
    const int warp = threadIdx.x >> 5;
    const int tile_n = blockIdx.y * kSmallMWmmaN;
    const int b = blockIdx.z;
    if (b >= batch) {
        return;
    }

    const __half* Ab = (const __half*)((const char*)A + (size_t)b * sa * sizeof(__half));
    const __half* Bb = (const __half*)((const char*)B + (size_t)b * sb * sizeof(__half));
    __half* Cb = (__half*)((char*)C + (size_t)b * sc * sizeof(__half));

    __shared__ __half As[kWmmaM * kWmmaK];
    __shared__ __half Bs[kWmmaK * kSmallMWmmaN];
    __shared__ float Cs[kWmmaM * kSmallMWmmaN];

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::col_major>
        b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int kk = 0; kk < k; kk += kWmmaK) {
        for (int idx = threadIdx.x; idx < kWmmaM * kWmmaK; idx += blockDim.x) {
            const int r = idx / kWmmaK;
            const int c = idx - r * kWmmaK;
            const int gc = kk + c;
            As[idx] = (r < m && gc < k) ? Ab[(size_t)gc + (size_t)r * lda] : __float2half(0.0f);
        }
        for (int idx = threadIdx.x; idx < kWmmaK * kSmallMWmmaN; idx += blockDim.x) {
            const int r = idx % kWmmaK;
            const int c = idx / kWmmaK;
            const int gr = kk + r;
            const int gc = tile_n + c;
            Bs[idx] = (gr < k && gc < n) ? Bb[(size_t)gr + (size_t)gc * ldb] : __float2half(0.0f);
        }
        __syncthreads();

        nvcuda::wmma::load_matrix_sync(a_frag, As, kWmmaK);
        nvcuda::wmma::load_matrix_sync(b_frag, Bs + warp * kWmmaN * kWmmaK, kWmmaK);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(
        Cs + (size_t)warp * kWmmaN * kWmmaM, c_frag, kWmmaM, nvcuda::wmma::mem_col_major);
    __syncthreads();

    for (int idx = threadIdx.x; idx < kWmmaM * kSmallMWmmaN; idx += blockDim.x) {
        const int r = idx % kWmmaM;
        const int c = idx / kWmmaM;
        const int gc = tile_n + c;
        if (r < m && gc < n) {
            Cb[(size_t)r + (size_t)gc * ldc] = __float2half(Cs[idx]);
        }
    }
#else
    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)sa;
    (void)B;
    (void)ldb;
    (void)sb;
    (void)C;
    (void)ldc;
    (void)sc;
    (void)batch;
#endif
}

__global__ void
k_hgemm_tn_64x64_splitk_part(
    int m, int n, int k, const void* A, int lda, const void* B, int ldb, float* partials) {
#if __CUDA_ARCH__ >= 700
    const int warp = threadIdx.x >> 5;
    const int warp_m = warp & 3;
    const int warp_n_pair = warp >> 2;
    const int tile_m = blockIdx.x * kBlockWmmaM;
    const int tile_n = blockIdx.y * kBlockWmmaN2;
    const int split = blockIdx.z;

    const __half* Ab = (const __half*)A;
    const __half* Bb = (const __half*)B;

    __shared__ __half As[kBlockWmmaM * kWmmaK];
    __shared__ __half Bs[kWmmaK * kBlockWmmaN2];
    __shared__ float Cs[kBlockWmmaM * kBlockWmmaN2];

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::col_major>
        b_frag0;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::col_major>
        b_frag1;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag0;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag1;
    nvcuda::wmma::fill_fragment(c_frag0, 0.0f);
    nvcuda::wmma::fill_fragment(c_frag1, 0.0f);

    const int k_tiles = (k + kWmmaK - 1) / kWmmaK;
    const int tiles_per_split = (k_tiles + kBlockSplitK - 1) / kBlockSplitK;
    const int kt0 = split * tiles_per_split;
    const int kt1 = min(k_tiles, kt0 + tiles_per_split);

    for (int kt = kt0; kt < kt1; ++kt) {
        const int kk = kt * kWmmaK;
        for (int idx = threadIdx.x; idx < kBlockWmmaM * kWmmaK; idx += blockDim.x) {
            const int r = idx / kWmmaK;
            const int c = idx - r * kWmmaK;
            const int gr = tile_m + r;
            const int gc = kk + c;
            As[idx] = (gr < m && gc < k) ? Ab[(size_t)gc + (size_t)gr * lda] : __float2half(0.0f);
        }
        for (int idx = threadIdx.x; idx < kWmmaK * kBlockWmmaN2; idx += blockDim.x) {
            const int r = idx % kWmmaK;
            const int c = idx / kWmmaK;
            const int gr = kk + r;
            const int gc = tile_n + c;
            Bs[idx] = (gr < k && gc < n) ? Bb[(size_t)gr + (size_t)gc * ldb] : __float2half(0.0f);
        }
        __syncthreads();

        const int b_col0 = warp_n_pair * 2 * kWmmaN;
        nvcuda::wmma::load_matrix_sync(a_frag, As + warp_m * kWmmaM * kWmmaK, kWmmaK);
        nvcuda::wmma::load_matrix_sync(b_frag0, Bs + b_col0 * kWmmaK, kWmmaK);
        nvcuda::wmma::load_matrix_sync(b_frag1, Bs + (b_col0 + kWmmaN) * kWmmaK, kWmmaK);
        nvcuda::wmma::mma_sync(c_frag0, a_frag, b_frag0, c_frag0);
        nvcuda::wmma::mma_sync(c_frag1, a_frag, b_frag1, c_frag1);
        __syncthreads();
    }

    const int c_col0 = warp_n_pair * 2 * kWmmaN;
    nvcuda::wmma::store_matrix_sync(
        Cs + warp_m * kWmmaM + (size_t)c_col0 * kBlockWmmaM, c_frag0, kBlockWmmaM,
        nvcuda::wmma::mem_col_major);
    nvcuda::wmma::store_matrix_sync(
        Cs + warp_m * kWmmaM + (size_t)(c_col0 + kWmmaN) * kBlockWmmaM, c_frag1, kBlockWmmaM,
        nvcuda::wmma::mem_col_major);
    __syncthreads();

    float* split_partials = partials + (size_t)split * m * n;
    for (int idx = threadIdx.x; idx < kBlockWmmaM * kBlockWmmaN2; idx += blockDim.x) {
        const int r = idx % kBlockWmmaM;
        const int c = idx / kBlockWmmaM;
        const int gr = tile_m + r;
        const int gc = tile_n + c;
        if (gr < m && gc < n) {
            split_partials[(size_t)gr + (size_t)gc * m] = Cs[idx];
        }
    }
#else
    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)partials;
#endif
}

__global__ void
k_hgemm_tn_64x64_splitk_reduce(int m, int n, const float* partials, void* C, int ldc) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
#pragma unroll
    for (int split = 0; split < kBlockSplitK; ++split) {
        acc += partials[(size_t)split * m * n + (size_t)row + (size_t)col * m];
    }
    ((__half*)C)[(size_t)row + (size_t)col * ldc] = __float2half(acc);
}

__global__ void
k_hgemm_tn_smallm_n64_splitk_part(
    int m, int n, int k, const void* A, int lda, const void* B, int ldb, float* partials) {
#if __CUDA_ARCH__ >= 700
    const int warp = threadIdx.x >> 5;
    const int tile_n = blockIdx.x * kSmallMWmmaN;
    const int tile_m = blockIdx.y * kWmmaM;
    const int split = blockIdx.z;

    const __half* Ab = (const __half*)A;
    const __half* Bb = (const __half*)B;

    __shared__ __half As[kWmmaM * kWmmaK];
    __shared__ __half Bs[kWmmaK * kSmallMWmmaN];
    __shared__ float Cs[kWmmaM * kSmallMWmmaN];

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, nvcuda::wmma::col_major>
        b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    const int k_tiles = (k + kWmmaK - 1) / kWmmaK;
    const int tiles_per_split = (k_tiles + kSmallMSplitK - 1) / kSmallMSplitK;
    const int kt0 = split * tiles_per_split;
    const int kt1 = min(k_tiles, kt0 + tiles_per_split);

    for (int kt = kt0; kt < kt1; ++kt) {
        const int kk = kt * kWmmaK;
        for (int idx = threadIdx.x; idx < kWmmaM * kWmmaK; idx += blockDim.x) {
            const int r = idx / kWmmaK;
            const int c = idx - r * kWmmaK;
            const int gr = tile_m + r;
            const int gc = kk + c;
            As[idx] = (gr < m && gc < k) ? Ab[(size_t)gc + (size_t)gr * lda] : __float2half(0.0f);
        }
        for (int idx = threadIdx.x; idx < kWmmaK * kSmallMWmmaN; idx += blockDim.x) {
            const int r = idx % kWmmaK;
            const int c = idx / kWmmaK;
            const int gr = kk + r;
            const int gc = tile_n + c;
            Bs[idx] = (gr < k && gc < n) ? Bb[(size_t)gr + (size_t)gc * ldb] : __float2half(0.0f);
        }
        __syncthreads();

        nvcuda::wmma::load_matrix_sync(a_frag, As, kWmmaK);
        nvcuda::wmma::load_matrix_sync(b_frag, Bs + warp * kWmmaN * kWmmaK, kWmmaK);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(
        Cs + (size_t)warp * kWmmaN * kWmmaM, c_frag, kWmmaM, nvcuda::wmma::mem_col_major);
    __syncthreads();

    float* split_partials = partials + (size_t)split * kSmallMSplitMaxM * n;
    for (int idx = threadIdx.x; idx < kWmmaM * kSmallMWmmaN; idx += blockDim.x) {
        const int r = idx % kWmmaM;
        const int c = idx / kWmmaM;
        const int gr = tile_m + r;
        const int gc = tile_n + c;
        if (gr < m && gc < n) {
            split_partials[(size_t)gr * n + gc] = Cs[idx];
        }
    }
#else
    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)partials;
#endif
}

__global__ void
k_hgemm_tn_smallm_splitk_reduce(int m, int n, const float* partials, void* C, int ldc) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
#pragma unroll
    for (int split = 0; split < kSmallMSplitK; ++split) {
        acc += partials[(size_t)split * kSmallMSplitMaxM * n + (size_t)row * n + col];
    }
    ((__half*)C)[(size_t)row + (size_t)col * ldc] = __float2half(acc);
}

__global__ void
k_hgemv_tn_warp(
    int m, int k, const void* A, int lda, long long sa, const void* B, int ldb, long long sb,
    void* C, int ldc, long long sc, int tc, float alpha, float beta, size_t esc, int batch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kGemvWarpsPerBlock + warp;
    const int b = blockIdx.z;
    if (row >= m || b >= batch) {
        return;
    }

    const __half* Ab = (const __half*)((const char*)A + (size_t)b * sa * sizeof(__half));
    const __half* Bb = (const __half*)((const char*)B + (size_t)b * sb * sizeof(__half));
    void* Cb = (char*)C + (size_t)b * sc * esc;

    float acc = 0.0f;
    const __half* Arow = Ab + (size_t)row * lda;
    const uintptr_t align = (uintptr_t)Arow | (uintptr_t)Bb;
    if ((align & 0x3u) == 0) {
        const int pairs = k >> 1;
        const __half2* A2 = (const __half2*)Arow;
        const __half2* B2 = (const __half2*)Bb;
        for (int pair = lane; pair < pairs; pair += 32) {
            const float2 av = __half22float2(__ldg(A2 + pair));
            const float2 bv = __half22float2(__ldg(B2 + pair));
            acc += av.x * bv.x + av.y * bv.y;
        }
        if ((k & 1) && lane == 0) {
            const int col = k - 1;
            acc += __half2float(__ldg(Arow + col)) * __half2float(__ldg(Bb + col));
        }
    } else {
        for (int col = lane; col < k; col += 32) {
            acc += __half2float(__ldg(Arow + col)) * __half2float(__ldg(Bb + col));
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        const size_t ci = (size_t)row;
        float out = alpha * acc + (beta != 0.0f ? beta * ld(Cb, ci, tc) : 0.0f);
        st(Cb, ci, tc, out);
    }
}

__global__ void
k_hgemv_tn_warp_hout(
    int m, int k, const void* A, int lda, long long sa, const void* B, int ldb, long long sb,
    void* C, int ldc, long long sc, int batch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kGemvWarpsPerBlock + warp;
    const int b = blockIdx.z;
    if (row >= m || b >= batch) {
        return;
    }

    const __half* __restrict__ Ab =
        (const __half*)((const char*)A + (size_t)b * sa * sizeof(__half));
    const __half* __restrict__ Bb =
        (const __half*)((const char*)B + (size_t)b * sb * sizeof(__half));
    __half* __restrict__ Cb = (__half*)((char*)C + (size_t)b * sc * sizeof(__half));

    float acc = 0.0f;
    const __half* __restrict__ Arow = Ab + (size_t)row * lda;
    const uintptr_t align = (uintptr_t)Arow | (uintptr_t)Bb;
    if ((align & 0x3u) == 0) {
        const int pairs = k >> 1;
        const __half2* A2 = (const __half2*)Arow;
        const __half2* B2 = (const __half2*)Bb;
        for (int pair = lane; pair < pairs; pair += 32) {
            const float2 av = __half22float2(__ldg(A2 + pair));
            const float2 bv = __half22float2(__ldg(B2 + pair));
            acc += av.x * bv.x + av.y * bv.y;
        }
        if ((k & 1) && lane == 0) {
            const int col = k - 1;
            acc += __half2float(__ldg(Arow + col)) * __half2float(__ldg(Bb + col));
        }
    } else {
        for (int col = lane; col < k; col += 32) {
            acc += __half2float(__ldg(Arow + col)) * __half2float(__ldg(Bb + col));
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        Cb[(size_t)row] = __float2half(acc);
    }
}

__global__ void
k_sgemv_tn_warp(
    int m, int k, const void* A, int lda, long long sa, const void* B, int ldb, long long sb,
    void* C, int ldc, long long sc, float alpha, float beta, int batch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kGemvWarpsPerBlock + warp;
    const int b = blockIdx.z;
    if (row >= m || b >= batch) {
        return;
    }

    const float* Ab = (const float*)((const char*)A + (size_t)b * sa * sizeof(float));
    const float* Bb = (const float*)((const char*)B + (size_t)b * sb * sizeof(float));
    float* Cb = (float*)((char*)C + (size_t)b * sc * sizeof(float));
    const float* Arow = Ab + (size_t)row * lda;

    float acc = 0.0f;
    for (int col = lane; col < k; col += 32) {
        acc += Arow[col] * Bb[col];
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        const size_t ci = (size_t)row;
        Cb[ci] = alpha * acc + (beta != 0.0f ? beta * Cb[ci] : 0.0f);
    }
}

__global__ void
k_hgemm_tn_smalln_warp(
    int m, int n, int k, const void* A, int lda, long long sa, const void* B, int ldb, long long sb,
    void* C, int ldc, long long sc, int batch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kGemvWarpsPerBlock + warp;
    const int b = blockIdx.z;
    if (row >= m || b >= batch) {
        return;
    }

    const __half* Ab = (const __half*)((const char*)A + (size_t)b * sa * sizeof(__half));
    const __half* Bb = (const __half*)((const char*)B + (size_t)b * sb * sizeof(__half));
    __half* Cb = (__half*)((char*)C + (size_t)b * sc * sizeof(__half));
    const __half* Arow = Ab + (size_t)row * lda;

    float acc[kSmallNMax] = {0.0f};
    uintptr_t align = (uintptr_t)Arow;
    for (int j = 0; j < kSmallNMax; ++j) {
        if (j < n) {
            align |= (uintptr_t)(Bb + (size_t)j * ldb);
        }
    }
    if ((align & 0x3u) == 0) {
        const int pairs = k >> 1;
        const __half2* A2 = (const __half2*)Arow;
        for (int pair = lane; pair < pairs; pair += 32) {
            const float2 av = __half22float2(A2[pair]);
#pragma unroll
            for (int j = 0; j < kSmallNMax; ++j) {
                if (j < n) {
                    const __half2* B2 = (const __half2*)(Bb + (size_t)j * ldb);
                    const float2 bv = __half22float2(B2[pair]);
                    acc[j] += av.x * bv.x + av.y * bv.y;
                }
            }
        }
        if ((k & 1) && lane == 0) {
            const int col = k - 1;
            const float av = __half2float(Arow[col]);
#pragma unroll
            for (int j = 0; j < kSmallNMax; ++j) {
                if (j < n) {
                    acc[j] += av * __half2float(Bb[(size_t)col + (size_t)j * ldb]);
                }
            }
        }
    } else {
        for (int col = lane; col < k; col += 32) {
            const float av = __half2float(Arow[col]);
#pragma unroll
            for (int j = 0; j < kSmallNMax; ++j) {
                if (j < n) {
                    acc[j] += av * __half2float(Bb[(size_t)col + (size_t)j * ldb]);
                }
            }
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int j = 0; j < kSmallNMax; ++j) {
            acc[j] += __shfl_down_sync(0xffffffffu, acc[j], offset);
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < kSmallNMax; ++j) {
            if (j < n) {
                Cb[(size_t)row + (size_t)j * ldc] = __float2half(acc[j]);
            }
        }
    }
}

// Tiled shared-memory GEMM for LARGE outputs. One block computes a 64x64 C
// tile (16x16 threads, 4x4 register micro-tile each); A/B tiles are staged in
// fp32 shared memory via ld(), which already encodes the dtype + N/T index
// math identically to gemm_one, so results are numerically equivalent (fp32
// accumulation, same index formulas). Profiling on Thor: this wins ~4x on the
// large subsampling-conv GEMMs (k-reuse bound) but LOSES ~2.5x on the small
// streaming GEMMs (which need parallelism, not reuse) — so launch() picks per
// shape: tiled iff the aggregate batched output is large enough to fill the
// GPU (see EDGE_SHIM_TILE_MIN).
#define EDGE_TM 64
#define EDGE_TN 64
#define EDGE_TK 16
__device__ __forceinline__ void
gemm_tile(
    int m, int n, int k, int opA, int opB, const void* A, int lda, int ta, const void* B, int ldb,
    int tb, void* C, int ldc, int tc, float alpha, float beta) {
    __shared__ float As[EDGE_TK][EDGE_TM];  // As[c][r] = A[bi0+r, kk+c]
    __shared__ float Bs[EDGE_TK][EDGE_TN];  // Bs[c][t] = B[kk+c, bj0+t]
    const int bi0 = blockIdx.x * EDGE_TM, bj0 = blockIdx.y * EDGE_TN;
    const int tx = threadIdx.x, ty = threadIdx.y;  // 0..15
    const int tid = ty * 16 + tx;                  // 0..255
    float acc[4][4] = {{0.f}};
    for (int kk = 0; kk < k; kk += EDGE_TK) {
        // Cooperative load: 1024 A + 1024 B elems, 4 each across 256 threads.
        // idx = c*TM + r keeps consecutive r (m-index) on consecutive tid, so
        // OP_N (col-major) global reads coalesce.
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int idx = tid + e * 256;
            int c = idx / EDGE_TM, r = idx % EDGE_TM;
            int gi = bi0 + r, gl = kk + c;
            As[c][r] = (gi < m && gl < k) ? ld(A,
                                               opA == OP_N ? (size_t)gi + (size_t)gl * lda
                                                           : (size_t)gl + (size_t)gi * lda,
                                               ta)
                                          : 0.f;
        }
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int idx = tid + e * 256;
            int c = idx / EDGE_TN, t = idx % EDGE_TN;
            int gl = kk + c, gj = bj0 + t;
            Bs[c][t] = (gl < k && gj < n) ? ld(B,
                                               opB == OP_N ? (size_t)gl + (size_t)gj * ldb
                                                           : (size_t)gj + (size_t)gl * ldb,
                                               tb)
                                          : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int cc = 0; cc < EDGE_TK; cc++) {
            float a[4], bv[4];
#pragma unroll
            for (int r = 0; r < 4; r++) a[r] = As[cc][ty * 4 + r];
#pragma unroll
            for (int t = 0; t < 4; t++) bv[t] = Bs[cc][tx * 4 + t];
#pragma unroll
            for (int r = 0; r < 4; r++)
#pragma unroll
                for (int t = 0; t < 4; t++) acc[r][t] += a[r] * bv[t];
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; r++) {
        int gi = bi0 + ty * 4 + r;
        if (gi >= m)
            continue;
#pragma unroll
        for (int t = 0; t < 4; t++) {
            int gj = bj0 + tx * 4 + t;
            if (gj >= n)
                continue;
            size_t ci = (size_t)gi + (size_t)gj * ldc;
            float out = alpha * acc[r][t] + (beta != 0.f ? beta * ld(C, ci, tc) : 0.f);
            st(C, ci, tc, out);
        }
    }
}
__global__ void
k_strided_tiled(
    int m, int n, int k, int opA, int opB, const void* A, int lda, long long sa, int ta,
    const void* B, int ldb, long long sb, int tb, void* C, int ldc, long long sc, int tc,
    float alpha, float beta, size_t esa, size_t esb, size_t esc, int batch) {
    int b = blockIdx.z;
    if (b >= batch)
        return;
    gemm_tile(
        m, n, k, opA, opB, (const char*)A + (size_t)b * sa * esa, lda, ta,
        (const char*)B + (size_t)b * sb * esb, ldb, tb, (char*)C + (size_t)b * sc * esc, ldc, tc,
        alpha, beta);
}
// Use the tiled kernel only when the output is large enough that 64x64 tiling
// still yields enough blocks to fill the GPU AND the k-reuse pays for the
// reduced parallelism. Empirical crossover on Thor (see comment on gemm_tile).
inline bool
use_tiled(int m, int n, int batch) {
    // Crossover between the naive (parallelism-bound, small outputs) and the
    // tiled (k-reuse-bound, large batched outputs). Overridable for tuning.
    static const size_t min_mn = [] {
        const char* e = getenv("EDGE_SHIM_TILE_MIN");
        return e ? (size_t)atoll(e) : (size_t)64 * 1024;
    }();
    return (size_t)m * (size_t)n * (size_t)batch >= min_mn;
}
inline size_t
wmma_min_mn() {
    static const size_t min_mn = [] {
        const char* e = getenv("EDGE_SHIM_WMMA_MIN_MN");
        return e ? (size_t)atoll(e) : (size_t)1;
    }();
    return min_mn;
}
inline bool
use_wmma_tn(int m, int n, int opA, int opB, int ta, int tb, int tc) {
    return opA == OP_T && opB == OP_N && ta == R_16F && tb == R_16F &&
           (tc == R_16F || tc == R_32F || tc == R_16BF) && n > 1 &&
           (size_t)m * (size_t)n >= wmma_min_mn();
}
inline bool
use_hgemv_tn(int n, int opA, int opB, int ta, int tb, int tc) {
    return n == 1 && opA == OP_T && opB == OP_N && ta == R_16F && tb == R_16F &&
           (tc == R_16F || tc == R_32F || tc == R_16BF);
}
inline bool
use_sgemv_tn(int n, int opA, int opB, int ta, int tb, int tc) {
    return n == 1 && opA == OP_T && opB == OP_N && ta == R_32F && tb == R_32F && tc == R_32F;
}
inline bool
trace_shapes() {
    static const bool enabled = getenv("EDGE_SHIM_TRACE_SHAPES") != nullptr;
    return enabled;
}
inline void
trace_launch(
    const char* path, int m, int n, int k, int opA, int opB, int ta, int tb, int tc, float alpha,
    float beta, int batch) {
    if (!trace_shapes()) {
        return;
    }
    fprintf(
        stderr,
        "nemo_speech_cublas_shim path=%s m=%d n=%d k=%d opA=%d opB=%d ta=%d tb=%d tc=%d alpha=%g "
        "beta=%g batch=%d\n",
        path, m, n, k, opA, opB, ta, tb, tc, alpha, beta, batch);
}
inline size_t
esz(int dt) {
    return dt == R_32F ? 4 : 2;
}
inline float
host_scalar(const void* p, int ct) {
    if (ct == COMPUTE_16F) {
        uint16_t h = *(const uint16_t*)p;
        uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
        if (e == 0) {
            if (m == 0)
                bits = s << 31;
            else {
                e = 127 - 15 + 1;
                while (!(m & 0x400)) {
                    m <<= 1;
                    e--;
                }
                m &= 0x3ff;
                bits = (s << 31) | (e << 23) | (m << 13);
            }
        } else if (e == 0x1f)
            bits = (s << 31) | 0x7f800000u | (m << 13);
        else
            bits = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
        float f;
        __builtin_memcpy(&f, &bits, 4);
        return f;
    }
    return *(const float*)p;
}
inline void
launch(
    int m, int n, int k, int opA, int opB, const void* A, int lda, long long sa, int ta,
    const void* B, int ldb, long long sb, int tb, void* C, int ldc, long long sc, int tc,
    float alpha, float beta, int batch, cudaStream_t s, ShimHandle* sh) {
    int bb = batch > 0 ? batch : 1;
    if (use_hgemv_tn(n, opA, opB, ta, tb, tc)) {
        if (tc == R_16F && alpha == 1.0f && beta == 0.0f) {
            trace_launch("hgemv_tn_warp_hout", m, n, k, opA, opB, ta, tb, tc, alpha, beta, bb);
            dim3 blk(32 * kGemvWarpsPerBlock, 1, 1),
                grd((m + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock, 1, bb);
            k_hgemv_tn_warp_hout<<<grd, blk, 0, s>>>(m, k, A, lda, sa, B, ldb, sb, C, ldc, sc, bb);
            return;
        }
        trace_launch("hgemv_tn_warp", m, n, k, opA, opB, ta, tb, tc, alpha, beta, bb);
        dim3 blk(32 * kGemvWarpsPerBlock, 1, 1),
            grd((m + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock, 1, bb);
        k_hgemv_tn_warp<<<grd, blk, 0, s>>>(
            m, k, A, lda, sa, B, ldb, sb, C, ldc, sc, tc, alpha, beta, esz(tc), bb);
        return;
    }
    if (use_sgemv_tn(n, opA, opB, ta, tb, tc)) {
        trace_launch("sgemv_tn_warp", m, n, k, opA, opB, ta, tb, tc, alpha, beta, bb);
        dim3 blk(32 * kGemvWarpsPerBlock, 1, 1),
            grd((m + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock, 1, bb);
        k_sgemv_tn_warp<<<grd, blk, 0, s>>>(
            m, k, A, lda, sa, B, ldb, sb, C, ldc, sc, alpha, beta, bb);
        return;
    }
    if (use_wmma_tn(m, n, opA, opB, ta, tb, tc)) {
        if (tc == R_16F && alpha == 1.0f && beta == 0.0f) {
            if (m <= kSmallMSplitMaxM && n >= kSmallMWmmaN && n <= kSmallMSplitMaxN && k >= 1024 &&
                bb == 1) {
                std::lock_guard<std::mutex> lock(sh->mutex);
                void* splitk_workspace = splitk_workspace_for_stream_locked(sh, s);
                if (splitk_workspace != nullptr) {
                    trace_launch(
                        "wmma_smallm_n64_splitk", m, n, k, opA, opB, ta, tb, tc, alpha, beta, bb);
                    dim3 blk_part(32 * kSmallMWmmaWarps, 1, 1),
                        grd_part(
                            (n + kSmallMWmmaN - 1) / kSmallMWmmaN, (m + kWmmaM - 1) / kWmmaM,
                            kSmallMSplitK);
                    k_hgemm_tn_smallm_n64_splitk_part<<<grd_part, blk_part, 0, s>>>(
                        m, n, k, A, lda, B, ldb, (float*)splitk_workspace);
                    dim3 blk_reduce(16, 16, 1), grd_reduce((m + 15) / 16, (n + 15) / 16, 1);
                    k_hgemm_tn_smallm_splitk_reduce<<<grd_reduce, blk_reduce, 0, s>>>(
                        m, n, (const float*)splitk_workspace, C, ldc);
                    return;
                }
            }
            if (m <= 8 && n >= kSmallMWmmaN) {
                trace_launch("wmma16x64_tn_facc", m, n, k, opA, opB, ta, tb, tc, alpha, beta, bb);
                dim3 blk(32 * kSmallMWmmaWarps, 1, 1),
                    grd(1, (n + kSmallMWmmaN - 1) / kSmallMWmmaN, bb);
                k_hgemm_tn_wmma16x64_facc<<<grd, blk, 0, s>>>(
                    m, n, k, A, lda, sa, B, ldb, sb, C, ldc, sc, bb);
                return;
            }
            if (n <= kSmallNMax) {
                trace_launch(
                    "hgemm_tn_smalln_warp", m, n, k, opA, opB, ta, tb, tc, alpha, beta, bb);
                dim3 blk(32 * kGemvWarpsPerBlock, 1, 1),
                    grd((m + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock, 1, bb);
                k_hgemm_tn_smalln_warp<<<grd, blk, 0, s>>>(
                    m, n, k, A, lda, sa, B, ldb, sb, C, ldc, sc, bb);
                return;
            }
            if (m >= kBlockWmmaM && n >= 48 && (size_t)m * (size_t)n <= kBlockSplitMaxElems &&
                k >= 512 && bb == 1) {
                std::lock_guard<std::mutex> lock(sh->mutex);
                void* splitk_workspace = splitk_workspace_for_stream_locked(sh, s);
                if (splitk_workspace != nullptr) {
                    trace_launch(
                        "wmma64x64_tn_splitk", m, n, k, opA, opB, ta, tb, tc, alpha, beta, bb);
                    dim3 blk_part(32 * kBlockWmmaWarps, 1, 1),
                        grd_part(
                            (m + kBlockWmmaM - 1) / kBlockWmmaM,
                            (n + kBlockWmmaN2 - 1) / kBlockWmmaN2, kBlockSplitK);
                    k_hgemm_tn_64x64_splitk_part<<<grd_part, blk_part, 0, s>>>(
                        m, n, k, A, lda, B, ldb, (float*)splitk_workspace);
                    dim3 blk_reduce(16, 16, 1), grd_reduce((m + 15) / 16, (n + 15) / 16, 1);
                    k_hgemm_tn_64x64_splitk_reduce<<<grd_reduce, blk_reduce, 0, s>>>(
                        m, n, (const float*)splitk_workspace, C, ldc);
                    return;
                }
            }
            if (m >= kBlockWmmaM && n >= 48) {
                trace_launch("wmma64x64_tn_facc", m, n, k, opA, opB, ta, tb, tc, alpha, beta, bb);
                dim3 blk(32 * kBlockWmmaWarps, 1, 1),
                    grd((m + kBlockWmmaM - 1) / kBlockWmmaM, (n + kBlockWmmaN2 - 1) / kBlockWmmaN2,
                        bb);
                k_hgemm_tn_wmma64x64_facc<<<grd, blk, 0, s>>>(
                    m, n, k, A, lda, sa, B, ldb, sb, C, ldc, sc, bb);
                return;
            }
            if (m >= kBlockWmmaM && n >= kWmmaN) {
                trace_launch("wmma64x32_tn_facc", m, n, k, opA, opB, ta, tb, tc, alpha, beta, bb);
                dim3 blk(32 * kBlockWmmaWarps, 1, 1), grd((m + kBlockWmmaM - 1) / kBlockWmmaM,
                                                          (n + kBlockWmmaN - 1) / kBlockWmmaN, bb);
                k_hgemm_tn_wmma64x32_facc<<<grd, blk, 0, s>>>(
                    m, n, k, A, lda, sa, B, ldb, sb, C, ldc, sc, bb);
                return;
            }
            trace_launch("wmma16x16_tn_facc", m, n, k, opA, opB, ta, tb, tc, alpha, beta, bb);
            dim3 blk(32, 1, 1), grd((m + kWmmaM - 1) / kWmmaM, (n + kWmmaN - 1) / kWmmaN, bb);
            k_hgemm_tn_wmma_facc<<<grd, blk, 0, s>>>(
                m, n, k, A, lda, sa, B, ldb, sb, C, ldc, sc, bb);
            return;
        }
        trace_launch("wmma_tn", m, n, k, opA, opB, ta, tb, tc, alpha, beta, bb);
        dim3 blk(32, 1, 1), grd((m + kWmmaM - 1) / kWmmaM, (n + kWmmaN - 1) / kWmmaN, bb);
        k_hgemm_tn_wmma<<<grd, blk, 0, s>>>(
            m, n, k, A, lda, sa, B, ldb, sb, C, ldc, sc, tc, alpha, beta, esz(tc), bb);
        return;
    }
    if (use_tiled(m, n, bb)) {
        trace_launch("tiled", m, n, k, opA, opB, ta, tb, tc, alpha, beta, bb);
        dim3 blk(16, 16, 1), grd((m + EDGE_TM - 1) / EDGE_TM, (n + EDGE_TN - 1) / EDGE_TN, bb);
        k_strided_tiled<<<grd, blk, 0, s>>>(
            m, n, k, opA, opB, A, lda, sa, ta, B, ldb, sb, tb, C, ldc, sc, tc, alpha, beta, esz(ta),
            esz(tb), esz(tc), bb);
        return;
    }
    trace_launch("scalar", m, n, k, opA, opB, ta, tb, tc, alpha, beta, bb);
    dim3 blk(16, 16, 1), grd((m + 15) / 16, (n + 15) / 16, bb);
    k_strided<<<grd, blk, 0, s>>>(
        m, n, k, opA, opB, A, lda, sa, ta, B, ldb, sb, tb, C, ldc, sc, tc, alpha, beta, esz(ta),
        esz(tb), esz(tc), bb);
}
}  // namespace

extern "C" {

cublasStatus_t
cublasCreate_v2(cublasHandle_t* h) {
    auto* sh = new ShimHandle{};
    if (cudaGetDevice(&sh->device) != cudaSuccess) {
        sh->device = -1;
        cudaGetLastError();
    }
    *h = sh;
    return STATUS_SUCCESS;
}
cublasStatus_t
cublasDestroy_v2(cublasHandle_t h) {
    ShimHandle* sh = (ShimHandle*)h;
    if (sh == nullptr) {
        return STATUS_SUCCESS;
    }

    std::unordered_map<cudaStream_t, ShimHandle::SplitKWorkspace> splitk_workspaces;
    {
        std::lock_guard<std::mutex> lock(sh->mutex);
        splitk_workspaces.swap(sh->splitk_workspaces);
    }

    if (!splitk_workspaces.empty()) {
        if (sh->device >= 0 && cudaSetDevice(sh->device) != cudaSuccess) {
            cudaGetLastError();
        }
        cudaDeviceSynchronize();
        cudaGetLastError();
    }

    for (auto& entry : splitk_workspaces) {
        free_splitk_workspace(entry.second);
    }
    delete sh;
    return STATUS_SUCCESS;
}
cublasStatus_t
cublasSetStream_v2(cublasHandle_t h, cudaStream_t s) {
    ShimHandle* sh = (ShimHandle*)h;
    std::lock_guard<std::mutex> lock(sh->mutex);
    sh->stream = s;
    return STATUS_SUCCESS;
}
cublasStatus_t
cublasSetMathMode(cublasHandle_t, cublasMath_t) {
    return STATUS_SUCCESS;
}
const char*
cublasGetStatusString(cublasStatus_t) {
    return "EDGE_SHIM_OK";
}

cublasStatus_t
cublasGemmEx(
    cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k,
    const void* alpha, const void* A, cudaDataType ta, int lda, const void* B, cudaDataType tb,
    int ldb, const void* beta, void* C, cudaDataType tc, int ldc, cublasComputeType_t ct,
    cublasGemmAlgo_t) {
    ShimHandle* sh = (ShimHandle*)h;
    const cudaStream_t stream = stream_for_handle(sh);
    launch(
        m, n, k, opA, opB, A, lda, 0, ta, B, ldb, 0, tb, C, ldc, 0, tc, host_scalar(alpha, ct),
        host_scalar(beta, ct), 1, stream, sh);
    return STATUS_SUCCESS;
}
cublasStatus_t
cublasGemmStridedBatchedEx(
    cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k,
    const void* alpha, const void* A, cudaDataType ta, int lda, long long sa, const void* B,
    cudaDataType tb, int ldb, long long sb, const void* beta, void* C, cudaDataType tc, int ldc,
    long long sc, int batch, cublasComputeType_t ct, cublasGemmAlgo_t) {
    ShimHandle* sh = (ShimHandle*)h;
    const cudaStream_t stream = stream_for_handle(sh);
    launch(
        m, n, k, opA, opB, A, lda, sa, ta, B, ldb, sb, tb, C, ldc, sc, tc, host_scalar(alpha, ct),
        host_scalar(beta, ct), batch, stream, sh);
    return STATUS_SUCCESS;
}
cublasStatus_t
cublasGemmBatchedEx(
    cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k,
    const void* alpha, const void* const Aarray[], cudaDataType ta, int lda,
    const void* const Barray[], cudaDataType tb, int ldb, const void* beta, void* const Carray[],
    cudaDataType tc, int ldc, int batch, cublasComputeType_t ct, cublasGemmAlgo_t) {
    ShimHandle* sh = (ShimHandle*)h;
    const cudaStream_t stream = stream_for_handle(sh);
    dim3 blk(16, 16, 1), grd((m + 15) / 16, (n + 15) / 16, batch);
    k_ptrs<<<grd, blk, 0, stream>>>(
        m, n, k, opA, opB, Aarray, lda, ta, Barray, ldb, tb, Carray, ldc, tc,
        host_scalar(alpha, ct), host_scalar(beta, ct), batch);
    return STATUS_SUCCESS;
}
cublasStatus_t
cublasSgemm_v2(
    cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta,
    float* C, int ldc) {
    ShimHandle* sh = (ShimHandle*)h;
    const cudaStream_t stream = stream_for_handle(sh);
    launch(
        m, n, k, opA, opB, A, lda, 0, R_32F, B, ldb, 0, R_32F, C, ldc, 0, R_32F, *alpha, *beta, 1,
        stream, sh);
    return STATUS_SUCCESS;
}
cublasStatus_t
cublasSgemmStridedBatched(
    cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k,
    const float* alpha, const float* A, int lda, long long sa, const float* B, int ldb,
    long long sb, const float* beta, float* C, int ldc, long long sc, int batch) {
    ShimHandle* sh = (ShimHandle*)h;
    const cudaStream_t stream = stream_for_handle(sh);
    launch(
        m, n, k, opA, opB, A, lda, sa, R_32F, B, ldb, sb, R_32F, C, ldc, sc, R_32F, *alpha, *beta,
        batch, stream, sh);
    return STATUS_SUCCESS;
}
cublasStatus_t
cublasStrsmBatched(
    cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
    int, const float*, const float* const[], int, float* const[], int, int) {
    fprintf(stderr, "nemo-speech cublas shim: cublasStrsmBatched called (unsupported)\n");
    abort();
}

}  // extern "C"
