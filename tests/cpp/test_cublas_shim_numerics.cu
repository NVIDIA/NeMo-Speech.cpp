// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool
check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "FAIL: %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool
check_cublas(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "FAIL: %s: cuBLAS status %d\n", operation, (int)status);
    return false;
}

bool
run_cancellation_case(cublasHandle_t handle, int m, int n, int k) {
    std::vector<__half> a((size_t)m * k, __float2half(100.0f));
    std::vector<__half> b((size_t)n * k);
    std::vector<__half> c((size_t)m * n);

    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < k; ++row) {
            b[(size_t)col * k + row] = __float2half(row < k / 2 ? 10.0f : -10.0f);
        }
    }

    __half* device_a = nullptr;
    __half* device_b = nullptr;
    __half* device_c = nullptr;
    bool ok = check_cuda(cudaMalloc(&device_a, a.size() * sizeof(__half)), "cudaMalloc(A)") &&
              check_cuda(cudaMalloc(&device_b, b.size() * sizeof(__half)), "cudaMalloc(B)") &&
              check_cuda(cudaMalloc(&device_c, c.size() * sizeof(__half)), "cudaMalloc(C)");
    if (!ok) {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        return false;
    }

    ok = check_cuda(
             cudaMemcpy(device_a, a.data(), a.size() * sizeof(__half), cudaMemcpyHostToDevice),
             "cudaMemcpy(A)") &&
         check_cuda(
             cudaMemcpy(device_b, b.data(), b.size() * sizeof(__half), cudaMemcpyHostToDevice),
             "cudaMemcpy(B)");

    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    if (ok) {
        ok = check_cublas(
            cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, device_a, CUDA_R_16F, k,
                device_b, CUDA_R_16F, k, &beta, device_c, CUDA_R_16F, m, CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmEx");
    }
    if (ok) {
        ok = check_cuda(
            cudaMemcpy(c.data(), device_c, c.size() * sizeof(__half), cudaMemcpyDeviceToHost),
            "cudaMemcpy(C)");
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    if (!ok) {
        return false;
    }
    for (size_t i = 0; i < c.size(); ++i) {
        const float value = __half2float(c[i]);
        if (!std::isfinite(value) || std::fabs(value) > 0.5f) {
            std::fprintf(
                stderr, "FAIL: m=%d n=%d k=%d output[%zu]=%g, expected zero\n", m, n, k, i, value);
            return false;
        }
    }
    return true;
}

bool
run_batched_f32_case(cublasHandle_t handle) {
    constexpr int m = 64;
    constexpr int n = 111;
    constexpr int k = 111;
    constexpr int batch = 12;
    constexpr long long stride_a = (long long)m * k;
    constexpr long long stride_b = (long long)n * k;
    constexpr long long stride_c = (long long)m * n;

    std::vector<float> a((size_t)batch * stride_a, 1.0f);
    std::vector<float> b((size_t)batch * stride_b);
    std::vector<float> c((size_t)batch * stride_c);
    for (int item = 0; item < batch; ++item) {
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < k; ++row) {
                b[(size_t)item * stride_b + (size_t)col * k + row] = row % 2 == 0 ? 1.0f : -1.0f;
            }
        }
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    bool ok = check_cuda(cudaMalloc(&device_a, a.size() * sizeof(float)), "cudaMalloc(A f32)") &&
              check_cuda(cudaMalloc(&device_b, b.size() * sizeof(float)), "cudaMalloc(B f32)") &&
              check_cuda(cudaMalloc(&device_c, c.size() * sizeof(float)), "cudaMalloc(C f32)");
    if (!ok) {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        return false;
    }

    ok = check_cuda(
             cudaMemcpy(device_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy(A f32)") &&
         check_cuda(
             cudaMemcpy(device_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy(B f32)");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (ok) {
        ok = check_cublas(
            cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, device_a, CUDA_R_32F, k,
                stride_a, device_b, CUDA_R_32F, k, stride_b, &beta, device_c, CUDA_R_32F, m,
                stride_c, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmStridedBatchedEx");
    }
    if (ok) {
        ok = check_cuda(
            cudaMemcpy(c.data(), device_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy(C f32)");
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    if (!ok) {
        return false;
    }
    for (size_t i = 0; i < c.size(); ++i) {
        if (std::fabs(c[i] - 1.0f) > 1.0e-5f) {
            std::fprintf(stderr, "FAIL: batched f32 output[%zu]=%g, expected one\n", i, c[i]);
            return false;
        }
    }
    return true;
}

// Pointer-array f32 GEMM with a transposed B, as ggml's OUT_PROD issues it.
bool
run_pointer_batched_f32_case(cublasHandle_t handle) {
    constexpr int m = 37;
    constexpr int n = 29;
    constexpr int k = 13;
    constexpr int batch = 5;
    constexpr size_t size_a = (size_t)m * k;  // m x k, column-major
    constexpr size_t size_b = (size_t)n * k;  // n x k, used transposed
    constexpr size_t size_c = (size_t)m * n;

    std::vector<float> a(batch * size_a), b(batch * size_b), c(batch * size_c);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (float)((i * 7) % 11) - 5.0f;
    for (size_t i = 0; i < b.size(); ++i) b[i] = (float)((i * 5) % 13) - 6.0f;

    float* device = nullptr;
    const float** device_ptrs = nullptr;
    const size_t total = a.size() + b.size() + c.size();
    bool ok = check_cuda(cudaMalloc(&device, total * sizeof(float)), "cudaMalloc(batched)") &&
              check_cuda(cudaMalloc(&device_ptrs, 3 * batch * sizeof(float*)), "cudaMalloc(ptrs)");
    float* device_a = device;
    float* device_b = device_a + a.size();
    float* device_c = device_b + b.size();
    std::vector<const float*> ptrs(3 * batch);
    for (int item = 0; item < batch; ++item) {
        ptrs[item] = device_a + item * size_a;
        ptrs[batch + item] = device_b + item * size_b;
        ptrs[2 * batch + item] = device_c + item * size_c;
    }
    ok = ok &&
         check_cuda(
             cudaMemcpy(device_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy(A batched)") &&
         check_cuda(
             cudaMemcpy(device_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy(B batched)") &&
         check_cuda(
             cudaMemcpy(
                 device_ptrs, ptrs.data(), ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice),
             "cudaMemcpy(ptrs)");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (ok) {
        ok = check_cublas(
            cublasSgemmBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, device_ptrs, m,
                device_ptrs + batch, n, &beta, (float* const*)(device_ptrs + 2 * batch), m, batch),
            "cublasSgemmBatched");
    }
    if (ok) {
        ok = check_cuda(
            cudaMemcpy(c.data(), device_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy(C batched)");
    }
    cudaFree(device);
    cudaFree(device_ptrs);
    if (!ok) {
        return false;
    }
    for (int item = 0; item < batch; ++item) {
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                float expected = 0.0f;
                for (int i = 0; i < k; ++i) {
                    expected += a[item * size_a + (size_t)i * m + row] *
                                b[item * size_b + (size_t)i * n + col];
                }
                const float actual = c[item * size_c + (size_t)col * m + row];
                if (!std::isfinite(actual) || std::fabs(actual - expected) > 1.0e-3f) {
                    std::fprintf(
                        stderr, "FAIL: pointer-batched f32 [%d,%d,%d]=%g, expected %g\n", item, row,
                        col, actual, expected);
                    return false;
                }
            }
        }
    }
    return true;
}

// More batches than CUDA's grid.z limit (65535), and a zero-sized no-op.
bool
run_large_batch_case(cublasHandle_t handle) {
    constexpr int batch = 70000;
    constexpr int dim = 2;
    constexpr size_t size = (size_t)dim * dim;
    std::vector<float> a(batch * size), b(batch * size), c(batch * size);
    for (int item = 0; item < batch; ++item) {
        for (size_t i = 0; i < size; ++i) {
            a[item * size + i] = (float)(item % 7) + (float)i;
            b[item * size + i] = i % 3 == 0 ? 1.0f : 0.5f;
        }
    }

    float* device = nullptr;
    const float** device_ptrs = nullptr;
    bool ok = check_cuda(cudaMalloc(&device, 3 * a.size() * sizeof(float)), "cudaMalloc(large)") &&
              check_cuda(cudaMalloc(&device_ptrs, 3 * batch * sizeof(float*)), "cudaMalloc(ptrs)");
    float* device_a = device;
    float* device_b = device_a + a.size();
    float* device_c = device_b + b.size();
    std::vector<const float*> ptrs(3 * batch);
    for (int item = 0; item < batch; ++item) {
        ptrs[item] = device_a + item * size;
        ptrs[batch + item] = device_b + item * size;
        ptrs[2 * batch + item] = device_c + item * size;
    }
    ok = ok &&
         check_cuda(
             cudaMemcpy(device_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy(A large)") &&
         check_cuda(
             cudaMemcpy(device_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy(B large)") &&
         check_cuda(
             cudaMemcpy(
                 device_ptrs, ptrs.data(), ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice),
             "cudaMemcpy(ptrs large)");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    std::vector<float> expected(c.size());
    for (int item = 0; item < batch; ++item) {
        for (int col = 0; col < dim; ++col) {
            for (int row = 0; row < dim; ++row) {
                float sum = 0.0f;
                for (int i = 0; i < dim; ++i) {
                    sum += a[item * size + (size_t)i * dim + row] *
                           b[item * size + (size_t)col * dim + i];
                }
                expected[item * size + (size_t)col * dim + row] = sum;
            }
        }
    }
    const auto check_output = [&](const char* label) {
        if (!check_cuda(
                cudaMemcpy(c.data(), device_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost),
                "cudaMemcpy(C large)")) {
            return false;
        }
        for (size_t i = 0; i < c.size(); ++i) {
            if (!std::isfinite(c[i]) || std::fabs(c[i] - expected[i]) > 1.0e-4f) {
                std::fprintf(
                    stderr, "FAIL: %s output[%zu]=%g, expected %g\n", label, i, c[i], expected[i]);
                return false;
            }
        }
        return true;
    };

    if (ok) {
        ok = check_cuda(cudaMemset(device_c, 0xff, c.size() * sizeof(float)), "cudaMemset(C)") &&
             check_cublas(
                 cublasSgemmBatched(
                     handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, device_ptrs, dim,
                     device_ptrs + batch, dim, &beta, (float* const*)(device_ptrs + 2 * batch), dim,
                     batch),
                 "cublasSgemmBatched(large)") &&
             check_output("pointer-batched large");
    }
    if (ok) {
        ok = check_cuda(cudaMemset(device_c, 0xff, c.size() * sizeof(float)), "cudaMemset(C)") &&
             check_cublas(
                 cublasSgemmStridedBatched(
                     handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, device_a, dim,
                     (long long)size, device_b, dim, (long long)size, &beta, device_c, dim,
                     (long long)size, batch),
                 "cublasSgemmStridedBatched(large)") &&
             check_output("strided-batched large");
    }
    // Zero-sized problems complete without touching C.
    if (ok) {
        ok = check_cublas(
                 cublasSgemmBatched(
                     handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, device_ptrs, dim,
                     device_ptrs + batch, dim, &beta, (float* const*)(device_ptrs + 2 * batch), dim,
                     0),
                 "cublasSgemmBatched(batch=0)") &&
             check_cublas(
                 cublasSgemmStridedBatched(
                     handle, CUBLAS_OP_N, CUBLAS_OP_N, 0, dim, dim, &alpha, device_a, dim, 0,
                     device_b, dim, 0, &beta, device_c, dim, 0, 1),
                 "cublasSgemmStridedBatched(m=0)") &&
             check_output("zero-sized no-op");
    }
    if (ok && cublasSgemmStridedBatched(
                  handle, CUBLAS_OP_N, CUBLAS_OP_N, -1, dim, dim, &alpha, device_a, dim, 0,
                  device_b, dim, 0, &beta, device_c, dim, 0, 1) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: a negative dimension was not rejected\n");
        ok = false;
    }

    cudaFree(device);
    cudaFree(device_ptrs);
    return ok;
}

bool
run_stream_churn_case(cublasHandle_t handle) {
    bool ok = true;
    for (int i = 0; i < 12 && ok; ++i) {
        cudaStream_t stream = nullptr;
        ok = check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate") &&
             check_cublas(cublasSetStream(handle, stream), "cublasSetStream") &&
             run_cancellation_case(handle, 24, 432, 1296) &&
             check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        if (stream != nullptr) {
            ok &= check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
        }
    }
    ok &= check_cublas(cublasSetStream(handle, nullptr), "cublasSetStream(default)");
    return ok;
}

}  // namespace

int
main() {
    cublasHandle_t handle = nullptr;
    if (!check_cublas(cublasCreate(&handle), "cublasCreate")) {
        return 1;
    }

    bool ok = true;
    ok &= run_cancellation_case(handle, 1, 68, 256);
    ok &= run_cancellation_case(handle, 16, 16, 256);
    ok &= run_cancellation_case(handle, 68, 32, 256);
    ok &= run_cancellation_case(handle, 68, 64, 256);
    ok &= run_cancellation_case(handle, 24, 432, 1296);
    ok &= run_cancellation_case(handle, 768, 111, 768);
    ok &= run_batched_f32_case(handle);
    ok &= run_pointer_batched_f32_case(handle);
    ok &= run_large_batch_case(handle);
    ok &= run_stream_churn_case(handle);

    ok &= check_cublas(cublasDestroy(handle), "cublasDestroy");
    return ok ? 0 : 1;
}
