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
    ok &= run_stream_churn_case(handle);

    ok &= check_cublas(cublasDestroy(handle), "cublasDestroy");
    return ok ? 0 : 1;
}
