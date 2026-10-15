




#pragma once


#if defined(KITTENS_AMPERE)
#define KITTENS_SM80
#endif
#if defined(KITTENS_HOPPER)
#define KITTENS_SM90
#endif
#if defined(KITTENS_BLACKWELL)
#define KITTENS_SM100
#endif


#if defined(KITTENS_SM80) + defined(KITTENS_SM90) + defined(KITTENS_SM100) + defined(KITTENS_SM103) + defined(KITTENS_SM120) != 1
#error "Define exactly one of: KITTENS_SM80, KITTENS_SM90, KITTENS_SM100, KITTENS_SM103, KITTENS_SM120"
#endif


#if defined(KITTENS_SM100) || defined(KITTENS_SM103)
#define KITTENS_SM10X
#endif


#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#ifdef KITTENS_NO_HOST
namespace std { using namespace cuda::std; }
using uint = uint32_t;
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 13
struct alignas(128) CUtensorMap { char __opaque[128]; };
#else
struct alignas(64) CUtensorMap { char __opaque[128]; };
#endif
#endif


#include <cuda_fp16.h>
#include <cuda_bf16.h>
#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
#include <cuda_fp8.h>
#endif
#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)
#include <cuda_fp4.h>
#endif


#ifndef KITTENS_NO_HOST
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>
#endif


#include "common/common.cuh"
#include "types/types.cuh"
#include "ops/ops.cuh"
