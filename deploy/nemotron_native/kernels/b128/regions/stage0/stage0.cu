#define MK2_BATCH 128
#define STAGES 3
#define KCHUNK 128
#define MK2_NO_MAIN 1
#define MK2_DIRECT_INPUT 1
#define MK2_B1_LOCAL_BARRIERS 1
#define MK2_SKIP_HIDDEN_SNAPSHOT 1
#define MK2_ACTIVE_EXPERTS 1
#define MK2_SSM_SPLITS 1
#define NMULT 2
#define MK2_OUT_MULT 2
#define SSM_PREFETCH_MAX 12
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp8.h>
#if MK2_INFLIGHT
#include <curand_kernel.h>
#endif
#include <fstream>
#include <math_constants.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <kittens.cuh>
namespace b128_stage0 {
#include "native/stage0/predictor_mk2.cu"
}
