#define MK2_BATCH 128
#define MK2_BATCH_TILE 64
#define MK2_PREFILL_ONLY 1
#define STAGES 3
#define KCHUNK 128
#define MK2_GEMM_CONSUMERS 3
#define MK2_THREADS 512
#define MK2_NO_MAIN 1
#define MK2_DIRECT_INPUT 1
#define MK2_B1_LOCAL_BARRIERS 1
#define MK2_SKIP_HIDDEN_SNAPSHOT 1
#define MK2_ACTIVE_EXPERTS 1
#define MK2_SSM_SPLITS 1
#define PF_SSM_PW_MULTI 12
#define PF_ACTIVE_TOKEN_EXPERTS 1
#include <array>
#include <cstring>
#include "dedup_predictor_mk2.cuh"
