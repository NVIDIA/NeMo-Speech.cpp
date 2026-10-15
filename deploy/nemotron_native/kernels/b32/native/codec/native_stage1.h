#pragma once
#include <stddef.h>
#include <stdint.h>
struct NativeStage1Weights { const void *weight; const float *bias; const float *alpha; };
struct NativeStage1Status { int32_t code, row, frame, detail; };
#ifdef __cplusplus
extern "C" {
#endif
void *native_stage1_create(int batch, int max_frames, int state_slots,
                          int resident_blocks, const NativeStage1Weights *weights26);
int native_stage1_run(void *handle, const int64_t *codes, const int32_t *lengths,
                     const int32_t *slots, const void *const *states26,
                     void *const *next_states26, float *pcm, int32_t *samples,
                     NativeStage1Status *status, int frames, void *stream);
int native_stage1_run_profile(void *handle, const int64_t *codes, const int32_t *lengths,
                     const int32_t *slots, const void *const *states26,
                     void *const *next_states26, float *pcm, int32_t *samples,
                     NativeStage1Status *status, int frames, void *stream,
                     uint64_t *phase_timestamps30);
int native_stage1_destroy(void *handle);
const char *native_stage1_last_error();
size_t native_stage1_scratch_bytes(void *handle);
int native_stage1_resident_blocks(void *handle);
#ifdef __cplusplus
}
#endif
