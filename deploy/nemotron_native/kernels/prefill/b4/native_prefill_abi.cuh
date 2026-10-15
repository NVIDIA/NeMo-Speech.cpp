#pragma once
#include "mk2_abi.cuh"
#include <cstdint>
extern "C" {
struct MK2NativePrefill;
struct MK2NativePrefillConfig {
    int batch;
    int prompt_positions;
    int max_sequence_tokens;
    int tokens_per_block;
};
struct MK2NativePrefillInfo {
    int version;
    int batch;
    int prompt_positions;
    int total_prompt_rows;
    int padded_prompt_rows;
    int max_sequence_tokens;
    int tokens_per_block;
    int blocks_per_sequence;
    int conv_slot_elements;
    int ssm_slot_elements;
    std::uint64_t kv_pool_bytes_per_layer;
    std::uint64_t prefill_scratch_bytes;
};
MK2NativePrefill *mk2_native_prefill_create(
    const MK2Weights *predictor, const MK2BackboneWeights *backbone,
    const MK2NativePrefillConfig *config, const void *prompt, void *final_hidden);
int mk2_native_prefill_run(MK2NativePrefill *p, void *stream);
int mk2_native_prefill_get_state(
    const MK2NativePrefill *p, MK2Stage0Bindings *out, MK2NativePrefillInfo *info);
void mk2_native_prefill_destroy(MK2NativePrefill *p);
}
