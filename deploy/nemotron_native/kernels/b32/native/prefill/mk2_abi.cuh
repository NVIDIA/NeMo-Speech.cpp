#pragma once
#include <cstdint>
extern "C" {
struct MK2Predictor;
struct MK2Weights {
    const void *qkv[3];
    const void *o[3];
    const void *ff_in[3];
    const void *ff_out[3];
    const void *norm_self[3];
    const void *norm_ff[3];
    const void *pos_emb;
    const void *audio_emb[16];
    const void *head_w[16];
    const void *head_b[16];
};
MK2Predictor *mk2_create_from_weights(const MK2Weights *w, int batch);
MK2Predictor *mk2_create(const char *raw_dir, int batch);
int mk2_run(MK2Predictor *p, const void *hidden_fp16, const void *gumbel_fp32,
            float temperature, void *codes_int64, void *stream);
int mk2_run_masked(MK2Predictor *p, const void *hidden_fp16, const void *gumbel_fp32,
                   float temperature, const void *audio_valid_bool,
                   void *codes_int64, void *stream);
void mk2_destroy(MK2Predictor *p);
struct MK2BuildInfo {
    int version, batch, batch_pad, stages, kchunk;
    int direct_input, b1_local_barriers, skip_hidden_snapshot;
    int threads, dynamic_smem, static_smem, registers;
    int device, sms, blocks_per_sm, grid;
};
int mk2_build_info(MK2BuildInfo *out);
struct MK2Stage0Bindings {
    void *conv_state[14];
    int conv_slot_stride[14];
    void *ssm_state[14];
    int ssm_slot_stride[14];
    const void *slot_map;
    void *kv_pool_base[3];
    const void *kv_block_table[3];
    int kv_max_blocks_per_seq;
    int kv_tokens_per_block;
    const void *seq_len;
    const void *embedding;
    const void *gumbel;
    float temperature;
    const void *moe_w_inject_debug;
    void *codes_out;
    void *phoneme_out;
};
struct MK2BackboneWeights {
    const void *norm[31];
    const void *norm_f;
    const void *mb_in_proj[14];
    const void *mb_out_proj[14];
    const void *mb_conv_w[14];
    const void *mb_conv_b[14];
    const void *mb_A[14];
    const void *mb_D[14];
    const void *mb_dt_bias[14];
    const void *mb_gnorm[14];
    const void *at_qkv[3];
    const void *at_o[3];
    const void *moe_exp_up[14];
    const void *moe_exp_down[14];
    const void *moe_sh_up[14];
    const void *moe_sh_down[14];
    const void *moe_gate[14];
    const void *moe_gate_bias[14];
    const void *phoneme_w;
    const void *phoneme_b;
    const char *pattern;
};
MK2Predictor *mk2_create_stage0(const MK2Weights *predictor,
                                const MK2BackboneWeights *backbone, int batch);
int mk2_run_stage0(MK2Predictor *p, const MK2Stage0Bindings *b, void *stream);
int mk2_run_native_stage0(MK2Predictor *p, const MK2Stage0Bindings *b, void *stream);
unsigned long long mk2_stage0_launches(MK2Predictor *p);
int mk2_stage0_timing(MK2Predictor *p, unsigned long long *out);
int mk2_batch(const MK2Predictor *p);
int mk2_positions(const MK2Predictor *p);
const char *mk2_last_error(void);
}
