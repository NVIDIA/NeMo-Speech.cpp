#include "native_prefill_abi.cuh"
struct MK2NativePrefill {
    MK2Predictor *base;
    MK2Stage0Bindings state;
    MK2NativePrefillInfo info;
};
namespace {
struct NativePrefillBaseDeleter {
    void operator()(MK2Predictor *p) const noexcept { mk2_destroy(p); }
};
static void native_prefill_check_config(const MK2NativePrefillConfig &c) {
    if (c.batch != BATCH || !(BATCH == 1 || BATCH == 16 || BATCH == 32 || BATCH == 64))
        throw std::runtime_error("native prefill requires matching batch 1/16/32/64");
    if (c.prompt_positions <= 0 || c.max_sequence_tokens <= c.prompt_positions ||
        c.max_sequence_tokens > INT_MAX - 31 || c.tokens_per_block != 32)
        throw std::runtime_error("native prefill requires positive prompt, decode capacity and 32-token pages");
    const std::uint64_t rows = std::uint64_t(BATCH) * c.prompt_positions;
    const std::uint64_t padded = ((rows + BATCH_PAD - 1) / BATCH_PAD) * BATCH_PAD;
    if (padded > INT_MAX / ((BB_EXPERTS + 1) * D_MODEL))
        throw std::runtime_error("native prefill token rows exceed safe device loop bounds");
    const std::uint64_t pages = (std::uint64_t(c.max_sequence_tokens) + 31) / 32;
    if (pages * BATCH * 2 > INT_MAX)
        throw std::runtime_error("native prefill primary page indices exceed int32");
#if PF_SKIP_MAMBA || PF_SKIP_ATTN || PF_SKIP_MOE || BB_SKIP_MAMBA || BB_SKIP_ATTN || BB_SKIP_MOE || BB_ATTN_SKIP_KV_WRITE || BB_ATTN_LEGACY_CUR_KV
    throw std::runtime_error("native prefill refuses arithmetic ablation builds");
#endif
#if defined(PF_CONV_ZERO_STATE) || defined(PF_CONV_NO_SILU) || defined(PF_CONV_STATE_NO_SHIFT)
    throw std::runtime_error("native prefill refuses convolution ablation builds");
#endif
}
}
extern "C" MK2NativePrefill *mk2_native_prefill_create(
    const MK2Weights *pw, const MK2BackboneWeights *bw,
    const MK2NativePrefillConfig *config, const void *prompt, void *final_hidden) {
    try {
        if (!config || !pw || !bw) throw std::runtime_error("null native prefill config/weights");
        native_prefill_check_config(*config);
        int device = -1;
        mk2_check(cudaGetDevice(&device));
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        mk2_check(cudaStreamIsCapturing(nullptr, &capture));
        if (capture != cudaStreamCaptureStatusNone)
            throw std::runtime_error("native prefill construction must occur outside capture");
        mk2_stage0_device_pointer(prompt, device, "native prompt");
        mk2_stage0_device_pointer(final_hidden, device, "native final_hidden");
        const size_t rows = size_t(BATCH) * config->prompt_positions;
        const size_t padded = ((rows + BATCH_PAD - 1) / BATCH_PAD) * BATCH_PAD;
        const size_t output_bytes = rows * D_MODEL * sizeof(bf16);
        const auto input_address = reinterpret_cast<std::uintptr_t>(prompt);
        const auto output_address = reinterpret_cast<std::uintptr_t>(final_hidden);
        const auto distance = input_address < output_address
            ? output_address - input_address : input_address - output_address;
        if (distance < output_bytes)
            throw std::runtime_error("native prompt/final_hidden buffers must not overlap");
        std::unique_ptr<MK2Predictor, NativePrefillBaseDeleter> base(
            mk2_create_stage0(pw, bw, BATCH));
        if (!base) return nullptr;
        auto *st = reinterpret_cast<MK2Stage0State *>(base.get());
        MK2Allocations owned;
        MK2Stage0Bindings binding{};
        const int conv_elements = MB_CONV_DIM * (MB_CONV_K - 1);
        const int ssm_elements = MB_HEADS * MB_HEAD_DIM * MB_STATE;
        const int blocks = (config->max_sequence_tokens + 31) / 32;
        const size_t page_elements = size_t(32) * BB_KV_HEADS * BB_HEAD_DIM;
        const size_t pool_bytes = size_t(BATCH) * 2 * blocks * page_elements * sizeof(bf16);
        binding.kv_max_blocks_per_seq = blocks;
        binding.kv_tokens_per_block = 32;
        for (int i = 0; i < 14; ++i) {
            bf16 *conv = nullptr;
            float *ssm = nullptr;
            mk2_check(owned.alloc(&conv, size_t(BATCH) * conv_elements * sizeof(bf16)));
            mk2_check(owned.alloc(&ssm, size_t(BATCH) * ssm_elements * sizeof(float)));
            mk2_check(cudaMemset(conv, 0, size_t(BATCH) * conv_elements * sizeof(bf16)));
            mk2_check(cudaMemset(ssm, 0, size_t(BATCH) * ssm_elements * sizeof(float)));
            binding.conv_state[i] = conv;
            binding.conv_slot_stride[i] = conv_elements;
            bf16 *private_state = nullptr;
            mk2_check(owned.alloc(&private_state, size_t(BATCH) * ssm_elements * sizeof(bf16)));
            mk2_check(cudaMemset(private_state, 0, size_t(BATCH) * ssm_elements * sizeof(bf16)));
            st->mb_host[i].b32_private = private_state;
            binding.ssm_state[i] = reinterpret_cast<float *>(private_state);
            binding.ssm_slot_stride[i] = ssm_elements;
            st->mb_host[i].conv_state = conv;
            st->mb_host[i].conv_slot_stride = conv_elements;
            st->mb_host[i].ssm_state = ssm;
            st->mb_host[i].ssm_slot_stride = ssm_elements;
            mk2_check(owned.alloc(&st->mb_host[i].r8_canonical_ssm, ssm_elements * sizeof(float)));
        }
        std::vector<int> table(size_t(BATCH) * 2 * blocks);
        for (size_t i = 0; i < table.size(); ++i) table[i] = static_cast<int>(i);
        for (int i = 0; i < 3; ++i) {
            bf16 *pool = nullptr;
            int *page_table = nullptr;
            mk2_check(owned.alloc(&pool, pool_bytes));
            mk2_check(cudaMemset(pool, 0, pool_bytes));
            mk2_check(owned.alloc(&page_table, table.size() * sizeof(int)));
            mk2_check(cudaMemcpy(page_table, table.data(), table.size() * sizeof(int),
                                cudaMemcpyHostToDevice));
            binding.kv_pool_base[i] = pool;
            binding.kv_block_table[i] = page_table;
            st->at_host[i].pool_base = pool;
            st->at_host[i].block_table = page_table;
            st->at_host[i].max_blocks_per_seq = blocks;
            st->at_host[i].tokens_per_block = 32;
        }
        std::vector<int> slots(BATCH), lengths(BATCH, config->prompt_positions);
        for (int i = 0; i < BATCH; ++i) slots[i] = i;
        int *device_slots = nullptr, *device_lengths = nullptr;
        mk2_check(owned.alloc(&device_slots, BATCH * sizeof(int)));
        mk2_check(owned.alloc(&device_lengths, BATCH * sizeof(int)));
        mk2_check(cudaMemcpy(device_slots, slots.data(), BATCH * sizeof(int), cudaMemcpyHostToDevice));
        mk2_check(cudaMemcpy(device_lengths, lengths.data(), BATCH * sizeof(int), cudaMemcpyHostToDevice));
        binding.slot_map = device_slots;
        binding.seq_len = device_lengths;
        Backbone &hb = st->host_bb;
        size_t scratch_bytes = 0;
        auto bf = [&](bf16 **p, size_t columns) {
            const size_t bytes = padded * columns * sizeof(bf16);
            mk2_check(owned.alloc(p, bytes));
            mk2_check(cudaMemset(*p, 0, bytes));
            scratch_bytes += bytes;
        };
        auto fp = [&](float **p, size_t columns) {
            const size_t bytes = padded * columns * sizeof(float);
            mk2_check(owned.alloc(p, bytes));
            mk2_check(cudaMemset(*p, 0, bytes));
            scratch_bytes += bytes;
        };
        bf(&hb.pf_h, D_MODEL); bf(&hb.pf_r, D_MODEL); bf(&hb.pf_normed, D_MODEL);
        bf(&hb.pf_zxbcdt, MB_IN_PROJ); bf(&hb.pf_xbc_act, MB_CONV_DIM);
        bf(&hb.pf_gated, MB_INNER); bf(&hb.pf_qkv, BB_QKV);
        bf(&hb.pf_moe_h, BB_EXPERTS * BB_MOE_INTER);
        bf(&hb.pf_moe_sh, BB_SHARED_INTER);
        fp(&hb.pf_moe_out, (BB_EXPERTS + 1) * D_MODEL);
        fp(&hb.pf_dtda, 2 * MB_HEADS);
        {
            const size_t estride = padded;
            const size_t grows = size_t(BB_EXPERTS) * estride;
            mk2_check(owned.alloc(&hb.pf_moe_gath, grows * D_MODEL * sizeof(bf16)));
            mk2_check(cudaMemset(hb.pf_moe_gath, 0, grows * D_MODEL * sizeof(bf16)));
            mk2_check(owned.alloc(&hb.pf_moe_hc, grows * BB_MOE_INTER * sizeof(bf16)));
            mk2_check(cudaMemset(hb.pf_moe_hc, 0, grows * BB_MOE_INTER * sizeof(bf16)));
            mk2_check(owned.alloc(&hb.pf_moe_gidx, grows * sizeof(int)));
            mk2_check(cudaMemset(hb.pf_moe_gidx, 0, grows * sizeof(int)));
            mk2_check(owned.alloc(&hb.pf_moe_ecnt, BB_EXPERTS * sizeof(int)));
            mk2_check(cudaMemset(hb.pf_moe_ecnt, 0, BB_EXPERTS * sizeof(int)));
            mk2_check(owned.alloc(&hb.pf_moe_ntiles, BB_EXPERTS * sizeof(int)));
            mk2_check(cudaMemset(hb.pf_moe_ntiles, 0, BB_EXPERTS * sizeof(int)));
            hb.pf_moe_estride = int(estride);
            hb.pf_moe_gath_v = act_gl{hb.pf_moe_gath, nullptr, nullptr, grows,
                                      size_t(D_MODEL)};
            hb.pf_moe_hc_v = act_gl{hb.pf_moe_hc, nullptr, nullptr, grows,
                                    size_t(BB_MOE_INTER)};
            hb.pf_moe_gath_vn = act_gl_n{hb.pf_moe_gath, nullptr, nullptr, grows,
                                         size_t(D_MODEL)};
            hb.pf_moe_hc_vn = act_gl_n{hb.pf_moe_hc, nullptr, nullptr, grows,
                                       size_t(BB_MOE_INTER)};
            scratch_bytes += grows * (D_MODEL + BB_MOE_INTER) * sizeof(bf16);
        }
        fp(&hb.pf_moe_w, BB_EXPERTS);
        hb.pf_final = static_cast<bf16 *>(final_hidden);
        hb.pf_normed_v = act_gl{hb.pf_normed, nullptr, nullptr, padded, size_t(D_MODEL)};
        hb.pf_gated_v = act_gl{hb.pf_gated, nullptr, nullptr, padded, size_t(MB_INNER)};
        hb.pf_qkv_v = act_gl{hb.pf_qkv, nullptr, nullptr, padded, size_t(BB_QKV)};
        hb.pf_moe_h_v = act_gl{hb.pf_moe_h, nullptr, nullptr, padded,
                              size_t(BB_EXPERTS * BB_MOE_INTER)};
        hb.pf_moe_sh_v = act_gl{hb.pf_moe_sh, nullptr, nullptr, padded, size_t(BB_SHARED_INTER)};
        hb.n_prefill_tokens = static_cast<int>(rows);
        hb.prefill_slot = 0;
        hb.pf_n_seq = BATCH;
        hb.pf_fuse_decode = 0;
        hb.prompt = static_cast<const bf16 *>(prompt);
        hb.pf_seq_slot = device_slots;
        hb.slot_map = device_slots;
        hb.seq_len = device_lengths;
        hb.pf_layer_out = nullptr;
        hb.moe_w_inject = nullptr;
        hb.moe_w_audit = nullptr;
        hb.native_prefill_reset = 1;
        mk2_check(owned.alloc(&hb.native_pf_cohort_check, 2 * sizeof(unsigned int)));
        mk2_check(cudaMemset(hb.native_pf_cohort_check, 0, 2 * sizeof(unsigned int)));
#if PF_ACTIVE_TOKEN_EXPERTS
        mk2_check(owned.alloc(&hb.native_pf_expert_masks, (padded/BATCH_PAD)*sizeof(unsigned int)));
#endif
        mk2_check(cudaMemcpy(const_cast<MambaLayer *>(hb.mamba), st->mb_host.data(),
                            14 * sizeof(MambaLayer), cudaMemcpyHostToDevice));
        mk2_check(cudaMemcpy(const_cast<AttnLayer *>(hb.attn), st->at_host.data(),
                            3 * sizeof(AttnLayer), cudaMemcpyHostToDevice));
        mk2_check(cudaMemcpy(st->d_bb, &hb, sizeof(Backbone), cudaMemcpyHostToDevice));
        MK2NativePrefillInfo info{1, BATCH, config->prompt_positions,
            static_cast<int>(rows), static_cast<int>(padded), config->max_sequence_tokens,
            32, blocks, conv_elements, ssm_elements, pool_bytes, scratch_bytes};
        std::unique_ptr<MK2NativePrefill> handle(new MK2NativePrefill{base.get(), binding, info});
        st->owned.reserve(st->owned.size() + owned.ptrs.size());
        st->owned.insert(st->owned.end(), owned.ptrs.begin(), owned.ptrs.end());
        owned.ptrs.clear();
        base.release();
        g_last_error.clear();
        return handle.release();
    } catch (const std::exception &e) { g_last_error = e.what(); return nullptr; }
      catch (...) { g_last_error = "unknown native prefill construction failure"; return nullptr; }
}
extern "C" int mk2_native_prefill_run(MK2NativePrefill *p, void *stream) {
    try {
        if (!p || !p->base) throw std::runtime_error("null native prefill handle");
        auto *st = reinterpret_cast<MK2Stage0State *>(p->base);
        int device = -1;
        mk2_check(cudaGetDevice(&device));
        if (device != st->pred.device || st->pred.batch != BATCH ||
            st->pred.kind != 1 || !st->host_bb.native_prefill_reset)
            throw std::runtime_error("native prefill handle device or mode mismatch");
        predictor_mk2<<<st->pred.grid, THREADS, st->pred.smem,
                         static_cast<cudaStream_t>(stream)>>>(
            st->pred.g, st->d_bb, N_LAYERS, N_POS, BB_LAYERS);
        mk2_check(cudaGetLastError());
        g_last_error.clear();
        return 0;
    } catch (const std::exception &e) { g_last_error = e.what(); return 1; }
      catch (...) { g_last_error = "unknown native prefill enqueue failure"; return 1; }
}
extern "C" int mk2_native_prefill_get_state(
    const MK2NativePrefill *p, MK2Stage0Bindings *out, MK2NativePrefillInfo *info) {
    if (!p || !out) { g_last_error = "null native prefill handle/state output"; return 1; }
    *out = p->state;
    if (info) *info = p->info;
    g_last_error.clear();
    return 0;
}
extern "C" void mk2_native_prefill_destroy(MK2NativePrefill *p) {
    if (!p) return;
    mk2_destroy(p->base);
    delete p;
}
extern "C" int mk2_native_prefill_effective_sequences(MK2NativePrefill *p, int *out) {
    try {
        if (!p || !out) throw std::runtime_error("null effective-sequence query");
        auto *st = reinterpret_cast<MK2Stage0State *>(p->base);
        mk2_check(cudaMemcpy(out, st->host_bb.native_pf_cohort_check + 1,
                            sizeof(int), cudaMemcpyDeviceToHost));
        return 0;
    } catch (const std::exception &e) { g_last_error = e.what(); return 1; }
}
extern "C" int mk2_r8_initial_state(MK2NativePrefill *p, void **initial, void **ready) {
    if (!p || !initial || !ready) return 1;
    auto *st = reinterpret_cast<MK2Stage0State *>(p->base);
    for (int i = 0; i < 14; ++i) initial[i] = st->mb_host[i].r8_canonical_ssm;
    *ready = st->host_bb.native_pf_cohort_check + 1;
    return 0;
}
