#pragma once
#if MK2_INFLIGHT
extern "C" int mk2_native_inflight_abi_version() { return 1; }
extern "C" int mk2_native_inflight_batch() { return BATCH; }
extern "C" int mk2_set_native_progress_mask(MK2Predictor *p,
                                            const unsigned char *mask) {
    try {
        if (!p || reinterpret_cast<MK2State *>(p)->kind != 1 || !mask)
            throw std::runtime_error("progress mask requires a full Stage0 handle and mask");
        auto *st = reinterpret_cast<MK2Stage0State *>(p);
        if (st->bindings_initialized && st->pred.native_progress_mask != mask)
            throw std::runtime_error("progress mask pointer cannot change after first Stage0 enqueue");
        mk2_stage0_device_pointer(mask, st->pred.device, "native_progress_mask");
        st->pred.native_progress_mask = mask;
        g_last_error.clear();
        return 0;
    } catch (const std::exception &e) { g_last_error = e.what(); return 1; }
      catch (...) { g_last_error = "unknown progress mask binding failure"; return 1; }
}
__global__ void native_inflight_noise(
    const unsigned long long *seeds, const int *generations,
    const unsigned char *mask, float *out) {
    constexpr int groups = N_POS * TOP_K / 4;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= BATCH * groups) return;
    const int row = index / groups;
    if (!mask[row]) return;
    const int group = index % groups;
    curandStatePhilox4_32_10_t state;
    curand_init(seeds[row], (unsigned long long)group,
                (unsigned long long)(unsigned int)generations[row] * 4ull, &state);
    const float4 uniform = curand_uniform4(&state);
    const float maximum = 0x1.fffffep-1f;
    const float4 noise = make_float4(
        -logf(-logf(fminf(uniform.x, maximum))),
        -logf(-logf(fminf(uniform.y, maximum))),
        -logf(-logf(fminf(uniform.z, maximum))),
        -logf(-logf(fminf(uniform.w, maximum))));
    reinterpret_cast<float4 *>(out)[index] = noise;
}
extern "C" int mk2_inflight_noise(
    const unsigned long long *seeds, const int *generations,
    const unsigned char *mask, float *out, void *stream) {
    try {
        int device = -1;
        mk2_check(cudaGetDevice(&device));
        mk2_stage0_device_pointer(seeds, device, "request_seeds");
        mk2_stage0_device_pointer(generations, device, "request_generations");
        mk2_stage0_device_pointer(mask, device, "request_progress_mask");
        mk2_stage0_device_pointer(out, device, "request_noise");
        native_inflight_noise<<<(BATCH * N_POS * TOP_K / 4 + 255) / 256, 256, 0,
                                 static_cast<cudaStream_t>(stream)>>>(
            seeds, generations, mask, out);
        mk2_check(cudaGetLastError());
        g_last_error.clear();
        return 0;
    } catch (const std::exception &e) { g_last_error = e.what(); return 1; }
      catch (...) { g_last_error = "unknown request noise failure"; return 1; }
}
struct MK2InflightResetLayout {
    unsigned long long row_stride, source_row_stride, elements;
    int rank, item_bytes, source_compact;
    unsigned long long sizes[6], strides[6];
};
static_assert(sizeof(MK2InflightResetLayout) == 136);
template<class Element>
__global__ void native_inflight_reset_rows(
    Element *destination, const Element *source, const int *slot_ids,
    const int *device_count, int capacity, unsigned long long fill_bits,
    MK2InflightResetLayout layout) {
    const int count = *device_count;
    if (count < 0 || count > capacity) return;
    const unsigned long long total = (unsigned long long)count * layout.elements;
    for (unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (unsigned long long)gridDim.x * blockDim.x) {
        const unsigned long long selected = index / layout.elements;
        const int row = slot_ids[selected];
        if (row < 0 || row >= capacity) continue;
        const unsigned long long inner = index % layout.elements;
        unsigned long long remaining = inner, offset = (unsigned long long)row * layout.row_stride;
        for (int axis = layout.rank - 1; axis >= 0; --axis) {
            offset += remaining % layout.sizes[axis] * layout.strides[axis];
            remaining /= layout.sizes[axis];
        }
        destination[offset] = source
            ? source[(layout.source_compact ? selected : (unsigned long long)row) *
                     layout.source_row_stride + inner]
            : static_cast<Element>(fill_bits);
    }
}
extern "C" int mk2_inflight_reset_rows(
    void *destination, const void *source, const int *slot_ids, const int *device_count,
    int capacity, unsigned long long fill_bits, const MK2InflightResetLayout *layout,
    void *stream) {
    try {
        if (!layout || capacity != BATCH || layout->rank < 1 || layout->rank > 6 ||
            !layout->row_stride || !layout->elements ||
            (layout->source_compact != 0 && layout->source_compact != 1))
            throw std::runtime_error("invalid request reset layout");
        unsigned long long elements = 1, span = 0;
        for (int axis = 0; axis < layout->rank; ++axis) {
            if (!layout->sizes[axis] || !layout->strides[axis] ||
                layout->sizes[axis] > ULLONG_MAX / elements ||
                layout->sizes[axis] - 1 > (ULLONG_MAX - span) / layout->strides[axis])
                throw std::runtime_error("invalid request reset shape or strides");
            elements *= layout->sizes[axis];
            span += (layout->sizes[axis] - 1) * layout->strides[axis];
        }
        if (elements != layout->elements || span >= layout->row_stride ||
            elements > ULLONG_MAX / (unsigned long long)capacity ||
            layout->row_stride > ULLONG_MAX / (unsigned long long)capacity ||
            (source && layout->source_row_stride && layout->source_row_stride < elements))
            throw std::runtime_error("request reset extent exceeds row storage");
        int device = -1;
        mk2_check(cudaGetDevice(&device));
        mk2_stage0_device_pointer(destination, device, "reset_destination");
        if (source) mk2_stage0_device_pointer(source, device, "reset_source");
        mk2_stage0_device_pointer(slot_ids, device, "reset_slot_ids");
        mk2_stage0_device_pointer(device_count, device, "reset_count");
        const unsigned long long work = (unsigned long long)capacity * elements;
        const int blocks = (int)std::min(132ull, (work - 1ull) / 256ull + 1ull);
        const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
#define MK2_RESET_DISPATCH(bytes, type) \
        case bytes: native_inflight_reset_rows<type><<<blocks, 256, 0, cuda_stream>>>( \
            static_cast<type *>(destination), static_cast<const type *>(source), slot_ids, \
            device_count, capacity, fill_bits, *layout); break
        switch (layout->item_bytes) {
            MK2_RESET_DISPATCH(1, unsigned char);
            MK2_RESET_DISPATCH(2, unsigned short);
            MK2_RESET_DISPATCH(4, unsigned int);
            MK2_RESET_DISPATCH(8, unsigned long long);
            default: throw std::runtime_error("unsupported request reset element size");
        }
#undef MK2_RESET_DISPATCH
        mk2_check(cudaGetLastError());
        g_last_error.clear();
        return 0;
    } catch (const std::exception &e) { g_last_error = e.what(); return 1; }
      catch (...) { g_last_error = "unknown request reset failure"; return 1; }
}
#endif
