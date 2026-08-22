// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
#include <algorithm>
#include <chrono>
#include <mutex>
#include <unordered_set>

#include "runtime.h"

namespace ggml_runtime {

Session::~Session() = default;

Session::Session(BackendManager& backend_manager, Module* module, GGUFLoader* gguf_loader) {
    this->backend_manager_ = &backend_manager;
    this->params = backend_manager.get_params();
    this->root_module = module;
    this->gguf_loader = gguf_loader;
}

// Launches without forcing each graph boundary to wait for the device. The
// caller synchronizes once after any queued host transfers.
static bool
ggml_graph_compute_helper_async(
    ggml_backend_sched_t sched, struct ggml_cgraph* graph, int n_threads) {
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

        auto* fn_set_n_threads = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_set_n_threads");
        if (fn_set_n_threads) {
            fn_set_n_threads(backend, n_threads);
        }
    }

    return ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS;
}

// A scheduler split can cover a graph range containing nodes that are not
// active for the current run. Before bypassing the scheduler on a cache hit,
// verify the backend can execute every active node in the full graph. This is
// especially important for accelerator backends such as BLAS, which support
// large matrix multiplications but rely on the CPU backend for ops such as PAD.
static bool
backend_supports_compute_graph(ggml_backend_t backend, ggml_cgraph* graph) {
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(graph, i);
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 &&
            !ggml_backend_supports_op(backend, node)) {
            return false;
        }
    }
    return true;
}

// Cache-aware encoder graphs grow with the true batch dimension. Keep enough
// scheduler capacity for large batches while bounding its proportional hash,
// backend-ID, and graph-copy allocations. Near-capacity graphs fail explicitly.
static constexpr size_t kSchedGraphSize = 65536;
// Leave headroom for scheduler-inserted nodes.
static constexpr double kSchedGraphWarnFrac = 0.80;
static constexpr double kSchedGraphErrorFrac = 0.95;

static bool
memstats_enabled() {
    static const bool on = std::getenv("NEMO_SPEECH_MEMSTATS") != nullptr;
    return on;
}

void
Session::init_schedule() {
    sched.reset(ggml_backend_sched_new(
        backends.data(), nullptr, backends.size(), kSchedGraphSize, false, false));
    // A re-setup() replaces the scheduler and model tensors; cached runs
    // reference the old ones, and the fresh sched's pools start at 0 so the
    // size tracker would never flag a change.
    run_cache_.clear();
    run_cache_lru_.clear();
    sched_pool_sizes_.clear();
    sched_pool_generation_ = 0;
    // Graph-sizing scratch is allocated on demand.
    if (memstats_enabled()) {
        fprintf(stderr, "[memstats] session-init sched_graph_size=%zu\n", kSchedGraphSize);
    }
}

// Builds graph metadata in caller-owned storage without allocating tensors.
static ggml_cgraph*
expand_graph_into(
    TensorBag& input_tensors, TensorBag& output_tensors, size_t capacity,
    std::vector<uint8_t>& meta) {
    meta.assign(ggml_graph_overhead_custom(capacity, false) + 4096, 0);
    struct ggml_init_params params = {
        /*.mem_size   =*/meta.size(),
        /*.mem_buffer =*/meta.data(),
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx = ggml_init(params);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, capacity, false);
    for (size_t i = 0; i < input_tensors.tensor_count(); ++i) {
        ggml_set_input(input_tensors.get_tensor(i).tensor);
    }
    for (size_t i = 0; i < output_tensors.tensor_count(); ++i) {
        ggml_bf_tensor bf = output_tensors.get_tensor(i);
        // Flag the whole view chain: when the scheduler's graph allocator
        // places intermediates (sched_managed mode), an output that is a view
        // owns no memory of its own — the viewed tensor must also be pinned
        // (never reused) or the data would be clobbered before readback.
        for (ggml_tensor* t = bf.tensor; t != nullptr; t = t->view_src) {
            ggml_set_output(t);
        }
        ggml_build_forward_expand(gf, bf.tensor);
    }
    ggml_free(ctx);  // gf lives in meta, not in ctx
    return gf;
}

// Probes the node count, then rebuilds into fitted caller-owned storage.
static ggml_cgraph*
build_graph_into(
    Module* /*root_module*/, TensorBag input_tensors, TensorBag output_tensors,
    std::vector<uint8_t>& scratch, std::vector<uint8_t>& dest_meta) {
    ggml_cgraph* probe_gf =
        expand_graph_into(input_tensors, output_tensors, kSchedGraphSize, scratch);
    const size_t n_nodes = (size_t)ggml_graph_n_nodes(probe_gf);
    const size_t fitted = std::min(kSchedGraphSize, 2 * n_nodes + 2048);
    return expand_graph_into(input_tensors, output_tensors, fitted, dest_meta);
}

// Tensor- and Input-based cache keys must use the same signature.
static inline void
hash_tensor_sig(uint64_t& h, const char* name, ggml_type type, const int64_t ne[GGML_MAX_DIMS]) {
    auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ULL;
    };
    for (const char* p = name; *p; ++p) mix((uint64_t)(uint8_t)*p);
    mix(0x0);
    mix((uint64_t)type);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) mix((uint64_t)ne[d]);
}

static uint64_t
hash_input_shapes(const TensorBag& bag) {
    uint64_t h = 14695981039346656037ULL;  // FNV-1a offset basis
    for (size_t i = 0; i < bag.tensor_count(); ++i) {
        ggml_tensor* t = bag.get_tensor(i).tensor;
        if (!t) {
            h ^= 0xDEADBEEF;
            h *= 1099511628211ULL;
            continue;
        }
        hash_tensor_sig(h, t->name, t->type, t->ne);
    }
    return h;
}

// Persistent inputs use their declared signature; per-call shapes are padded to four dims.
static uint64_t
hash_inputs(Session* s, const std::vector<Session::Input>& inputs) {
    uint64_t h = 14695981039346656037ULL;
    for (const auto& in : inputs) {
        if (in.persistent) {
            ggml_tensor* t = s->model_tensor_container->get_tensor_by_name(in.name).tensor;
            hash_tensor_sig(h, t->name, t->type, t->ne);
        } else {
            int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
            for (size_t d = 0; d < in.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = in.shape[d];
            hash_tensor_sig(h, in.name.c_str(), in.dtype, ne);
        }
        if (in.device_tensor != nullptr) {
            // Hashing runs before create_input_tensor's validation.
            if (!in.device_tensor->valid()) {
                throw std::runtime_error(
                    format("Session::Input '%s': invalid device tensor handle", in.name.c_str()));
            }
            const uintptr_t address = reinterpret_cast<uintptr_t>(in.device_tensor->tensor->data) +
                                      in.device_tensor->byte_offset;
            h ^= static_cast<uint64_t>(address);
            h *= 1099511628211ULL;
        }
    }
    return h;
}


void
Session::dump_schedule(std::ostream& out, const std::string& session_label) const {
    out << "== " << session_label << " ==\n";
    // A Session may dispatch several graph topologies (the RNNT decoder has
    // encoder-projection, predictor, and joint-tail stages). Aggregate every
    // resident entry so a clean final joint run cannot hide fallback in a
    // different cached stage.
    std::vector<NodeAssignment> cached_schedule_snapshot;
    size_t graph_count = 0;
    for (uint64_t key : run_cache_lru_) {
        auto it = run_cache_.find(key);
        if (it == run_cache_.end() || it->second.schedule.empty())
            continue;
        ++graph_count;
        cached_schedule_snapshot.insert(
            cached_schedule_snapshot.end(), it->second.schedule.begin(), it->second.schedule.end());
    }
    if (cached_schedule_snapshot.empty()) {
        out << "  <no snapshot — call Session::run() first>\n\n";
        return;
    }
    const std::vector<NodeAssignment>& last_schedule_snapshot_ = cached_schedule_snapshot;
    out << "cached graphs: " << graph_count << "\n";

    std::map<std::string, size_t> counts;
    std::vector<std::string> first_seen;  // preserve insertion order
    for (const auto& a : last_schedule_snapshot_) {
        if (counts.find(a.backend) == counts.end()) {
            first_seen.push_back(a.backend);
        }
        counts[a.backend] += 1;
    }
    out << "backend summary:\n";
    size_t max_name = 0;
    for (const auto& n : first_seen) max_name = std::max(max_name, n.size());
    for (const auto& n : first_seen) {
        out << "  " << n << ":";
        for (size_t i = n.size(); i < max_name; ++i) out << ' ';
        out << " " << counts[n] << " nodes\n";
    }

    const size_t sample_n = std::min<size_t>(20, last_schedule_snapshot_.size());
    out << "sample (first " << sample_n << " of " << last_schedule_snapshot_.size() << " nodes):\n";
    size_t op_w = 0, name_w = 0;
    for (size_t i = 0; i < sample_n; ++i) {
        op_w = std::max(op_w, last_schedule_snapshot_[i].op.size());
        name_w = std::max(name_w, last_schedule_snapshot_[i].name.size());
    }
    for (size_t i = 0; i < sample_n; ++i) {
        const auto& a = last_schedule_snapshot_[i];
        out << "  " << a.op;
        for (size_t k = a.op.size(); k < op_w; ++k) out << ' ';
        out << "  " << a.name;
        for (size_t k = a.name.size(); k < name_w; ++k) out << ' ';
        out << "  -> " << a.backend << "\n";
    }

    bool has_non_cpu = false;
    for (const auto& n : first_seen) {
        if (n != std::string("CPU")) {
            has_non_cpu = true;
            break;
        }
    }
    if (!has_non_cpu) {
        out << "CPU-only run (no GPU backend to fall back from) ✓\n\n";
        return;
    }
    std::map<std::string, size_t> cpu_op_counts;
    std::map<std::string, std::string> cpu_op_first_name;
    for (const auto& a : last_schedule_snapshot_) {
        if (a.backend == std::string("CPU")) {
            if (cpu_op_counts.find(a.op) == cpu_op_counts.end()) {
                cpu_op_first_name[a.op] = a.name;
            }
            cpu_op_counts[a.op] += 1;
        }
    }
    if (cpu_op_counts.empty()) {
        out << "CPU-fallback ops (none) ✓\n\n";
    } else {
        out << "CPU-fallback ops ✗:\n";
        for (const auto& kv : cpu_op_counts) {
            out << "  " << kv.first << "  x" << kv.second;
            const std::string& fn = cpu_op_first_name[kv.first];
            if (!fn.empty())
                out << "  (e.g. " << fn << ")";
            out << "\n";
        }
        out << "\n";
    }
}


void
Session::load_weight(const std::string& gguf_key) {
    ggml_bf_tensor t = model_tensor_container->get_tensor_by_name(gguf_key);
    const ggml_type mem = t.tensor->type;
    const ggml_type disk = gguf_loader->get_tensor_type(gguf_key);
    const size_t mem_bytes = ggml_nbytes(t.tensor);

    // Model-specific storage annotations (e.g. the ASR encoder's serialized
    // tensor-planar Q8 flag) are applied by the owner-installed hook; this
    // runtime stays agnostic of model metadata schemas.
    if (weight_load_hook_) {
        weight_load_hook_(gguf_key, t, gguf_loader);
    }

    if (mem == disk) {
        // Verbatim: tensor was declared at the on-disk dtype, so ggml_nbytes
        // matches the on-disk byte size for any format (F32/F16/BF16/Q*).
        const char* data = gguf_loader->get_tensor_file_data(gguf_key, mem_bytes);
        ggml_backend_tensor_set(t.tensor, data, 0, mem_bytes);
    } else if (disk == GGML_TYPE_F32 && mem == GGML_TYPE_F16) {
        // Converter quirk: some Conv weights are F32 on disk but wanted F16 in
        // memory. On-disk is 2x the F16 byte size.
        const char* data = gguf_loader->get_tensor_file_data(gguf_key, mem_bytes * 2);
        std::vector<char> fp16(mem_bytes);
        ggml_fp32_to_fp16_row(
            reinterpret_cast<const float*>(data), reinterpret_cast<ggml_fp16_t*>(fp16.data()),
            mem_bytes / 2);
        ggml_backend_tensor_set(t.tensor, fp16.data(), 0, mem_bytes);
    } else {
        throw std::runtime_error(
            "load_weight(" + gguf_key + "): cannot load on-disk dtype " +
            std::to_string(int(disk)) + " into in-memory dtype " + std::to_string(int(mem)) +
            " (only same-dtype copy or F32->F16 convert supported)");
    }
}

int
Session::setup() {
    std::lock_guard<std::mutex> compute_lock(backend_manager_->compute_mutex());
    buft_list = backend_manager_->get_buft_list();
    backends = backend_manager_->get_backends();

    // Weight declarations mutate modules, so this container cannot use the
    // two-pass sizing used for activation graphs.
    model_tensor_container =
        std::make_unique<TensorContainer>(buft_list, TensorContainer::ArenaSizes{});
    // State tensors (declared via create_state_tensor_*) live in this sibling
    // container, which is deliberately NEVER allocate_tensors_on_backend_buffers()'d:
    // each per-stream SessionState supplies their device backing per run. The
    // objects exist for the graph to reference; bind_state() points them at the
    // active stream's buffers.
    state_tensor_container =
        std::make_unique<TensorContainer>(buft_list, TensorContainer::ArenaSizes{});
    root_module->define_tensors(this);
    model_tensor_container->free_temp_ctx();
    model_tensor_container->allocate_tensors_on_backend_buffers();
    root_module->set_data(this);
    init_schedule();
    // Weights are uploaded; drop the loader's file handle + shared read
    // buffer (sized to the largest tensor). The loader reopens on demand if
    // a later Session (lazily built against the same GGUF) loads weights.
    if (gguf_loader != nullptr) {
        gguf_loader->release_file_resources();
    }
    if (memstats_enabled()) {
        fprintf(
            stderr, "[memstats] model-container device=%.2f MB\n",
            model_tensor_container->total_backend_buffer_bytes() / 1048576.0);
    }
    return 0;
}

namespace {

ggml_bf_tensor
create_input_tensor(TensorContainer* tc, const Session::Input& in) {
    const auto& s = in.shape;
    ggml_bf_tensor result = [&]() {
        switch (s.size()) {
            case 1:
                return tc->create_tensor_1d(in.name, in.dtype, s[0]);
            case 2:
                return tc->create_tensor_2d(in.name, in.dtype, s[0], s[1]);
            case 3:
                return tc->create_tensor_3d(in.name, in.dtype, s[0], s[1], s[2]);
            case 4:
                return tc->create_tensor_4d(in.name, in.dtype, s[0], s[1], s[2], s[3]);
            default:
                throw std::runtime_error(format(
                    "Session::Input '%s': shape rank %zu unsupported (need 1..4)", in.name.c_str(),
                    s.size()));
        }
    }();

    if (in.device_tensor != nullptr) {
        const DeviceTensor& source = *in.device_tensor;
        if (in.persistent || in.upload) {
            throw std::runtime_error(format(
                "Session::Input '%s': device input cannot be persistent or uploaded",
                in.name.c_str()));
        }
        if (!source.valid() || source.tensor->type != in.dtype) {
            throw std::runtime_error(format(
                "Session::Input '%s': invalid device tensor or dtype mismatch", in.name.c_str()));
        }
        size_t want = ggml_type_size(in.dtype);
        for (const int64_t dim : in.shape) want *= static_cast<size_t>(dim);
        const size_t have = ggml_nbytes(source.tensor);
        if (source.byte_offset > have || want > have - source.byte_offset) {
            throw std::runtime_error(format(
                "Session::Input '%s': device range [%zu, %zu) exceeds source size %zu",
                in.name.c_str(), source.byte_offset, source.byte_offset + want, have));
        }
        if (result.buft != source.buft) {
            throw std::runtime_error(format(
                "Session::Input '%s': device buffer type does not match session placement",
                in.name.c_str()));
        }
        result.tensor->buffer = source.tensor->buffer;
        result.tensor->data = static_cast<char*>(source.tensor->data) + source.byte_offset;
    }
    return result;
}

size_t
nbytes_of(const Session::Input& in) {
    size_t n = ggml_type_size(in.dtype);
    for (auto d : in.shape) n *= static_cast<size_t>(d);
    return n;
}

ggml_backend_t
backend_for_tensor(const std::vector<ggml_backend_t>& backends, const ggml_tensor* tensor) {
    const ggml_tensor* storage = tensor;
    while (storage != nullptr && storage->buffer == nullptr) {
        storage = storage->view_src;
    }
    if (storage == nullptr || storage->buffer == nullptr) {
        return nullptr;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(storage->buffer);
    ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
    for (ggml_backend_t backend : backends) {
        if (ggml_backend_get_device(backend) == device &&
            ggml_backend_supports_buft(backend, buft)) {
            return backend;
        }
    }
    return nullptr;
}

}  // namespace

// Adapts Input and Output values to the callback-based implementation.
void
Session::run(const std::vector<Input>& inputs, std::vector<Output>& outputs, SessionState* state) {
    const uint64_t key = hash_inputs(this, inputs);
    run_impl(
        key,
        [&inputs](Session* s, TensorContainer* tc) {
            TensorBag bag;
            for (const auto& in : inputs) {
                if (in.persistent && in.device_tensor != nullptr) {
                    throw std::runtime_error(format(
                        "Session::Input '%s': persistent and device bindings are mutually "
                        "exclusive",
                        in.name.c_str()));
                }
                if (in.persistent) {
                    // Tensor was declared in Module::define_tensors on
                    // model_tensor_container. Surface its metadata into the
                    // bag so the run-cache key picks up its name/type/shape.
                    bag.add_tensor(s->model_tensor_container->get_tensor_by_name(in.name));
                } else {
                    bag.add_tensor(create_input_tensor(tc, in));
                }
            }
            return bag;
        },
        [&inputs](Session* s, TensorContainer* tc) {
            bool transferred_host_data = false;
            for (const auto& in : inputs) {
                if (in.device_tensor != nullptr) {
                    continue;
                }
                if (!in.upload) {
                    continue;  // device contents already current (see Input::upload)
                }
                auto t = in.persistent ? s->model_tensor_container->get_tensor_by_name(in.name)
                                       : tc->get_tensor_by_name(in.name);
                // Validated before the device write: a size mismatch inside
                // ggml_backend_tensor_set asserts (process abort) rather than
                // throwing.
                const size_t have = nbytes_of(in);
                const size_t want = ggml_nbytes(t.tensor);
                if (have != want) {
                    throw std::runtime_error(format(
                        "Session::Input '%s': host size %zu != declared tensor size %zu",
                        in.name.c_str(), have, want));
                }
                ggml_backend_t backend = backend_for_tensor(s->backends, t.tensor);
                if (backend != nullptr && !in.synchronous_upload) {
                    ggml_backend_tensor_set_async(backend, t.tensor, in.host_data, 0, have);
                    transferred_host_data = true;
                } else {
                    ggml_backend_tensor_set(t.tensor, in.host_data, 0, have);
                }
            }
            return transferred_host_data;
        },
        [&outputs](Session* s, TensorBag out_bag, TensorContainer* tc) {
            bool transferred_host_data = false;
            for (auto& out : outputs) {
                if ((out.index < 0) == out.name.empty()) {
                    throw std::runtime_error(
                        "Session::Output: exactly one of `index` (>=0) or `name` must be set");
                }
                if (out.index >= 0 && static_cast<size_t>(out.index) >= out_bag.tensor_count()) {
                    throw std::runtime_error(format(
                        "Session::Output index %d out of range (graph has %zu outputs)", out.index,
                        out_bag.tensor_count()));
                }
                ggml_bf_tensor t = (out.index >= 0) ? out_bag.get_tensor(out.index)
                                                    : tc->get_tensor_by_name(out.name);
                const size_t want = ggml_nbytes(t.tensor);
                if (out.device_tensor != nullptr) {
                    // May reference scheduler-owned storage: valid only until
                    // the next run on this Session (async chaining contract).
                    *out.device_tensor = DeviceTensor{t.tensor, t.buft, 0};
                }
                if (out.host_buffer != nullptr) {
                    if (out.nbytes < want) {
                        throw std::runtime_error(format(
                            "Session::Output buffer too small for tensor (have %zu, need %zu)",
                            out.nbytes, want));
                    }
                    ggml_backend_t backend = backend_for_tensor(s->backends, t.tensor);
                    if (backend != nullptr) {
                        ggml_backend_tensor_get_async(backend, t.tensor, out.host_buffer, 0, want);
                    } else {
                        ggml_backend_tensor_get(t.tensor, out.host_buffer, 0, want);
                    }
                    transferred_host_data = true;
                } else if (out.device_tensor == nullptr) {
                    throw std::runtime_error(
                        "Session::Output requires a host buffer or a device tensor destination");
                }
                for (int i = 0; i < 4; ++i) out.out_shape[i] = t.tensor->ne[i];
            }
            return transferred_host_data;
        },
        state);
}

ggml_bf_tensor
Session::create_state_tensor_1d(const std::string& name, ggml_type t, int64_t ne0) {
    state_specs_.push_back({name, t, 1, {ne0, 1, 1, 1}});
    return state_tensor_container->create_tensor_1d(name, t, ne0);
}
ggml_bf_tensor
Session::create_state_tensor_2d(const std::string& name, ggml_type t, int64_t ne0, int64_t ne1) {
    state_specs_.push_back({name, t, 2, {ne0, ne1, 1, 1}});
    return state_tensor_container->create_tensor_2d(name, t, ne0, ne1);
}
ggml_bf_tensor
Session::create_state_tensor_3d(
    const std::string& name, ggml_type t, int64_t ne0, int64_t ne1, int64_t ne2) {
    state_specs_.push_back({name, t, 3, {ne0, ne1, ne2, 1}});
    return state_tensor_container->create_tensor_3d(name, t, ne0, ne1, ne2);
}

SessionState
Session::make_session_state() {
    // Match the template tensor shape and placement for each stream.
    std::lock_guard<std::mutex> compute_lock(backend_manager_->compute_mutex());
    auto tc = std::make_unique<TensorContainer>(buft_list, TensorContainer::ArenaSizes{});
    for (const auto& s : state_specs_) {
        switch (s.ndim) {
            case 1:
                tc->create_tensor_1d(s.name, s.type, s.ne[0]);
                break;
            case 2:
                tc->create_tensor_2d(s.name, s.type, s.ne[0], s.ne[1]);
                break;
            case 3:
                tc->create_tensor_3d(s.name, s.type, s.ne[0], s.ne[1], s.ne[2]);
                break;
            default:
                throw std::runtime_error("SessionState: unsupported state tensor rank");
        }
    }
    tc->free_temp_ctx();
    tc->allocate_tensors_on_backend_buffers();
    for (const auto& s : state_specs_) {
        ggml_tensor* t = tc->get_tensor_by_name(s.name).tensor;
        ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    }
    return SessionState(std::move(tc));
}

void
Session::reset_session_state(SessionState& state) {
    if (!state.valid())
        throw std::invalid_argument("Session::reset_session_state: invalid handle");
    std::lock_guard<std::mutex> compute_lock(backend_manager_->compute_mutex());
    TensorContainer* tc = state.container();
    for (const auto& s : state_specs_) {
        ggml_tensor* t = tc->get_tensor_by_name(s.name).tensor;
        ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    }
}

void
Session::bind_state(SessionState* state) {
    if (!state)
        return;  // a model with state tensors must pass one; one without has none to bind
    TensorContainer* tc = state->container();
    for (const auto& s : state_specs_) {
        ggml_tensor* tmpl = state_tensor_container->get_tensor_by_name(s.name).tensor;
        ggml_tensor* buf = tc->get_tensor_by_name(s.name).tensor;
        tmpl->buffer = buf->buffer;
        tmpl->data = buf->data;
    }
}

void
Session::collect_state_views(ggml_cgraph* gf, std::vector<ggml_tensor*>& out) const {
    out.clear();
    if (state_specs_.empty())
        return;
    // A graph tensor is state-derived when its view chain reaches a template tensor.
    std::unordered_set<const ggml_tensor*> roots;
    roots.reserve(state_specs_.size());
    for (const auto& s : state_specs_) {
        roots.insert(state_tensor_container->get_tensor_by_name(s.name).tensor);
    }
    auto is_state_view = [&](ggml_tensor* t) {
        for (ggml_tensor* v = t->view_src; v != nullptr; v = v->view_src) {
            if (roots.count(v))
                return true;
        }
        return false;
    };
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor* t = ggml_graph_node(gf, i);
        if (is_state_view(t))
            out.push_back(t);
    }
}

void
Session::read_model_tensor(const std::string& name, void* host_buffer, size_t nbytes) {
    std::lock_guard<std::mutex> compute_lock(backend_manager_->compute_mutex());
    auto t = model_tensor_container->get_tensor_by_name(name);
    const size_t want = ggml_nbytes(t.tensor);
    if (nbytes < want) {
        throw std::runtime_error(format(
            "Session::read_model_tensor('%s'): buffer too small (have %zu, need %zu)", name.c_str(),
            nbytes, want));
    }
    ggml_backend_tensor_get(t.tensor, host_buffer, 0, want);
}

void
Session::run_impl(
    uint64_t key, const std::function<TensorBag(Session*, TensorContainer*)>& define_input_tensors,
    const std::function<bool(Session*, TensorContainer*)>& set_input_data,
    const std::function<bool(Session*, TensorBag, TensorContainer*)>& return_output,
    SessionState* state) {
    // Cache mutation, allocation, and compute share backend state.
    std::lock_guard<std::mutex> compute_lock(backend_manager_->compute_mutex());

    // Bind shared state-tensor objects to this stream for the duration of the run.
    bind_state(state);

    // The probe container measures arena usage on cache misses.
    constexpr size_t kProbeArenaBytes = 64 * 1024 * 1024;
    auto make_probe = [&]() {
        TensorContainer::ArenaSizes probe_sizes;
        probe_sizes.temp_ctx_bytes = kProbeArenaBytes;
        probe_sizes.default_buft_bytes = kProbeArenaBytes;
        return std::make_unique<TensorContainer>(buft_list, probe_sizes);
    };

    std::unique_ptr<TensorContainer> probe;
    TensorBag probe_inputs;
    bool fresh = false;  // entry built this call, no successful run yet
    auto it = run_cache_.find(key);
    if (it != run_cache_.end() && it->second.pool_generation != sched_pool_generation_) {
        // The shared scheduler pool grew after this entry was cached (a larger
        // shape resized a backend's compute buffer), so its device placements
        // are stale - re-running it raises CUDA invalid-argument. Rebuild.
        auto lru_it = std::find(run_cache_lru_.begin(), run_cache_lru_.end(), key);
        if (lru_it != run_cache_lru_.end())
            run_cache_lru_.erase(lru_it);
        run_cache_.erase(it);
        it = run_cache_.end();
    }
    if (it == run_cache_.end()) {
        // Evict before allocating the replacement so its device storage is available.
        while (run_cache_.size() >= run_cache_capacity_ && !run_cache_lru_.empty()) {
            uint64_t evict = run_cache_lru_.front();
            run_cache_lru_.erase(run_cache_lru_.begin());
            run_cache_.erase(evict);
        }

        if (!probe) {
            probe = make_probe();
            probe_inputs = define_input_tensors(this, probe.get());
        }

        // Probe graph construction must be idempotent because the result is discarded.
        (void)root_module->build_graph(this, probe_inputs, probe.get());
        const auto measurement = probe->measure();
        probe.reset();

        auto fit_size = [](size_t used) -> size_t {
            constexpr size_t kPage = 4096;
            constexpr size_t kHeadroom = 64 * 1024;
            const size_t target = static_cast<size_t>(used * 1.05) + kHeadroom;
            return ((target + kPage - 1) / kPage) * kPage;
        };
        TensorContainer::ArenaSizes fit_sizes;
        // Scheduler-managed intermediates share storage by liveness. Pool
        // growth invalidates older cached placements through the generation check.
        fit_sizes.sched_managed = true;
        fit_sizes.temp_ctx_bytes = fit_size(measurement.temp_ctx_bytes);
        // Default for any buft the probe didn't touch but the real build
        // might create (shouldn't happen given idempotency, but be safe).
        fit_sizes.default_buft_bytes = 4 * 1024 * 1024;
        for (const auto& kv : measurement.per_buft_bytes) {
            fit_sizes.per_buft[kv.first] = fit_size(kv.second);
        }

        CachedRun cr;
        cr.container = std::make_unique<TensorContainer>(buft_list, fit_sizes);
        cr.input_tensors = define_input_tensors(this, cr.container.get());
        cr.output_tensors = root_module->build_graph(this, cr.input_tensors, cr.container.get());
        cr.container->allocate_tensors_on_backend_buffers();
        cr.container->free_temp_ctx();
        cr.gf = build_graph_into(
            root_module, cr.input_tensors, cr.output_tensors, sched_meta, cr.sched_meta);
        collect_state_views(cr.gf, cr.state_views);

        const int actual_nodes = ggml_graph_n_nodes(cr.gf);
        if (actual_nodes > static_cast<int>(kSchedGraphErrorFrac * kSchedGraphSize)) {
            throw std::runtime_error(format(
                "graph has %d nodes, above %.0f%% of scheduler cap %zu — bump "
                "kSchedGraphSize in session.cpp",
                actual_nodes, 100.0 * kSchedGraphErrorFrac, kSchedGraphSize));
        }
        if (actual_nodes > static_cast<int>(kSchedGraphWarnFrac * kSchedGraphSize)) {
            GGMLF_LOG_WARN(
                "graph has %d nodes, above %.0f%% of scheduler cap %zu — "
                "consider bumping kSchedGraphSize\n",
                actual_nodes, 100.0 * kSchedGraphWarnFrac, kSchedGraphSize);
        }

        if (memstats_enabled()) {
            const char* out0 =
                (cr.output_tensors.tensor_count() > 0 && cr.output_tensors.get_tensor(0).tensor)
                    ? cr.output_tensors.get_tensor(0).tensor->name
                    : "?";
            fprintf(
                stderr,
                "[memstats] cache-entry out=%-20s nodes=%d graph_meta=%.2f MB "
                "device_arenas=%.2f MB (entries now %zu)\n",
                out0, actual_nodes, cr.sched_meta.size() / 1048576.0,
                cr.container->total_backend_buffer_bytes() / 1048576.0, run_cache_.size() + 1);
            cr.container->dump_largest_tensors(12);
        }
        it = run_cache_.emplace(key, std::move(cr)).first;
        run_cache_lru_.push_back(key);
        fresh = true;
    } else {
        auto lru_it = std::find(run_cache_lru_.begin(), run_cache_lru_.end(), key);
        if (lru_it != run_cache_lru_.end()) {
            run_cache_lru_.erase(lru_it);
            run_cache_lru_.push_back(key);
        }
    }

    CachedRun& cr = it->second;

    // Reset the scheduler on every exit path once alloc_graph has run: an
    // un-reset sched trips GGML_ASSERT(!sched->is_alloc) on the next run and
    // aborts the process.
    struct SchedResetGuard {
        ggml_backend_sched_t s = nullptr;
        ~SchedResetGuard() {
            if (s)
                ggml_backend_sched_reset(s);
        }
    } sched_guard;

    static const bool t_log = std::getenv("NEMO_SPEECH_TIMING") != nullptr;
    using _clk = std::chrono::high_resolution_clock;
    auto _t0 = _clk::now();
    // Transactional: an entry that has never completed a run is evicted on
    // failure rather than retried as a cache hit; a deterministic failure
    // (bad graph, undersized output) would otherwise never rebuild.
    try {
        const bool host_input_pending = set_input_data(this, cr.container.get());
        auto _t1 = _clk::now();

        // Valid single-backend placements can bypass scheduler allocation.
        const bool direct = !fresh && cr.direct_ok && state == nullptr && cr.state_views.empty();
        if (direct) {
            auto _t2d = _clk::now();
            if (ggml_backend_graph_compute_async(cr.direct_backend, cr.gf) != GGML_STATUS_SUCCESS) {
                GGMLF_LOG_ERROR("Failed to compute graph (direct)\n");
                throw std::runtime_error("failed to compute graph");
            }
            auto _t3d = _clk::now();
            const bool host_output_pending =
                return_output(this, cr.output_tensors, cr.container.get());
            if (host_input_pending || host_output_pending) {
                ggml_backend_synchronize(cr.direct_backend);
            }
            auto _t4d = _clk::now();
            if (t_log) {
                auto ms = [](auto a, auto b) {
                    return std::chrono::duration<double, std::milli>(b - a).count();
                };
                const char* out0 =
                    (cr.output_tensors.tensor_count() > 0 && cr.output_tensors.get_tensor(0).tensor)
                        ? cr.output_tensors.get_tensor(0).tensor->name
                        : "?";
                fprintf(
                    stderr,
                    "[timing] session out=%-20s nodes=%d in=%.2f direct compute=%.2f out=%.2f "
                    "total=%.2f ms\n",
                    out0, ggml_graph_n_nodes(cr.gf), ms(_t0, _t1), ms(_t2d, _t3d), ms(_t3d, _t4d),
                    ms(_t0, _t4d));
            }
            return;
        }

        // Clear state views so allocation binds them to the current SessionState.
        if (state != nullptr) {
            for (ggml_tensor* v : cr.state_views) {
                v->buffer = nullptr;
                v->data = nullptr;
            }
        }

        sched_guard.s = sched.get();
        if (!ggml_backend_sched_alloc_graph(sched.get(), cr.gf)) {
            GGMLF_LOG_ERROR("Failed to allocate graph\n");
            throw std::runtime_error("failed to allocate graph");
        }

        // Record direct-compute eligibility for subsequent hits: one split,
        // one non-CPU backend that supports every active node in the complete
        // graph. The CPU path routes thread-count setup through
        // ggml_graph_compute_helper, so keep it on the scheduler.
        cr.direct_ok = false;
        cr.direct_backend = nullptr;
        if (ggml_backend_sched_get_n_splits(sched.get()) == 1 && ggml_graph_n_nodes(cr.gf) > 0) {
            ggml_backend_t b =
                ggml_backend_sched_get_tensor_backend(sched.get(), ggml_graph_node(cr.gf, 0));
            if (b != nullptr) {
                ggml_backend_dev_t dev = ggml_backend_get_device(b);
                if (dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU &&
                    backend_supports_compute_graph(b, cr.gf)) {
                    cr.direct_ok = true;
                    cr.direct_backend = b;
                }
            }
        }

        // Any pool-size change invalidates other placements; sizes can also redistribute.
        if (sched_pool_sizes_.size() < backends.size())
            sched_pool_sizes_.resize(backends.size(), 0);
        bool pool_changed = false;
        for (size_t b = 0; b < backends.size(); ++b) {
            const size_t sz = ggml_backend_sched_get_buffer_size(sched.get(), backends[b]);
            if (sz != sched_pool_sizes_[b]) {
                sched_pool_sizes_[b] = sz;
                pool_changed = true;
            }
        }
        if (pool_changed)
            sched_pool_generation_++;
        cr.pool_generation = sched_pool_generation_;

        // Capture assignments after allocation and before scheduler reset.
        static const bool capture_schedule = std::getenv("NEMO_SPEECH_SCHEDULE_CAPTURE") != nullptr;
        if (capture_schedule && cr.schedule.empty()) {
            if (memstats_enabled()) {
                for (size_t b = 0; b < backends.size(); ++b) {
                    const size_t sz = ggml_backend_sched_get_buffer_size(sched.get(), backends[b]);
                    if (sz > 0) {
                        fprintf(
                            stderr, "[memstats] sched-compute-buffer backend=%s size=%.2f MB\n",
                            ggml_backend_name(backends[b]), sz / 1048576.0);
                    }
                }
            }
            const int n_nodes = ggml_graph_n_nodes(cr.gf);
            cr.schedule.reserve(n_nodes);
            for (int i = 0; i < n_nodes; ++i) {
                ggml_tensor* node = ggml_graph_node(cr.gf, i);
                NodeAssignment a;
                a.op = ggml_op_name(node->op);
                a.name = node->name[0] ? node->name : "";
                ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched.get(), node);
                a.backend = b ? ggml_backend_name(b) : "<unassigned>";
                cr.schedule.push_back(std::move(a));
            }
        }

        auto _t2 = _clk::now();
        if (!ggml_graph_compute_helper_async(sched.get(), cr.gf, 4)) {
            GGMLF_LOG_ERROR("Failed to compute graph\n");
            throw std::runtime_error("failed to compute graph");
        }
        auto _t3 = _clk::now();

        // Outputs are read before the guard resets the scheduler: in
        // sched_managed mode non-named output tensors live in the scheduler's
        // allocator.
        const bool host_output_pending = return_output(this, cr.output_tensors, cr.container.get());
        // Scheduler-managed storage cannot be reset while the graph is still
        // using it. Cache hits eligible for asynchronous chaining take the
        // direct path above.
        ggml_backend_sched_synchronize(sched.get());
        auto _t4 = _clk::now();
        if (t_log) {
            auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            const char* out0 =
                (cr.output_tensors.tensor_count() > 0 && cr.output_tensors.get_tensor(0).tensor)
                    ? cr.output_tensors.get_tensor(0).tensor->name
                    : "?";
            fprintf(
                stderr,
                "[timing] session out=%-20s nodes=%d in=%.2f alloc=%.2f compute=%.2f out=%.2f "
                "total=%.2f ms\n",
                out0, ggml_graph_n_nodes(cr.gf), ms(_t0, _t1), ms(_t1, _t2), ms(_t2, _t3),
                ms(_t3, _t4), ms(_t0, _t4));
        }
    }
    catch (...) {
        if (fresh) {
            auto lru_it = std::find(run_cache_lru_.begin(), run_cache_lru_.end(), key);
            if (lru_it != run_cache_lru_.end())
                run_cache_lru_.erase(lru_it);
            run_cache_.erase(key);
        }
        throw;
    }
}


}  // namespace ggml_runtime
