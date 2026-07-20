// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
#pragma once

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifdef __GNUC__
#ifdef __MINGW32__
#define GGMLF_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define GGMLF_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define GGMLF_ATTRIBUTE_FORMAT(...)
#endif

namespace ggml_runtime {
GGMLF_ATTRIBUTE_FORMAT(5, 6)
void log_internal(
    ggml_log_level level, const char* file, int line, const char* func, const char* format, ...);
void log_callback_default(ggml_log_level level, const char* text, void* user_data);
}  // namespace ggml_runtime

// Kept in the global namespace because llama_file and other global-scope
// helpers in loader.cpp call it for std::runtime_error messages.
std::string format(const char* fmt, ...);

#define GGMLF_LOG_ERROR(...) \
    ::ggml_runtime::log_internal(GGML_LOG_LEVEL_ERROR, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define GGMLF_LOG_WARN(...) \
    ::ggml_runtime::log_internal(GGML_LOG_LEVEL_WARN, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define GGMLF_LOG_INFO(...) \
    ::ggml_runtime::log_internal(GGML_LOG_LEVEL_INFO, __FILE__, __LINE__, __func__, __VA_ARGS__)

struct llama_file;

namespace ggml_runtime {

class GGUFLoader {
   public:
    explicit GGUFLoader(const std::string& path);
    // Out-of-line so the unique_ptr<llama_file> deleter sees the complete
    // llama_file type (defined in loader.cpp).
    ~GGUFLoader();

    const char* get_tensor_file_data(const std::string& tensor_name, size_t size);
    // Releases file-backed resources while retaining metadata; tensor reads reopen on demand.
    void release_file_resources();
    ggml_type get_tensor_type(const std::string& tensor_name);
    // Returns the on-disk rank (1-4), or 0 when the tensor is absent.
    int get_tensor_n_dims(const std::string& tensor_name) const;
    bool has_tensor(const std::string& tensor_name) const;

    // Metadata accessors (return `def` if key missing).
    uint32_t get_u32(const std::string& key, uint32_t def = 0) const;
    int32_t get_i32(const std::string& key, int32_t def = 0) const;
    float get_f32(const std::string& key, float def = 0.0f) const;
    bool get_bool(const std::string& key, bool def = false) const;
    std::string get_str(const std::string& key, const std::string& def = "") const;
    std::vector<int32_t> get_i32_array(const std::string& key) const;
    std::vector<std::string> get_str_array(const std::string& key) const;
    bool has_key(const std::string& key) const;

   private:
    std::string m_path;
    gguf_context_ptr m_context;
    std::unique_ptr<llama_file> m_file;
    std::map<std::string, std::tuple<ggml_type, uint64_t>> m_tensor_infos;
    // Per-tensor on-disk dimensionality.
    std::map<std::string, int> m_tensor_n_dims;
    std::vector<char> m_tensor_buffer;
};

using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

struct ggml_bf_tensor {
    ggml_tensor* tensor;
    ggml_backend_buffer_type_t buft;

    ggml_bf_tensor(ggml_tensor* tensor, ggml_backend_buffer_type_t buft)
        : tensor(tensor), buft(buft) {}
};
using ggml_bf_tensor_t = ggml_bf_tensor*;

struct ggml_bf_context {
    ggml_context* ctx;
    ggml_backend_buffer_type_t buft;

    ggml_bf_context(ggml_context* ctx, ggml_backend_buffer_type_t buft) : ctx(ctx), buft(buft) {}
};
using ggml_bf_context_t = ggml_bf_context*;

using buft_ctx_map_t = std::map<ggml_backend_buffer_type_t, ggml_bf_context>;

class TensorBag {
   public:
    TensorBag();
    ~TensorBag() = default;

    void add_tensor(ggml_bf_tensor tensor);
    ggml_bf_tensor get_tensor(size_t index) const;
    size_t tensor_count() const;
    void set_first_tensor(ggml_bf_tensor tensor);

   private:
    std::vector<ggml_bf_tensor> tensors;
};

class Session;
class TensorContainer;

class Module {
   public:
    virtual ~Module() = default;
    // Declare persistent model tensors.
    virtual void define_tensors(Session* session) = 0;
    // Build the forward graph for one input shape. Must be idempotent
    // and side-effect-free outside the per-call TensorContainer — the
    // runtime calls build_graph TWICE per cache miss (probe pass to
    // measure arena size, real pass to populate the cached container).
    virtual TensorBag build_graph(
        Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) = 0;
    // Upload values into tensors declared by define_tensors().
    virtual void set_data(Session* session) = 0;
};

struct Params {
    bool use_gpu = false;
    int gpu_device_idx = 0;
    char* pe_bin_path = nullptr;
};

// Owns ggml backend handles shared by one or more Sessions.
class BackendManager {
   public:
    explicit BackendManager(Params params);
    // Out-of-line so ggml_backend_ptr's deleter is instantiated in the implementation TU.
    ~BackendManager();

    BackendManager(const BackendManager&) = delete;
    BackendManager& operator=(const BackendManager&) = delete;

    void init_backends();
    buft_list_t get_buft_list();
    // Non-owning view of the registered backend handles, suitable for
    // ggml_backend_sched_new. The returned ggml_backend_t handles remain
    // owned by this BackendManager.
    std::vector<ggml_backend_t> get_backends();
    const Params& get_params() const { return params; }

    // Sessions sharing a manager serialize backend setup and compute. Use
    // separate managers for independent execution lanes.
    std::mutex& compute_mutex() { return compute_mu_; }

    // Non-owning handle to this manager's GPU backend, or nullptr for CPU-only use.
    ggml_backend_t gpu_backend_handle() const { return gpu_backend; }

   private:
    Params params;
    std::vector<ggml_backend_ptr> backends;
    // Non-owning alias into `backends`.
    ggml_backend_t gpu_backend = nullptr;
    buft_list_t buft_list;
    std::mutex compute_mu_;
};

class TensorContainer {
   public:
    // Arena sizes for metadata and buffer-type-specific graph contexts.
    struct ArenaSizes {
        size_t temp_ctx_bytes = 64 * 1024 * 1024;      // 64 MB virtual default
        size_t default_buft_bytes = 64 * 1024 * 1024;  // 64 MB virtual default
        std::map<ggml_backend_buffer_type_t, size_t> per_buft;
        // Leave unnamed intermediates to the scheduler's liveness allocator.
        bool sched_managed = false;
    };

    TensorContainer(buft_list_t buft_list, ArenaSizes sizes);
    ~TensorContainer() = default;

    // Reports metadata arena use after graph construction.
    struct Measurement {
        size_t temp_ctx_bytes;
        std::map<ggml_backend_buffer_type_t, size_t> per_buft_bytes;
    };
    Measurement measure() const;

    size_t total_backend_buffer_bytes() const;

    void dump_largest_tensors(size_t top_n) const;

    ggml_context* get_temp_ctx();
    ggml_bf_context get_ctx_of_buffer_type(ggml_backend_buffer_type_t buft);
    void allocate_tensors_on_backend_buffers();
    void free_temp_ctx();
    ggml_bf_tensor get_tensor_by_name(const std::string& name);
    bool has_tensor_by_name(const std::string& name);
    void cache_tensor(std::string name, ggml_bf_tensor tensor);

    // Declares tensors on the primary device's main buffer type.
    ggml_bf_tensor create_tensor_1d(std::string name, ggml_type data_type, int64_t ne0);
    ggml_bf_tensor create_tensor_2d(
        std::string name, ggml_type data_type, int64_t ne0, int64_t ne1);
    ggml_bf_tensor create_tensor_3d(
        std::string name, ggml_type data_type, int64_t ne0, int64_t ne1, int64_t ne2);
    ggml_bf_tensor create_tensor_4d(
        std::string name, ggml_type data_type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

   private:
    ArenaSizes sizes_;
    // Lazily materializes tensor metadata before backend buffer allocation.
    ggml_context_ptr temp_ctx;
    buft_list_t buft_list;
    // Per-buffer-type owning storage for the activation/weight contexts
    // built in get_ctx_of_buffer_type(). ctx_map below holds raw views
    // (the ggml_bf_context.ctx field) into these contexts.
    std::vector<ggml_context_ptr> owned_ctxs_;
    buft_ctx_map_t ctx_map;
    std::vector<ggml_backend_buffer_ptr> backend_buffers;
    std::map<std::string, ggml_bf_tensor> tensor_lookup;

    ggml_bf_tensor m_create_tensor(ggml_tensor* meta, std::string& name);
};

// Per-stream device storage for a Session's state tensors. The state must
// outlive any run that binds it.
class SessionState {
   public:
    SessionState() = default;
    explicit SessionState(std::unique_ptr<TensorContainer> tc) : tc_(std::move(tc)) {}
    bool valid() const { return tc_ != nullptr; }
    TensorContainer* container() const { return tc_.get(); }

   private:
    std::unique_ptr<TensorContainer> tc_;
};

class Session {
   public:
    Params params;

    explicit Session(BackendManager& backend_manager, Module* module, GGUFLoader* gguf_loader);
    // Out-of-line so ggml deleters are instantiated in the implementation TU.
    ~Session();

    int setup();

    // Allows exact dtype copies and F32-to-F16 conversion.
    void load_weight(const std::string& gguf_key);

    // Invoked before upload so model code can annotate destination storage.
    // Install before setup(), which loads the model weights.
    using WeightLoadHook =
        std::function<void(const std::string& gguf_key, ggml_bf_tensor& t, GGUFLoader* loader)>;
    void set_weight_load_hook(WeightLoadHook hook) { weight_load_hook_ = std::move(hook); }

    // Persistent inputs must already exist in model_tensor_container.
    struct Input {
        std::string name;
        ggml_type dtype;
        const void* host_data;
        std::vector<int64_t> shape;  // ne[0..N], 1 <= N <= 4
        // Bind an existing named model tensor instead of allocating a per-call input.
        bool persistent = false;
        // False when persistent device contents are already current.
        bool upload = true;
    };

    // Select by exactly one of TensorBag index or cached name. nbytes is the
    // host-buffer capacity; out_shape receives the resolved dimensions.
    struct Output {
        int index = -1;
        std::string name;
        void* host_buffer = nullptr;
        size_t nbytes = 0;
        int64_t out_shape[4] = {0, 0, 0, 0};
    };

    // Reuse a SessionState across calls to preserve device-updated stream state;
    // separate handles isolate streams.
    void run(
        const std::vector<Input>& inputs, std::vector<Output>& outputs,
        SessionState* state = nullptr);

    // Declares persistent graph state backed by a per-run SessionState.
    ggml_bf_tensor create_state_tensor_1d(const std::string& name, ggml_type t, int64_t ne0);
    ggml_bf_tensor create_state_tensor_2d(
        const std::string& name, ggml_type t, int64_t ne0, int64_t ne1);
    ggml_bf_tensor create_state_tensor_3d(
        const std::string& name, ggml_type t, int64_t ne0, int64_t ne1, int64_t ne2);

    // Call after setup().
    SessionState make_session_state();

    // Zero an existing state's buffers without reallocating them.
    void reset_session_state(SessionState& state);

    // Reads a persistent model tensor under the backend compute lock; nbytes is
    // the destination capacity and must cover the tensor.
    void read_model_tensor(const std::string& name, void* host_buffer, size_t nbytes);

    // Reports cached graph placement and CPU fallbacks.
    void dump_schedule(std::ostream& out, const std::string& session_label) const;

    // Module-internal state; external callers should use run().
    std::unique_ptr<TensorContainer> model_tensor_container;
    // Declared state tensors live here (objects the graph references); their
    // device buffers come from a per-run SessionState, not from this container's
    // bulk allocation (it is never allocate_tensors_on_backend_buffers()'d).
    std::unique_ptr<TensorContainer> state_tensor_container;
    GGUFLoader* gguf_loader;

   private:
    void init_schedule();

    WeightLoadHook weight_load_hook_;

    // Called under the compute mutex before graph allocation.
    void bind_state(SessionState* state);

    // Finds graph views that must be rebound for each SessionState.
    void collect_state_views(ggml_cgraph* gf, std::vector<ggml_tensor*>& out) const;

    struct StateSpec {
        std::string name;
        ggml_type type;
        int ndim;
        int64_t ne[4];
    };
    std::vector<StateSpec> state_specs_;

    void run_impl(
        uint64_t key,
        const std::function<TensorBag(Session*, TensorContainer*)>& define_input_tensors,
        const std::function<void(Session*, TensorContainer*)>& set_input_data,
        const std::function<void(Session*, TensorBag, TensorContainer*)>& return_output,
        SessionState* state);

    BackendManager* backend_manager_;
    Module* root_module;
    buft_list_t buft_list;
    std::vector<ggml_backend_t> backends;

    ggml_backend_sched_ptr sched;
    std::vector<uint8_t> sched_meta;

    // Captured when a graph is built and reused by dump_schedule().
    struct NodeAssignment {
        std::string op;
        std::string name;
        std::string backend;
    };

    // Per-input-shape graph cache, bounded by LRU eviction.
    struct CachedRun {
        std::unique_ptr<TensorContainer> container;
        TensorBag input_tensors;
        TensorBag output_tensors;
        std::vector<uint8_t> sched_meta;  // backing for gf
        ggml_cgraph* gf;
        std::vector<NodeAssignment> schedule;
        // Cleared before allocation so views bind to the current SessionState.
        std::vector<ggml_tensor*> state_views;
        // Entries with stale scheduler-pool placements are rebuilt.
        uint64_t pool_generation = 0;
        // Single-backend graphs without state views can bypass the scheduler.
        bool direct_ok = false;
        ggml_backend_t direct_backend = nullptr;
    };
    std::unordered_map<uint64_t, CachedRun> run_cache_;
    std::vector<uint64_t> run_cache_lru_;  // back() = most-recent
    size_t run_cache_capacity_ = 4;
    // Pool growth invalidates cached device placements.
    std::vector<size_t> sched_pool_sizes_;
    uint64_t sched_pool_generation_ = 0;

   public:
    // Maximum number of cached input shapes.
    void set_run_cache_capacity(size_t cap) { run_cache_capacity_ = cap; }
};

}  // namespace ggml_runtime
