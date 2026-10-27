// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "ctc_timestamp.h"

#include <sentencepiece_processor.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

#include "base64.h"
#include "ggml-backend.h"
#include "nvtx_utils.h"
#include "pee_rope_transformer.h"
#include "types.h"

namespace nemo_speech::speechlm {
using namespace ggml_runtime;
namespace {
TensorBag
one(ggml_tensor* t, ggml_backend_buffer_type_t b) {
    TensorBag bag;
    bag.add_tensor({t, b});
    return bag;
}
std::string
unbase64(const std::string& s) {
    auto out = nemo_speech::decode_base64(s);
    if (!out)
        throw std::runtime_error("invalid CTC tokenizer base64");
    return std::move(*out);
}
struct CtcGraph : Module {
    nemo_speech::speechlm::RopeTransformerConfig cfg;
    int vocab;
    std::string name;  // ctc.model_name (the checkpoint's), empty = derived from the vocabulary
    float residual_scale;
    int upsampling;                    // CTC frames per encoder frame
    std::unique_ptr<Linear> upsample;  // with upsampling > 1
    std::unique_ptr<LayerNorm> embed, final;
    std::vector<std::unique_ptr<nemo_speech::speechlm::RopeTransformerBlock>> layers;
    std::unique_ptr<Linear> classifier;
    explicit CtcGraph(GGUFLoader& l) {
        if (!ggml_runtime::gguf_holds(l, "ctc_timestamp"))
            throw std::runtime_error("expected ctc_timestamp GGUF");
        cfg.d_model = l.get_u32("ctc.d_model");
        cfg.n_heads = l.get_u32("ctc.n_heads");
        cfg.n_layers = l.get_u32("ctc.n_layers");
        cfg.ff_hidden = l.get_u32("ctc.ff_hidden");
        cfg.rotary_dims = l.get_u32("ctc.rotary_dims");
        cfg.qk_norm = l.get_bool("ctc.qk_norm");
        cfg.qkv_bias = l.get_bool("ctc.qkv_bias");
        vocab = l.get_u32("ctc.vocab_size");
        residual_scale = l.get_f32("ctc.residual_scale");
        name = l.get_str("ctc.model_name");
        upsampling = int(l.get_u32("ctc.upsampling_factor", 1));
        if (cfg.d_model <= 0 || cfg.n_heads <= 0 || cfg.d_model % cfg.n_heads ||
            cfg.n_layers <= 0 || vocab < 2 || upsampling < 1)
            throw std::runtime_error("invalid CTC adapter dimensions");
        if (upsampling > 1)
            upsample =
                std::make_unique<Linear>("ctc.upsample", cfg.d_model, upsampling * cfg.d_model);
        const int64_t shape[] = {cfg.d_model, 1, 1, 1};
        embed = std::make_unique<LayerNorm>("ctc.transformer.embed_norm", shape);
        final = std::make_unique<LayerNorm>("ctc.transformer.final_norm", shape);
        for (int i = 0; i < cfg.n_layers; ++i)
            layers.emplace_back(new nemo_speech::speechlm::RopeTransformerBlock(
                "ctc.transformer.layers." + std::to_string(i), cfg));
        classifier = std::make_unique<Linear>("ctc.classifier", cfg.d_model, vocab);
    }
    void define_tensors(Session* s) override {
        if (upsample)
            upsample->define_tensors(s);
        embed->define_tensors(s);
        final->define_tensors(s);
        for (auto& l : layers) l->define_tensors(s);
        classifier->define_tensors(s);
    }
    void set_data(Session* s) override {
        if (upsample)
            upsample->set_data(s);
        embed->set_data(s);
        final->set_data(s);
        for (auto& l : layers) l->set_data(s);
        classifier->set_data(s);
    }
    TensorBag build_graph(Session* s, TensorBag, TensorContainer* tc) override {
        auto w = s->model_tensor_container->get_tensor_by_name("ctc.classifier.weight").tensor;
        auto b = s->model_tensor_container->get_tensor_by_name("ctc.classifier.bias").tensor;
        if (tc->has_tensor_by_name("input.ids")) {
            auto x = tc->get_tensor_by_name("input.states");
            auto* ctx = tc->get_ctx_of_buffer_type(x.buft).ctx;
            auto ids = tc->get_tensor_by_name("input.ids").tensor;
            auto selected = ggml_get_rows(ctx, w, ids);
            auto bias = ggml_get_rows(ctx, ggml_reshape_2d(ctx, b, 1, vocab), ids);
            auto y = ggml_mul_mat(ctx, selected, x.tensor);
            ggml_prec_set_acc(y, GGML_PREC_F32);
            y = ggml_add(ctx, y, ggml_reshape_1d(ctx, bias, ids->ne[0]));
            // No softmax: its per-frame normalizer is identical for every CTC path.
            return one(y, x.buft);
        }
        if (tc->has_tensor_by_name("input.classes")) {
            // Logits of evidence rows: h rows gathered from the device store, F per batch item
            // (input.gather (F, B)), against each item's classes (input.classes (K, B)): (K, F, B).
            // The log-sum-exp is subtracted on the host.
            auto rows = tc->get_tensor_by_name("input.rows");
            auto* ctx = tc->get_ctx_of_buffer_type(rows.buft).ctx;
            auto gather = tc->get_tensor_by_name("input.gather").tensor;
            auto classes = tc->get_tensor_by_name("input.classes").tensor;
            const int64_t D = rows.tensor->ne[0], F = gather->ne[0], K = classes->ne[0],
                          B = classes->ne[1];
            auto flat = ggml_reshape_1d(ctx, classes, K * B);
            auto h = ggml_reshape_3d(
                ctx, ggml_get_rows(ctx, rows.tensor, ggml_reshape_1d(ctx, gather, F * B)), D, F, B);
            auto y =
                ggml_mul_mat(ctx, ggml_reshape_3d(ctx, ggml_get_rows(ctx, w, flat), D, K, B), h);
            ggml_prec_set_acc(y, GGML_PREC_F32);
            auto bias = ggml_get_rows(ctx, ggml_reshape_2d(ctx, b, 1, vocab), flat);
            return one(ggml_add(ctx, y, ggml_reshape_3d(ctx, bias, K, 1, B)), rows.buft);
        }
        // The adapter over a batch of windows of encoder frames (D, T, B): input.states, or rows
        // gathered from the device store (input.rows, input.gather (T, B)). Positions and the key
        // mask (input.pos, input.mask) are in CTC frames, upsampling x T.
        const bool gathered = tc->has_tensor_by_name("input.gather");
        auto in = tc->get_tensor_by_name(gathered ? "input.rows" : "input.states");
        auto* ctx = tc->get_ctx_of_buffer_type(in.buft).ctx;
        ggml_tensor* x = in.tensor;
        if (gathered) {
            auto gather = tc->get_tensor_by_name("input.gather").tensor;
            x = ggml_reshape_3d(
                ctx,
                ggml_get_rows(ctx, x, ggml_reshape_1d(ctx, gather, gather->ne[0] * gather->ne[1])),
                x->ne[0], gather->ne[0], gather->ne[1]);
        }
        if (upsample) {
            // ConvTranspose1d with kernel = stride = upsampling, as one product: each encoder
            // frame's upsampling CTC frames are consecutive D-rows of (upsampling x D, T, B), so a
            // reshape is the (D, upsampling x T, B) sequence. The residual adds this input.
            auto y = upsample->build_graph(s, one(x, in.buft), tc).get_tensor(0).tensor;
            x = ggml_reshape_3d(ctx, y, x->ne[0], upsampling * x->ne[1], x->ne[2]);
        }
        auto bag = embed->build_graph(s, one(x, in.buft), tc);
        bag.add_tensor(tc->get_tensor_by_name("input.pos"));
        auto km = tc->get_tensor_by_name("input.mask").tensor;
        auto shape = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, x->ne[1], x->ne[1], 1, x->ne[2]);
        bag.add_tensor({ggml_cast(ctx, ggml_repeat(ctx, km, shape), GGML_TYPE_F16), in.buft});
        for (auto& l : layers) bag = l->build_graph(s, bag, tc);
        auto y =
            final->build_graph(s, one(bag.get_tensor(0).tensor, in.buft), tc).get_tensor(0).tensor;
        auto h = ggml_add(ctx, x, ggml_scale(ctx, y, residual_scale));
        if (!gathered)
            return one(h, in.buft);
        // The head's transcript-independent part: h, and per frame the log-sum-exp and maximum of
        // the logits of every class. The class projection runs in chunks of kClassChunk classes, so
        // only one chunk's logits are live at a time.
        const int64_t N = h->ne[1] * h->ne[2];
        auto h2 = ggml_reshape_2d(ctx, h, h->ne[0], N);
        constexpr int64_t kClassChunk = 4096;
        // Row maxima (n, N) -> (1, N): argmax picks each row's column (a block per row) and
        // get_rows reads it (a 1 x n max pool runs one thread per row: most of the head's time).
        const auto row_max = [ctx](ggml_tensor* x) {
            const int64_t n = x->ne[0], rows = x->ne[1];
            auto column = ggml_reshape_2d(ctx, ggml_argmax(ctx, x), 1, rows);
            return ggml_reshape_2d(
                ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, x, 1, n, rows), column), 1, rows);
        };
        ggml_tensor *peaks = nullptr, *sums = nullptr;  // (chunks, N)
        for (int64_t c0 = 0; c0 < vocab; c0 += kClassChunk) {
            const int64_t n = std::min<int64_t>(kClassChunk, vocab - c0);
            auto logits = ggml_mul_mat(
                ctx, ggml_view_2d(ctx, w, w->ne[0], n, w->nb[1], c0 * w->nb[1]), h2);  // (n, N)
            ggml_prec_set_acc(logits, GGML_PREC_F32);
            logits = ggml_add(ctx, logits, ggml_view_1d(ctx, b, n, c0 * b->nb[0]));
            auto peak = row_max(logits);
            auto sum = ggml_sum_rows(ctx, ggml_exp(ctx, ggml_sub(ctx, logits, peak)));
            peaks = peaks ? ggml_concat(ctx, peaks, peak, 0) : peak;
            sums = sums ? ggml_concat(ctx, sums, sum, 0) : sum;
        }
        auto peak = row_max(peaks);
        auto lse = ggml_add(
            ctx,
            ggml_log(
                ctx,
                ggml_sum_rows(ctx, ggml_mul(ctx, sums, ggml_exp(ctx, ggml_sub(ctx, peaks, peak))))),
            peak);
        TensorBag out;
        out.add_tensor({h, in.buft});
        out.add_tensor({lse, in.buft});
        out.add_tensor({peak, in.buft});
        return out;
    }
};

float
bf16_round(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof bits);
    if ((bits & 0x7f800000u) == 0x7f800000u)
        return v;  // inf / nan unchanged
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&v, &bits, sizeof v);
    return v;
}

// Grow-only device storage for evidence rows (dimension() floats each). Chunks are allocated as
// needed and kept, never moved, while the model or any evidence lives, so the steady state neither
// allocates nor frees device memory; a request's rows are one contiguous range of one chunk.
struct RowPool {
    struct Chunk {
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor* rows = nullptr;      // (dim, capacity) f32
        std::map<int64_t, int64_t> free;  // start -> length
        DeviceTensor device() const { return {rows, ggml_backend_buffer_get_type(buffer), 0}; }
    };
    ggml_backend_buffer_type_t buft;
    int dim;
    std::mutex mutex;
    std::vector<std::unique_ptr<Chunk>> chunks;
    RowPool(ggml_backend_buffer_type_t b, int d) : buft(b), dim(d) {}
    ~RowPool() {
        for (auto& c : chunks) {
            ggml_backend_buffer_free(c->buffer);
            ggml_free(c->ctx);
        }
    }
    // First fit; a new chunk (at least twice the largest so far) when nothing fits.
    std::pair<Chunk*, int64_t> take(int64_t n) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& c : chunks)
            for (auto it = c->free.begin(); it != c->free.end(); ++it)
                if (it->second >= n) {
                    const int64_t start = it->first, left = it->second - n;
                    c->free.erase(it);
                    if (left)
                        c->free[start + n] = left;
                    return {c.get(), start};
                }
        constexpr int64_t kMinRows = 16384;
        int64_t capacity = std::max(n, kMinRows);
        for (auto& c : chunks) capacity = std::max(capacity, 2 * c->rows->ne[1]);
        auto c = std::make_unique<Chunk>();
        ggml_init_params params{ggml_tensor_overhead(), nullptr, true};
        c->ctx = ggml_init(params);
        c->rows = ggml_new_tensor_2d(c->ctx, GGML_TYPE_F32, dim, capacity);
        c->buffer = ggml_backend_alloc_ctx_tensors_from_buft(c->ctx, buft);
        if (!c->buffer) {
            ggml_free(c->ctx);
            throw std::runtime_error(
                "CTC evidence: cannot allocate " + std::to_string(capacity) + " rows");
        }
        if (capacity > n)
            c->free[n] = capacity - n;
        int64_t used = n;
        for (const auto& k : chunks) {
            used += k->rows->ne[1];
            for (const auto& [start, length] : k->free) used -= length;
        }
        std::fprintf(
            stderr,
            "[ctc] word-timestamp evidence: +%lld frames on the device (%.0f MiB), %lld in use\n",
            static_cast<long long>(capacity), ggml_backend_buffer_get_size(c->buffer) / 1048576.0,
            static_cast<long long>(used));
        chunks.push_back(std::move(c));
        return {chunks.back().get(), 0};
    }
    void give(Chunk* c, int64_t start, int64_t n) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = c->free.emplace(start, n).first;
        if (auto next = std::next(it); next != c->free.end() && start + n == next->first) {
            it->second += next->second;
            c->free.erase(next);
        }
        if (it != c->free.begin()) {
            auto prev = std::prev(it);
            if (prev->first + prev->second == it->first) {
                prev->second += it->second;
                c->free.erase(it);
            }
        }
    }
};
}  // namespace

struct CtcEvidence::Rows {
    std::shared_ptr<RowPool> pool;
    RowPool::Chunk* chunk;
    int64_t start, n;
    Rows(std::shared_ptr<RowPool> p, int64_t count) : pool(std::move(p)), n(count) {
        std::tie(chunk, start) = pool->take(n);
    }
    ~Rows() { pool->give(chunk, start, n); }
    size_t offset(int64_t row) const { return size_t(start + row) * pool->dim * sizeof(float); }
};
CtcEvidence::~CtcEvidence() = default;
int
TimestampEvidence::ctc_frames() const {
    return head ? head->frames() : 0;
}

namespace {
// `bytes` of `src` from byte `src_offset` to `dst` from byte `dst_offset`, queued on `backend`.
void
copy_bytes(
    ggml_backend_t backend, ggml_tensor* src, size_t src_offset, ggml_tensor* dst,
    size_t dst_offset, size_t bytes) {
    ggml_init_params params{2 * ggml_tensor_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(params);
    auto view = [&](ggml_tensor* t, size_t offset) {
        auto v = ggml_view_1d(ctx, t, int64_t(bytes / sizeof(float)), offset);
        ggml_backend_view_init(v);
        return v;
    };
    ggml_backend_tensor_copy_async(backend, backend, view(src, src_offset), view(dst, dst_offset));
    ggml_free(ctx);
}

int
pow2_at_least(int64_t n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

int
pow2_at_most(int64_t n) {
    int p = 1;
    while (2 * int64_t(p) <= n) p *= 2;
    return p;
}

ggml_backend_buffer_type_t
main_buft(const buft_list_t& list) {
    for (const auto& b : list)
        if (b.second == ggml_backend_dev_buffer_type(b.first))
            return b.second;
    return list.back().second;
}
}  // namespace

struct CtcTimestampModel::Impl {
    // Its own execution lane (backend and compute lock): the head runs beside the encoder instead
    // of queueing behind its batches.
    BackendManager lane;
    GGUFLoader loader;
    CtcGraph graph;
    Session session;
    sentencepiece::SentencePieceProcessor tokenizer;
    std::mutex mutex;  // one session: serialize its runs
    std::shared_ptr<RowPool> pool;
    // Background head runs (finish()), one at a time in order; the queue owns a job's evidence
    // until its run is done.
    struct HeadJob {
        std::vector<std::shared_ptr<CtcEvidence>> evidence;
        TimestampOptions options;
        std::promise<void> done;
    };
    std::mutex jobs_mutex;
    std::condition_variable jobs_cv;
    std::deque<HeadJob> jobs;
    bool stopping = false;
    std::thread worker;
    Impl(BackendManager& bm, const std::string& path)
        : lane(bm.get_params()), loader(path), graph(loader), session(lane, &graph, &loader) {
        auto status =
            tokenizer.LoadFromSerializedProto(unbase64(loader.get_str("ctc.tokenizer.spm_model")));
        if (!status.ok() || tokenizer.GetPieceSize() != graph.vocab - 1)
            throw std::runtime_error("CTC tokenizer does not match classifier");
        session.setup();
        // Head and projection shapes are bucketed (frames, batch and classes in powers of two), and
        // runs are serialized and download their host outputs before returning: one shared arena.
        session.set_run_cache_capacity(64);
        session.set_shared_named_arena(true);
        pool = std::make_shared<RowPool>(main_buft(lane.get_buft_list()), graph.cfg.d_model);
        worker = std::thread([this] { work(); });
    }
    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(jobs_mutex);
            stopping = true;
        }
        jobs_cv.notify_all();
        worker.join();
    }
    void work() {
        for (;;) {
            HeadJob job;
            {
                std::unique_lock<std::mutex> lock(jobs_mutex);
                jobs_cv.wait(lock, [this] { return stopping || !jobs.empty(); });
                if (jobs.empty())
                    return;
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            ggml_nvtx::range nvtx_r("ctc.head");
            try {
                heads(job.evidence, job.options);
                job.done.set_value();
            }
            catch (...) {
                job.done.set_exception(std::current_exception());
            }
        }
    }
    ggml_backend_t backend() {
        auto b = lane.gpu_backend_handle();
        return b ? b : lane.get_backends().front();
    }
    std::vector<float> bridge(const float* host, int T, const std::vector<int>& lens) {
        std::lock_guard<std::mutex> lock(mutex);
        const int B = int(lens.size()), D = graph.cfg.d_model, U = graph.upsampling;
        if (T <= 0 || B <= 0)
            throw std::runtime_error("invalid CTC bridge batch");
        std::vector<int32_t> positions(size_t(U) * T);
        std::iota(positions.begin(), positions.end(), 0);
        std::vector<float> mask(size_t(U) * T * B, -1e9f);
        for (int b = 0; b < B; ++b) {
            if (lens[b] < 1 || lens[b] > T)
                throw std::runtime_error("invalid CTC bridge length");
            std::fill_n(mask.data() + size_t(b) * U * T, U * lens[b], 0.f);
        }
        std::vector<Session::Input> inputs{
            {"input.states", GGML_TYPE_F32, host, {D, T, B}},
            {"input.pos", GGML_TYPE_I32, positions.data(), {U * T}},
            {"input.mask", GGML_TYPE_F32, mask.data(), {U * T, 1, 1, B}}};
        std::vector<float> out(size_t(D) * U * T * B);
        std::vector<Session::Output> outputs(1);
        outputs[0].index = 0;
        outputs[0].host_buffer = out.data();
        outputs[0].nbytes = out.size() * sizeof(float);
        session.run(inputs, outputs);
        return out;
    }

    // The head over every window of `evidence`'s staged states, windows batched by padded length
    // (encoder frames to 64, batch to a power of two); then h rows (U per staged row) replace the
    // staged ones.
    void heads(
        const std::vector<std::shared_ptr<CtcEvidence>>& evidence,
        const TimestampOptions& options) {
        const int D = graph.cfg.d_model, U = graph.upsampling;
        struct Window {
            CtcEvidence* ev;
            timestamps::HeadWindow w;
            int t_pad;
        };
        std::vector<Window> windows;
        for (const auto& ev : evidence) {
            ev->h_ = std::make_unique<CtcEvidence::Rows>(pool, ev->frames());
            ev->lse_.assign(ev->frames(), 0.f);
            ev->max_.assign(ev->frames(), 0.f);
            for (const auto& w : timestamps::head_windows(
                     ev->encoder_frames_, options.head_window, options.head_left_context,
                     options.head_right_context))
                windows.push_back({ev.get(), w, (w.context_end - w.context_start + 63) / 64 * 64});
        }
        std::stable_sort(windows.begin(), windows.end(), [](const Window& a, const Window& b) {
            return std::make_pair(a.ev->staging_->chunk, a.t_pad) <
                   std::make_pair(b.ev->staging_->chunk, b.t_pad);
        });
        // At most kFramesPerCall padded CTC frames per call (one class chunk's logits and the
        // activations scale with them).
        constexpr int kFramesPerCall = 4096;
        std::lock_guard<std::mutex> lock(mutex);
        for (size_t first = 0; first < windows.size();) {
            const int t_pad = windows[first].t_pad, ctc_pad = U * t_pad;
            RowPool::Chunk* chunk = windows[first].ev->staging_->chunk;
            const int cap = pow2_at_most(std::max(1, kFramesPerCall / ctc_pad));
            size_t last = first;
            while (last < windows.size() && windows[last].t_pad == t_pad &&
                   windows[last].ev->staging_->chunk == chunk && int(last - first) < cap)
                ++last;
            const int B = pow2_at_least(int64_t(last - first));
            // Padded keys and empty batch items read a valid row and are masked.
            const auto& w0 = windows[first];
            std::vector<int32_t> gather(
                size_t(t_pad) * B, int32_t(w0.ev->staging_->start + w0.w.context_start));
            std::vector<float> mask(size_t(ctc_pad) * B, -1e9f);
            for (size_t i = first; i < last; ++i) {
                const auto& w = windows[i].w;
                const size_t slot = i - first;
                for (int f = 0; f < w.context_end - w.context_start; ++f)
                    gather[slot * t_pad + f] =
                        int32_t(windows[i].ev->staging_->start + w.context_start + f);
                std::fill_n(
                    mask.data() + slot * ctc_pad, U * (w.context_end - w.context_start), 0.f);
            }
            std::vector<int32_t> positions(ctc_pad);
            std::iota(positions.begin(), positions.end(), 0);
            const DeviceTensor rows = chunk->device();
            Session::Input staged{"input.rows", GGML_TYPE_F32, nullptr, {D, rows.tensor->ne[1]}};
            staged.upload = false;
            staged.device_tensor = &rows;
            std::vector<Session::Input> inputs{
                staged,
                {"input.gather", GGML_TYPE_I32, gather.data(), {t_pad, B}},
                {"input.pos", GGML_TYPE_I32, positions.data(), {ctc_pad}},
                {"input.mask", GGML_TYPE_F32, mask.data(), {ctc_pad, 1, 1, B}}};
            const size_t N = size_t(ctc_pad) * B;
            std::vector<float> lse(N), peak(N);
            DeviceTensor h;
            std::vector<Session::Output> outputs(3);
            outputs[0].index = 0;
            outputs[0].device_tensor = &h;
            outputs[1].index = 1;
            outputs[1].host_buffer = lse.data();
            outputs[1].nbytes = N * sizeof(float);
            outputs[2].index = 2;
            outputs[2].host_buffer = peak.data();
            outputs[2].nbytes = N * sizeof(float);
            session.run(inputs, outputs);
            if (outputs[0].out_shape[0] != D || outputs[0].out_shape[1] != ctc_pad ||
                outputs[0].out_shape[2] != B || outputs[1].out_shape[1] != int64_t(N))
                throw std::runtime_error("unexpected CTC head output shape");
            // Window cores to their requests (in CTC frames): h rows on the lane's stream (ordered
            // before the next run reuses the output), the normalizers on the host.
            for (size_t i = first; i < last; ++i) {
                const auto& w = windows[i].w;
                CtcEvidence& ev = *windows[i].ev;
                const size_t row = (i - first) * ctc_pad + size_t(U) * (w.start - w.context_start);
                const int start = U * w.start, n = U * (w.end - w.start);
                copy_bytes(
                    backend(), h.tensor, row * D * sizeof(float), ev.h_->chunk->rows,
                    ev.h_->offset(start), size_t(n) * D * sizeof(float));
                std::copy_n(lse.data() + row, n, ev.lse_.data() + start);
                std::copy_n(peak.data() + row, n, ev.max_.data() + start);
            }
            first = last;
        }
        ggml_backend_synchronize(backend());
        for (const auto& ev : evidence) ev->staging_.reset();
    }

    // Logits of each request's classes over its h rows, in tiles of up to kTileFrames frames
    // batched by (frames, classes) padded to powers of two; returns per request frames x
    // columns, minus the log-sum-exp.
    std::vector<std::vector<float>> project(
        const std::vector<const CtcEvidence*>& evidence,
        const std::vector<std::vector<int>>& columns, const TimestampOptions& options) {
        const int D = graph.cfg.d_model, blank = graph.vocab - 1;
        constexpr int kTileFrames = 4096, kFramesPerCall = 8192, kLogitsPerCall = 1 << 23;
        struct Tile {
            size_t req;
            int f0, n, f_pad, k_pad;
            RowPool::Chunk* chunk;
        };
        std::vector<Tile> tiles;
        std::vector<std::vector<float>> out(evidence.size());
        for (size_t r = 0; r < evidence.size(); ++r) {
            const CtcEvidence& ev = *evidence[r];
            if (columns[r].empty())
                throw std::runtime_error("invalid CTC head input");
            for (int id : columns[r])
                if (id < 0 || id >= graph.vocab)
                    throw std::runtime_error("invalid CTC class ID");
            out[r].resize(size_t(ev.frames()) * columns[r].size());
            const int k_pad = std::max(64, pow2_at_least(int64_t(columns[r].size())));
            for (int f0 = 0; f0 < ev.frames(); f0 += kTileFrames) {
                const int n = std::min(kTileFrames, ev.frames() - f0);
                tiles.push_back({r, f0, n, std::max(64, pow2_at_least(n)), k_pad, ev.h_->chunk});
            }
        }
        std::stable_sort(tiles.begin(), tiles.end(), [](const Tile& a, const Tile& b) {
            return std::make_tuple(a.chunk, a.f_pad, a.k_pad) <
                   std::make_tuple(b.chunk, b.f_pad, b.k_pad);
        });
        std::lock_guard<std::mutex> lock(mutex);
        for (size_t first = 0; first < tiles.size();) {
            const Tile& t0 = tiles[first];
            const int f_pad = t0.f_pad, k_pad = t0.k_pad;
            const int cap = pow2_at_most(std::max<int64_t>(
                1, std::min<int64_t>(
                       kFramesPerCall / f_pad, kLogitsPerCall / (int64_t(f_pad) * k_pad))));
            size_t last = first;
            while (last < tiles.size() && tiles[last].chunk == t0.chunk &&
                   tiles[last].f_pad == f_pad && tiles[last].k_pad == k_pad &&
                   int(last - first) < cap)
                ++last;
            const int B = pow2_at_least(int64_t(last - first));
            // Padded frames and empty batch items read a valid row and the blank; dropped below.
            std::vector<int32_t> gather(
                size_t(f_pad) * B, int32_t(evidence[t0.req]->h_->start + t0.f0));
            std::vector<int32_t> classes(size_t(k_pad) * B, blank);
            for (size_t i = first; i < last; ++i) {
                const Tile& t = tiles[i];
                const size_t slot = i - first;
                for (int f = 0; f < t.n; ++f)
                    gather[slot * f_pad + f] = int32_t(evidence[t.req]->h_->start + t.f0 + f);
                std::copy(
                    columns[t.req].begin(), columns[t.req].end(), classes.begin() + slot * k_pad);
            }
            const DeviceTensor rows = t0.chunk->device();
            Session::Input h{"input.rows", GGML_TYPE_F32, nullptr, {D, rows.tensor->ne[1]}};
            h.upload = false;
            h.device_tensor = &rows;
            std::vector<Session::Input> inputs{
                h,
                {"input.gather", GGML_TYPE_I32, gather.data(), {f_pad, B}},
                {"input.classes", GGML_TYPE_I32, classes.data(), {k_pad, B}}};
            std::vector<float> logits(size_t(k_pad) * f_pad * B);
            std::vector<Session::Output> outputs(1);
            outputs[0].index = 0;
            outputs[0].host_buffer = logits.data();
            outputs[0].nbytes = logits.size() * sizeof(float);
            session.run(inputs, outputs);
            if (outputs[0].out_shape[0] != k_pad || outputs[0].out_shape[1] != f_pad ||
                outputs[0].out_shape[2] != B)
                throw std::runtime_error("unexpected CTC projection output shape");
            for (size_t i = first; i < last; ++i) {
                const Tile& t = tiles[i];
                const CtcEvidence& ev = *evidence[t.req];
                const size_t K = columns[t.req].size();
                for (int f = 0; f < t.n; ++f) {
                    const float* src = logits.data() + ((i - first) * f_pad + f) * k_pad;
                    const float lse = ev.lse_[t.f0 + f];
                    float* dst = out[t.req].data() + size_t(t.f0 + f) * K;
                    for (size_t k = 0; k < K; ++k) {
                        const float v = src[k] - lse;
                        dst[k] = options.bf16_emissions ? bf16_round(v) : v;
                    }
                }
            }
            first = last;
        }
        return out;
    }
};
CtcTimestampModel::CtcTimestampModel(BackendManager& bm, const std::string& path)
    : impl_(new Impl(bm, path)) {}
CtcTimestampModel::~CtcTimestampModel() = default;
int
CtcTimestampModel::dimension() const {
    return impl_->graph.cfg.d_model;
}
int
CtcTimestampModel::blank_id() const {
    return impl_->graph.vocab - 1;
}
int
CtcTimestampModel::upsampling() const {
    return impl_->graph.upsampling;
}
std::string
CtcTimestampModel::model_name() const {
    if (!impl_->graph.name.empty())
        return impl_->graph.name;
    return "Nemotron3-CTC-Timestamp-Adapter_" + std::to_string((impl_->graph.vocab - 1) / 1024) +
           "k";
}

std::shared_ptr<CtcEvidence>
CtcTimestampModel::reserve(int encoder_frames) {
    if (encoder_frames < 1)
        throw std::runtime_error("CTC evidence needs at least one frame");
    std::shared_ptr<CtcEvidence> ev(new CtcEvidence());
    ev->encoder_frames_ = encoder_frames;
    ev->upsampling_ = upsampling();
    ev->staging_ = std::make_unique<CtcEvidence::Rows>(impl_->pool, encoder_frames);
    return ev;
}

void
CtcTimestampModel::stage(
    ggml_backend_t backend, const DeviceTensor& src, size_t src_row, CtcEvidence& evidence,
    int dst_row, int rows) {
    const size_t row_bytes = size_t(dimension()) * sizeof(float);
    if (!evidence.staging_ || dst_row < 0 || rows < 0 ||
        dst_row + rows > evidence.encoder_frames_ || !src.valid() ||
        src.tensor->type != GGML_TYPE_F32 || src.tensor->ne[0] != dimension() ||
        src.byte_offset + (src_row + rows) * row_bytes > ggml_nbytes(src.tensor))
        throw std::runtime_error("invalid CTC evidence rows");
    if (rows)
        copy_bytes(
            backend, src.tensor, src.byte_offset + src_row * row_bytes,
            evidence.staging_->chunk->rows, evidence.staging_->offset(dst_row), rows * row_bytes);
}

void
CtcTimestampModel::finish(
    const std::vector<std::shared_ptr<CtcEvidence>>& evidence, const TimestampOptions& options) {
    if (evidence.empty())
        return;
    for (const auto& ev : evidence)
        if (!ev || !ev->staging_ || ev->ready_.valid())
            throw std::runtime_error("CTC evidence finished twice");
    Impl::HeadJob job{evidence, options, {}};
    const auto ready = job.done.get_future().share();
    for (const auto& ev : evidence) ev->ready_ = ready;
    {
        std::lock_guard<std::mutex> lock(impl_->jobs_mutex);
        impl_->jobs.push_back(std::move(job));
    }
    impl_->jobs_cv.notify_one();
}

std::vector<std::vector<float>>
CtcTimestampModel::emissions(
    const std::vector<const CtcEvidence*>& evidence, const std::vector<std::vector<int>>& columns,
    const TimestampOptions& options) {
    if (columns.size() != evidence.size())
        throw std::invalid_argument("emissions: one column list per evidence");
    for (const CtcEvidence* ev : evidence) {
        if (!ev)
            throw std::runtime_error("missing CTC evidence");
        ev->wait();
    }
    return impl_->project(evidence, columns, options);
}

std::vector<float>
CtcTimestampModel::head_log_probs(
    const float* states, int frames, const std::vector<int>& columns,
    const TimestampOptions& options) {
    return std::move(head_log_probs({{states, frames, columns}}, options).front());
}

std::vector<std::vector<float>>
CtcTimestampModel::head_log_probs(
    const std::vector<HeadRequest>& requests, const TimestampOptions& options) {
    std::vector<std::shared_ptr<CtcEvidence>> evidence;
    std::vector<const CtcEvidence*> views;
    std::vector<std::vector<int>> columns;
    for (const auto& r : requests) {
        if (!r.states || r.frames < 1 || r.columns.empty())
            throw std::runtime_error("invalid CTC head input");
        auto ev = reserve(r.frames);
        ggml_backend_tensor_set(
            ev->staging_->chunk->rows, r.states, ev->staging_->offset(0),
            size_t(r.frames) * dimension() * sizeof(float));
        evidence.push_back(ev);
        views.push_back(ev.get());
        columns.push_back(r.columns);
    }
    finish(evidence, options);
    return emissions(views, columns, options);
}

timestamps::AlignmentTargets
CtcTimestampModel::targets(const std::string& transcript, int ctc_frames) const {
    return timestamps::alignment_targets(impl_->tokenizer, transcript, blank_id(), ctc_frames);
}

std::string
CtcTimestampModel::word_timestamps(
    const TimestampEvidence& evidence, const std::string& transcript, asr::SpeechLmDetails& details,
    const TimestampOptions& options) {
    std::vector<asr::SpeechLmDetails*> out{&details};
    return word_timestamps({evidence}, {transcript}, out, options).front();
}

std::vector<std::string>
CtcTimestampModel::word_timestamps(
    const std::vector<TimestampEvidence>& evidence, const std::vector<std::string>& transcripts,
    const std::vector<asr::SpeechLmDetails*>& details, const TimestampOptions& options) {
    namespace ts = ::nemo_speech::speechlm::timestamps;
    const size_t n = evidence.size();
    if (transcripts.size() != n || details.size() != n)
        throw std::invalid_argument("word_timestamps: one transcript and result per request");
    std::vector<std::string> errors(n);
    constexpr double kLabelSeconds = 0.01;  // NeMo's diarization_frame_seconds
    // The 10 ms diarization segments of a request (NeMo's diarization_labels: activity > 0.5),
    // returned with every result that has the encoder outputs, aligned or not.
    const auto set_diarization = [&](size_t i, const std::vector<ts::Segment>* segments) {
        const auto& ev = evidence[i];
        auto& d = *details[i];
        const bool labels = ev.activity10 && ev.speakers > 0 && ev.activity10_frames > 0;
        const auto own = labels && !segments ? ts::diarization_segments(
                                                   ev.activity10, ev.activity10_frames, ev.speakers,
                                                   kLabelSeconds, ev.duration_seconds)
                                             : std::vector<ts::Segment>{};
        d.diarization.clear();
        for (const auto& s : segments ? *segments : own)
            d.diarization.push_back(
                {s.speaker, ts::round_decimals(s.start, 3), ts::round_decimals(s.end, 3)});
        d.diarization_frame_seconds = labels ? kLabelSeconds : 0.0;
    };
    const auto unavailable = [&](size_t i, const char* reason, std::string error,
                                 bool diarization) {
        auto& d = *details[i];
        d.alignment_status = "unavailable";
        d.alignment_model.clear();
        d.failure_reason = reason;
        d.frame_stride_seconds = 0.0;
        d.words.clear();
        d.unaligned_words = 0;
        d.speaker_words.clear();
        d.diarization.clear();
        d.diarization_frame_seconds = 0.0;
        if (diarization)
            set_diarization(i, nullptr);
        errors[i] = std::move(error);
    };
    // Targets first (NeMo's _prepare), then one projection for every request with words, then
    // each alignment.
    struct Prepared {
        bool ok = false;
        ts::AlignmentTargets targets;
        size_t head = 0;  // index in `heads` (requests with speaker streams)
    };
    std::vector<Prepared> prepared(n);
    std::vector<const CtcEvidence*> heads;
    std::vector<std::vector<int>> head_columns;
    const int blank = blank_id();
    for (size_t i = 0; i < n; ++i) {
        const auto& ev = evidence[i];
        const auto transcript = ts::clean_sot_transcript(transcripts[i], options.special_tokens);
        if (transcript.empty()) {
            unavailable(i, "empty_transcript", "empty transcript", false);
            continue;
        }
        if (!ev.head) {
            unavailable(i, "ctc_evidence_unavailable", "no ASR states", false);
            continue;
        }
        auto& p = prepared[i];
        try {
            if (ev.encoder_frames < 1)
                throw std::runtime_error("no CTC frames to align");
            if (ev.head->encoder_frames() != ev.encoder_frames)
                throw std::runtime_error("CTC evidence frames do not match");
            ev.head->wait();
            if (!(ev.duration_seconds > 0.0) || !std::isfinite(ev.duration_seconds))
                throw std::runtime_error("invalid audio duration");
            p.targets = ts::alignment_targets(impl_->tokenizer, transcript, blank, ev.ctc_frames());
            if (!p.targets.streams.empty()) {
                p.head = heads.size();
                heads.push_back(ev.head.get());
                head_columns.push_back(p.targets.columns);
            }
            p.ok = true;
        }
        catch (const std::exception& e) {
            unavailable(i, "ctc_alignment_failed", e.what(), true);
        }
    }
    std::vector<std::vector<float>> emissions;
    if (!heads.empty()) {
        try {
            emissions = this->emissions(heads, head_columns, options);
        }
        catch (const std::exception& e) {
            for (size_t i = 0; i < n; ++i)
                if (prepared[i].ok && !prepared[i].targets.streams.empty()) {
                    prepared[i].ok = false;
                    unavailable(i, "ctc_alignment_failed", e.what(), true);
                }
        }
    }
    ts::AlignmentConfig config;
    config.max_word_frames = 12 * upsampling();  // NeMo: 12 encoder frames in CTC frames
    for (size_t i = 0; i < n; ++i) {
        auto& p = prepared[i];
        if (!p.ok)
            continue;
        const auto& ev = evidence[i];
        auto& d = *details[i];
        try {
            const int frames = ev.ctc_frames();
            // The speaker probabilities on the encoder grid, as NeMo's encoder aligns them
            // (_align_diar_frames: the last row repeated, or cut); the alignment resamples them to
            // the CTC frames.
            std::vector<float> speaker;
            if (ev.activity && ev.speakers > 0 && ev.activity_frames > 0) {
                const size_t row = static_cast<size_t>(ev.speakers);
                speaker.resize(static_cast<size_t>(ev.encoder_frames) * row);
                for (int t = 0; t < ev.encoder_frames; ++t)
                    std::copy_n(
                        ev.activity + std::min(t, ev.activity_frames - 1) * row, row,
                        speaker.begin() + t * row);
            }
            const std::vector<float> blank_only(p.targets.streams.empty() ? frames : 0, 0.0f);
            ts::AlignmentInputs inputs;
            inputs.log_probs =
                p.targets.streams.empty() ? blank_only.data() : emissions[p.head].data();
            inputs.frames = frames;
            inputs.speaker_probs = speaker.empty() ? nullptr : speaker.data();
            inputs.speaker_frames = ev.encoder_frames;
            inputs.speakers = ev.speakers;
            inputs.activity10 = ev.activity10;
            inputs.label_frames = ev.activity10 ? ev.activity10_frames : 0;
            inputs.label_speakers = ev.speakers;
            inputs.label_seconds = kLabelSeconds;
            inputs.duration = ev.duration_seconds;
            const auto result = ts::align_record(p.targets, inputs, config);
            d.alignment_status = "aligned";
            d.alignment_model = model_name();
            d.failure_reason.clear();
            d.frame_stride_seconds = result.ctc_step;
            d.words.clear();
            d.unaligned_words = 0;
            d.speaker_words.clear();
            for (const auto& row : ts::public_words(result.words)) {
                const auto& unit = p.targets.words[row.word];
                asr::SpeechLmWord out;
                out.word = unit.word;
                out.display = unit.display;
                out.joins_previous = unit.joins_previous;
                out.word_index = unit.index;
                out.start = row.start;
                out.end = row.end;
                out.speaker = row.tag;
                out.speaker_slot = row.column;
                out.speaker_confidence = row.speaker_confidence;
                d.speaker_words[ts::speaker_label(row.tag)].push_back(out);
                d.words.push_back(std::move(out));
            }
            for (const auto& failure : result.failed_streams) {
                d.unaligned_words += static_cast<int>(failure.words.size());
                d.warnings.push_back(
                    "word timestamps: " + std::to_string(failure.words.size()) + " words of " +
                    ts::speaker_label(failure.tag) + " not aligned (" + failure.reason + ")");
            }
            set_diarization(i, &result.diarization);
        }
        catch (const std::exception& e) {
            unavailable(i, "ctc_alignment_failed", e.what(), true);
        }
    }
    return errors;
}
std::vector<float>
CtcTimestampModel::bridge(const float* x, int t, const std::vector<int>& lengths) {
    return impl_->bridge(x, t, lengths);
}
std::vector<float>
CtcTimestampModel::project(const std::vector<float>& states, int T, const std::vector<int>& ids) {
    if (T <= 0 || states.size() != size_t(T) * dimension() || ids.empty())
        throw std::runtime_error("invalid CTC projection shape");
    for (int id : ids)
        if (id < 0 || id >= impl_->graph.vocab)
            throw std::runtime_error("invalid CTC class ID");
    std::vector<float> out(size_t(T) * ids.size());
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<Session::Input> inputs{
        {"input.states", GGML_TYPE_F32, states.data(), {dimension(), T}},
        {"input.ids", GGML_TYPE_I32, ids.data(), {int64_t(ids.size())}}};
    std::vector<Session::Output> outputs(1);
    outputs[0].index = 0;
    outputs[0].host_buffer = out.data();
    outputs[0].nbytes = out.size() * sizeof(float);
    impl_->session.run(inputs, outputs);
    return out;
}

}  // namespace nemo_speech::speechlm
