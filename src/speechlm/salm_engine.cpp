// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "salm_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <regex>
#include <stdexcept>
#include <string_view>

#include "audio_resampler.h"
#include "common.h"
#include "features/fe.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"
#include "nvtx_utils.h"
#include "pee/pee_model.h"
#include "pee/timestamp_alignment.h"
#include "runtime.h"
#include "salm_guards.h"
#include "salm_policy.h"
#include "speculative.h"
#include "types.h"

// The batched serving path needs the patched llama.cpp/ggml (patches/) and a CUDA GPU; other builds
// use only upstream llama.cpp behavior.
#if defined(NEMO_SPEECH_GGML_PATCHED) && defined(GGML_USE_CUDA)
#define SALM_FAST_PATH 1
#include "ggml-cuda.h"
#else
#define SALM_FAST_PATH 0
#endif

namespace nemo_speech::speechlm {

#if !defined(_WIN32)
// The backtrace ggml prints on abort, declared in its internal ggml-impl.h. Shared libraries export
// it on Linux and macOS; a Windows ggml.dll exports only GGML_API symbols, and ggml prints no
// backtrace on Windows anyway.
extern "C" void ggml_print_backtrace(void);
#endif

namespace {

double
audio_seconds_of(const SalmRequest& r) {
    return r.seconds > 0 ? r.seconds : r.count / static_cast<double>(r.sample_rate);
}

// The result of a request dropped by SalmEngine::cancel (generated_tokens keeps the work done).
void
cancel_result(SalmResult& r) {
    r.cancelled = true;
    r.error = "request cancelled";
    r.text.clear();
    r.details.reset();
    r.timestamp_error.clear();
}

// Unified-memory GPUs (such as DGX Spark) register as integrated; count them like the backend
// manager does so asr.backend.gpu indexes the same device.
bool
is_gpu_device(ggml_backend_dev_t dev) {
    const auto type = ggml_backend_dev_type(dev);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

const char* kTurnEnd = "<|im_end|>";

// ggml reports a CUDA error, then aborts. When the error is "out of memory" (the CUDA memory pool
// grows while a graph runs and aborts when it cannot), the abort becomes a clear error and exit
// status 1 instead of a crash.
std::atomic<bool> g_cuda_out_of_memory{false};
ggml_abort_callback_t g_previous_abort = nullptr;

void
llama_log_cb(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_ERROR && text && std::strstr(text, "CUDA error: out of memory"))
        g_cuda_out_of_memory.store(true);
    if (level >= GGML_LOG_LEVEL_WARN)
        std::fputs(text, stderr);
}

void
abort_cb(const char* message) {
    if (g_cuda_out_of_memory.load() && std::strstr(message, "CUDA error")) {
        std::fprintf(
            stderr,
            "%s\nSALM: the GPU ran out of memory while running the model. The decode slots "
            "(asr.salm.max_sequences x asr.salm.n_ctx_per_sequence) leave too little memory for "
            "the compute buffers; lower either.\n",
            message);
        std::fflush(stderr);
        std::_Exit(EXIT_FAILURE);
    }
    if (g_previous_abort) {
        g_previous_abort(message);
        return;
    }
    // ggml's own behaviour without a callback
    std::fprintf(stderr, "%s\n", message);
#if !defined(_WIN32)
    ggml_print_backtrace();
#endif
}

// No BOS: the chat template starts the prompt.
std::vector<llama_token>
tokenize(const llama_vocab* vocab, const std::string& text) {
    if (text.empty())
        return {};
    int n = llama_tokenize(
        vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, false, true);
    if (n < 0)
        n = -n;
    std::vector<llama_token> out(n);
    const int r = llama_tokenize(
        vocab, text.c_str(), static_cast<int32_t>(text.size()), out.data(), n, false, true);
    if (r < 0)
        throw std::runtime_error("llama_tokenize failed");
    out.resize(r);
    return out;
}

std::string
piece(const llama_vocab* vocab, llama_token t) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    if (n >= 0)
        return std::string(buf, n);
    std::string s(-n, '\0');
    llama_token_to_piece(vocab, t, s.data(), -n, 0, true);
    return s;
}

int
argmax(const float* logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > bv) {
            bv = logits[i];
            best = i;
        }
    return best;
}

// Scheduling constants of the batched path.
// Distinct prompt prefixes kept computed in extra sequences; admitted requests copy their prefix.
constexpr int kPromptCacheSequences = 16;
// Frames per sequence per prefill pass (round-robin over the sequences of a pass).
constexpr int kPrefillChunk = 2048;
// Free slots are refilled in groups of max_sequences / kAdmissionGroups: each refill runs the
// encoder and the prefill of the admitted requests while the active sequences pause.
constexpr int kAdmissionGroups = 12;
// Upstream llama.cpp's micro-batch: its attention reads a dense mask of (KV pool x micro-batch)
// cells, so the pool's long requests need short micro-batches (its own default).
constexpr int kUpstreamUbatch = 512;
// Encoder windows are padded to a multiple of this many mel frames (fewer graph shapes).
constexpr int kEncoderBucketFrames = 128;
// Requests encoded per encoder call once some are queued (larger calls give the length buckets
// more same-length windows to batch).
constexpr size_t kEncoderAhead = 256;

}  // namespace

struct SalmEngine::Impl {
    SalmConfig cfg;
    llama_model_tensor_buft_override embedding_overrides[2] = {};
    ggml_backend_dev_t decoder_devices[2] = {};  // the decoder's GPU (null-terminated list)
    ggml_runtime::BackendManager bm;
    // Batched serving path (SALM_FAST_PATH builds on a CUDA GPU with a Nemotron-H MoE decoder):
    // ragged batches with pinned recurrent cells, masks and prefill embeddings on the device,
    // pipelined decode steps, MTP with replay rollback, two encoder jobs overlapping decoding.
    // Otherwise only upstream llama.cpp behavior is used.
    bool fast = false;
    bool overlap_encoder = false;  // fast path: the encoder runs on its own thread while decoding
#if SALM_FAST_PATH
    ggml_backend_cuda_serving cuda_encoder{}, cuda_decoder{};
#endif
    // Fast path: the mel frontend gets its own backend manager (stream + compute lock), so its
    // small graphs do not queue behind the other encoder job's PEE batches on the shared compute
    // lock.
    std::unique_ptr<ggml_runtime::BackendManager> fe_bm;
    PeeModel pee;
    std::unique_ptr<MelSpectrogramExtractor> fe;
    int hop = 0;

    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_context* ctx_dft = nullptr;  // draft context (MTP: same model; DFlash: model_dft)
    llama_model* model_dft = nullptr;  // DFlash2 drafter, null with MTP
    common_speculative* spec = nullptr;
    const llama_vocab* vocab = nullptr;
    int n_embd = 0, n_vocab = 0;
    uint32_t n_ctx_seq = 0;  // longest sequence (prompt + generated tokens)
    uint32_t kv_pool = 0;    // KV cells shared by all sequences
    // A user turn, tokenized: template + text before the audio locator, text after it + generation
    // prompt.
    struct Prompt {
        std::vector<llama_token> prefix, suffix;
        std::vector<llama_token> text_prompt;  // prefix + suffix, handed to the speculative driver
    };
    std::map<std::string, std::shared_ptr<Prompt>> prompts;  // by user turn
    std::shared_ptr<Prompt> default_prompt;
    // Fast path: prompt prefixes are computed once into cache sequences (ids max_sequences..),
    // which admitted sequences copy (shared KV cells; the recurrent state is read from the cache
    // sequence's cell) instead of recomputing them. An entry holds a whole prefix, or the leading
    // tokens that several prompts share (the rest of each prefix is computed with the request).
    struct CacheEntry {
        std::vector<llama_token> tokens;
        int seq = -1;
        std::vector<uint8_t> draft_state;  // drafter state after the tokens (copied to sequences)
        uint64_t used = 0;                 // admission round of the last use (LRU eviction)
    };
    bool prefix_cache = false;                             // fast path
    std::vector<std::shared_ptr<CacheEntry>> cache_owner;  // per prompt-cache sequence
    uint64_t admission_round = 0;
    size_t kv_cached = 0;    // cells of the computed prefixes
    size_t kv_reserved = 0;  // cells reserved by active sequences
    // Fast path: a sequence reserves its prompt, a first share of its output budget and the rows a
    // step may add (verify rows, plus the pipelined next step), and grows by kKvGrowCells as it
    // generates, up to its worst case (prompt + budget + verify rows). Every reservation covers the
    // cells its sequence holds (its whole prompt, prefix-cache cells included), so the pool never
    // runs out inside llama_decode. Admission also keeps one more share free for each sequence in
    // flight; when a sequence cannot grow, the most recently admitted one is preempted.
    static constexpr size_t kKvGrowCells = 128;
    uint64_t admissions = 0;  // admission counter: the most recently admitted sequence goes first
    llama_token im_end = LLAMA_TOKEN_NULL;  // <|im_end|>: the only token that ends generation
    bool script_mask = false;               // the script guard's token mask is installed
    std::vector<std::string> piece_memo;    // commit: token id -> piece
    std::vector<uint8_t> piece_state;       // 0 unknown, 1 text, 2 end of turn
    std::string piece_tmp;
    llama_batch tok_batch{}, emb_batch{};
    float* emb_batch_heap = nullptr;  // llama_batch_init's buffer, restored before llama_batch_free
    // upstream path on a GPU: pinned replacement, so prefill embeddings upload at DMA speed
    ggml_backend_buffer_t emb_pinned = nullptr;
    // fast path: prefill rows are staged (own stream) in one of two device tensors while the
    // previous pass still runs, and the graph reads them in place (llama_batch.embd_input_device).
    // The encoder output never leaves the device: its rows are copied from the encoder's blocks,
    // token rows from emb_zero.
    ggml_backend_t emb_copy_backend = nullptr;
    ggml_context* emb_dev_ctx = nullptr;
    ggml_backend_buffer_t emb_dev_buf = nullptr;
    ggml_tensor* emb_dev[2] = {nullptr, nullptr};
    ggml_tensor* emb_zero = nullptr;       // zero rows
    ggml_context* emb_view_ctx = nullptr;  // the two views of one staging copy
    int emb_dev_turn = 0;
    std::vector<llama_pos>
        dft_pos;  // tok_batch positions shifted per slot for the MTP draft context
    std::vector<llama_sampler*>
        seq_samplers;  // chain attached to each slot: greedy (GPU argmax) or temperature sampling
    std::vector<char> seq_greedy;  // seq_samplers[i] is a greedy chain
    std::vector<llama_sampler_seq_config> seq_sampler_cfg;

    // Word-timestamp evidence of one request: the CTC head's output for its frames (on the
    // device), speaker activity on the 80 ms grid (frames x n_spk) and the native 10 ms activity.
    struct TimestampInput {
        std::shared_ptr<CtcEvidence> head;
        std::vector<float> probs80, probs10;
        int frames = 0;
    };
    std::vector<std::pair<size_t, std::shared_ptr<TimestampInput>>> timestamp_ready;
    std::vector<std::shared_ptr<Prompt>> row_prompt;  // serve: prompt of each request row
    std::vector<int> row_budget;  // serve: generation budget of each request row (0 = default)
    std::vector<size_t>
        finished_reqs;  // request rows complete since the last flush (text, timestamps, errors)
    // One encoded utterance ready for the decoder.
    struct Encoded {
        size_t req = 0;
        std::shared_ptr<TimestampInput> timestamps;
        std::vector<float> proj;                     // (frames, n_embd); upstream path
        std::vector<PeeModel::ProjSpan> proj_spans;  // fast path: the frames on the device
        int frames = 0;
        double audio_seconds = 0;
        std::string error;
        bool invalid_request = false;  // error is the request's fault
        bool speech = true;            // the speaker branch found a voice (else: no decode)
    };

    // One in-flight sequence.
    struct Slot {
        bool active = false;
        int seq_id = 0;
        size_t req = 0;
        std::shared_ptr<Prompt> prompt;
        std::shared_ptr<CacheEntry> cached;  // fast path: the cached prefix it starts from, or null
        int budget = 0;                      // generated tokens allowed
        size_t kv_cells = 0;                 // KV cells reserved (fast path: grows up to kv_full)
        size_t kv_full = 0;                  // the cap (prompt + budget + verify rows)
        uint64_t admitted = 0;               // admission counter value
        std::vector<llama_token> tokens;  // fast path: committed output tokens (for a preemption)
        std::vector<llama_token> resume;  // output tokens prefilled after the prompt (readmission)
        llama_pos pos = 0;
        llama_token last = 0;  // next input token (also the newest output)
        int n_generated = 0;
        bool ended = false;  // finished at <|im_end|>
        std::string text;
        std::function<void(std::string_view)> on_text;  // the request's text stream, or empty
        size_t streamed = 0;                            // bytes of `text` handed to on_text
        std::shared_ptr<TimestampInput> timestamps;
        std::vector<llama_token> pending;  // tokens produced this step, to be committed
        std::vector<llama_token> draft;    // MTP draft tokens for the current step
        std::vector<int32_t> i_batch;      // batch indices of [last, draft...] in the verify batch
        llama_pos dft_offset = 0;  // target position of the first token the MTP draft context sees
        float temperature = 0;  // the request's sampling (greedy below policy::kGreedyTemperature)
        std::optional<uint64_t> seed;
        bool latin_script = false;  // the script guard applies
        guards::LoopGuard loop;
        // SalmRequest::fallback_prompt: the request's encoder output and the fallback turn, kept
        // until the response shows text (dropped then) or ends empty (decoded again from them).
        std::optional<Encoded> fallback;
        std::shared_ptr<Prompt> fallback_prompt;
        bool cancelled = false;  // SalmEngine::cancel: commit releases it without its tokens
    };
    std::vector<Encoded> fallback_ready;  // ended empty: decode again (serve queues them first)
    // SalmEngine::cancel: tickets posted by other threads; the serve loop checks `cancel_posted`
    // once per iteration (no lock unless something was posted).
    std::mutex cancel_mutex;
    std::vector<size_t> cancel_posted_tickets;
    std::atomic<bool> cancel_posted{false};
    size_t cancelled_total = 0;  // since the engine was created (logged)
    // Fast path: a sequence preempted for KV cells keeps what it had generated with its request row
    // until it is admitted again (encoded again, then prefilled from its prompt and these tokens).
    struct Preempted {
        std::vector<llama_token> tokens;  // committed output tokens, prefilled after the prompt
        std::string text;
        size_t streamed = 0;
        int n_generated = 0;
        guards::LoopGuard loop;
        llama_sampler* sampler = nullptr;  // temperature sampling: its chain and random state
        std::shared_ptr<TimestampInput> timestamps;
        bool fallback = false;  // the empty-response fallback is still armed
        size_t kv_share = 0;    // its first share, held back from admission until it is encoded
    };
    std::map<size_t, Preempted> preempted;  // by request row
    std::vector<size_t> reencode;           // preempted request rows for the encoder (serve)
    size_t reencoding = 0;                  // preempted requests in the encoder, not yet ready
    // preempted requests that still hold their encoder output (fallback armed): no encoder pass
    std::vector<Encoded> preempted_ready;
    size_t kv_resuming = 0;  // first shares of the requests in `reencode` / the encoder (admission)
    size_t preemptions = 0;
    // fast path: while decode step N runs on the GPU, step N+1 is prepared (llama_decode_prepare)
    // for the same live lanes; its token ids are set at launch. Lanes that finish at N ride along
    // as zombies (outputs ignored; their freed cells are rewritten harmlessly). Only prepared when
    // no refill is close (prepare_ok, set by serve).
    std::vector<Slot*> pend_lanes;
    std::vector<llama_token> pend_tok;
    bool pend_ready = false;
    bool prepare_ok = false;

    explicit Impl(SalmConfig c)
        : cfg(std::move(c)), bm(make_params(cfg)), pee(bm, cfg.encoder_path, false, cfg.ctc_path) {
#if SALM_FAST_PATH
        // the serving kernels need an NVIDIA GPU of compute capability 8.0 or newer
        fast = cfg.gpu >= 0 && bm.gpu_backend_handle() != nullptr &&
               ggml_backend_is_cuda(bm.gpu_backend_handle()) &&
               ggml_backend_cuda_serving_supported(bm.gpu_backend_handle());
#endif
        // MTP rolls drafts back through the batched path's replay rows; on upstream llama.cpp the
        // engine compacts slot ids, which cannot carry the drafter's state.
        if (!fast)
            cfg.mtp_draft = 0;
        // The encoder overlaps decoding on its own thread and stream (fast path). For word
        // timestamps it keeps the CTC head's output of those requests only, on the device; the
        // alignment runs after decoding.
        overlap_encoder = fast;
#if SALM_FAST_PATH
        if (fast) {
            cuda_encoder.fused = true;
            ggml_backend_cuda_set_serving(bm.gpu_backend_handle(), &cuda_encoder);
            // The encoder stream runs at a higher priority than the decoder (-5, clamped to the
            // device range): its small kernels go ahead of the decoder ones, so admission waits
            // less for encoded requests.
            ggml_backend_cuda_set_stream_priority(bm.gpu_backend_handle(), -5);
        }
#endif
        pee.set_bucket_frames(kEncoderBucketFrames);
        pee.set_max_batch(cfg.encoder_max_batch);
        // The decoder only reads the projection (and speaker probs for timestamps): skip the
        // fused-state download (d_model floats per frame) and its stitching.
        pee.set_emit_fused(false);
        const auto& pc = pee.cfg();
        if (pc.proj_dim <= 0)
            throw std::runtime_error("encoder GGUF has no fusion.proj (pee.fusion.proj_dim)");

        MelSpecConfig fe_cfg;
        fe_cfg.sample_rate = pc.sample_rate;
        fe_cfg.n_fft = pc.n_fft;
        fe_cfg.n_mels = pc.n_mels;
        fe_cfg.window_size = pc.window_size;
        fe_cfg.window_stride = pc.window_stride;
        fe_cfg.normalize_per_feature =
            false;  // raw log-mel; the ASR expert normalizes per utterance
        if (fast) {
            fe_bm = std::make_unique<ggml_runtime::BackendManager>(make_params(cfg));
#if SALM_FAST_PATH
            ggml_backend_cuda_set_serving(fe_bm->gpu_backend_handle(), &cuda_encoder);
#endif
        }
        fe = std::make_unique<MelSpectrogramExtractor>(
            fe_cfg, cfg.gpu >= 0 ? (fe_bm ? fe_bm.get() : &bm) : nullptr);
        fe->set_mel_basis(pee.mel_basis().data(), pc.n_mels, pc.n_fft / 2 + 1);
        hop = fe->hop_length();

        llama_log_set(llama_log_cb, nullptr);
        static std::once_flag abort_once;
        std::call_once(abort_once, [] { g_previous_abort = ggml_set_abort_callback(abort_cb); });
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = cfg.gpu >= 0 ? 999 : 0;
        mp.load_mtp = cfg.mtp_draft > 0 && cfg.dflash_path.empty();
        if (cfg.gpu >= 0) {
            mp.load_mode = LLAMA_LOAD_MODE_NONE;
            // The requested GPU alone: by default llama.cpp splits the layers over every GPU.
            for (size_t i = 0, gpu_index = 0; i < ggml_backend_dev_count(); ++i) {
                auto dev = ggml_backend_dev_get(i);
                if (is_gpu_device(dev) && static_cast<int>(gpu_index++) == cfg.gpu) {
                    decoder_devices[0] = dev;
                    break;
                }
            }
            if (!decoder_devices[0])
                throw std::runtime_error("GPU " + std::to_string(cfg.gpu) + " not found");
            mp.devices = decoder_devices;
            mp.split_mode = LLAMA_SPLIT_MODE_NONE;
            mp.main_gpu = 0;
        }
        if (fast) {  // token embeddings on the GPU (mixed text/audio rows are looked up there)
            if (!decoder_devices[0])
                throw std::runtime_error("embedding GPU device not found");
            embedding_overrides[0] = {
                "token_embd\\.weight", ggml_backend_dev_buffer_type(decoder_devices[0])};
            mp.tensor_buft_overrides = embedding_overrides;
        }
        model = llama_model_load_from_file(cfg.llm_path.c_str(), mp);
        if (!model)
            throw std::runtime_error("failed to load LLM " + cfg.llm_path);
        vocab = llama_model_get_vocab(model);
        if (const auto t = tokenize(vocab, kTurnEnd); t.size() == 1)
            im_end = t[0];
        else
            throw std::runtime_error(
                std::string("the LLM vocabulary has no single ") + kTurnEnd + " token");
        n_embd = llama_model_n_embd(model);
        n_vocab = llama_vocab_n_tokens(vocab);
        if (pc.proj_dim != n_embd)
            throw std::runtime_error(
                "encoder projection dim " + std::to_string(pc.proj_dim) +
                " does not match LLM n_embd " + std::to_string(n_embd));
        if (cfg.max_sequences < 1)
            throw std::runtime_error("max_sequences must be >= 1");

        char arch[64] = {};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        if (fast && std::string(arch) != "nemotron_h_moe") {
            fast = false;  // the batched serving path is Nemotron-H MoE only
            cfg.mtp_draft = 0;
        }

        if (cfg.mtp_draft > 0 && cfg.dflash_path.empty() && llama_model_n_layer_nextn(model) <= 0) {
            std::fprintf(
                stderr,
                "[salm] the GGUF carries no NextN (MTP) layers: decoding without MTP drafts\n");
            cfg.mtp_draft = 0;
        }
        if (!fast && cfg.max_sequences > 256) {
            cfg.max_sequences = 256;  // upstream llama.cpp's LLAMA_MAX_SEQ
        }

        // a decode step holds every slot's verify rows (last token + drafts)
        if (cfg.n_batch < cfg.max_sequences * (cfg.mtp_draft + 1))
            throw std::invalid_argument(
                "asr.salm.n_batch (" + std::to_string(cfg.n_batch) +
                ") must be at least max_sequences x (mtp_draft + 1) = " +
                std::to_string(cfg.max_sequences * (cfg.mtp_draft + 1)));

        llama_context_params cp = llama_context_default_params();
        // fast path: extra sequences (ids max_sequences..) hold computed prompt prefixes; new
        // sequences copy them (shared KV cells; the recurrent state is read from the prefix's
        // pinned cell) instead of recomputing them.
        prefix_cache = fast;
        if (prefix_cache) {
            const int max_seq = static_cast<int>(llama_max_parallel_sequences());
            if (cfg.max_sequences >= max_seq)
                throw std::invalid_argument(
                    "asr.salm.max_sequences must be below " + std::to_string(max_seq));
            cache_owner.assign(
                std::min(kPromptCacheSequences, max_seq - cfg.max_sequences), nullptr);
        }
        cp.n_seq_max = cfg.max_sequences + static_cast<int>(cache_owner.size());
        cp.n_ctx = static_cast<uint32_t>(cfg.n_ctx_per_sequence) * cfg.max_sequences;
        cp.n_batch = cfg.n_batch;
        cp.n_ubatch = fast ? cfg.n_batch : std::min(cfg.n_batch, kUpstreamUbatch);
        cp.n_threads = cfg.threads;
        cp.n_threads_batch = cfg.threads;
        // One KV pool on every path: a sequence may use cells up to max_context while the pool
        // holds them (the batched path's ragged batches need it too).
        cp.kv_unified = true;
#if SALM_FAST_PATH
        if (fast) {
            cuda_decoder.fused = true;
            cp.serving.batched = true;
            // rejected drafts roll back natively (pinned cells + ReplaySSM), no state snapshots
            cp.serving.mtp_replay = cfg.mtp_draft > 0 ? cfg.mtp_draft + 1 : 0;
            // the KV length is padded to multiples of 4096 cells so decode graphs are reused; the
            // masks reserved at init cover at most 262144 cells and the allocator grows when a step
            // needs more
            cp.serving.kv_bucket = 4096;
            cp.serving.kv_reserve_cells = 262144;
            cp.serving.cuda_serving = &cuda_decoder;
        }
#endif
        // Reserve every verification output for backend sampling, including MTP.
        cp.n_outputs_max_per_seq = std::max(1, cfg.mtp_draft + 1);
        cp.n_outputs_max = cp.n_seq_max * cp.n_outputs_max_per_seq;
        // Sample on the backend through per-slot sampler chains (greedy: GPU argmax) instead of
        // copying every sequence's logits to the host each step.
        for (int i = 0; i < cfg.max_sequences; i++) {
            llama_sampler* chain = sampler_chain(0, std::nullopt);
            seq_samplers.push_back(chain);
            seq_greedy.push_back(1);
            seq_sampler_cfg.push_back({i, chain});
        }
        cp.samplers = seq_sampler_cfg.data();
        cp.n_samplers = seq_sampler_cfg.size();
        ctx = llama_init_from_model(model, cp);
        if (!ctx)
            throw std::runtime_error("failed to create llama context");
#if SALM_FAST_PATH
        if (fast) {
            // Tokens whose text is not Latin script, including partial UTF-8 characters.
            const int n = llama_vocab_n_tokens(vocab);
            std::vector<uint8_t> banned(n, 0);
            size_t n_banned = 0;
            for (int id = 0; id < n; ++id) {
                banned[id] = !guards::is_latin_script(piece(vocab, id));
                n_banned += banned[id];
            }
            llama_set_token_mask(ctx, banned.data(), n);
            script_mask = true;
            std::fprintf(
                stderr, "[salm] script guard: %zu of %d tokens are not Latin script\n", n_banned,
                n);
        }
#endif
        if (cfg.mtp_draft > 0 && !cfg.dflash_path.empty()) {
            // DFlash2: its own model and KV cache (6 layers), the target's token embeddings and LM
            // head through ctx_other; it reads 6 target layers per processed row
            llama_model_params mpd = llama_model_default_params();
            mpd.n_gpu_layers = mp.n_gpu_layers;
            mpd.devices = mp.devices;
            mpd.split_mode = mp.split_mode;
            mpd.main_gpu = mp.main_gpu;
            mpd.load_mode = mp.load_mode;
            model_dft = llama_model_load_from_file(cfg.dflash_path.c_str(), mpd);
            if (!model_dft)
                throw std::runtime_error("failed to load the DFlash drafter " + cfg.dflash_path);
            // same pool, sequences and serving options as the target; no recurrent state,
            // samplers or outputs
            llama_context_params cpd = cp;
            cpd.n_outputs_max = cp.n_seq_max;
            cpd.n_outputs_max_per_seq = 1;
            cpd.samplers = nullptr;
            cpd.n_samplers = 0;
            cpd.ctx_other = ctx;
#if SALM_FAST_PATH
            cpd.serving.mtp_replay = 0;
#endif
            ctx_dft = llama_init_from_model(model_dft, cpd);
            if (!ctx_dft)
                throw std::runtime_error("failed to create the DFlash draft context");
            common_params_speculative sp;
            sp.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
            sp.draft.n_max = cfg.mtp_draft;
            sp.draft.n_min = 0;
            sp.draft.p_min = 0.0f;
            sp.draft.ctx_tgt = ctx;
            sp.draft.ctx_dft = ctx_dft;
            spec = common_speculative_init(sp, cp.n_seq_max);
            if (!spec)
                throw std::runtime_error("failed to initialise DFlash speculative decoding");
        } else if (cfg.mtp_draft > 0) {
            llama_context_params cpd = cp;
            cpd.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            cpd.n_outputs_max = cp.n_seq_max;
            cpd.n_outputs_max_per_seq = 1;
            cpd.samplers = nullptr;  // the drafter samples on its own; chains can be attached to
                                     // one context only
            cpd.n_samplers = 0;
            ctx_dft = llama_init_from_model(model, cpd);
            if (!ctx_dft)
                throw std::runtime_error("failed to create the MTP draft context");
            common_params_speculative sp;
            sp.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
            sp.draft.n_max = cfg.mtp_draft;
            sp.draft.n_min = 0;
            sp.draft.p_min = 0.0f;
            sp.draft.ctx_tgt = ctx;
            sp.draft.ctx_dft = ctx_dft;
            spec =
                common_speculative_init(sp, cp.n_seq_max);  // includes the prompt-template sequence
            if (!spec)
                throw std::runtime_error("failed to initialise MTP speculative decoding");
        }
        // one unified KV stream: any sequence may use cells up to max_context, as long as the pool
        // holds them
        kv_pool = llama_n_ctx(ctx);
        n_ctx_seq = static_cast<uint32_t>(std::clamp<int64_t>(cfg.max_context, 1, kv_pool));
        tok_batch = llama_batch_init(cfg.n_batch, 0, cfg.max_sequences);
        dft_pos.resize(cfg.n_batch);
        emb_batch = llama_batch_init(cfg.n_batch, n_embd, cfg.max_sequences);
        if (cfg.gpu >= 0) {
            int gpu_index = 0;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto dev = ggml_backend_dev_get(i);
                if (!is_gpu_device(dev) || gpu_index++ != cfg.gpu)
                    continue;
                if (fast) {
                    emb_copy_backend = ggml_backend_dev_init(dev, nullptr);
                    ggml_init_params ip = {3 * ggml_tensor_overhead(), nullptr, true};
                    emb_dev_ctx = ggml_init(ip);
                    for (auto& t : emb_dev)
                        t = ggml_new_tensor_2d(emb_dev_ctx, GGML_TYPE_F32, n_embd, cfg.n_batch);
                    emb_zero = ggml_new_tensor_2d(emb_dev_ctx, GGML_TYPE_F32, n_embd, 64);
                    emb_dev_buf = ggml_backend_alloc_ctx_tensors_from_buft(
                        emb_dev_ctx, ggml_backend_dev_buffer_type(dev));
                    if (!emb_copy_backend || !emb_dev_buf)
                        throw std::runtime_error("SALM: device staging of prefill rows failed");
                    ggml_backend_buffer_clear(emb_dev_buf, 0);
                    ggml_init_params vp = {2 * ggml_tensor_overhead(), nullptr, true};
                    emb_view_ctx = ggml_init(vp);
                    // the encoder output stays on the device (same GPU) for the staging copies
                    pee.set_proj_on_device(true);
                } else if (auto host = ggml_backend_dev_host_buffer_type(dev)) {
                    emb_pinned = ggml_backend_buft_alloc_buffer(
                        host, (size_t)cfg.n_batch * n_embd * sizeof(float));
                    if (emb_pinned) {
                        emb_batch_heap = emb_batch.embd;
                        emb_batch.embd =
                            static_cast<float*>(ggml_backend_buffer_get_base(emb_pinned));
                    }
                }
                break;
            }
        }

        default_prompt = get_prompt(cfg.prompt);
        log_configuration();
    }

    // The path the engine runs and its main settings.
    void log_configuration() const {
        std::fprintf(
            stderr, "[salm] path=%s mtp_draft=%d%s slots=%d n_batch=%d\n",
            fast ? "batched (patched ggml, CUDA)" : "upstream llama.cpp", cfg.mtp_draft,
            model_dft ? " (DFlash2 drafter)" : "", cfg.max_sequences, cfg.n_batch);
    }

    // The tokenized form of a user turn (kept while in use or computed, plus a bounded number of
    // recent ones).
    std::shared_ptr<Prompt> get_prompt(const std::string& turn) {
        if (auto it = prompts.find(turn); it != prompts.end())
            return it->second;
        const size_t loc = turn.find(policy::kAudioLocator);
        if (loc == std::string::npos ||
            turn.find(policy::kAudioLocator, loc + 1) != std::string::npos)
            throw std::invalid_argument(
                "prompt must contain exactly one " + std::string(policy::kAudioLocator));
        auto p = std::make_shared<Prompt>();
        p->prefix = tokenize(vocab, std::string(policy::kChatPrefix) + turn.substr(0, loc));
        p->suffix = tokenize(
            vocab,
            turn.substr(loc + policy::kAudioLocator.size()) + std::string(policy::kChatSuffix));
        p->text_prompt = p->prefix;
        p->text_prompt.insert(p->text_prompt.end(), p->suffix.begin(), p->suffix.end());
        constexpr size_t kMaxPrompts = 4096;
        if (prompts.size() >= kMaxPrompts) {
            for (auto it = prompts.begin(); it != prompts.end();) {
                if (it->second.use_count() == 1)
                    it = prompts.erase(it);
                else
                    ++it;
            }
        }
        prompts.emplace(turn, p);
        return p;
    }

    // Fast path: the prompt-cache plan of an admission round (`uses`: its distinct prompts and
    // their requests). Each prompt starts from an entry of its whole prefix, resident ones first
    // (nothing to recompute), then the most requested. With more distinct prompts than cache
    // sequences, the last sequence holds the next prompt's prefix or the leading tokens all
    // remaining prompts share, whichever saves more rows; the rest of each prefix is computed with
    // the request. Missing entries are computed in one pass; entries used in this round are never
    // evicted.
    std::map<const Prompt*, std::shared_ptr<CacheEntry>> plan_prompt_cache(
        const std::vector<std::pair<std::shared_ptr<Prompt>, int>>& uses) {
        std::vector<std::shared_ptr<CacheEntry>> planned;
        auto find = [&](const std::vector<llama_token>& tokens) {
            for (const auto& e : planned)
                if (e->tokens == tokens)
                    return e;
            for (const auto& owner : cache_owner)
                if (owner && owner->tokens == tokens)
                    return owner;
            return std::shared_ptr<CacheEntry>();
        };
        auto entry = [&](const std::vector<llama_token>& tokens) {
            auto e = find(tokens);
            if (!e) {
                e = std::make_shared<CacheEntry>();
                e->tokens = tokens;
            }
            if (std::find(planned.begin(), planned.end(), e) == planned.end())
                planned.push_back(e);
            e->used = admission_round;
            return e;
        };
        auto whole = uses;
        std::stable_sort(whole.begin(), whole.end(), [&](const auto& a, const auto& b) {
            const bool ra = bool(find(a.first->prefix)), rb = bool(find(b.first->prefix));
            return ra != rb ? ra : a.second > b.second;
        });
        const size_t seqs = cache_owner.size();
        // with more prompts than sequences the last one holds a whole entry or the shared one
        const size_t n_whole = whole.size() <= seqs ? whole.size() : seqs - 1;
        const std::vector<std::pair<std::shared_ptr<Prompt>, int>> rest(
            whole.begin() + n_whole, whole.end());
        std::map<const Prompt*, std::shared_ptr<CacheEntry>> plan;
        for (size_t k = 0; k < n_whole; ++k)
            plan[whole[k].first.get()] = entry(whole[k].first->prefix);
        if (!rest.empty()) {
            std::vector<llama_token> head = rest[0].first->prefix;
            size_t requests = 0;
            for (const auto& u : rest) {
                const auto& prefix = u.first->prefix;
                head.resize(
                    std::mismatch(head.begin(), head.end(), prefix.begin(), prefix.end()).first -
                    head.begin());
                requests += u.second;
            }
            // rows saved in this round (a new entry is computed once) by the shared entry, and by
            // the next whole entry in its place
            auto saved = [&](const std::vector<llama_token>& tokens, size_t requests) {
                return tokens.size() * (requests - (find(tokens) ? 0 : 1));
            };
            const size_t shared = head.empty() ? 0 : saved(head, requests);
            if (shared > 0 && shared >= saved(rest[0].first->prefix, rest[0].second)) {
                auto e = entry(head);
                for (const auto& u : rest) plan[u.first.get()] = e;
            } else {
                plan[rest[0].first.get()] = entry(rest[0].first->prefix);
            }
        }
        // computes the new entries
        std::vector<char> taken(cache_owner.size(), 0);
        std::vector<std::shared_ptr<CacheEntry>> todo;
        for (const auto& e : planned) {
            if (e->seq >= 0)
                continue;
            // a free cache sequence, else the least recently used one not needed now
            int best = -1;
            for (size_t c = 0; c < cache_owner.size(); ++c) {
                if (taken[c])
                    continue;
                if (!cache_owner[c]) {
                    best = static_cast<int>(c);
                    break;
                }
                if (cache_owner[c]->used >= admission_round)
                    continue;
                if (best < 0 || cache_owner[c]->used < cache_owner[best]->used)
                    best = static_cast<int>(c);
            }
            if (best < 0)
                throw std::logic_error("SALM: prompt cache admission");
            taken[best] = 1;
            llama_memory_t mem = llama_get_memory(ctx);
            const llama_seq_id seq = cfg.max_sequences + best;
            if (auto& old =
                    cache_owner[best]) {  // its cells stay with the sequences that copied them
                llama_memory_seq_rm(mem, seq, -1, -1);
                if (spec)
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, -1, -1);
                kv_cached -= old->tokens.size();
                old->seq = -1;
                old->draft_state.clear();
            }
            cache_owner[best] = e;
            e->seq = seq;
            e->draft_state.clear();
            kv_cached += e->tokens.size();
            todo.push_back(e);
        }
        if (todo.empty())
            return plan;
        size_t n = 0;
        auto flush = [&]() {
            tok_batch.n_tokens = static_cast<int32_t>(n);
            decode(tok_batch);
            if (spec)
                common_speculative_process(spec, tok_batch);  // the drafter ingests the prefix too
            n = 0;
        };
        for (const auto& e : todo) {
            if (n + e->tokens.size() > static_cast<size_t>(cfg.n_batch))
                flush();
            for (size_t i = 0; i < e->tokens.size(); ++i) {
                if (n == static_cast<size_t>(cfg.n_batch))
                    flush();
                tok_batch.token[n] = e->tokens[i];
                tok_batch.pos[n] = static_cast<llama_pos>(i);
                tok_batch.n_seq_id[n] = 1;
                tok_batch.seq_id[n][0] = e->seq;
                // nothing is sampled here; the drafter reads dense NextN states
                tok_batch.logits[n] = 0;
                ++n;
            }
        }
        if (n)
            flush();
        llama_synchronize(ctx);
        return plan;
    }

    // Drop every computed prefix (the decoder memory was cleared).
    void reset_prompt_cache() {
        for (auto& owner : cache_owner) {
            if (owner) {
                owner->seq = -1;
                owner->draft_state.clear();
            }
            owner.reset();
        }
        kv_cached = 0;
    }

    ~Impl() {
        if (emb_dev_buf)
            ggml_backend_buffer_free(emb_dev_buf);
        if (emb_dev_ctx)
            ggml_free(emb_dev_ctx);
        if (emb_view_ctx)
            ggml_free(emb_view_ctx);
        if (emb_copy_backend)
            ggml_backend_free(emb_copy_backend);
        llama_batch_free(tok_batch);
        if (emb_pinned) {
            emb_batch.embd = emb_batch_heap;
            ggml_backend_buffer_free(emb_pinned);
        }
        llama_batch_free(emb_batch);
        if (spec)
            common_speculative_free(spec);
        if (ctx_dft)
            llama_free(ctx_dft);
        if (model_dft)
            llama_model_free(model_dft);
        if (ctx)
            llama_free(ctx);
        for (auto* smpl : seq_samplers) llama_sampler_free(smpl);
        if (model)
            llama_model_free(model);
    }

    static ggml_runtime::Params make_params(const SalmConfig& c) {
        ggml_runtime::Params p;
        p.use_gpu = c.gpu >= 0;
        p.gpu_device_idx = std::max(c.gpu, 0);
        return p;
    }

    // ------------------------------------------------------------ encoder --

    // Raw log-mel frames an encoder job keeps on the device at once (256 MiB of 128-bin frames,
    // about 1.45 hours of audio; a steady-state job of a few hundred requests fits); larger jobs
    // encode in parts.
    static constexpr size_t kMelStoreFrames = size_t(1) << 19;
    // Encode `reqs[first:first+count)` with cross-utterance window batching.
    std::vector<Encoded> encode(const std::vector<SalmRequest>& reqs, size_t first, size_t count) {
        ggml_nvtx::range nvtx_r("salm.encode");
        const auto& pc = pee.cfg();
        struct Utt {
            size_t mel_frame = 0;  // first raw log-mel frame in the part's MelStore
            int t_mel = 0;
            PeeModel::NormStats stats;
            std::vector<PeeModel::Window> plan;
            std::vector<PeeModel::WindowOut> outs;
            std::shared_ptr<CtcEvidence> evidence;  // word timestamps: joined ASR state cores
            std::vector<int> core_row;              // each window's first joined row
        };
        std::vector<Utt> utts(count);
        std::vector<Encoded> enc(count);
        // Input samples at the model rate, and the length the frontend pads each clip to: a
        // multiple of fe_bucket samples so consecutive clips reuse the frontend's cached graph
        // (frames past the real audio are dropped: t_mel). A clip already at a multiple gets no
        // zero tail, so its frames are those of the unpadded clip.
        struct Clip {
            std::vector<float> resampled;
            const float* samples = nullptr;
            size_t n = 0, n_fe = 0;
        };
        std::vector<Clip> clips(count);
        constexpr size_t fe_bucket = 4000;
        std::vector<size_t> ok;  // clips that reach the encoder
        for (size_t i = 0; i < count; i++) {
            const SalmRequest& r = reqs[first + i];
            enc[i].req = first + i;
            enc[i].audio_seconds = audio_seconds_of(r);
            try {
                Clip& c = clips[i];
                c.samples = r.samples;
                c.n = r.count;
                if (r.sample_rate != pc.sample_rate) {
                    c.resampled =
                        audio::resample_audio(r.samples, r.count, r.sample_rate, pc.sample_rate);
                    c.samples = c.resampled.data();
                    c.n = c.resampled.size();
                }
                // The frontend zero-pads (NeMo's constant-mode STFT), so a clip shorter than n_fft
                // works; the per-feature normalization's unbiased std needs two mel frames, the
                // API's minimum (320 samples at 16 kHz).
                if (c.n < 2 * static_cast<size_t>(hop))
                    throw std::invalid_argument("audio too short");
                c.n_fe = (c.n + fe_bucket - 1) / fe_bucket * fe_bucket;
                // center=True frame count
                utts[i].t_mel = std::min(fe->frames_for(c.n_fe), static_cast<int>(c.n / hop) + 1);
                ok.push_back(i);
            }
            catch (const std::invalid_argument& e) {
                enc[i].error = e.what();
                enc[i].invalid_request = true;
            }
            catch (const std::exception& e) {
                enc[i].error = e.what();
            }
        }
        // Front end of a part: each clip's raw log-mel into the store at its mel_frame, and its
        // NeMo per_feature normalization statistics.
        auto frontend = [&](const std::vector<size_t>& part, PeeModel::MelStore& store) {
            if (!fe->uses_gpu()) {
                for (size_t i : part) {
                    try {
                        const Clip& c = clips[i];
                        Utt& u = utts[i];
                        std::vector<float> padded, mel;
                        if (c.n_fe != c.n) {
                            padded.assign(c.samples, c.samples + c.n);
                            padded.resize(c.n_fe, 0.0f);
                        }
                        int t = 0;
                        fe->compute(
                            padded.empty() ? c.samples : padded.data(), c.n_fe, mel, t, true,
                            false);
                        if (t < u.t_mel)
                            throw std::runtime_error("frontend returned too few frames");
                        u.stats = pee.norm_stats(mel.data(), u.t_mel);
                        pee.upload_mel(store, u.mel_frame, mel.data(), u.t_mel);
                    }
                    catch (const std::exception& e) {
                        enc[i].error = e.what();
                    }
                }
                return;
            }
            // Up to fe_batch utterances per frontend run, sorted by length and zero-padded to the
            // group's longest, in whole seconds (fewer graph shapes).
            constexpr size_t fe_batch = 64;
            constexpr size_t max_samples = (size_t)4 << 20;  // samples per frontend group
            const size_t pad_to = static_cast<size_t>(pc.sample_rate);
            std::vector<size_t> order(part);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return clips[a].n_fe < clips[b].n_fe;
            });
            std::vector<MelSpectrogramExtractor::PaddedItem> items;
            std::vector<float> mean, var;
            for (size_t a = 0; a < order.size();) {
                size_t b = a, n_pad = 0;
                while (b < order.size() && b - a < fe_batch) {
                    const size_t p = (clips[order[b]].n_fe + pad_to - 1) / pad_to * pad_to;
                    if (b > a && (b - a + 1) * p > max_samples)
                        break;
                    n_pad = p;
                    ++b;
                }
                items.clear();
                for (size_t k = a; k < b; k++) {
                    const Clip& c = clips[order[k]];
                    const Utt& u = utts[order[k]];
                    items.push_back({c.samples, c.n, c.n_fe, u.t_mel, store.offset(u.mel_frame)});
                }
                const size_t n_mels = static_cast<size_t>(pc.n_mels);
                mean.resize(n_mels * items.size());
                var.resize(n_mels * items.size());
                try {
                    fe->compute_padded_batch(items, n_pad, store.device(), mean.data(), var.data());
                    for (size_t k = a; k < b; k++) {
                        PeeModel::NormStats& st = utts[order[k]].stats;
                        const float* m = mean.data() + (k - a) * n_mels;
                        const float* v = var.data() + (k - a) * n_mels;
                        st.mean.assign(m, m + n_mels);
                        st.inv_std.resize(n_mels);
                        for (size_t j = 0; j < n_mels; j++)
                            st.inv_std[j] = 1.0f / (std::sqrt(std::max(v[j], 0.0f)) + 1e-5f);
                    }
                }
                catch (const std::exception& e) {
                    for (size_t k = a; k < b; k++) enc[order[k]].error = e.what();
                }
                a = b;
            }
        };
        // Windows of multi-window utterances fuse the speaker probabilities of the streaming
        // speaker steps, so they batch apart from single-window utterances.
        const bool stateful = pee.stateful_speakers();
        std::vector<PeeModel::SpeakerUtt> spk(count);
        for (size_t a = 0; a < ok.size();) {
            // a part: consecutive clips whose frames fit the store budget (at least one clip)
            size_t b = a, part_frames = 0;
            while (b < ok.size() && (b == a || part_frames + utts[ok[b]].t_mel <= kMelStoreFrames))
                part_frames += utts[ok[b++]].t_mel;
            const std::vector<size_t> part(ok.begin() + a, ok.begin() + b);
            a = b;
            std::shared_ptr<PeeModel::MelStore> store;
            try {
                store = pee.mel_store(part_frames);
            }
            catch (const std::exception& e) {  // these requests fail; the others continue
                for (size_t i : part) enc[i].error = e.what();
                continue;
            }
            size_t next_frame = pee.zero_frames();
            for (size_t i : part) {
                utts[i].mel_frame = next_frame;
                next_frame += utts[i].t_mel;
            }
            std::optional<ggml_nvtx::range> stage_range(std::in_place, "enc.fe");
            frontend(part, *store);
            // (utt, window) of every window, by kind
            std::vector<std::pair<size_t, size_t>> windows[2];
            for (size_t i : part) {
                Utt& u = utts[i];
                if (!enc[i].error.empty())
                    continue;
                u.plan = pee.plan_windows(u.t_mel);
                u.outs.resize(u.plan.size());
                const bool external = stateful && u.plan.size() > 1;
                for (size_t w = 0; w < u.plan.size(); w++) windows[external].push_back({i, w});
            }
            // Word-timestamp requests: device rows for their joined ASR state cores (the CTC head
            // runs over them once the windows are encoded). Without rows a request's alignment
            // reports the evidence unavailable.
            for (size_t i : part) {
                Utt& u = utts[i];
                if (!pee.ctc() || !reqs[first + i].timestamps || !enc[i].error.empty() ||
                    u.plan.empty())
                    continue;
                int frames = 0;
                for (const auto& w : u.plan) {
                    u.core_row.push_back(frames);
                    frames += w.core;
                }
                try {
                    u.evidence = pee.ctc()->reserve(frames);
                }
                catch (const std::exception& e) {
                    std::fprintf(stderr, "[salm] word-timestamp evidence: %s\n", e.what());
                }
            }
            stage_range.emplace("enc.batches");
            // Streaming speaker steps of the multi-window utterances, one window index at a time
            // (batched across them).
            std::vector<PeeModel::SpeakerUtt*> speakers;
            for (const auto& [i, w] : windows[1]) {
                if (w != 0)
                    continue;
                spk[i].mel_frame = utts[i].mel_frame;
                spk[i].plan = &utts[i].plan;
                spk[i].want_p10 = reqs[first + i].timestamps && pee.ctc();
                speakers.push_back(&spk[i]);
            }
            pee.speaker_steps(*store, speakers);
            for (const bool external : {false, true}) {
                const auto& all = windows[external];
                std::vector<int> all_lens(all.size());
                for (size_t k = 0; k < all.size(); k++)
                    all_lens[k] = utts[all[k].first].plan[all[k].second].len();
                for (const auto& group : pee.plan_batches(all_lens)) {
                    const int blen = pee.bucket_len(all_lens[group.front()]);
                    std::vector<std::pair<size_t, size_t>> batch;
                    for (size_t k : group) batch.push_back(all[k]);
                    std::vector<size_t> frames(batch.size());
                    std::vector<const PeeModel::NormStats*> stats(batch.size());
                    std::vector<int> lens(batch.size());
                    std::vector<PeeModel::SpeakerProbs> p80(external ? batch.size() : 0);
                    std::vector<PeeModel::EvidenceSink> sinks(batch.size());
                    bool evidence = false;  // a window of a word-timestamp request is in this batch
                    for (size_t b = 0; b < batch.size(); b++) {
                        evidence = evidence || reqs[first + batch[b].first].timestamps;
                        const Utt& u = utts[batch[b].first];
                        const size_t wi = batch[b].second;
                        const auto& w = u.plan[wi];
                        frames[b] = u.mel_frame + static_cast<size_t>(w.enc_stt);
                        stats[b] = &u.stats;
                        lens[b] = w.len();
                        if (external)
                            p80[b] = spk[batch[b].first].p80[wi];
                        if (u.evidence)
                            sinks[b] = {u.evidence.get(), u.core_row[wi], w.left_drop, w.core};
                    }
                    evidence = evidence && pee.ctc();
                    const int b_pad = pee.batch_pad(static_cast<int>(batch.size()));
                    auto outs = pee.encode_batch(
                        *store, frames, stats, lens, blen, evidence, external ? &p80 : nullptr,
                        evidence ? &sinks : nullptr, b_pad);
                    for (size_t b = 0; b < batch.size(); b++)
                        utts[batch[b].first].outs[batch[b].second] = std::move(outs[b]);
                }
            }
        }
        ggml_nvtx::range fuse_range("enc.fuse");
        std::vector<std::shared_ptr<CtcEvidence>> heads;  // this job's word-timestamp evidence
        for (size_t i = 0; i < count; i++) {
            if (!enc[i].error.empty())
                continue;
            const bool evidence = reqs[first + i].timestamps && pee.ctc();
            std::vector<const std::vector<float>*> p10;
            for (const auto& p : spk[i].p10) p10.push_back(&p);
            auto out = pee.fuse_windows(
                utts[i].plan, utts[i].outs, evidence, spk[i].p80.empty() ? nullptr : &p10);
            enc[i].frames = out.n_frames;
            enc[i].proj = std::move(out.proj);
            enc[i].proj_spans = std::move(out.proj_spans);
            // Audio without a voice (noise, tones) makes the decoder invent a transcript: the
            // speaker branch's activity (as for diarization, above the fusion threshold in some
            // 80 ms frame) says whether anyone speaks.
            enc[i].speech = std::any_of(out.spk_probs.begin(), out.spk_probs.end(), [&](float p) {
                return p > pc.speaker_activity_threshold;
            });
            if (evidence) {
                auto ts = std::make_shared<TimestampInput>();
                ts->head = utts[i].evidence;
                if (ts->head)
                    heads.push_back(ts->head);
                ts->probs80 = std::move(out.spk_probs);
                ts->probs10 = std::move(out.spk_probs10);
                ts->frames = out.n_frames;
                ts->probs10.resize(std::min(
                    ts->probs10.size(), static_cast<size_t>(utts[i].t_mel) * pc.num_speakers));
                enc[i].timestamps = std::move(ts);
            }
        }
        // The CTC head over the job's evidence runs in the background on its own lane; alignment
        // waits for it.
        try {
            if (!heads.empty())
                pee.ctc()->finish(heads);
        }
        catch (const std::exception& e) {
            std::fprintf(stderr, "[salm] word-timestamp evidence: %s\n", e.what());
        }
        return enc;
    }

    // ------------------------------------------------------------ decoder --

    // The drafter ingests the target's rows of a pass; positions handed to the driver are shifted
    // by the slot's dft_offset. `slot_of` maps seq id -> Slot.
    void spec_process(const std::vector<Slot*>& slot_of) {
        ggml_nvtx::range nvtx_r("mtp.ingest");
        llama_batch view = tok_batch;
        for (int i = 0; i < tok_batch.n_tokens; i++) {
            const Slot* s = slot_of[tok_batch.seq_id[i][0]];
            dft_pos[i] = tok_batch.pos[i] - s->dft_offset;
        }
        view.pos = dft_pos.data();
        common_speculative_process(spec, view);
    }

    void decode(llama_batch& b) {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "llama_decode n=%d%s", b.n_tokens, b.embd ? " embd" : "");
        ggml_nvtx::range nvtx_r(tag);
        const int rc = llama_decode(ctx, b);
        if (rc != 0)
            throw std::runtime_error("llama_decode failed with " + std::to_string(rc));
    }

#if SALM_FAST_PATH
    // Copy `rows` rows of `src` from row `src_row` to row `dst_row` of a staging buffer, on the
    // staging stream.
    void stage_copy(
        ggml_tensor* dst, size_t dst_row, ggml_tensor* src, size_t src_row, size_t rows) {
        const size_t row_bytes = static_cast<size_t>(n_embd) * sizeof(float);
        ggml_reset(emb_view_ctx);  // the views only describe this copy
        ggml_tensor* from = ggml_view_1d(emb_view_ctx, src, rows * n_embd, src_row * row_bytes);
        ggml_tensor* to = ggml_view_1d(emb_view_ctx, dst, rows * n_embd, dst_row * row_bytes);
        ggml_backend_tensor_copy_async(emb_copy_backend, emb_copy_backend, from, to);
    }

    // Pack text, audio and suffix rows into the same model pass. Audio rows use token=-1; token
    // rows have a zero supplied vector and use embedding lookup. With device staging the rows never
    // pass through the host: encoder rows are copied on the device, and the graph reads the staging
    // buffer in place (llama_batch.embd_input_device).
    void prefill_packed(
        std::vector<Slot*>& slots, const std::vector<const Encoded*>& encs,
        std::vector<Slot*>& active) {
        std::vector<Slot*> slot_of(cfg.max_sequences, nullptr);
        std::vector<Flagged> flagged;
        for (Slot* s : active) slot_of[s->seq_id] = s;
        for (Slot* s : slots) slot_of[s->seq_id] = s;
        size_t n = 0;
        // The next pass is packed while the GPU runs the current one: the pass's sampled rows are
        // read just before the next decode (decode has already uploaded its inputs, so the buffers
        // are free).
        std::vector<Flagged> deferred;
        // Cached prompt prefixes: every new sequence shares its prompt's prefix cells and starts
        // from its recurrent state, and the drafter gets the prefix KV and pending hidden row.
        // Done before the first pass: the host fallback for the drafter's row waits for the
        // target's stream, which would stall the overlap of packing with a running pass.
        for (Slot* s : slots) {
            s->dft_offset = 0;
            if (!s->cached)
                continue;
            CacheEntry& c = *s->cached;
            llama_memory_seq_cp(llama_get_memory(ctx), c.seq, s->seq_id, -1, -1);
            if (spec) {
                llama_memory_seq_cp(llama_get_memory(ctx_dft), c.seq, s->seq_id, -1, -1);
                // copied on the GPU, queued with the drafter's other row moves
                if (!common_speculative_copy_state(spec, c.seq, s->seq_id)) {
                    if (c.draft_state.empty())
                        common_speculative_get_state(spec, c.seq, c.draft_state);
                    common_speculative_set_state(spec, s->seq_id, c.draft_state);
                }
            }
            s->pos = static_cast<llama_pos>(c.tokens.size());
        }
        // Attention scratch grows with tokens per pass x the longest sequence in the cache. Cap the
        // pass so the scratch never exceeds a full pass over 8192 cells (long sessions get shorter
        // passes).
        size_t longest = 1;
        for (const Slot* s : active)
            longest = std::max<size_t>(longest, static_cast<size_t>(s->pos));
        for (size_t k = 0; k < slots.size() && k < encs.size(); ++k)
            longest = std::max<size_t>(
                longest, slots[k]->prompt->prefix.size() + static_cast<size_t>(encs[k]->frames) +
                             slots[k]->prompt->suffix.size() + slots[k]->resume.size());
        constexpr size_t kScratchCells = size_t(16384) * 8192;
        const size_t capacity = std::min<size_t>(
            static_cast<size_t>(cfg.n_batch),
            std::max<size_t>(static_cast<size_t>(kPrefillChunk), kScratchCells / longest));
        auto flush = [&]() {
            tok_batch.n_tokens = static_cast<int32_t>(n);
            tok_batch.embd_input = emb_batch.embd;
            // the pass's rows were staged on the device while it was packed; the target and the
            // drafter read them in place
            emb_dev_turn ^= 1;
            ggml_backend_synchronize(emb_copy_backend);
            tok_batch.embd_input_device = emb_dev[emb_dev_turn];
            // This pass is prepared (batch, memory, graph) while the previous one still runs; the
            // previous pass sampled rows stay readable until the launch, so they are read in
            // between and the GPU does not wait for the host to build the pass.
            bool prepared = false;
            if (!deferred.empty()) {
                {
                    ggml_nvtx::range nvtx_p("llama_decode_prepare");
                    prepared = llama_decode_prepare(ctx, tok_batch) == 0;
                }
                sample(deferred);
            }
            if (prepared) {
                char tag[64];
                std::snprintf(tag, sizeof(tag), "llama_decode_launch n=%zu", n);
                ggml_nvtx::range nvtx_l(tag);
                const int rc = llama_decode_launch(ctx, nullptr, static_cast<int32_t>(n));
                if (rc != 0)
                    throw std::runtime_error(
                        "llama_decode_launch failed with " + std::to_string(rc));
            } else {
                decode(tok_batch);
            }
            if (spec)
                spec_process(slot_of);  // reads this pass's rows: before the next pass is packed
            tok_batch.embd_input_device = nullptr;
            tok_batch.embd_input = nullptr;
            deferred.swap(flagged);
            flagged.clear();
            n = 0;
        };
        // `count` rows of slot s from row n on: tokens (zero supplied vectors), or encoder rows
        // from `block` row `src`; `output` asks for the last row's logits. Each span is copied into
        // the staging buffer of the pass being packed (the other one: the pass before the running
        // one has finished reading it, and so has the drafter's ingest of it, which the running
        // pass's ingest waited for).
        auto add_rows = [&](Slot* s, size_t count, const llama_token* tokens,
                            const PeeModel::DeviceBlock* block, size_t src, bool output) {
            for (size_t done = 0; done < count;) {
                const size_t take = std::min(count - done, capacity - n);
                ggml_tensor* staging = emb_dev[emb_dev_turn ^ 1];
                if (block)
                    stage_copy(staging, n, block->tensor(), src + done, take);
                for (size_t z = 0; !block && z < take; z += emb_zero->ne[1])
                    stage_copy(
                        staging, n + z, emb_zero, 0, std::min<size_t>(take - z, emb_zero->ne[1]));
                for (size_t k = 0; k < take; ++k, ++n) {
                    tok_batch.token[n] = tokens ? tokens[done + k] : LLAMA_TOKEN_NULL;
                    tok_batch.pos[n] = s->pos++;
                    tok_batch.n_seq_id[n] = 1;
                    tok_batch.seq_id[n][0] = s->seq_id;
                    tok_batch.logits[n] = output && done + k + 1 == count;
                    if (tok_batch.logits[n])
                        flagged.push_back({s, static_cast<int32_t>(n)});
                }
                done += take;
                if (n == capacity)
                    flush();
            }
        };
        for (size_t j = 0; j < slots.size(); ++j) {
            Slot* s = slots[j];
            const Encoded& e = *encs[j];
            const Prompt& p = *s->prompt;
            const size_t cached = s->cached ? s->cached->tokens.size() : 0;
            add_rows(s, p.prefix.size() - cached, p.prefix.data() + cached, nullptr, 0, false);
            for (const auto& span : e.proj_spans)
                add_rows(s, span.rows, nullptr, span.block.get(), span.row, false);
            // a preempted sequence continues from its output so far: its last row samples the
            // token after them
            add_rows(s, p.suffix.size(), p.suffix.data(), nullptr, 0, s->resume.empty());
            if (!s->resume.empty())
                add_rows(s, s->resume.size(), s->resume.data(), nullptr, 0, true);
        }
        if (n)
            flush();
        if (!deferred.empty())
            sample(deferred);
        for (Slot* s : slots) {
            s->pending.assign(1, s->last);
            s->cached.reset();  // its cells are shared now; the entry may be evicted
            s->resume.clear();
        }
    }
#endif  // SALM_FAST_PATH

    // Prefill freshly admitted slots: prefix tokens for all, then all audio
    // frames, then suffix tokens with logits on each slot's last token.
    void prefill(
        std::vector<Slot*>& slots_in, const std::vector<const Encoded*>& encs_in,
        std::vector<Slot*>& active) {
        ggml_nvtx::range nvtx_r("salm.prefill");
#if SALM_FAST_PATH
        if (fast) {
            prefill_packed(slots_in, encs_in, active);
            return;
        }
#endif
        const std::vector<Slot*>& slots = slots_in;
        const std::vector<const Encoded*>& encs = encs_in;
        const size_t n_new = slots.size();
        // (1) prefix tokens, all slots
        {
            size_t n = 0;
            auto flush = [&]() {
                tok_batch.n_tokens = static_cast<int32_t>(n);
                decode(tok_batch);
                n = 0;
            };
            for (size_t s = 0; s < n_new; s++) {
                for (llama_token t : slots[s]->prompt->prefix) {
                    tok_batch.token[n] = t;
                    tok_batch.pos[n] = slots[s]->pos++;
                    tok_batch.n_seq_id[n] = 1;
                    tok_batch.seq_id[n][0] = slots[s]->seq_id;
                    tok_batch.logits[n] = 0;
                    if (++n == static_cast<size_t>(cfg.n_batch))
                        flush();
                }
            }
            if (n)
                flush();
        }
        // (2) audio embeddings, all slots, n_batch frames per decode
        {
            // Round-robin over fixed frame chunks: batch k holds frames [k*C, (k+1)*C) of
            // every sequence, so all sequences in a micro-batch have equal token counts
            // and llama.cpp runs one wide micro-batch (n_seqs x C) rather than a
            // staircase of small remainders.
            size_t n = 0;
            auto flush = [&]() {
                emb_batch.n_tokens = static_cast<int32_t>(n);
                decode(emb_batch);
                n = 0;
            };
            int max_frames = 0;
            for (const Encoded* e : encs) max_frames = std::max(max_frames, e->frames);
            const int chunk = std::max(
                1,
                std::min(kPrefillChunk, cfg.n_batch / std::max<int>(1, static_cast<int>(n_new))));
            for (int f0 = 0; f0 < max_frames; f0 += chunk) {
                for (size_t s = 0; s < n_new; s++) {
                    const Encoded& e = *encs[s];
                    const int f1 = std::min(e.frames, f0 + chunk);
                    for (int f = f0; f < f1; f++) {
                        std::memcpy(
                            emb_batch.embd + n * n_embd,
                            e.proj.data() + static_cast<size_t>(f) * n_embd,
                            static_cast<size_t>(n_embd) * sizeof(float));
                        emb_batch.pos[n] = slots[s]->pos++;
                        emb_batch.n_seq_id[n] = 1;
                        emb_batch.seq_id[n][0] = slots[s]->seq_id;
                        emb_batch.logits[n] = 0;
                        if (++n == static_cast<size_t>(cfg.n_batch))
                            flush();
                    }
                }
                if (n)
                    flush();  // one batch per chunk level keeps the per-sequence counts equal
            }
        }
        // (3) suffix tokens; the last one of each slot produces that slot's first logits
        {
            size_t n = 0;
            std::vector<Flagged> flagged;
            auto flush = [&]() {
                tok_batch.n_tokens = static_cast<int32_t>(n);
                decode(tok_batch);
                sample(flagged);
                flagged.clear();
                n = 0;
            };
            for (size_t s = 0; s < n_new; s++) {
                const auto& suffix = slots[s]->prompt->suffix;
                for (size_t k = 0; k < suffix.size(); k++) {
                    const bool last = k + 1 == suffix.size();
                    tok_batch.token[n] = suffix[k];
                    tok_batch.pos[n] = slots[s]->pos++;
                    tok_batch.n_seq_id[n] = 1;
                    tok_batch.seq_id[n][0] = slots[s]->seq_id;
                    tok_batch.logits[n] = last ? 1 : 0;
                    if (last)
                        flagged.push_back({slots[s], static_cast<int32_t>(n)});
                    if (++n == static_cast<size_t>(cfg.n_batch))
                        flush();
                }
            }
            if (n)
                flush();
        }
        for (Slot* s : slots) s->pending.assign(1, s->last);
    }

    // A request's sampler chain: greedy, or temperature scaling and a seeded draw from the full
    // distribution.
    static llama_sampler* sampler_chain(float temperature, std::optional<uint64_t> seed) {
        llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (temperature < policy::kGreedyTemperature) {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
            return chain;
        }
        uint32_t seed32 = LLAMA_DEFAULT_SEED;  // random
        if (seed) {  // fold the 64-bit seed (splitmix64 finalizer) into the 32-bit seed llama.cpp
                     // takes
            uint64_t z = *seed + 0x9e3779b97f4a7c15ull;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
            seed32 = static_cast<uint32_t>(z ^ (z >> 31));
            if (seed32 == LLAMA_DEFAULT_SEED)
                seed32 = 0;  // that value would mean "random"
        }
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed32));
        return chain;
    }

    // Attach `chain` to sequence `seq`, owning it, and free the chain it replaces (an attached
    // chain cannot be attached again, so every change builds a new one).
    void attach_sampler(int seq, llama_sampler* chain, bool greedy) {
        if (!llama_set_sampler(ctx, seq, chain)) {
            llama_sampler_free(chain);
            throw std::runtime_error("SALM: the sampler cannot run on the backend");
        }
        llama_sampler_free(seq_samplers[seq]);
        seq_samplers[seq] = chain;
        seq_greedy[seq] = greedy;
    }

    // Attach the slot's sampler to its sequence. Greedy sequences keep their chain, so all-greedy
    // serving never changes the graph.
    void bind_sampler(Slot* s) {
        const bool greedy = s->temperature < policy::kGreedyTemperature;
        if (greedy && seq_greedy[s->seq_id])
            return;
        attach_sampler(s->seq_id, sampler_chain(s->temperature, s->seed), greedy);
    }

    // Greedy-pick the next token for each slot whose logits were requested.
    // llama_get_logits_ith(i) takes the *batch entry index* of a token that had
    // logits enabled (it maps through output_ids), so callers pass those indices.
    struct Flagged {
        Slot* slot;
        int32_t batch_idx;
    };
    // Tokens of several batch rows: pick_rows() reads the backend-sampled ids of `pick_idx` after
    // one wait for the step; picked_at(i) is the token of pick_idx[i]. Upstream llama.cpp may leave
    // a row unsampled (a sampler chain it could not offload): its logits are decoded on the host.
    std::vector<int32_t> pick_idx;
    std::vector<llama_token> picked;
    void pick_rows() {
        picked.assign(pick_idx.size(), LLAMA_TOKEN_NULL);
#if SALM_FAST_PATH
        llama_get_sampled_tokens_ith(
            ctx, pick_idx.data(), static_cast<int32_t>(pick_idx.size()), picked.data());
#else
        for (size_t i = 0; i < pick_idx.size(); ++i)
            picked[i] = llama_get_sampled_token_ith(ctx, pick_idx[i]);
#endif
    }
    llama_token picked_at(size_t i) {
        if (picked[i] == LLAMA_TOKEN_NULL) {
            if (fast)  // the batched path samples every row on the device
                throw std::runtime_error(
                    "no sampled token for batch entry " + std::to_string(pick_idx[i]));
            const float* logits = llama_get_logits_ith(ctx, pick_idx[i]);
            if (!logits)
                throw std::runtime_error(
                    "missing logits for batch entry " + std::to_string(pick_idx[i]));
            picked[i] = argmax(logits, n_vocab);
        }
        return picked[i];
    }

    void sample(const std::vector<Flagged>& flagged) {
        ggml_nvtx::range nvtx_r("salm.sample");
        const size_t n = flagged.size();
        pick_idx.resize(n);
        for (size_t i = 0; i < n; ++i) pick_idx[i] = flagged[i].batch_idx;
        pick_rows();
        for (size_t i = 0; i < n; ++i) flagged[i].slot->last = picked_at(i);
    }

    // Commit the sampled token(s) of each slot in `sampled`: append text, detect end
    // of turn. Returns the new active set (all of `all_active` minus finished),
    // sorted and compacted by sequence id.
    std::vector<Slot*> commit(
        const std::vector<Slot*>& sampled, const std::vector<Slot*>& all_active,
        std::vector<SalmResult>& results, bool compact = true) {
        ggml_nvtx::range nvtx_r("salm.commit");
        std::vector<Slot*> finished;
        // per token id: its piece and whether it ends the turn (commit runs between decode steps on
        // the critical path)
        if (piece_memo.size() != static_cast<size_t>(n_vocab)) {
            piece_memo.assign(n_vocab, std::string());
            piece_state.assign(n_vocab, 0);
        }
        for (Slot* s : sampled) {
            const bool cancelled = s->cancelled;
            bool done = cancelled;
            if (cancelled)
                s->pending.clear();  // released without this step's tokens
            for (llama_token t : s->pending) {
                // commit the draw behind each kept token (MTP verification draws for rejected rows
                // are redrawn)
                if (!seq_greedy[s->seq_id])
                    llama_sampler_accept(seq_samplers[s->seq_id], t);
                const bool in_vocab = t >= 0 && t < n_vocab;
                if (in_vocab && piece_state[t] == 0) {
                    piece_memo[t] = piece(vocab, t);
                    piece_state[t] = t == im_end ? 2 : 1;
                }
                if (!in_vocab)
                    piece_tmp = piece(vocab, t);
                const std::string& p = in_vocab ? piece_memo[t] : piece_tmp;
                const bool eog = t == im_end;
                if (!eog) {
                    s->text += p;
                    s->n_generated++;
                    if (fast)
                        s->tokens.push_back(t);
                    if (s->fallback_prompt && !policy::response_text(p).empty()) {
                        s->fallback.reset();
                        s->fallback_prompt.reset();
                    }
                }
                const bool runaway = !eog && s->loop.push(t);
                if (eog || runaway || s->n_generated >= s->budget ||
                    s->pos + 1 >= static_cast<llama_pos>(n_ctx_seq)) {
                    s->ended = eog;
                    done = true;
                    break;
                }
            }
            // Stream what is settled: all of the text but an incomplete trailing character. A
            // settled piece is followed by more text, so an incomplete sequence at its end is
            // invalid (utf8_lossy); only the end of the generation drops one (utf8_finish).
            if (s->on_text && !cancelled) {
                const std::string_view text = s->text;
                const size_t end = done ? text.size() : text.size() - policy::utf8_pending(text);
                if (end > s->streamed) {
                    const auto piece = text.substr(s->streamed, end - s->streamed);
                    const std::string settled =
                        done ? policy::utf8_finish(piece) : policy::utf8_lossy(piece);
                    s->streamed = end;
                    if (!settled.empty())
                        s->on_text(settled);
                }
            }
            s->pending.clear();
            if (done) {
                s->on_text = nullptr;
                kv_reserved -= s->kv_cells;
                s->kv_cells = 0;
                s->prompt.reset();
                SalmResult& r = results[s->req];
                r.end_of_turn = s->ended;
                r.text = policy::utf8_finish(s->text);
                r.generated_tokens = s->n_generated;
                if (cancelled) {  // no fallback decode or alignment; the evidence is freed
                    cancel_result(r);
                    s->timestamps.reset();
                    s->cancelled = false;
                    finished_reqs.push_back(s->req);
                } else if (s->fallback_prompt && s->fallback) {
                    // the response is empty: decode again
                    r.first_attempt = SalmResult::Attempt{
                        r.audio_tokens, r.prompt_tokens, r.generated_tokens, r.end_of_turn};
                    row_prompt[s->req] = std::move(s->fallback_prompt);
                    fallback_ready.push_back(std::move(*s->fallback));
                    s->fallback.reset();
                } else if (s->timestamps)
                    timestamp_ready.emplace_back(s->req, std::move(s->timestamps));
                else
                    finished_reqs.push_back(s->req);
                s->fallback.reset();
                s->fallback_prompt.reset();
                s->text.clear();
                s->tokens.clear();
                finished.push_back(s);
            }
        }
        llama_memory_t mem = llama_get_memory(ctx);
        for (Slot* f : finished) {
            llama_memory_seq_rm(mem, f->seq_id, -1, -1);
            if (spec)
                llama_memory_seq_rm(llama_get_memory(ctx_dft), f->seq_id, -1, -1);
            f->active = false;
        }
        std::vector<Slot*> keep;  // the finished slots were just marked inactive
        keep.reserve(all_active.size());
        for (Slot* s : all_active)
            if (s->active)
                keep.push_back(s);
        std::sort(keep.begin(), keep.end(), [](const Slot* a, const Slot* b) {
            return a->seq_id < b->seq_id;
        });
        // fast path: recurrent states stay in their sequence's pinned cell, so ids may have holes
        // (no migrations).
        for (size_t i = 0; compact && !fast && i < keep.size(); i++) {
            if (keep[i]->seq_id == static_cast<int>(i))
                continue;
            // Fill this hole with the highest live id. Shifting every following
            // sequence renames thousands of KV cells for every finished request.
            // The remaining pointers stay ordered, so subsequent holes are still
            // found by the same walk; at most one migration per hole is needed.
            Slot* moved = keep.back();
            keep.pop_back();
            keep.insert(keep.begin() + i, moved);
            const int old_id = keep[i]->seq_id;
            llama_memory_seq_cp(mem, old_id, static_cast<int>(i), -1, -1);
            llama_memory_seq_rm(mem, old_id, -1, -1);
            keep[i]->seq_id = static_cast<int>(i);
            if (!seq_greedy[old_id])  // the sampler moves too, with its random state
                attach_sampler(
                    static_cast<int>(i), llama_sampler_clone(seq_samplers[old_id]), false);
            else
                bind_sampler(keep[i]);
        }
        return keep;
    }

    // Fast path: rows a sequence's next step may add to the cells it holds (the last token and the
    // drafts, plus one for the step prepared while the current one runs).
    size_t kv_step_rows() const { return static_cast<size_t>(cfg.mtp_draft) + 2; }

    // Fast path: remove the computed prompt prefixes from both memories (cells that admitted
    // sequences copied stay with them).
    void drop_prompt_cache() {
        for (const auto& owner : cache_owner) {
            if (!owner)
                continue;
            llama_memory_seq_rm(llama_get_memory(ctx), owner->seq, -1, -1);
            if (spec)
                llama_memory_seq_rm(llama_get_memory(ctx_dft), owner->seq, -1, -1);
        }
        reset_prompt_cache();
    }

    // Fast path: free a sequence's KV cells and recurrent state, and keep what it generated for its
    // readmission (requeued by serve ahead of requests that have not started): its request is
    // encoded again, unless it still holds its encoder output for the empty-response fallback, and
    // prefilled from its prompt and output tokens, so greedy decoding continues as it would have.
    void preempt(Slot* v) {
        Preempted p;
        const size_t prompt = v->kv_full - static_cast<size_t>(v->budget) - cfg.mtp_draft - 1;
        p.kv_share =
            std::min(v->kv_full, prompt + v->tokens.size() + kv_step_rows() + kKvGrowCells);
        p.tokens = std::move(v->tokens);
        p.text = std::move(v->text);
        p.streamed = v->streamed;
        p.n_generated = v->n_generated;
        p.loop = std::move(v->loop);
        p.timestamps = std::move(v->timestamps);
        if (!seq_greedy[v->seq_id])
            p.sampler = llama_sampler_clone(seq_samplers[v->seq_id]);
        p.fallback = v->fallback_prompt && v->fallback;
        if (p.fallback) {
            preempted_ready.push_back(std::move(*v->fallback));
            p.kv_share = 0;
        } else {
            reencode.push_back(v->req);
            ++reencoding;
            kv_resuming += p.kv_share;
        }
        // as for a finished sequence; a prepared step that holds it runs it as a zombie
        llama_memory_seq_rm(llama_get_memory(ctx), v->seq_id, -1, -1);
        if (spec)
            llama_memory_seq_rm(llama_get_memory(ctx_dft), v->seq_id, -1, -1);
        kv_reserved -= v->kv_cells;
        v->kv_cells = 0;
        v->active = false;
        v->prompt.reset();
        v->on_text = nullptr;
        v->fallback.reset();
        v->fallback_prompt.reset();
        v->text.clear();
        v->tokens.clear();
        v->pending.clear();
        ++preemptions;
        preempted[v->req] = std::move(p);
    }

    // Fast path, after a step's commit: grow every sequence's reservation to cover its next step,
    // oldest first, by kKvGrowCells when the pool has them. When it cannot cover the step, the most
    // recently admitted sequence is preempted (the growing one, when that is the newest) until it
    // can. A sequence alone always fits: its worst case fits the context and so the pool, once the
    // prompt cache gives its cells back. Returns the preempted request rows.
    std::vector<size_t> grow_kv(std::vector<Slot*>& active) {
        std::vector<size_t> out;
        if (!fast || active.empty())
            return out;
        std::vector<Slot*> order(active);
        std::sort(order.begin(), order.end(), [](const Slot* a, const Slot* b) {
            return a->admitted < b->admitted;
        });
        auto take = [&](Slot* v) {
            out.push_back(v->req);
            preempt(v);
        };
        for (Slot* s : order) {
            if (!s->active)
                continue;
            const size_t need = std::min(s->kv_full, static_cast<size_t>(s->pos) + kv_step_rows());
            if (s->kv_cells >= need)
                continue;
            while (kv_cached + kv_reserved + (need - s->kv_cells) > kv_pool) {
                Slot* newest = nullptr;
                for (auto it = order.rbegin(); it != order.rend() && !newest; ++it)
                    if ((*it)->active)
                        newest = *it;
                const bool alone = std::none_of(
                    order.begin(), order.end(), [&](const Slot* o) { return o != s && o->active; });
                if (!alone) {
                    take(newest);
                    if (newest == s)
                        break;
                } else if (kv_cached > 0) {
                    drop_prompt_cache();
                } else {
                    throw std::logic_error("SALM: KV reservation exceeds the pool");
                }
            }
            if (!s->active)
                continue;
            const size_t want = std::min(s->kv_full, need + kKvGrowCells);
            const size_t grant =
                kv_cached + kv_reserved + (want - s->kv_cells) <= kv_pool ? want : need;
            kv_reserved += grant - s->kv_cells;
            s->kv_cells = grant;
        }
        if (!out.empty())
            active.erase(
                std::remove_if(
                    active.begin(), active.end(), [](const Slot* a) { return !a->active; }),
                active.end());
        return out;
    }

    // One generation step for every active slot (in ascending id order, so the
    // sequential split sees one run of consecutive ids = one micro-batch).
    void step(std::vector<Slot*>& active) {
        char tag[48];
        std::snprintf(tag, sizeof(tag), "salm.step active=%zu", active.size());
        ggml_nvtx::range nvtx_r(tag);
        std::sort(active.begin(), active.end(), [](const Slot* a, const Slot* b) {
            return a->seq_id < b->seq_id;
        });
        std::vector<Slot*>
            lanes;  // batch rows of this step (live lanes, plus zombies of a prepared step)
        if (pend_ready) {
            lanes = pend_lanes;
            pend_tok.resize(lanes.size());
            for (size_t i = 0; i < lanes.size(); ++i)
                pend_tok[i] = lanes[i]->active ? lanes[i]->last : 0;
            char tag2[64];
            std::snprintf(tag2, sizeof(tag2), "llama_decode n=%zu launch", lanes.size());
            ggml_nvtx::range nvtx_l(tag2);
#if SALM_FAST_PATH
            const int rc =
                llama_decode_launch(ctx, pend_tok.data(), static_cast<int32_t>(pend_tok.size()));
            if (rc != 0)
                throw std::runtime_error("llama_decode_launch failed with " + std::to_string(rc));
#endif
            pend_ready = false;
        } else {
            size_t n = 0;
            for (Slot* s : active) {
                tok_batch.token[n] = s->last;
                tok_batch.pos[n] = s->pos++;
                tok_batch.n_seq_id[n] = 1;
                tok_batch.seq_id[n][0] = s->seq_id;
                tok_batch.logits[n] = 1;
                n++;
            }
            tok_batch.n_tokens = static_cast<int32_t>(n);
            decode(tok_batch);
            lanes = active;
        }
#if SALM_FAST_PATH
        if (fast && prepare_ok && !spec) {
            // the next step: this step's live lanes (some may finish now and become zombies there)
            std::vector<Slot*> next;
            for (Slot* s : lanes)
                if (s->active)
                    next.push_back(s);
            if (!next.empty()) {
                size_t n = 0;
                for (Slot* s : next) {
                    tok_batch.token[n] = s->last;  // placeholder; launch sets the sampled ids
                    tok_batch.pos[n] = s->pos;
                    tok_batch.n_seq_id[n] = 1;
                    tok_batch.seq_id[n][0] = s->seq_id;
                    tok_batch.logits[n] = 1;
                    n++;
                }
                tok_batch.n_tokens = static_cast<int32_t>(n);
                ggml_nvtx::range nvtx_p("llama_decode_prepare");
                if (llama_decode_prepare(ctx, tok_batch) == 0) {
                    for (Slot* s : next) s->pos++;
                    pend_lanes = std::move(next);
                    pend_ready = true;
                }
            }
        }
#endif
        std::vector<Flagged> flagged;
        flagged.reserve(lanes.size());
        for (size_t i = 0; i < lanes.size(); i++) {
            if (lanes[i]->active)
                flagged.push_back({lanes[i], static_cast<int32_t>(i)});
        }
        sample(flagged);
        for (Slot* s : active) s->pending.assign(1, s->last);
    }

    // One MTP step: draft up to mtp_draft tokens per slot with the model's NextN
    // head, verify them all in a single target decode, keep the longest agreeing
    // prefix (greedy), roll the rejected tail out of the target memory.
    void draft_mtp(const std::vector<Slot*>& active) {
        for (Slot* s : active) {
            s->draft.clear();
            common_speculative_get_draft_params(spec, s->seq_id) = {
                /* .drafting = */ true,
                /* .n_max    = */ cfg.mtp_draft,
                /* .pos0     = */ s->pos - s->dft_offset,
                /* .id_last  = */ s->last,
                /* .prompt   = */ &s->prompt->text_prompt,
                /* .result   = */ &s->draft,
            };
        }
        common_speculative_draft(spec);
        // Drafting extended the draft context with the candidate tokens; process()
        // will re-decode the verified rows at those positions, so drop them first
        // (the server does the same right after drafting).
        {
            llama_memory_t dmem = llama_get_memory(ctx_dft);
            for (Slot* s : active) llama_memory_seq_rm(dmem, s->seq_id, s->pos - s->dft_offset, -1);
        }
    }

    void append_verify(const std::vector<Slot*>& active, size_t& n) {
        for (Slot* s : active) {
            s->i_batch.clear();
            const llama_pos p0 = s->pos;
            auto add = [&](llama_token t, llama_pos pos) {
                tok_batch.token[n] = t;
                tok_batch.pos[n] = pos;
                tok_batch.n_seq_id[n] = 1;
                tok_batch.seq_id[n][0] = s->seq_id;
                tok_batch.logits[n] = 1;
                s->i_batch.push_back(static_cast<int32_t>(n));
                n++;
            };
            add(s->last, p0);
            for (size_t j = 0; j < s->draft.size(); j++)
                add(s->draft[j], p0 + 1 + static_cast<llama_pos>(j));
        }
    }

    void accept_mtp(const std::vector<Slot*>& active) {
        llama_memory_t mem = llama_get_memory(ctx);
        // every verify row of every slot, read after one wait for the step
        pick_idx.clear();
        for (Slot* s : active)
            pick_idx.insert(pick_idx.end(), s->i_batch.begin(), s->i_batch.end());
        pick_rows();
        size_t row = 0;
        for (Slot* s : active) {
            const size_t first = row;
            row += s->i_batch.size();
            std::vector<llama_token> accepted;
            accepted.push_back(picked_at(first));
            for (size_t j = 0; j < s->draft.size() && s->draft[j] == accepted.back(); j++)
                accepted.push_back(picked_at(first + j + 1));
            const size_t n_acc = accepted.size() - 1;  // draft tokens confirmed
            // memory holds [.., p0 (last), p0+1 .. p0+n_draft (drafts)]; keep p0 .. p0+n_acc
            if (n_acc < s->draft.size()) {
                const llama_pos cut = s->pos + 1 + static_cast<llama_pos>(n_acc);
                llama_memory_seq_rm(mem, s->seq_id, cut, -1);
                llama_memory_seq_rm(llama_get_memory(ctx_dft), s->seq_id, cut - s->dft_offset, -1);
            }
            common_speculative_accept(spec, s->seq_id, static_cast<uint16_t>(n_acc));
            s->pos += static_cast<llama_pos>(accepted.size());
            s->last = accepted.back();
            s->pending = std::move(accepted);
        }
    }

    void step_mtp(std::vector<Slot*>& active) {
        char tag[48];
        std::snprintf(tag, sizeof(tag), "salm.step_mtp active=%zu", active.size());
        ggml_nvtx::range nvtx_r(tag);
        std::sort(active.begin(), active.end(), [](const Slot* a, const Slot* b) {
            return a->seq_id < b->seq_id;
        });
        {
            ggml_nvtx::range r("mtp.draft");
            draft_mtp(active);
        }
        size_t n = 0;
        append_verify(active, n);
        if (n > static_cast<size_t>(cfg.n_batch))
            throw std::runtime_error("salm: verify batch exceeds n_batch");
        tok_batch.n_tokens = static_cast<int32_t>(n);
#if SALM_FAST_PATH
        // the one MTP verify decode: its rows stay uncommitted until the next verify accepts them;
        // every other decode (prefill) commits its rows
        llama_set_mtp_verify(ctx, true);
        try {
            decode(tok_batch);
        }
        catch (...) {
            llama_set_mtp_verify(ctx, false);
            throw;
        }
        llama_set_mtp_verify(ctx, false);
#else
        decode(tok_batch);
#endif
        std::vector<Slot*> slot_of(cfg.max_sequences, nullptr);
        for (Slot* s : active) slot_of[s->seq_id] = s;
        spec_process(slot_of);
        {
            ggml_nvtx::range r("mtp.accept");
            accept_mtp(active);
        }
    }

    // Persistent in-flight batching over a live request source (see SalmEngine::serve).
    // Request rows are recycled once their result has been handed out, so a long
    // serve() holds only in-flight requests.
    void serve(const SalmPull& pull, const SalmDone& done, SalmStats* stats) {
        ggml_nvtx::range run_range("salm.run");
        SalmStats st;
        timestamp_ready.clear();
        finished_reqs.clear();
        llama_memory_clear(llama_get_memory(ctx), true);
        if (spec)
            llama_memory_clear(llama_get_memory(ctx_dft), true);
        reset_prompt_cache();
        kv_reserved = 0;
        for (auto& [row, p] : preempted)
            if (p.sampler)
                llama_sampler_free(p.sampler);
        preempted.clear();
        reencode.clear();
        preempted_ready.clear();
        reencoding = kv_resuming = preemptions = 0;
        admissions = 0;
        row_prompt.clear();
        row_budget.clear();
        {  // tickets posted between serve() calls belonged to the previous one
            std::lock_guard<std::mutex> lock(cancel_mutex);
            cancel_posted_tickets.clear();
            cancel_posted.store(false, std::memory_order_relaxed);
        }
        std::vector<Slot> slots(cfg.max_sequences);
        for (int i = 0; i < cfg.max_sequences; i++)
            slots[i].seq_id = i;  // fixed ids; lowest free id is reused first
        std::vector<Slot*> active;
        const int group = std::max(1, cfg.max_sequences / kAdmissionGroups);

        std::vector<SalmRequest> reqs;
        std::vector<SalmResult> results;
        std::vector<size_t> tickets, free_rows;
        // per row: holds a request (pulled, result not yet delivered); SalmEngine::cancel reached
        // it
        std::vector<char> row_live, row_cancel;
        std::deque<size_t> waiting;  // pulled, not yet encoded
        std::vector<SalmRequest> incoming;
        size_t next_ticket = 0;
        bool open = true;
        auto take = [&](size_t max, bool block) {
            if (!open || max == 0)
                return;
            incoming.clear();
            open = pull(incoming, max, block);
            for (auto& r : incoming) {
                size_t i;
                if (free_rows.empty()) {
                    i = reqs.size();
                    reqs.emplace_back();
                    results.emplace_back();
                    tickets.push_back(0);
                    row_prompt.emplace_back();
                    row_budget.push_back(0);
                    row_live.push_back(0);
                    row_cancel.push_back(0);
                } else {
                    i = free_rows.back();
                    free_rows.pop_back();
                }
                row_live[i] = 1;
                row_cancel[i] = 0;
                results[i] = SalmResult{};
                results[i].id = r.id;
                results[i].audio_seconds = audio_seconds_of(r);
                row_budget[i] = r.max_new_tokens;
                tickets[i] = next_ticket++;
                try {
                    if (!(r.temperature >= 0) || !std::isfinite(r.temperature))
                        throw std::invalid_argument("temperature must be a finite number >= 0");
                    row_prompt[i] = r.prompt.empty() ? default_prompt : get_prompt(r.prompt);
                }
                catch (const std::exception& e) {  // this request fails; the others continue
                    results[i].error = e.what();
                    results[i].invalid_request =
                        dynamic_cast<const std::invalid_argument*>(&e) != nullptr;
                    row_prompt[i].reset();
                    reqs[i] = SalmRequest{};
                    finished_reqs.push_back(i);
                    continue;
                }
                reqs[i] = std::move(r);
                waiting.push_back(i);
            }
        };
        auto flush = [&]() {
            for (size_t i : finished_reqs) {
                done(tickets[i], std::move(results[i]));
                reqs[i] = SalmRequest{};
                row_prompt[i].reset();
                row_live[i] = 0;
                free_rows.push_back(i);
            }
            finished_reqs.clear();
        };
        // SalmEngine::cancel: a dropped request's result says so; it is delivered at the next
        // flush. One log line per kind and loop iteration.
        auto drop = [&](size_t i) {
            if (const auto it = preempted.find(i); it != preempted.end()) {
                kv_resuming -= it->second.kv_share;  // a preempted request waiting to resume
                if (it->second.sampler)
                    llama_sampler_free(it->second.sampler);
                preempted.erase(it);
            }
            cancel_result(results[i]);
            finished_reqs.push_back(i);
            ++cancelled_total;
        };
        auto log_cancel = [&](size_t n, const char* where) {
            std::fprintf(
                stderr,
                "[salm] cancelled %zu request(s) %s; active %zu/%d slots, KV cells reserved "
                "%zu/%u; %zu cancelled in total\n",
                n, where, active.size(), cfg.max_sequences, kv_reserved, kv_pool, cancelled_total);
        };

        // Encoder pipeline: utterances are encoded in groups of `group` (batched
        // windows), in a background thread on the encoder's own stream when
        // overlap is enabled, and queued until a decode slot is free.
        std::deque<Encoded> ready;
        // Requests whose response ended empty go back to the front of the queue for their fallback
        // decode (their encoder output is reused).
        fallback_ready.clear();
        auto requeue_fallbacks = [&]() {
            for (auto it = fallback_ready.rbegin(); it != fallback_ready.rend(); ++it)
                ready.push_front(std::move(*it));
            fallback_ready.clear();
        };
        // Fast path: two encoder calls in flight: one runs its GPU batch while the other does
        // feature extraction and window fusion on the host (the PEE batch itself is serialized
        // inside PeeModel).
        const size_t max_jobs = fast ? 2 : 1;
        std::deque<std::future<std::vector<Encoded>>> aheads;
        std::deque<double> ahead_seconds;  // audio seconds of each call in aheads
        // Preempted requests go back to the front of the queue, oldest first.
        auto requeue_front = [&](std::vector<Encoded>& encs) {
            std::stable_sort(encs.begin(), encs.end(), [&](const Encoded& a, const Encoded& b) {
                return tickets[a.req] < tickets[b.req];
            });
            for (auto it = encs.rbegin(); it != encs.rend(); ++it) ready.push_front(std::move(*it));
            encs.clear();
        };
        // A diarization-only request ends at its encoder output: the 10 ms speaker turns (NeMo's
        // diarization_labels, activity > 0.5), as a timestamp request returns them.
        auto finish_diarization = [&](Encoded& e) {
            SalmResult& r = results[e.req];
            r.audio_tokens = e.frames;
            const int n_spk = pee.cfg().num_speakers;
            if (!e.error.empty()) {
                r.error = std::move(e.error);
                r.invalid_request = e.invalid_request;
            } else if (!e.timestamps || n_spk <= 0) {
                r.error = "speaker diarization needs the CTC timestamp head";
            } else {
                constexpr double kLabelSeconds = 0.01;  // NeMo's diarization_frame_seconds
                const auto& p10 = e.timestamps->probs10;
                auto details = std::make_shared<asr::SpeechLmDetails>();
                for (const auto& s : timestamps::diarization_segments(
                         p10.data(), static_cast<int>(p10.size() / n_spk), n_spk, kLabelSeconds,
                         r.audio_seconds))
                    details->diarization.push_back(
                        {s.speaker, timestamps::round_decimals(s.start, 3),
                         timestamps::round_decimals(s.end, 3)});
                details->diarization_frame_seconds = kLabelSeconds;
                r.details = std::move(details);
            }
            finished_reqs.push_back(e.req);
        };
        // A request without a voice ends at its encoder output with an empty transcript: no
        // decode, no fallback and no token counts (no usage, as for near-silent audio); word
        // timestamps report it as they do a decoded empty transcript.
        auto finish_no_speech = [&](Encoded& e) {
            if (e.timestamps)
                timestamp_ready.emplace_back(e.req, std::move(e.timestamps));
            else
                finished_reqs.push_back(e.req);
        };
        // An encoder call's output into the queue: new requests at the back, preempted ones (their
        // share no longer held back from admission) at the front; diarization-only and voiceless
        // requests end.
        auto deliver = [&](std::vector<Encoded>&& encs) {
            std::vector<Encoded> again;
            size_t dropped = 0;
            for (auto& e : encs) {
                const auto it = preempted.find(e.req);
                if (it != preempted.end()) {
                    --reencoding;
                    kv_resuming -= it->second.kv_share;
                    it->second.kv_share = 0;
                }
                if (row_cancel[e.req]) {  // cancelled during the call: its output is discarded
                    drop(e.req);
                    ++dropped;
                } else if (reqs[e.req].diarization_only) {
                    finish_diarization(e);
                } else if (
                    it == preempted.end() && e.error.empty() && !e.speech &&
                    !reqs[e.req].decode_without_voice) {
                    finish_no_speech(e);
                } else if (it == preempted.end()) {
                    ready.push_back(std::move(e));
                } else {
                    again.push_back(std::move(e));
                }
            }
            if (dropped)
                log_cancel(dropped, "at the end of their encoder call");
            requeue_front(again);
        };
        // An encoder call over request rows (`again`: preempted requests, whose word-timestamp
        // evidence is kept from their first pass).
        auto submit = [&](const std::vector<size_t>& rows, bool again) {
            auto batch = std::make_shared<std::vector<SalmRequest>>();
            double batch_seconds = 0;
            for (size_t row : rows) {
                batch_seconds += audio_seconds_of(reqs[row]);
                batch->push_back(reqs[row]);
                if (again)
                    batch->back().timestamps = false;
            }
            auto run = [this, batch, rows]() {
                auto enc = encode(*batch, 0, batch->size());
                for (auto& e : enc) e.req = rows[e.req];
                return enc;
            };
            if (overlap_encoder) {
                aheads.emplace_back(std::async(std::launch::async, run));
                ahead_seconds.push_back(batch_seconds);
            } else {
                deliver(run());
            }
        };
        auto launch = [&]() {
            // preempted requests first, whatever is queued (their number bounds the call)
            if (!reencode.empty() && aheads.size() < max_jobs &&
                (overlap_encoder || aheads.empty())) {
                const std::vector<size_t> rows = std::move(reencode);
                reencode.clear();
                submit(rows, true);
            }
            while (aheads.size() < max_jobs) {
                if (ready.size() >=
                    static_cast<size_t>(group) * aheads.size() + (aheads.empty() ? group : 0))
                    return;
                // Larger encoder calls give the length buckets more same-length
                // windows to batch; the call blocks decoding for its duration.
                const bool cold = active.empty() && ready.empty();
                const size_t want = std::max<size_t>(
                    kEncoderAhead,
                    cold ? static_cast<size_t>(cfg.max_sequences) : static_cast<size_t>(group));
                if (waiting.size() < want)
                    take(want - waiting.size(), false);
                size_t count = std::min(want, waiting.size());
                if (count == 0)
                    return;
                if (!overlap_encoder && !aheads.empty())
                    return;
                // Keep the audio encoded ahead of admission (ready queue + calls in flight) within
                // encoder_ahead_seconds: its projection stays on the device until admitted. With
                // nothing queued or encoding, the first request goes regardless of its length.
                if (cfg.encoder_ahead_seconds > 0) {
                    double queued = 0;
                    for (const Encoded& e : ready) queued += e.audio_seconds;
                    for (double s : ahead_seconds) queued += s;
                    const bool idle = ready.empty() && aheads.empty();
                    size_t fit = 0;
                    for (double s = queued; fit < count; ++fit) {
                        s += audio_seconds_of(reqs[waiting[fit]]);
                        if (s > cfg.encoder_ahead_seconds && !(idle && fit == 0))
                            break;
                    }
                    if (fit == 0)
                        return;
                    count = fit;
                }
                std::vector<size_t> rows;
                for (size_t k = 0; k < count; ++k) {
                    rows.push_back(waiting.front());
                    waiting.pop_front();
                }
                submit(rows, false);
                if (!overlap_encoder)
                    return;
            }
        };
        auto harvest = [&](bool block) {
            while (!aheads.empty()) {
                auto& front = aheads.front();
                if (!block && front.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                    return;
                deliver(front.get());
                aheads.pop_front();
                ahead_seconds.pop_front();
                block = false;  // after the awaited job, take only finished ones
            }
        };

        // Word timestamps (the CTC head's projection of the transcript's classes + forced
        // alignment) run beside the decode loop on their own thread, one job at a time over the
        // requests finished since the last job (one batched projection for all of them); the loop
        // delivers them when the job is done. The CTC head has its own GPU lane.
        struct Aligned {
            size_t req = 0;
            std::shared_ptr<asr::SpeechLmDetails> details;
            std::string error;
        };
        struct AlignJob {
            std::vector<Aligned> done;
        };
        std::optional<std::future<AlignJob>> aligning;
        auto start_timestamps = [&]() {
            if (aligning || timestamp_ready.empty())
                return;
            struct Input {
                size_t req;
                std::shared_ptr<TimestampInput> evidence;
                std::string text;
                double seconds;
            };
            std::vector<Input> inputs;
            for (auto& [req, evidence] : timestamp_ready)
                inputs.push_back(
                    {req, std::move(evidence), results[req].text, results[req].audio_seconds});
            timestamp_ready.clear();
            aligning = std::async(std::launch::async, [this, inputs = std::move(inputs)]() {
                ggml_nvtx::range timestamp_range("salm.timestamps");
                const int n_spk = pee.cfg().num_speakers;
                std::vector<TimestampEvidence> evidence;
                std::vector<std::string> texts;
                std::vector<asr::SpeechLmDetails*> details;
                AlignJob job;
                for (const auto& in : inputs) {
                    const auto& ts = *in.evidence;
                    TimestampEvidence ev;
                    ev.head = ts.head;
                    ev.encoder_frames = ts.frames;
                    ev.activity = ts.probs80.data();
                    ev.activity_frames = static_cast<int>(ts.probs80.size() / n_spk);
                    ev.activity10 = ts.probs10.data();
                    ev.activity10_frames = static_cast<int>(ts.probs10.size() / n_spk);
                    ev.speakers = n_spk;
                    ev.duration_seconds = in.seconds;
                    evidence.push_back(ev);
                    texts.push_back(in.text);
                    job.done.push_back({in.req, std::make_shared<asr::SpeechLmDetails>(), {}});
                    details.push_back(job.done.back().details.get());
                }
                // one head pass for the whole job; a failed alignment does not throw
                auto errors = pee.ctc()->word_timestamps(evidence, texts, details);
                for (size_t i = 0; i < errors.size(); ++i) job.done[i].error = std::move(errors[i]);
                return job;
            });
        };
        // Deliver a finished job (waiting up to `wait` for it) and start the next.
        auto harvest_timestamps = [&](std::chrono::milliseconds wait) {
            if (aligning && aligning->wait_for(wait) == std::future_status::ready) {
                AlignJob job = aligning->get();
                aligning.reset();
                size_t dropped = 0;
                for (auto& a : job.done) {
                    if (row_cancel[a.req]) {  // cancelled during the alignment
                        drop(a.req);
                        ++dropped;
                        continue;
                    }
                    results[a.req].details = std::move(a.details);
                    results[a.req].timestamp_error = std::move(a.error);
                    finished_reqs.push_back(a.req);
                }
                if (dropped)
                    log_cancel(dropped, "at the end of their word alignment");
            }
            start_timestamps();
        };
        // The posted cancellations, at a step boundary (finished_reqs was just flushed): a request
        // waiting for the encoder (also a preempted one), for a slot or for its alignment is
        // dropped now; a decoding one releases its slot (commit); one inside an encoder call or the
        // alignment job is marked and dropped when that job is harvested.
        auto process_cancels = [&]() {
            ggml_nvtx::range r("salm.loop.cancel");
            std::vector<size_t> posted;
            {
                std::lock_guard<std::mutex> lock(cancel_mutex);
                posted.swap(cancel_posted_tickets);
                cancel_posted.store(false, std::memory_order_relaxed);
            }
            size_t queued = 0, in_job = 0, cells = 0;
            std::vector<Slot*> stop;
            for (const size_t ticket : posted) {
                size_t i = 0;
                while (i < reqs.size() && !(row_live[i] && tickets[i] == ticket)) ++i;
                if (i == reqs.size() || row_cancel[i])
                    continue;  // already delivered, or already cancelled
                row_cancel[i] = 1;
                if (auto it = std::find(waiting.begin(), waiting.end(), i); it != waiting.end()) {
                    waiting.erase(it);
                    drop(i);
                    ++queued;
                    continue;
                }
                if (auto it = std::find(reencode.begin(), reencode.end(), i);
                    it != reencode.end()) {
                    reencode.erase(it);  // preempted, not yet back in the encoder
                    --reencoding;
                    drop(i);
                    ++queued;
                    continue;
                }
                if (auto it = std::find_if(
                        ready.begin(), ready.end(), [i](const Encoded& e) { return e.req == i; });
                    it != ready.end()) {
                    ready.erase(it);  // frees its projection
                    drop(i);
                    ++queued;
                    continue;
                }
                if (auto it = std::find_if(
                        timestamp_ready.begin(), timestamp_ready.end(),
                        [i](const auto& t) { return t.first == i; });
                    it != timestamp_ready.end()) {
                    timestamp_ready.erase(it);
                    drop(i);
                    ++queued;
                    continue;
                }
                if (auto it = std::find_if(
                        active.begin(), active.end(), [i](const Slot* a) { return a->req == i; });
                    it != active.end()) {
                    (*it)->cancelled = true;
                    cells += (*it)->kv_cells;
                    stop.push_back(*it);
                    continue;
                }
                ++in_job;  // inside an encoder call or the alignment job
            }
            if (!stop.empty()) {
                // A prepared decode step keeps the released lanes as zombies, as for finished ones.
                active = commit(stop, active, results);
                cancelled_total += stop.size();
            }
            if (queued)
                log_cancel(queued, "before decoding");
            if (!stop.empty()) {
                char where[96];
                std::snprintf(
                    where, sizeof(where), "while decoding (slots and %zu KV cells released)",
                    cells);
                log_cancel(stop.size(), where);
            }
            if (in_job)
                std::fprintf(
                    stderr,
                    "[salm] %zu cancelled request(s) end with their encoder call or word "
                    "alignment\n",
                    in_job);
        };
        for (;;) {
            {
                ggml_nvtx::range r("salm.loop.flush");
                harvest_timestamps(std::chrono::milliseconds(0));
                flush();
            }
            // Tickets posted by SalmEngine::cancel: one atomic load per iteration, no lock.
            if (cancel_posted.load(std::memory_order_acquire)) {
                process_cancels();
                flush();
            }
            // keep the encoder busy one group ahead. Below half the slots (fast path), wait for the
            // encoder instead of stepping a few sequences: a decode step costs nearly the same at
            // 100 or 700 rows (expert weights are read once per step), and waiting leaves the GPU
            // to the encoder.
            const double min_fill = fast ? 0.5 : 0.0;
            // a prepared decode step is launched next: no blocking waits, no admission before it
            const bool pend_now = pend_ready;
            const bool starved = !pend_now && ready.empty() && !aheads.empty() &&
                                 static_cast<double>(active.size()) < min_fill * cfg.max_sequences;
            {
                ggml_nvtx::range r("salm.loop.harvest");
                const bool cold = active.empty() && ready.empty();
                // nothing to decode while preempted requests are encoded again: wait for them
                const bool resuming = active.empty() && reencoding > 0;
                harvest(/*block=*/!pend_now && (cold || starved || resuming));
            }
            {
                ggml_nvtx::range r("salm.loop.launch");
                launch();
            }
            if (active.empty() && ready.empty() && waiting.empty() && aheads.empty() && !pend_now &&
                reencode.empty()) {
                // only word timestamps in flight (also queued ones, such as a voiceless request's):
                // keep taking new work meanwhile
                if (aligning || !timestamp_ready.empty()) {
                    take(cfg.max_sequences, /*block=*/false);
                    if (waiting.empty())
                        harvest_timestamps(std::chrono::milliseconds(1));
                    continue;
                }
                if (!open)
                    break;
                flush();  // diarization-only results end without a decode step
                take(cfg.max_sequences, /*block=*/true);  // idle: wait for work or closure
                continue;
            }

            // ---- admit encoded utterances into free slots (lowest ids first), prefill together
            std::vector<Slot*> fresh;
            std::vector<const Encoded*> fresh_enc;
            std::vector<Encoded> taken;
            const int free_slots = cfg.max_sequences - static_cast<int>(active.size());
            // admit when slots are free and either the decoder is idle, a group's worth
            // of slots is free, or the queue can fill every free slot
            const bool admit_now = !pend_now && !ready.empty() && free_slots > 0 &&
                                   (active.empty() || free_slots >= group);
            if (admit_now) {
                ++admission_round;
                // Take ready requests in order while each gets a slot and its KV cells from the
                // shared pool (fast path: its prompt, the output a preempted request had generated
                // and a first share of the budget, leaving every sequence in flight one more share;
                // other paths: the worst case); the rest wait for a later round. Prompt prefixes
                // come from the prompt cache where it has room.
                std::vector<int> budgets;
                std::vector<size_t> cells, fulls;
                size_t kv_round = 0, kv_spare = 0;
                if (fast)
                    for (const Slot* a : active)
                        kv_spare += std::min(kKvGrowCells, a->kv_full - a->kv_cells);
                int n_admit = 0;
                while (!ready.empty() && n_admit < free_slots) {
                    Encoded& e = ready.front();
                    SalmResult& r = results[e.req];
                    const auto& prompt = row_prompt[e.req];
                    r.audio_tokens = e.frames;
                    r.prompt_tokens =
                        static_cast<int>(prompt->prefix.size() + prompt->suffix.size()) + e.frames;
                    int budget = 0;
                    size_t need = 0, full = 0;
                    if (e.error.empty()) {
                        const int room =
                            static_cast<int>(n_ctx_seq) - r.prompt_tokens - 1 - cfg.mtp_draft;
                        if (room < 1) {
                            e.error = "audio too long: the prompt takes " +
                                      std::to_string(r.prompt_tokens) +
                                      " tokens, the context holds " + std::to_string(n_ctx_seq);
                            e.invalid_request = true;
                        } else {
                            budget = row_budget[e.req] > 0
                                         ? row_budget[e.req]
                                         : std::min(
                                               cfg.max_new_tokens,
                                               32 + static_cast<int>(r.audio_seconds * 10.0));
                            budget = std::min(budget, room);
                        }
                    }
                    if (e.error.empty() && fast) {
                        const auto rec = preempted.find(e.req);
                        const size_t resumed =
                            rec == preempted.end() ? 0 : rec->second.tokens.size();
                        const size_t prompt = static_cast<size_t>(r.prompt_tokens);
                        // the worst case fits the context (room), and so the pool
                        full = prompt + static_cast<size_t>(budget) + cfg.mtp_draft + 1;
                        need = std::min(full, prompt + resumed + kv_step_rows() + kKvGrowCells);
                        if (kv_cached + full > kv_pool && kv_reserved + kv_round == 0)
                            drop_prompt_cache();  // its cells go to the request
                        if (kv_cached + full > kv_pool) {
                            if (kv_reserved + kv_round > 0)
                                break;  // waits for cells held by other sequences
                            e.error = "audio too long for the KV pool (" + std::to_string(full) +
                                      " of " + std::to_string(kv_pool - kv_cached) + " cells)";
                            e.invalid_request = true;
                            need = full = 0;
                        } else if (
                            kv_cached + kv_reserved + kv_resuming + kv_round + need + kv_spare >
                            kv_pool) {
                            break;  // waits for cells held by other sequences
                        } else {
                            kv_spare += std::min(kKvGrowCells, full - need);
                        }
                    } else if (e.error.empty()) {
                        // other paths reserve the worst case now (no growth or preemption); it fits
                        // the context, and so the pool, alone
                        full = need = static_cast<size_t>(r.prompt_tokens) +
                                      static_cast<size_t>(budget) + cfg.mtp_draft + 1;
                        if (kv_reserved + kv_round + need > kv_pool)
                            break;  // waits for cells held by other sequences
                    }
                    if (e.error.empty())
                        ++n_admit;
                    budgets.push_back(budget);
                    cells.push_back(need);
                    fulls.push_back(full);
                    kv_round += need;
                    taken.push_back(std::move(e));
                    ready.pop_front();
                }
                // the prompts of this round and their requests, for the prompt-cache plan
                std::vector<std::pair<std::shared_ptr<Prompt>, int>> round_prompts;
                for (const Encoded& e : taken) {
                    if (!e.error.empty() || !fast)
                        continue;
                    const auto& prompt = row_prompt[e.req];
                    auto it = std::find_if(
                        round_prompts.begin(), round_prompts.end(),
                        [&](const auto& u) { return u.first == prompt; });
                    if (it == round_prompts.end())
                        round_prompts.push_back({prompt, 1});
                    else
                        ++it->second;
                }
                // the ready queue just drained: start the next encoder jobs now so they run during
                // this prefill
                launch();
                std::map<const Prompt*, std::shared_ptr<CacheEntry>> cache_plan;
                if (fast)
                    cache_plan = plan_prompt_cache(round_prompts);
                // With pinned recurrent cells (fast path) ids keep their holes: admit into the
                // lowest free ids.
                const bool pinned_ids = fast;
                std::vector<char> id_used(cfg.max_sequences, 0);
                for (const Slot* a : active) id_used[a->seq_id] = 1;
                int next_free = 0;
                std::vector<std::pair<Slot*, Encoded*>> keep_encoded;  // fallback candidates
                for (size_t k = 0; k < taken.size(); ++k) {
                    Encoded& e = taken[k];
                    SalmResult& r = results[e.req];
                    auto rec = preempted.find(e.req);
                    if (!e.error.empty() && rec != preempted.end()) {  // its encoder pass failed
                        if (rec->second.sampler)
                            llama_sampler_free(rec->second.sampler);
                        preempted.erase(rec);
                        rec = preempted.end();
                    }
                    if (!e.error.empty() && r.first_attempt) {
                        // the fallback decode does not fit: the first decode's result stands
                        r.audio_tokens = r.first_attempt->audio_tokens;
                        r.prompt_tokens = r.first_attempt->prompt_tokens;
                        r.generated_tokens = r.first_attempt->generated_tokens;
                        r.end_of_turn = r.first_attempt->end_of_turn;
                        r.first_attempt.reset();
                        finished_reqs.push_back(e.req);
                        continue;
                    }
                    if (!e.error.empty()) {
                        r.error = e.error;
                        r.invalid_request = e.invalid_request;
                        finished_reqs.push_back(e.req);
                        continue;
                    }
                    Slot* s = nullptr;
                    for (auto& cand : slots)
                        if (!cand.active) {
                            s = &cand;
                            break;
                        }
                    s->active = true;
                    if (pinned_ids) {
                        while (id_used[next_free]) ++next_free;
                        id_used[next_free] = 1;
                        s->seq_id = next_free;
                    } else {
                        s->seq_id = static_cast<int>(active.size() + fresh.size());
                    }
                    s->req = e.req;
                    s->prompt = row_prompt[e.req];
                    if (auto it = cache_plan.find(s->prompt.get()); it != cache_plan.end())
                        s->cached = it->second;
                    s->budget = budgets[k];
                    s->kv_cells = cells[k];
                    s->kv_full = fulls[k];
                    kv_reserved += cells[k];
                    s->admitted = ++admissions;
                    s->timestamps = std::move(e.timestamps);
                    s->temperature = reqs[e.req].temperature;
                    s->seed = reqs[e.req].seed;
                    if (rec != preempted.end() && rec->second.sampler) {
                        attach_sampler(s->seq_id, rec->second.sampler, false);
                        rec->second.sampler = nullptr;
                    } else {
                        bind_sampler(s);
                    }
                    s->latin_script = reqs[e.req].latin_script;
                    s->on_text = reqs[e.req].on_text;
                    s->streamed = 0;
                    s->fallback.reset();
                    s->fallback_prompt.reset();
                    if (!reqs[e.req].fallback_prompt.empty() && !reqs[e.req].timestamps &&
                        !r.first_attempt && (rec == preempted.end() || rec->second.fallback)) {
                        try {
                            s->fallback_prompt = get_prompt(reqs[e.req].fallback_prompt);
                            keep_encoded.push_back({s, &e});
                        }
                        catch (const std::exception&) {  // no fallback for this request
                        }
                    }
#if SALM_FAST_PATH
                    if (script_mask)
                        llama_set_seq_token_mask(ctx, s->seq_id, s->latin_script);
#endif
                    s->loop.reset();
                    s->pos = 0;
                    s->n_generated = 0;
                    s->ended = false;
                    s->text.clear();
                    s->pending.clear();
                    s->tokens.clear();
                    s->resume.clear();
                    fresh.push_back(s);
                    fresh_enc.push_back(&e);
                    if (rec != preempted.end()) {
                        // a preempted request continues where it stopped (its usage counted once)
                        Preempted& p = rec->second;
                        s->resume = p.tokens;
                        s->tokens = std::move(p.tokens);
                        s->text = std::move(p.text);
                        s->streamed = p.streamed;
                        s->n_generated = p.n_generated;
                        s->loop = std::move(p.loop);
                        if (p.timestamps)
                            s->timestamps = std::move(p.timestamps);
                        preempted.erase(rec);
                    }
                }
                if (!fresh.empty()) {
                    // prefill batches need increasing ids within a batch
                    std::vector<size_t> ord(fresh.size());
                    for (size_t i = 0; i < ord.size(); i++) ord[i] = i;
                    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
                        return fresh[a]->seq_id < fresh[b]->seq_id;
                    });
                    std::vector<Slot*> fs;
                    std::vector<const Encoded*> fe;
                    for (size_t i : ord) {
                        fs.push_back(fresh[i]);
                        fe.push_back(fresh_enc[i]);
                    }
                    prefill(fs, fe, active);
                    // prefill has read the encoder outputs: fallback candidates keep theirs
                    for (auto& [slot, e] : keep_encoded) slot->fallback = std::move(*e);
                    std::vector<Slot*> merged = active;
                    merged.insert(merged.end(), fs.begin(), fs.end());
                    active = commit(fs, merged, results);
                    requeue_fallbacks();
                    st.refills++;
                    st.peak_sequences =
                        std::max(st.peak_sequences, static_cast<int>(active.size()));
                }
                // nothing fit (KV cells or prompt-cache sequences are held by active sequences):
                // decode a step
                if (!taken.empty())
                    continue;
            }
            if (active.empty() && !pend_now)
                continue;  // waiting for the encoder
            // prepare the following step during this one only when no refill is close (it would
            // wait a step): fewer free slots than a refill group, with a margin
            constexpr int kPrepareMargin = 40;
            prepare_ok =
                cfg.max_sequences - static_cast<int>(active.size()) + kPrepareMargin < group;
            // ---- one decode step for all active sequences
            if (spec)
                step_mtp(active);
            else
                step(active);
            active = commit(active, active, results);
            // the sequences' next steps get their cells (fast path), preempting the newest ones
            // when the pool is short
            const std::vector<size_t> evicted = grow_kv(active);
            for (size_t k = 0; k < evicted.size(); ++k) {
                const size_t nth = preemptions - evicted.size() + k + 1;
                if (nth == 1 || nth % 50 == 0)
                    std::fprintf(
                        stderr,
                        "[salm] KV pool full: preempted %s after %zu output tokens (%zu "
                        "preemptions); it is encoded and prefilled again. A larger "
                        "asr.salm.n_ctx_per_sequence avoids the recomputation.\n",
                        reqs[evicted[k]].id.empty() ? "a request" : reqs[evicted[k]].id.c_str(),
                        preempted.at(evicted[k]).tokens.size(), nth);
            }
            requeue_front(preempted_ready);
            requeue_fallbacks();
        }
        while (aligning || !timestamp_ready.empty())
            harvest_timestamps(std::chrono::milliseconds(1));
        flush();
        if (stats)
            *stats = st;
    }
};

SalmEngine::SalmEngine(SalmConfig cfg) : impl_(std::make_unique<Impl>(std::move(cfg))) {}
SalmEngine::~SalmEngine() = default;

void
SalmEngine::serve(const SalmPull& pull, const SalmDone& done, SalmStats* stats) {
    impl_->serve(pull, done, stats);
}

void
SalmEngine::cancel(size_t ticket) {
    std::lock_guard<std::mutex> lock(impl_->cancel_mutex);
    impl_->cancel_posted_tickets.push_back(ticket);
    impl_->cancel_posted.store(true, std::memory_order_release);
}

const SalmConfig&
SalmEngine::config() const {
    return impl_->cfg;
}

bool
SalmEngine::batched() const {
    return impl_->fast;
}

int
SalmEngine::sample_rate() const {
    return impl_->pee.cfg().sample_rate;
}

std::string
SalmEngine::special_token_in(const std::string& text) const {
    // The model's own transcript vocabulary, which instructions name (the speaker prompt does).
    static const std::regex transcript_tag(R"(<spk:\d+>|<ev:[a-z_]+>|<sty:(?:stt|end):[a-z_]+>)");
    const llama_vocab* vocab = impl_->vocab;
    for (const llama_token t : tokenize(vocab, text)) {
        if (!(llama_vocab_get_attr(vocab, t) &
              (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)))
            continue;
        const std::string token = llama_vocab_get_text(vocab, t);
        if (token != policy::kAudioLocator && !std::regex_match(token, transcript_tag))
            return token;
    }
    return {};
}

}  // namespace nemo_speech::speechlm
