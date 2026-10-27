// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Parallel Expert Encoder (PEE): Transformer-AED ASR expert + 8-speaker
// Sortformer expert on a shared mel, fused into one (d_model, T') encoder
// state stream for a speech-LM decoder. Port of NeMo's
// `ParallelExpertEncoderPT` (asr_encoder_type: transformer), GGUF arch "pee"
// (conversion/pee.py).
//
// Graph (one Session, batch of B same-length windows):
//
//   input.mel_asr (128, T_mel, B) per-feature normalized -> asr RoPE transformer
//                                                          => asr states (1280, T', B)
//   input.mel     (128, T_mel, B) raw                     -> diar RoPE transformer
//      -> encoder_proj 512->192 -> subpixel_upsample x8 (Conv1d k3) -> relu
//      -> Linear -> relu -> Linear(8) -> sigmoid          => probs (8, 8*T', B) @10ms
//
// Long-form audio follows ParallelExpertEncoder._forward_online for the ASR
// branch: fixed windows of `online_inference_length` output frames with
// left/right context, overlap-and-trim in output-frame space. Windows are the
// unit of scheduling: every window of every utterance is the same kind of
// job, and same-length windows batch together (encode_batch), so a long file
// never holds a short one hostage beyond one batch.
//
// Speaker branch across windows: NeMo's streaming Sortformer (sync mode). Window
// 0 is an offline pass; every later window runs on [speaker cache | FIFO |
// window] pre-encoded rows, and the cache/FIFO are updated (and the cache
// compressed) after each window, so speaker columns keep their identity across
// the whole file. With a GPU backend the state and its update/compression stay
// on the device inside each step's graph, so the steps of a file run back to
// back without host synchronization. Single-window utterances take the one fused
// graph; longer ones run speaker steps (batched across utterances at the same
// window index) and then the ASR/fusion graph with those speaker probabilities.
//
// Everything after the mel is on the graph: per-utterance per-feature
// normalization of the ASR input (host supplies 128 mean / inv-std pairs),
// mask-weighted avg-pool of the probs to the 80 ms grid (exact for partial
// last windows), threshold, LayerNorm(8), @ sinusoid kernel (8, 1280),
// + LayerNorm(1280)(asr states). The host only trims window cores and
// concatenates; fusion is per-frame so it commutes with overlap-and-trim.
#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ctc_timestamp.h"
#include "nn.h"
#include "pee_rope_transformer.h"
#include "runtime.h"

namespace nemo_speech::speechlm {

struct PeeConfig {
    nemo_speech::speechlm::RopeTransformerConfig asr;
    nemo_speech::speechlm::RopeTransformerConfig diar;
    int num_speakers = 8;
    int tf_d_model = 192;
    int upsample_factor = 8;
    float speaker_activity_threshold = 0.5f;
    std::string asr_normalize_type = "per_feature";
    // SALM modality projection fusion.proj (d_model -> LLM hidden); 0 = absent.
    int proj_dim = 0;
    // Long-form windowing, in output (80 ms) frames.
    int online_inference_length = 500;
    int chunk_left_context = 50;
    int chunk_right_context = 50;

    // FE (from GGUF pee.preprocessor.*)
    int sample_rate = 16000;
    float window_size = 0.025f;
    float window_stride = 0.01f;
    int n_fft = 512;
    int n_mels = 128;
};

// Per-run inputs (by name): input.mel (n_mels, T_mel, B) raw log-mel,
// input.mean / input.inv_std (n_mels, 1, B) per-utterance ASR normalization,
// input.valid (sub*T', 1, 1) 1.0 for 10 ms frames < T_mel, input.pos (T') i32.
// Output bag: [0] fused (d_model, T', B), [1] speaker probs @80 ms (n_spk, T', B),
// then asr states (d_model, T', B) only when built with want_asr_states, then
// proj (proj_dim, T', B) only when the GGUF carries fusion.proj; finally
// native speaker probabilities when requested by the timestamp path.
// With input.p80 (n_spk, T', B) the speaker branch is skipped and those 80 ms
// probabilities are fused (stateful windows); no native probabilities then.
// Speaker step (input.spk_mel (n_mels, T_mel, B) raw, optional input.spk_prev
// (d_diar, P, B) pre-encoded cache+FIFO rows, input.spk_pos (P+T') i32,
// optional input.spk_kmask (P+T', 1, 1, B)): [0] the window's pre-encoded rows
// (d_diar, T', B), [1] 80 ms probs (n_spk, P+T', B), [2] 10 ms probs
// (n_spk, 8(P+T'), B). Device-resident state (row lists as i32 inputs, per batch
// item): input.spk_out_rows (T', B) adds [3] the window's 80 ms probs on its own
// frames (n_spk, T', B; rows of [1] or zero) and input.spk_core_rows (T', B) adds
// [4] the core's 10 ms probs (8 n_spk, T', B); with input.spk_cache_rows (n, B),
// input.spk_cpred_rows (n, B), optional input.spk_fifo_rows (F', B) and
// input.spk_cpred (n_spk, S, B) the step also updates the state:
// [5] the next [cache | FIFO] rows (d_diar, S' + F', B), compressed to the
// cache length when n exceeds it, and [6] the cache's probs (n_spk, S', B).
class PeeGraph : public ggml_runtime::Module {
   public:
    // has_sil_emb: the GGUF carries diar.head.learnable_sil_emb (device speaker-state updates).
    PeeGraph(
        const PeeConfig& cfg, bool want_asr_states, bool want_native_probs = false,
        bool has_sil_emb = false);
    ~PeeGraph();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

   private:
    // Sortformer head: encoder output (d_diar, T, B) -> 10 ms speaker probabilities (n_spk, 8T, B).
    // kmask: the batch's additive key mask (T, 1, 1, B; 0 valid, -1e9 padding), null if unpadded.
    ggml_tensor* speaker_probs(
        ggml_runtime::Session* session, const ggml_runtime::ggml_bf_tensor& emb, ggml_tensor* kmask,
        ggml_runtime::TensorContainer* tc);
    ggml_runtime::TensorBag build_speaker_step(
        ggml_runtime::Session* session, ggml_runtime::TensorContainer* tc);
    // SortformerModules._compress_spkcache on the graph: n cache rows emb (d_diar, n, B) with
    // their 80 ms probs (n_spk, n, B) -> the cache length's rows and probs.
    std::pair<ggml_tensor*, ggml_tensor*> compress_cache(
        ggml_runtime::Session* session, ggml_context* ctx, ggml_tensor* emb, ggml_tensor* preds);

    PeeConfig cfg_;
    nemo_speech::speechlm::RopeTransformerEncoder* asr_;
    nemo_speech::speechlm::RopeTransformerEncoder* diar_;
    ggml_runtime::Linear* encoder_proj_;
    ggml_runtime::Conv1D* upsample_;
    ggml_runtime::Linear* head_hidden_;
    ggml_runtime::Linear* head_spks_;
    ggml_runtime::LayerNorm* asr_norm_;
    ggml_runtime::LayerNorm* diar_norm_;
    ggml_runtime::Linear* proj_ = nullptr;  // fusion.proj, only with cfg.proj_dim > 0
    bool want_asr_states_;
    bool want_native_probs_;
    bool has_sil_emb_;
};

class PeeModel {
   public:
    // want_asr_states: also return the pre-fusion ASR states (parity/debug;
    // costs an extra (d_model, T', B) download per batch). ctc_path attaches
    // the timestamp head, which takes the raw device states (EvidenceSink) without that download.
    PeeModel(
        ggml_runtime::BackendManager& bm, const std::string& gguf_path,
        bool want_asr_states = false, const std::string& ctc_path = "");
    ~PeeModel();

    const PeeConfig& cfg() const { return cfg_; }
    CtcTimestampModel* ctc() { return ctc_.get(); }
    // Trained mel filterbank (n_mels x (n_fft/2+1)) for the FE.
    const std::vector<float>& mel_basis() const { return mel_basis_; }
    int out_frames(int t_mel) const { return cfg_.asr.out_frames(t_mel); }
    // Max windows per graph run when encode() batches an utterance's windows.
    void set_max_batch(int b) { max_batch_ = b < 1 ? 1 : b; }
    // false: skip the fused-state download and stitching (Output::fused stays empty); for callers
    // that only use the projection / speaker outputs (SALM decoder input).
    void set_emit_fused(bool on) { emit_fused_ = on; }
    // true (GPU backend only): the projection stays on the device: BatchOut::proj_block instead of
    // a host download, Output::proj_spans instead of Output::proj.
    void set_proj_on_device(bool on) { proj_on_device_ = on && gpu_ != nullptr; }
    bool proj_on_device() const { return proj_on_device_; }
    // Graphs built for batch shapes that were not cached (diagnostics).
    size_t graph_misses() const { return session_->run_cache_misses(); }
    size_t graph_stale() const { return session_->run_cache_stale(); }

    // One context-extended window of an utterance (mel-frame slice and the
    // output-frame core kept after overlap-and-trim).
    struct Window {
        int enc_stt = 0;  // mel frame slice [enc_stt, enc_end)
        int enc_end = 0;
        int left_drop = 0;  // output frames dropped at the front
        int core = 0;       // output frames kept
        int right = 0;      // mel frames of right context
        int len() const { return enc_end - enc_stt; }
    };
    std::vector<Window> plan_windows(int t_mel) const;

    // Per-utterance ASR normalization statistics (n_mels mean and 1/(std+1e-5)).
    struct NormStats {
        std::vector<float> mean, inv_std;
    };
    NormStats norm_stats(const float* mel, int t_mel) const;

    // Host buffer that is page-locked (pinned) when a GPU backend is present, so
    // Session transfers run at DMA speed and do not stage through pageable memory.
    class HostBuffer {
       public:
        HostBuffer() = default;
        HostBuffer(ggml_backend_buffer_type_t host_buft, size_t bytes);
        ~HostBuffer();
        HostBuffer(const HostBuffer&) = delete;
        HostBuffer& operator=(const HostBuffer&) = delete;
        // Grow to at least `bytes` (contents not preserved).
        void reserve(ggml_backend_buffer_type_t host_buft, size_t bytes);
        float* data() const { return static_cast<float*>(base_); }
        size_t bytes() const { return bytes_; }

       private:
        ggml_backend_buffer_t buf_ = nullptr;
        void* base_ = nullptr;
        size_t bytes_ = 0;
        std::vector<uint8_t> fallback_;  // CPU-only path
    };

    // Device memory holding one graph launch's projection (proj_dim, T, B). Blocks come from a pool
    // and return to it: freeing device memory would synchronize the device.
    struct DevicePool;
    class DeviceBlock {
       public:
        DeviceBlock() = default;
        ~DeviceBlock();
        DeviceBlock(const DeviceBlock&) = delete;
        DeviceBlock& operator=(const DeviceBlock&) = delete;
        ggml_tensor* tensor() const { return tensor_; }  // F32, contiguous

       private:
        friend class PeeModel;
        std::shared_ptr<DevicePool> pool_;
        ggml_backend_buffer_t buf_ = nullptr;
        ggml_context* ctx_ = nullptr;
        ggml_tensor* tensor_ = nullptr;
    };
    // `rows` projected frames (proj_dim floats each) starting at frame `row` of a block.
    struct ProjSpan {
        std::shared_ptr<const DeviceBlock> block;
        size_t row = 0;
        int rows = 0;
    };

    // One graph launch's downloaded outputs, shared by its windows (no per-window copies).
    struct BatchOut {
        int T = 0, B = 0;  // padded output frames, batch
        size_t d = 0, n_spk = 0, dp = 0;
        HostBuffer fused;    // (d_model, T, B) -> [b][t][d]
        HostBuffer probs80;  // (n_spk, T, B)
        HostBuffer probs10;  // optional native speaker activity
        HostBuffer asr;      // (d_model, T, B), only with want_asr_states
        HostBuffer proj;     // (proj_dim, T, B), only with cfg.proj_dim > 0 (host projection)
        std::shared_ptr<const DeviceBlock> proj_block;  // the same on the device (proj_on_device)
    };
    // Per-window view into a BatchOut on the window's own 80 ms grid.
    struct WindowOut {
        std::shared_ptr<const BatchOut> batch;
        int index = 0;  // batch item
        int t_out = 0;  // valid frames = out_frames(len)
        const float* fused() const;
        const float* probs80() const;
        const float* asr() const;
        const float* proj() const;
    };

    // A job's raw log-mel on the encoder's device (F32, n_mels per frame, frame-major): the front
    // end writes it there and encode_batch / speaker_steps copy their windows from it device to
    // device, so the mel never round-trips through the host. The first zero_frames() frames are
    // zeros (window padding). Stores are pooled grow-only buffers: steady-state jobs allocate
    // nothing.
    class MelStore {
       public:
        ~MelStore();
        MelStore(const MelStore&) = delete;
        MelStore& operator=(const MelStore&) = delete;
        size_t offset(size_t frame) const { return frame * frame_bytes_; }  // bytes
        size_t frames() const { return frames_; }
        const ggml_runtime::DeviceTensor& device() const { return dev_; }

       private:
        friend class PeeModel;
        MelStore() = default;
        ggml_context* ctx_ = nullptr;
        ggml_backend_buffer_t buf_ = nullptr;
        ggml_runtime::DeviceTensor dev_;
        size_t frames_ = 0, frame_bytes_ = 0;
    };
    // A store with room for `frames` frames after the zero frames; back to the pool on release
    // (release before the PeeModel).
    std::shared_ptr<MelStore> mel_store(size_t frames);
    // Leading zero frames of every store: the longest padding a window can need.
    size_t zero_frames() const {
        const int sub = cfg_.asr.subsampling_factor;
        return static_cast<size_t>(bucket_len(
            (cfg_.online_inference_length + cfg_.chunk_left_context + cfg_.chunk_right_context) *
            sub));
    }
    // Host frames into a store (front end on the CPU).
    void upload_mel(MelStore& store, size_t frame, const float* mel, size_t frames);

    // A window's 80 ms speaker probabilities from the streaming speaker steps, to fuse:
    // out_frames(len) x n_spk floats (core rows set, context rows zero), in host memory or in a
    // device block (starting at byte `offset` of block->tensor()).
    struct SpeakerProbs {
        const float* host = nullptr;
        std::shared_ptr<const DeviceBlock> block;
        size_t offset = 0;
    };
    // A multi-window utterance of the streaming speaker branch (sync mode): its windows and, after
    // speaker_steps, each window's speaker probabilities.
    struct SpeakerUtt {
        size_t mel_frame = 0;                       // first raw log-mel frame in the MelStore
        const std::vector<Window>* plan = nullptr;  // plan_windows (more than one window)
        bool want_p10 = false;  // the window cores' native 10 ms probabilities (word timestamps)
        std::vector<SpeakerProbs> p80;             // per window
        std::vector<std::vector<float>> p10;       // per window: core x 8 x n_spk (want_p10)
        std::vector<std::vector<float>> p80_host;  // the host path's storage behind p80
    };
    // Needs the speaker cache's learned silence embedding in the GGUF; older conversions lack it.
    bool stateful_speakers() const { return !sil_emb_.empty(); }
    // Every window step of the utterances: window index by window index, batched across the
    // utterances at that index, each step updating its utterance's speaker cache and FIFO (no
    // update after the last window). On the device the steps are queued without waiting for
    // them (the p80 blocks are complete in stream order); want_p10 outputs are on the host on
    // return.
    void speaker_steps(const MelStore& store, const std::vector<SpeakerUtt*>& utts);
    // GPU backend: keep the speaker state on the device (default). false: the host reference
    // path (state uploaded, outputs downloaded and the state updated on the CPU at every step).
    void set_speaker_state_on_device(bool on) { speaker_on_device_ = on && gpu_ != nullptr; }
    bool speaker_state_on_device() const { return speaker_on_device_; }

    // Where a window's core of pre-fusion ASR states goes in its request's CTC evidence (word
    // timestamps): output frames [left_drop, left_drop + core) to joined rows [row, row + core).
    struct EvidenceSink {
        CtcEvidence* evidence = nullptr;  // none: the window's request has no word timestamps
        int row = 0;
        int left_drop = 0, core = 0;
    };

    // Run B windows in one graph launch, zero-padded to t_pad mel frames
    // (>= every lens[b]). Window b is lens[b] raw log-mel frames from `store` starting at frame
    // mel_frame[b]; stats[b] are that window's utterance statistics. Padded
    // frames are masked out of attention and of the 10 ms pooling; each
    // WindowOut covers only its own out_frames(lens[b]). `evidence` (with a CTC
    // adapter attached) also downloads the native 10 ms speaker activity, and `sinks` (one per
    // window) copy the windows' ASR state cores into their requests' CTC evidence on the device,
    // complete on return.
    // `p80` (stateful windows): the speaker probabilities to fuse, out_frames(lens[b]) x n_spk per
    // window, instead of the window's own speaker pass (all on the host or all on the device).
    // `b_pad` > B runs the graph at that batch size (copies of window 0 fill it; only the real
    // windows are downloaded).
    std::vector<WindowOut> encode_batch(
        const MelStore& store, const std::vector<size_t>& mel_frame,
        const std::vector<const NormStats*>& stats, const std::vector<int>& lens, int t_pad,
        bool evidence = false, const std::vector<SpeakerProbs>* p80 = nullptr,
        const std::vector<EvidenceSink>* sinks = nullptr, int b_pad = 0);

    // Bucketed padding: windows are padded up to a multiple of `frames` mel
    // frames so windows of different lengths share graph shapes and batch
    // together. 1 = exact lengths (no padding). Waste <= frames-1 per window.
    void set_bucket_frames(int frames) { bucket_frames_ = frames < 1 ? 1 : frames; }
    int bucket_len(int len) const {
        return (len + bucket_frames_ - 1) / bucket_frames_ * bucket_frames_;
    }

    // Graph batch size for n windows: n padded up a ladder of sizes (ratio about sqrt 2, at most
    // the max batch), so a bounded set of batch shapes recurs and their graphs (and CUDA graphs)
    // are reused instead of rebuilt for every batch size. encode_batch fills the padding.
    int batch_pad(int n) const;
    // Group windows of the given lengths (mel frames) into graph runs: each returned group lists
    // indices into `lens`, longest first; the run pads them to bucket_len(longest) frames and
    // batch_pad(size) windows. Minimizes runs x kRunCostFrames + padded frames over groups of
    // consecutive windows in length order (exact for that cost).
    std::vector<std::vector<size_t>> plan_batches(const std::vector<int>& lens) const;
    // A graph run's fixed cost (dispatch, synchronization, download set-up) expressed in padded mel
    // frames of compute: one more run costs about as much as 6000 more padded frames.
    static constexpr int64_t kRunCostFrames = 6000;

    struct Output {
        int n_frames = 0;                // T_asr (80 ms grid)
        std::vector<float> spk_probs10;  // native 10 ms speaker activity (evidence only)
        std::vector<float>
            asr_states;                // T_asr x d_model, frame-major (pre-fusion; want_asr_states)
        std::vector<float> spk_probs;  // T_asr x n_spk, 80 ms averaged probabilities
        std::vector<float> spk_binary;  // T_asr x n_spk, thresholded
        std::vector<float> fused;       // T_asr x d_model, frame-major (encoder output)
        std::vector<float>
            proj;  // T_asr x proj_dim, frame-major (LLM input); empty without fusion.proj
        std::vector<ProjSpan> proj_spans;  // the projection's frames on the device (proj_on_device)
    };

    // Trim each window to its core and concatenate. `outs` in plan order.
    // evidence: also join the timestamp evidence (every window must have been encoded with it).
    // probs10: the windows' core 10 ms speaker probabilities when they came from speaker steps.
    Output fuse_windows(
        const std::vector<Window>& plan, const std::vector<WindowOut>& outs, bool evidence = false,
        const std::vector<const std::vector<float>*>* probs10 = nullptr) const;

    // Convenience: whole utterance. mel: (t_mel, n_mels) frame-major,
    // UN-normalized log-mel. Plans windows, batches same-length windows, fuses.
    Output encode(const float* mel, int t_mel);

   private:
    PeeConfig cfg_;
    bool want_asr_states_ = false;
    bool emit_fused_ = true;
    bool proj_on_device_ = false;
    ggml_runtime::BackendManager* bm_ = nullptr;
    ggml_backend_t gpu_ = nullptr;
    std::shared_ptr<DevicePool> device_pool_;
    // Copy a graph output that is valid until the next run into a pooled block.
    std::shared_ptr<const DeviceBlock> keep_on_device(
        const ggml_runtime::DeviceTensor& out, int64_t items);
    // A pooled block of `floats` F32 elements (contents undefined).
    std::shared_ptr<DeviceBlock> device_block(size_t floats);
    bool speaker_on_device_ = false;
    void speaker_steps_host(const MelStore& store, const std::vector<SpeakerUtt*>& utts);
    void speaker_steps_device(const MelStore& store, const std::vector<SpeakerUtt*>& utts);
    std::unique_ptr<CtcTimestampModel> ctc_;
    int max_batch_ = 8;
    int bucket_frames_ = 1;
    std::vector<float> mel_basis_;
    std::vector<float> sil_emb_;  // diar.head.learnable_sil_emb; empty = stateless speaker windows
    // Persistent pinned upload scratch (one arena, sliced per input).
    ggml_backend_buffer_type_t host_buft_ = nullptr;
    ggml_backend_t backend_ = nullptr;  // the session's main backend (evidence copies, mel stores)
    std::mutex store_mu_;
    std::vector<std::unique_ptr<MelStore>> free_stores_;
    // Device copies of each window's frames (zero-padded to t_pad) into a (n_mels, t_pad, Bg)
    // input; items past the real windows repeat window 0.
    std::vector<ggml_runtime::Session::Input::DeviceRange> window_gather(
        const MelStore& store, const std::vector<size_t>& mel_frame, const std::vector<int>& lens,
        int t_pad, int Bg) const;
    HostBuffer in_arena_;
    std::mutex batch_mu_;
    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    std::unique_ptr<PeeGraph> graph_;
    std::unique_ptr<ggml_runtime::Session> session_;

    void read_host_tensor(const std::string& name, std::vector<float>& out, size_t n);
};

// NeMo `normalize_batch(..., "per_feature")`: per-mel-bin mean / unbiased std
// over the valid frames, std += 1e-5. mel is (t, n_mels) frame-major.
void per_feature_normalize(const float* mel, int t, int n_mels, float* out);

}  // namespace nemo_speech::speechlm
