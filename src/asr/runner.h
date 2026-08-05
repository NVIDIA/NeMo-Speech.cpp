// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Per-stream audio buffering, encoder execution, and head decoding.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "decoder.h"
#include "fastconformer.h"
#include "fe.h"
#include "model.h"
#include "parameter_parser.h"
#include "runtime.h"
#include "types.h"
#include "vad_endpointer.h"
#include "vad_masker.h"

namespace nemo_speech::asr {

struct RecognizerConfig;

// Defined in flashlight_decoder.h; non-flashlight builds hold it null.
struct FlashlightResources;

// Runtime latency/throughput policy, independent of model identity.
struct StreamingConfig {
    float chunk_size = 0.16f;         // seconds (CTC buffered window)
    float ctc_left_padding = 1.92f;   // seconds
    float ctc_right_padding = 1.92f;  // seconds
    // A trained cache-aware right-context mode in encoder frames. -1 uses the
    // value stored in the model.
    int rnnt_right_context = 1;
    void Register(common::ParameterParser& p) {
        p.Register("chunk_size", &chunk_size, "CTC buffered chunk size (s)", {"--chunk-sec"});
        p.Register(
            "ctc_left_padding", &ctc_left_padding, "CTC left context padding (s)",
            {"--left-pad-sec"});
        p.Register(
            "ctc_right_padding", &ctc_right_padding, "CTC right context padding (s)",
            {"--right-pad-sec"});
        p.Register(
            "rnnt_right_context", &rnnt_right_context,
            "RNNT cache-aware right context (encoder frames; -1 = model default)");
    }
};

// Snapshot returned by step() and finalize().
struct StreamingUpdate {
    // Tokens emitted by this call, in stream order.
    std::vector<int> new_token_ids;
    // Full transcript after this update.
    std::string transcript_so_far;
    // Per-word time spans (global encoder frames), populated when requested
    // explicitly or required for diarization. Scoped per utterance: a
    // mid-stream EOU final carries that utterance's words, then the buffer restarts.
    std::vector<WordTiming> words;
    // Additional final hypotheses, best-first and excluding the top result.
    std::vector<Hypothesis> extra_alternatives;
    bool is_final = false;
    // Mean token posterior for greedy CTC; 1.0 when the decoder has no posterior.
    float confidence = 1.0f;
    // Seconds of audio processed.
    float audio_processed_sec = 0.0f;
};

// Runner methods are single-threaded. Independent runners may execute
// concurrently; compatible work is batched below this layer and indexed state
// rows keep streams isolated.
class AsrRunner {
   public:
    virtual ~AsrRunner() = default;

    // Feed mono float32 audio at the model's sample rate.
    virtual void feed_audio(const float* samples, size_t n_samples) = 0;

    // Advance: process any chunks that are ready, return the new tokens.
    // Empty result means no chunk was ready (need more audio).
    virtual StreamingUpdate step() = 0;

    // No more audio is coming. Flush the tail and return the final result.
    virtual StreamingUpdate finalize() = 0;

    // Reset for a fresh utterance (same model, same runner).
    virtual void reset() = 0;

    // Set the language-ID prompt slot for prompt-conditioned multilingual
    // models (nemotron-3.5). No-op for non-prompt runners. -1 = no prompt.
    virtual void set_prompt_index(int /*prompt_index*/) {}

    // Language codes detected from model language-ID tags. Empty for
    // non-prompt models.
    virtual std::vector<std::string> detected_languages() const { return {}; }

    // Per-request options (word timestamps, boosting, ...). Call once after
    // construction, before feed_audio. Default no-op.
    virtual void set_request_options(const AsrRequestOptions& /*opts*/) {}

    // Finalize the current utterance on the next step, then continue the stream.
    // This works even when endpointing is disabled. Default no-op.
    virtual void force_eou() {}
};

// Buffered streaming over a non-cache-aware encoder.
//
// Each encoder pass sees [left_pad | chunk | right_pad] seconds of audio
// (RecognizerConfig::streaming), emits CTC output only for the center chunk.
// Latency ~ chunk_size + ctc_right_padding. The decoder, VAD masker, and
// endpointer are read from the same RecognizerConfig - there is no separate
// per-runner config to keep in sync.
class BufferedStreamRunner final : public AsrRunner {
   public:
    // `flashlight` shares the prebuilt KenLM and lexicon trie across streams.
    // When null, the decoder builds its own resources.
    explicit BufferedStreamRunner(
        CtcModel* model, const RecognizerConfig& cfg,
        std::shared_ptr<const FlashlightResources> flashlight = nullptr,
        std::shared_ptr<SileroVadModel> vad_model = nullptr);

    void feed_audio(const float* samples, size_t n_samples) override;
    StreamingUpdate step() override;
    StreamingUpdate finalize() override;
    void reset() override;
    void set_request_options(const AsrRequestOptions& opts) override;
    void force_eou() override;

   private:
    StreamingUpdate process_window(size_t win_start, size_t win_end, bool is_last);

    // Poll the endpointer on the decode clock (now = emit cursor; last speech
    // = VAD bits over decoded frames, or the decoder's last_emit_frame).
    // Called after each processed window so silence gaps interior to one feed
    // still fire. On EOU it runs the shared fire_eou and returns true; the
    // step loop breaks so the next utterance's windows decode on the next
    // step() call.
    bool poll_endpoint(StreamingUpdate& update);

    // Drop audio/mel/timeline prefixes nothing can read again (left context,
    // un-melled FE tail, and un-observed VAD audio are kept). Bounds per-stream
    // memory on endless streams; global indices (emit_start_, frame cursors)
    // are unaffected - only the storage base moves.
    void trim_buffers();

    // Extend mel_buf_ with raw log-mel for any audio frames newly available in
    // audio_buf_. Consecutive windows overlap ~96%, so without this each step
    // would recompute the whole window's mel; here only the new frames are
    // computed and appended. Windows slice from here in both the plain and
    // VAD-masking paths (masking mutates the slice copy, never the cache).
    void ensure_mel_cache();

    CtcModel* model_;
    std::unique_ptr<Decoder> head_;
    // Non-owning alias into head_ when the compact device-argmax CTC path is
    // active. Null for Flashlight, which requires full log probabilities.
    GreedyCtcDecoder* greedy_ctc_ = nullptr;
    AsrRequestOptions opts_;
    int64_t enc_frame_samples_ = 1;  // subsampling_factor * FE hop (for frame_offset)

    // Streaming chunk geometry - derived from chunk_sec/* and sample_rate at ctor.
    int sample_rate_;
    size_t chunk_samples_;
    size_t left_samples_;
    size_t right_samples_;
    int left_out_frames_;
    int chunk_out_frames_;

    // Optional Silero VAD (per-stream; null unless masking or VAD-driven
    // endpointing needs it). Owns the speech/silence timeline; observed once
    // per step and shared by the masker + endpointer (no double inference).
    std::unique_ptr<SileroVad> vad_;
    // Optional masker (null unless mask_enable). Borrows vad_; masks silence
    // mel frames in raw log-mel space before normalize + the encoder.
    std::unique_ptr<VadMasker> vad_masker_;
    size_t audio_fed_to_vad_ = 0;  // GLOBAL sample count already fed to vad_

    // Always constructed so force_eou() works when threshold endpointing is off.
    std::unique_ptr<VadEndpointer> endpointer_;
    // VAD-driven EOU scan state: next global mel frame to scan, and the last
    // speech mel frame seen at or before the decode cursor.
    int64_t vad_scan_frame_ = 0;
    int64_t vad_speech_seen_frame_ = -1;

    // audio_buf_ and mel_buf_ are sliding stores: audio_base_ and
    // mel_base_frame_ are the GLOBAL sample/mel-frame index of element 0, so
    // all cursors (emit_start_, window bounds, frame indices) stay global and
    // trim_buffers() only moves the storage base.
    std::vector<float> audio_buf_;
    size_t audio_base_ = 0;
    // Global raw log-mel cache (frame-major: n_mels inner, frames outer) and
    // the count of frames computed so far. Windows slice their frames from
    // here, optionally VAD-mask the slice copy, then apply per_feature
    // normalization before the encoder. The cache itself always stays raw.
    std::vector<float> mel_buf_;
    int64_t mel_base_frame_ = 0;
    int64_t total_mel_frames_produced_ = 0;
    size_t emit_start_;
    std::vector<int> all_tokens_;
    std::string transcript_;
    bool finalized_;
    // True once finalize() begins, so its internal drain step() doesn't fire EOU.
    bool finalizing_ = false;
};

// Unary/full-utterance path. Unlike BufferedStreamRunner/CacheStreamRunner it
// performs one non-cache-aware encoder pass over the entire recording. This is
// the full-utterance path for every head; streaming uses the buffered CTC or
// cache-aware transducer runners.
class OfflineRunner final : public AsrRunner {
   public:
    OfflineRunner(
        AsrModel* model, const RecognizerConfig& cfg,
        std::shared_ptr<const FlashlightResources> flashlight = nullptr);

    void feed_audio(const float* samples, size_t n_samples) override;
    StreamingUpdate step() override { return {}; }
    StreamingUpdate finalize() override;
    void reset() override;
    void set_prompt_index(int prompt_index) override { prompt_index_ = prompt_index; }
    std::vector<std::string> detected_languages() const override { return detected_languages_; }
    void set_request_options(const AsrRequestOptions& opts) override { opts_ = opts; }
    void force_eou() override {}

   private:
    AsrModel* model_;
    DecoderConfig decoder_cfg_;
    std::unique_ptr<Decoder> ctc_decoder_;
    AsrRequestOptions opts_;
    std::vector<float> audio_;
    int prompt_index_ = -1;
    bool finalized_ = false;
    std::vector<std::string> detected_languages_;
};

// Drives a fixed-shape encoder graph (chunk_size_mel in, chunk_enc_frames
// out) with per-layer K/V/conv caches threaded across chunks in a
// device-resident indexed cache row. Audio is buffered until at least one chunk's
// worth of mel frames is ready; each step uploads mel + a small validity mask
// and downloads only the joint-projected encoder output.
//
// The right-context value selects one of the model's trained latency modes;
// -1 uses its stored default. Mel-frame geometry is fixed by the model.
class CacheStreamRunner final : public AsrRunner {
   public:
    // Borrows shared model resources and owns a per-stream encoder cache row.
    explicit CacheStreamRunner(
        RnntModel* model, const RecognizerConfig& cfg,
        std::shared_ptr<SileroVadModel> vad_model = nullptr);

    ~CacheStreamRunner() override;

    void feed_audio(const float* samples, size_t n_samples) override;
    StreamingUpdate step() override;
    StreamingUpdate finalize() override;
    void reset() override;
    void set_prompt_index(int prompt_index) override { prompt_index_ = prompt_index; }
    std::vector<std::string> detected_languages() const override { return detected_languages_; }
    void set_request_options(const AsrRequestOptions& opts) override;
    void force_eou() override;

    // Diagnostic accessor: copy out the prompt-conditioned joint encoder
    // projection from the most recent chunk. Returns the number of frames.
    // Layout is column-major (joint_dim, T_out).
    int take_encoder_frames(std::vector<float>& out, int& T_out, int& d_model);

    int chunks_processed() const { return chunks_processed_; }
    int cache_filled_frames() const { return cache_filled_frames_; }
    RnntDecodeStats rnnt_decode_stats() const;

   private:
    void process_one_chunk(bool /*is_last*/);
    void finish_endpoint(StreamingUpdate& update, bool preserve_buffered_future);
    void upload_attn_mask();
    void zero_caches();
    // Poll the endpointer on the decode clock (now = encoder frames emitted;
    // last speech = VAD bits over decoded mel frames, or the decoder's
    // last_emit_frame). Called after each processed chunk. On EOU it runs the
    // normal EOS path, then starts the next utterance with fresh encoder and
    // predictor state. The chunk loop breaks when this returns true.
    bool poll_endpoint(StreamingUpdate& update, bool after_chunk);
    // Drop the audio prefix that FE and VAD have both consumed.
    void trim_buffers();
    void compact_mel_buffer();

    RnntModel* model_ = nullptr;
    AsrRequestOptions opts_;
    // GGUF-pinned cache-aware mel-frame geometry. These are NeMo's
    // CacheAwareStreamingConfig chunk_size / shift_size in mel frames
    // (setup_streaming_params, conformer_encoder.py):
    //   chunk_size_mel = pre_encode_cache_size + subsampling_factor*(1 + R)
    //   shift_size_mel = subsampling_factor*(1 + R - cache_drop_size)
    // These values are fixed by the model.
    int pre_encode_cache_size_ = 9;  // overlap, mel frames
    int cache_drop_size_ = 0;        // 0 for chunked_limited

    // Language-ID prompt slot for prompt-conditioned multilingual models
    // (nemotron-3.5), set per-stream from the request's language_code via
    // set_prompt_index(). -1 = no prompt fusion. Passed to encode_cache_aware.
    int prompt_index_ = -1;
    // Language codes stripped from transcript tags and exposed in results.
    std::vector<std::string> detected_languages_;

    // Cache-aware encoder shape copied from the model at construction.
    EncoderConfig enc_cfg_;

    // Optional Silero VAD (per-stream; null unless masking or VAD-driven
    // endpointing needs it). Owns the speech/silence timeline, observed once
    // per step and shared by the masker + endpointer (no double inference).
    std::unique_ptr<SileroVad> vad_;
    // Optional masker (null unless mask_enable). Borrows vad_; zeros silence
    // mel frames at the FE→mel hand-off in step().
    std::unique_ptr<VadMasker> vad_masker_;
    // GLOBAL sample count already fed to vad_.
    size_t audio_fed_to_vad_ = 0;

    // Always constructed so force_eou() works with threshold endpointing off. Polled
    // after each chunk; on fire the runner emits is_final and resets
    // per-utterance state. Encoder and predictor state are reset at the
    // boundary so delayed tokens cannot leak into the next utterance.
    std::unique_ptr<VadEndpointer> endpointer_;
    bool force_eou_pending_ = false;
    // VAD-driven EOU scan state: next global mel frame to scan, and the last
    // speech mel frame seen at or before the decode cursor.
    int64_t vad_scan_frame_ = 0;
    int64_t vad_speech_seen_frame_ = -1;

    // audio_buf_ is a sliding store: audio_base_ is the
    // GLOBAL sample index of element 0 (trim_buffers drops consumed prefix).
    std::vector<float> audio_buf_;
    size_t audio_base_ = 0;
    std::vector<float> mel_buf_;  // (n_mels, T) flat, frame-major
    size_t mel_offset_ = 0;
    // Total number of mel frames produced from audio_buf_ across all step()
    // calls (monotonic). NB: mel_buf_ also gets frames *consumed* from the
    // front each chunk, so its current size is smaller than this counter.
    int64_t total_mel_frames_produced_ = 0;

    // Per-stream K/V/conv cache, device-resident. Allocated lazily on first
    // encode (and re-allocated fresh on reset) via AsrModel::make_cache_state().
    // The encoder graph binds these buffers per run and updates them in-graph -
    // no host<->device copies, no cross-stream hand-off.
    CacheAwareEncoder::State cache_state_;
    // attn_mask is still a per-call Input (it grows with cache_filled_frames_), so
    // it lives host-side and is uploaded every chunk.
    std::vector<float> attn_mask_;

    int cache_filled_frames_ = 0;
    int chunks_processed_ = 0;
    int64_t total_frames_emitted_ = 0;
    bool finalized_ = false;
    // True once finalize() begins, so its internal drain step() doesn't fire EOU.
    bool finalizing_ = false;
    // Tracks whether we've prepended pre_encode_cache_size zero mel frames at
    // start-of-stream. NeMo's CacheAwareStreamingAudioBuffer does this once
    // per stream so the encoder's drop_extra_pre_encoded behaviour throws
    // away zero-pad rather than the leading ~160 ms of real audio.
    bool stream_zero_padded_ = false;

    // Encoder output from most recent chunk (column-major (d_model, T_out)).
    std::vector<float> last_enc_out_;
    int last_enc_T_ = 0;

    // RNNT head (null for CTC or when constructed via model_path ctor).
    // Runs the greedy emit loop on the encoder output of each chunk; per-stream
    // LSTM state lives inside it.
    std::unique_ptr<Decoder> head_;
    std::vector<int> all_tokens_;
    std::string transcript_;
    // Tokens emitted during the most recent step()/finalize() call.
    std::vector<int> last_step_new_tokens_;
};

}  // namespace nemo_speech::asr
