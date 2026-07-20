// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "runner.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <stdexcept>

#include "greedy_ctc_decoder.h"
#include "recognizer.h"
#ifdef NEMO_SPEECH_WITH_FLASHLIGHT
#include "flashlight_decoder.h"
#endif

namespace nemo_speech::asr {

namespace {
// Removes <lang> tags and returns their language codes.
std::vector<std::string>
extract_lang_tags(std::string& text) {
    static const std::regex tag_re(R"(<([A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,4})?)>)");
    std::vector<std::string> langs;
    for (std::sregex_iterator it(text.begin(), text.end(), tag_re), end; it != end; ++it) {
        const std::string code = (*it)[1].str();
        if (std::find(langs.begin(), langs.end(), code) == langs.end())
            langs.push_back(code);
    }
    if (!langs.empty()) {
        text = std::regex_replace(text, tag_re, "");
        text = std::regex_replace(text, std::regex(R"( {2,})"), " ");
        const size_t a = text.find_first_not_of(' ');
        const size_t b = text.find_last_not_of(' ');
        text = (a == std::string::npos) ? std::string() : text.substr(a, b - a + 1);
    }
    return langs;
}

// NeMo per_feature normalization (per mel bin across frames). Bit-identical
// to MelSpectrogramExtractor's internal normalize=true path (fe.cpp), so a
// no-mask window matches the non-VAD infer_ctc output exactly. Used by the
// VAD path, which computes raw log-mel (normalize=false), masks silence
// frames, then normalizes here before the encoder.
void
normalize_per_feature(std::vector<float>& feats, int n_mels, int n_frames, double stddev_floor) {
    if (n_frames <= 0)
        return;
    for (int m = 0; m < n_mels; m++) {
        double sum = 0.0;
        for (int f = 0; f < n_frames; f++)
            sum += feats[static_cast<size_t>(m) + static_cast<size_t>(f) * n_mels];
        const double mean = sum / n_frames;
        double var = 0.0;
        for (int f = 0; f < n_frames; f++) {
            const double d = feats[static_cast<size_t>(m) + static_cast<size_t>(f) * n_mels] - mean;
            var += d * d;
        }
        var /= n_frames;
        const double inv_std = 1.0 / (std::sqrt(var) + stddev_floor);
        for (int f = 0; f < n_frames; f++) {
            const size_t idx = static_cast<size_t>(m) + static_cast<size_t>(f) * n_mels;
            feats[idx] = static_cast<float>((feats[idx] - mean) * inv_std);
        }
    }
}

// Shared mid-stream EOU fire for both runners. Order is load-bearing:
// finalize() commits a beam head's in-flight hypothesis and word timings, so
// it precedes the transcript/words capture; the per-utterance clears come
// last. This helper performs the decoder's soft utterance reset; a runner that
// requires a hard model-state boundary resets the decoder again after capture.
void
fire_eou(
    Decoder* head, const AsrRequestOptions& opts, std::vector<int>& all_tokens,
    std::string& transcript, StreamingUpdate& update) {
    update.is_final = true;
    // Runner-owned transcript first (greedy heads return no final_transcript);
    // a beam head's committed-best hypothesis then overrides it.
    update.transcript_so_far = transcript;
    if (head) {
        head->finalize();
        const std::string ft = head->final_transcript();
        if (!ft.empty())
            update.transcript_so_far = ft;
        if (opts.needs_word_timings())
            update.words = head->word_timings();
        update.confidence = head->confidence();
        if (opts.max_alternatives > 1)
            update.extra_alternatives = head->additional_hypotheses(opts.max_alternatives);
    }
    all_tokens.clear();
    transcript.clear();
    if (head)
        head->reset_utterance();
}

// Shared VAD/endpointer wiring for both runner ctors. The Silero VAD is built
// when feature masking or VAD-driven endpointing needs it; one inference per
// step feeds both. The endpointer is built unconditionally: cfg.enable gates
// threshold fires inside poll(), and force_eou() works regardless.
void
build_vad_stack(
    AsrModel* model, const std::string& vad_model_path, const VadMaskerCfg& vcfg,
    const VadEndpointerCfg& ecfg, std::shared_ptr<SileroVadModel> shared_model,
    std::unique_ptr<SileroVad>& vad, std::unique_ptr<VadMasker>& masker,
    std::unique_ptr<VadEndpointer>& endpointer) {
    const bool want_mask = !vad_model_path.empty() && vcfg.mask_enable;
    const bool want_vad_eou = ecfg.enable && ecfg.vad_based && !vad_model_path.empty();
    if (want_mask || want_vad_eou) {
        if (!shared_model) {
            // Allow direct runner construction without a shared VAD model.
            shared_model =
                std::make_shared<SileroVadModel>(model->backend_manager(), vad_model_path);
        }
        vad = std::make_unique<SileroVad>(std::move(shared_model));
        vad->set_binarizer(vcfg.onset, vcfg.offset, model->fe().hop_length());
        if (want_mask) {
            masker = std::make_unique<VadMasker>(vad.get(), model->fe_config().n_mels, vcfg);
        }
    }
    endpointer = std::make_unique<VadEndpointer>(ecfg);
    if (ecfg.enable && ecfg.vad_based && !vad) {
        std::fprintf(
            stderr,
            "[asr] endpointing: vad_based EOU requested but no VAD model loaded; "
            "using decoder token-silence instead.\n");
    }
}

std::unique_ptr<Decoder>
make_ctc_decoder(
    CtcModel* model, const RecognizerConfig& cfg,
    std::shared_ptr<const FlashlightResources> flashlight) {
    if (cfg.decoder.kind == DecoderConfig::Kind::Flashlight) {
#ifdef NEMO_SPEECH_WITH_FLASHLIGHT
        if (cfg.decoder.lexicon_path.empty())
            throw std::runtime_error("flashlight decoder set but lexicon_path is empty");
        FlashlightCtcCfg fcfg;
        fcfg.lm_path = cfg.decoder.lm_path;
        fcfg.lexicon_path = cfg.decoder.lexicon_path;
        fcfg.tokenizer_path = cfg.decoder.tokenizer_path;
        fcfg.beam_size = cfg.decoder.beam_size;
        fcfg.beam_size_token = cfg.decoder.beam_size_token;
        fcfg.beam_threshold = cfg.decoder.beam_threshold;
        fcfg.lm_weight = cfg.decoder.lm_weight;
        fcfg.word_insertion_score = cfg.decoder.word_insertion_score;
        fcfg.max_boost = cfg.decoder.max_boost;
        return std::make_unique<FlashlightDecoder>(
            model->ctc_config(), model->vocab(), std::move(fcfg), std::move(flashlight));
#else
        (void)flashlight;
        throw std::runtime_error("flashlight decoder requires NEMO_SPEECH_WITH_FLASHLIGHT=ON");
#endif
    }
    (void)flashlight;
    return std::make_unique<GreedyCtcDecoder>(model->ctc_config(), model->vocab());
}
}  // namespace

BufferedStreamRunner::BufferedStreamRunner(
    CtcModel* model, const RecognizerConfig& cfg,
    std::shared_ptr<const FlashlightResources> flashlight,
    std::shared_ptr<SileroVadModel> vad_model)
    : model_(model), sample_rate_(model->sample_rate()),
      chunk_samples_(static_cast<size_t>(cfg.streaming.chunk_size * sample_rate_)),
      left_samples_(static_cast<size_t>(cfg.streaming.ctc_left_padding * sample_rate_)),
      right_samples_(static_cast<size_t>(cfg.streaming.ctc_right_padding * sample_rate_)),
      emit_start_(0), finalized_(false) {
    const float out_hz = 100.0f / static_cast<float>(model->subsampling_factor());
    left_out_frames_ = static_cast<int>(cfg.streaming.ctc_left_padding * out_hz);
    chunk_out_frames_ = static_cast<int>(cfg.streaming.chunk_size * out_hz);
    // Audio samples per encoder output frame, for the frame_offset passed to
    // the decoder (subsampling_factor * mel hop).
    enc_frame_samples_ =
        static_cast<int64_t>(model->subsampling_factor()) * model->fe().hop_length();

    head_ = make_ctc_decoder(model, cfg, std::move(flashlight));
    greedy_ctc_ = dynamic_cast<GreedyCtcDecoder*>(head_.get());

    build_vad_stack(
        model_, cfg.vad.model_path, cfg.vad.masker, cfg.endpointing, std::move(vad_model), vad_,
        vad_masker_, endpointer_);

    // last_emit_frame() is only polled for token-silence endpointing (enabled,
    // non-VAD). Tell the head so a beam decoder can skip the per-frame argmax it
    // would otherwise run only to feed that signal (greedy derives it for free).
    if (head_) {
        const bool token_silence_eou = cfg.endpointing.enable && !cfg.endpointing.vad_based;
        head_->set_track_speech_frame(token_silence_eou);
    }
}

void
BufferedStreamRunner::set_request_options(const AsrRequestOptions& opts) {
    opts_ = opts;
    if (head_)
        head_->set_request_options(opts);
    if (endpointer_)
        endpointer_->set_stop_history_eou_ms(opts.stop_history_eou_ms);
}

void
BufferedStreamRunner::force_eou() {
    if (endpointer_)
        endpointer_->force();
}

void
BufferedStreamRunner::feed_audio(const float* samples, size_t n_samples) {
    if (finalized_)
        return;
    audio_buf_.insert(audio_buf_.end(), samples, samples + n_samples);
}

StreamingUpdate
BufferedStreamRunner::step() {
    StreamingUpdate update;
    update.transcript_so_far = transcript_;
    if (finalized_)
        return update;

    // Feed the shared Silero VAD the new monotonic audio once (inference +
    // binarized timeline), consumed by the masker (apply) and the endpointer.
    const size_t audio_end = audio_base_ + audio_buf_.size();
    if (vad_ && audio_end > audio_fed_to_vad_) {
        vad_->observe_audio(
            audio_buf_.data() + (audio_fed_to_vad_ - audio_base_), audio_end - audio_fed_to_vad_);
        audio_fed_to_vad_ = audio_end;
    }

    // While enough audio is buffered to advance the emit cursor by one full
    // chunk with right_pad lookahead, process windows. The endpointer is
    // polled per window (not once per step) so a silence gap fully inside one
    // large feed still fires; an EOU breaks the loop, and the next
    // utterance's remaining windows decode on the next step() call.
    while (emit_start_ + chunk_samples_ + right_samples_ <= audio_end) {
        const size_t win_start = (emit_start_ >= left_samples_) ? (emit_start_ - left_samples_) : 0;
        const size_t win_end = emit_start_ + chunk_samples_ + right_samples_;

        auto chunk_update = process_window(win_start, win_end, /*is_last=*/false);
        update.new_token_ids.insert(
            update.new_token_ids.end(), chunk_update.new_token_ids.begin(),
            chunk_update.new_token_ids.end());
        update.transcript_so_far = transcript_;
        emit_start_ += chunk_samples_;
        if (poll_endpoint(update))
            break;
    }
    // Set on zero-window steps too: an EOU can fire without a new window.
    update.audio_processed_sec = static_cast<float>(emit_start_) / static_cast<float>(sample_rate_);
    // A pending force_eou() must fire even on a window-less step.
    if (!update.is_final)
        poll_endpoint(update);
    // On EOU finals the words were captured by fire_eou before
    // reset_utterance cleared the decoder's buffer.
    if (!update.is_final && opts_.needs_word_timings())
        update.words = head_->word_timings();
    trim_buffers();
    return update;
}

// Advance the VAD speech scan over newly-decoded mel frames up to decoded_mel,
// clamped to committed_frames() so it never latches the provisional in_speech_
// bit of a not-yet-binarized frame (the cursor would then skip it forever).
// Shared by both runners' VAD-based endpointing so the clamp can't drift.
static void
scan_vad_speech(SileroVad& vad, int64_t decoded_mel, int64_t& scan_frame, int64_t& speech_seen) {
    const int64_t scan_end = std::min(decoded_mel, vad.committed_frames());
    for (int64_t g = scan_frame; g < scan_end; ++g) {
        if (vad.frame_speech(g))
            speech_seen = g;
    }
    scan_frame = std::max(scan_frame, scan_end);
}

bool
BufferedStreamRunner::poll_endpoint(StreamingUpdate& update) {
    // No EOU during finalize() (it drains via step()): the end-of-stream final
    // is emitted by the caller; an EOU here would clear the buffer and drop it.
    if (!endpointer_ || finalizing_)
        return false;
    // Both signals run on the decode clock (the emit cursor), never the raw
    // audio frontier: the frontier leads decoding by chunk+right_pad (~2 s),
    // which exceeds the EOU threshold, so a frontier-clocked fire would
    // finalize before the utterance tail is decoded and leak its words into
    // the next utterance.
    double now_ms = 0.0, last_speech_ms = 0.0;
    if (vad_ && endpointer_->config().vad_based) {
        // VAD-driven: scan the Silero speech bits over the newly decoded mel
        // frames; last speech is the latest speech frame at/before the cursor.
        const int hop = model_->fe().hop_length();
        const double mel_ms = 1000.0 * hop / sample_rate_;
        const int64_t decoded_mel = static_cast<int64_t>(emit_start_) / hop;
        scan_vad_speech(*vad_, decoded_mel, vad_scan_frame_, vad_speech_seen_frame_);
        now_ms = static_cast<double>(decoded_mel) * mel_ms;
        last_speech_ms = (vad_speech_seen_frame_ < 0)
                             ? 0.0
                             : static_cast<double>(vad_speech_seen_frame_ + 1) * mel_ms;
    } else {
        // Token-silence: time since the decoder's last speech evidence (exact
        // emission frame - works for greedy AND beam heads, whose step()
        // returns no token ids).
        const double frame_ms = model_->ms_per_enc_frame();
        now_ms = static_cast<double>(emit_start_) / sample_rate_ * 1000.0;
        const int64_t lef = head_ ? head_->last_emit_frame() : -1;
        last_speech_ms = (lef < 0) ? 0.0 : static_cast<double>(lef + 1) * frame_ms;
    }
    if (!endpointer_->poll(now_ms, last_speech_ms))
        return false;
    // Keep the masker FSM + cursors: the stream continues into the next
    // utterance.
    fire_eou(head_.get(), opts_, all_tokens_, transcript_, update);
    return true;
}

void
BufferedStreamRunner::trim_buffers() {
    const int n_fft = model_->fe_config().n_fft;
    const int hop = model_->fe().hop_length();
    // Oldest sample any consumer can still read: the next window's left
    // context, the FE's incremental left edge, and un-observed VAD audio.
    size_t keep_from = (emit_start_ >= left_samples_) ? (emit_start_ - left_samples_) : 0;
    const int64_t fe_from = total_mel_frames_produced_ * hop - n_fft / 2;
    if (fe_from < static_cast<int64_t>(keep_from))
        keep_from = (fe_from > 0) ? static_cast<size_t>(fe_from) : 0;
    if (vad_ && audio_fed_to_vad_ < keep_from)
        keep_from = audio_fed_to_vad_;
    if (keep_from > audio_base_) {
        audio_buf_.erase(audio_buf_.begin(), audio_buf_.begin() + (keep_from - audio_base_));
        audio_base_ = keep_from;
    }
    // Mel cache (non-VAD path): windows never reach before the left context.
    const int64_t keep_frame = static_cast<int64_t>(
        ((emit_start_ >= left_samples_) ? (emit_start_ - left_samples_) : 0) / hop);
    if (keep_frame > mel_base_frame_) {
        const size_t n_mels = static_cast<size_t>(model_->fe_config().n_mels);
        const size_t drop = static_cast<size_t>(keep_frame - mel_base_frame_) * n_mels;
        if (drop <= mel_buf_.size()) {
            mel_buf_.erase(mel_buf_.begin(), mel_buf_.begin() + drop);
            mel_base_frame_ = keep_frame;
        }
    }
    // Silero timeline: keep a margin below the decode cursor so the masker's
    // neighbourhood scans never cross the trimmed base, whatever its config.
    if (vad_) {
        const int64_t margin =
            std::max<int64_t>(256, vad_masker_ ? 2 * vad_masker_->max_scan() : 0);
        const int64_t g = keep_frame - margin;
        if (g > 0)
            vad_->discard_timeline_before(g);
    }
}

// Fill the final-result fields common to every finalize() exit (both runners,
// early-return and tail paths). The caller is responsible for any
// head->finalize() (flushing the trailing word) before this, and computes
// audio_processed_sec from its own sample-rate source.
static void
fill_final_update(
    StreamingUpdate& u, Decoder* head, bool want_word_timings, const std::string& transcript,
    float audio_processed_sec, int max_alternatives) {
    u.is_final = true;
    u.transcript_so_far = transcript;
    u.audio_processed_sec = audio_processed_sec;
    if (head) {
        if (want_word_timings)
            u.words = head->word_timings();
        u.confidence = head->confidence();
        if (max_alternatives > 1)
            u.extra_alternatives = head->additional_hypotheses(max_alternatives);
    }
}

StreamingUpdate
BufferedStreamRunner::finalize() {
    StreamingUpdate update;
    if (finalized_) {
        fill_final_update(
            update, head_.get(), opts_.needs_word_timings(), transcript_,
            static_cast<float>(audio_base_ + audio_buf_.size()) / static_cast<float>(sample_rate_),
            opts_.max_alternatives);
        return update;
    }

    finalizing_ = true;
    // Drain any whole-chunks that have enough audio for full right_pad first.
    auto draining = step();
    update.new_token_ids.insert(
        update.new_token_ids.end(), draining.new_token_ids.begin(), draining.new_token_ids.end());

    // Flush Silero's trailing partial window BEFORE the tail decode so the
    // tail window's mask uses the refined speech bits.
    if (vad_)
        vad_->flush_timeline();

    // Then process the tail (whatever remains after emit_start_), with no
    // right_pad lookahead. The window ends at the audio frontier.
    const size_t audio_end = audio_base_ + audio_buf_.size();
    if (emit_start_ < audio_end) {
        const size_t win_start = (emit_start_ >= left_samples_) ? (emit_start_ - left_samples_) : 0;
        auto tail = process_window(win_start, audio_end, /*is_last=*/true);
        update.new_token_ids.insert(
            update.new_token_ids.end(), tail.new_token_ids.begin(), tail.new_token_ids.end());
        emit_start_ = audio_end;
    } else if (head_) {
        // No tail window (stream ended exactly on a chunk boundary): the head
        // still needs its finalize (beam decoders commit the in-flight words).
        head_->finalize();
        const std::string ft = head_->final_transcript();
        if (!ft.empty())
            transcript_ = ft;
    }

    finalized_ = true;
    // head_->finalize() already ran above (process_window is_last, or the
    // else branch) so word_timings include the trailing word.
    fill_final_update(
        update, head_.get(), opts_.needs_word_timings(), transcript_,
        static_cast<float>(audio_end) / static_cast<float>(sample_rate_), opts_.max_alternatives);
    return update;
}

void
BufferedStreamRunner::reset() {
    audio_buf_.clear();
    audio_base_ = 0;
    mel_buf_.clear();
    mel_base_frame_ = 0;
    total_mel_frames_produced_ = 0;
    emit_start_ = 0;
    all_tokens_.clear();
    transcript_.clear();
    finalized_ = false;
    finalizing_ = false;
    if (head_)
        head_->reset();
    if (vad_)
        vad_->reset();
    audio_fed_to_vad_ = 0;
    if (endpointer_)
        endpointer_->reset();
    vad_scan_frame_ = 0;
    vad_speech_seen_frame_ = -1;
}

void
BufferedStreamRunner::ensure_mel_cache() {
    std::vector<float> new_mel;
    const int n = produce_new_mel_frames(
        model_->fe(), audio_buf_, audio_base_, total_mel_frames_produced_, new_mel);
    if (n > 0) {
        mel_buf_.insert(mel_buf_.end(), new_mel.begin(), new_mel.end());
        total_mel_frames_produced_ += n;
    }
}

StreamingUpdate
BufferedStreamRunner::process_window(size_t win_start, size_t win_end, bool is_last) {
    StreamingUpdate update;
    const size_t win_len = win_end - win_start;

    std::vector<float> log_probs;
    std::vector<int32_t> best_ids;
    std::vector<float> best_probs;
    int T_out = 0, n_classes = 0;
    {
        // Slice this window's (win_len/hop + 1) frames from the incremental
        // cache rather than recomputing the ~96%-overlapping window's mel each
        // step. Only the ~2 leftmost frames differ from a per-window reflect
        // pad (real vs reflected left context); they fall in the discarded
        // left-context region. VAD masking, when enabled, mutates only the
        // sliced copy (the cache stays raw). The re-masking across overlapping
        // windows is deliberate, not redundant: decide_masked() scans forward
        // neighbours up to max_scan, whose Silero bits may still be provisional
        // when a frame is first produced; re-deciding it in a later window picks
        // up the now-committed forward context. The decision is a pure function
        // of the timeline by global index, so re-masking is safe/idempotent.
        static const bool t_log = std::getenv("NEMO_SPEECH_TIMING") != nullptr;
        using _clk = std::chrono::high_resolution_clock;
        auto _a = _clk::now();
        ensure_mel_cache();
        auto _b = _clk::now();
        const int n_mels = model_->fe_config().n_mels;
        const int hop = model_->fe().hop_length();
        const int64_t g0 = static_cast<int64_t>(win_start) / hop;
        const int64_t want = static_cast<int64_t>(win_len) / hop + 1;
        const int64_t avail = total_mel_frames_produced_ - g0;
        const int n_win = static_cast<int>(std::max<int64_t>(0, std::min(want, avail)));
        if (n_win > 0) {
            const int64_t s0 = g0 - mel_base_frame_;  // storage index of global frame g0
            std::vector<float> feats(
                mel_buf_.begin() + s0 * n_mels, mel_buf_.begin() + (s0 + n_win) * n_mels);
            auto _c = _clk::now();
            if (vad_masker_)
                vad_masker_->apply(feats.data(), n_win, g0);
            const double stddev_floor = vad_masker_ ? vad_masker_->config().stddev_floor : 1e-5;
            normalize_per_feature(feats, n_mels, n_win, stddev_floor);
            auto _d = _clk::now();
            if (greedy_ctc_) {
                model_->infer_ctc_greedy_from_mel(feats.data(), n_win, best_ids, best_probs, T_out);
                n_classes = model_->ctc_config().num_classes + 1;
            } else {
                model_->infer_ctc_from_mel(feats.data(), n_win, log_probs, T_out, n_classes);
            }
            auto _e = _clk::now();
            if (t_log) {
                auto ms = [](auto a, auto b) {
                    return std::chrono::duration<double, std::milli>(b - a).count();
                };
                fprintf(
                    stderr,
                    "[timing] buffered-chunk n_win=%d mel_cache=%.2f mask+norm=%.2f "
                    "infer=%.2f ms\n",
                    n_win, ms(_a, _b), ms(_c, _d), ms(_d, _e));
            }
        }
    }
    if (T_out == 0 || n_classes == 0)
        return update;

    // Emit only the center chunk's CTC output. Skip the model output frames
    // that correspond to audio in [win_start, emit_start_). In steady state
    // emit_start_ - win_start == left_samples_, so skip_out == left_out_frames_;
    // during startup win_start clamps at 0 while emit_start_ advances, so we
    // must compute skip_out from the live offset to avoid re-emitting the
    // window's leading frames every step.
    const int skip_out = static_cast<int>(
        static_cast<uint64_t>(emit_start_ - win_start) * chunk_out_frames_ / chunk_samples_);
    int T_eff;
    if (is_last) {
        T_eff = std::max(0, T_out - skip_out);
    } else {
        T_eff = std::min(chunk_out_frames_, std::max(0, T_out - skip_out));
    }
    if (T_eff <= 0)
        return update;

    // frame_offset is the GLOBAL ENCODER-frame index of the first log-prob
    // column we hand the decoder (the center chunk start = emit_start_), so the
    // decoder can stamp token timestamps as frame_offset + local_t.
    const int64_t frame_offset =
        static_cast<int64_t>(emit_start_) / std::max<int64_t>(1, enc_frame_samples_);
    std::vector<int> new_ids;
    if (greedy_ctc_) {
        new_ids = greedy_ctc_->step_compact(
            best_ids.data() + skip_out, best_probs.data() + skip_out, T_eff, frame_offset);
    } else {
        new_ids =
            head_->step(log_probs.data() + skip_out * n_classes, n_classes, T_eff, frame_offset);
    }
    for (int id : new_ids) all_tokens_.push_back(id);
    update.new_token_ids = std::move(new_ids);

    if (is_last) {
        head_->finalize();
    }

    // Heads that own decoding internally (e.g. FlashlightDecoder with KenLM)
    // produce the transcript directly. Otherwise detokenize the accumulated
    // greedy token stream.
    const std::string head_transcript =
        is_last ? head_->final_transcript() : head_->partial_transcript();
    if (!head_transcript.empty()) {
        transcript_ = head_transcript;
    } else {
        transcript_ = detokenize_sentencepiece(all_tokens_, head_->vocab());
    }
    update.transcript_so_far = transcript_;
    if (opts_.needs_word_timings())
        update.words = head_->word_timings();
    return update;
}

OfflineRunner::OfflineRunner(
    AsrModel* model, const RecognizerConfig& cfg,
    std::shared_ptr<const FlashlightResources> flashlight)
    : model_(model) {
    if (!model_)
        throw std::invalid_argument("OfflineRunner: null model");
    if (model_->head_kind() == HeadKind::Ctc)
        ctc_decoder_ = make_ctc_decoder(static_cast<CtcModel*>(model_), cfg, std::move(flashlight));
}

void
OfflineRunner::feed_audio(const float* samples, size_t n_samples) {
    if (finalized_)
        throw std::runtime_error("OfflineRunner: feed_audio after finalize");
    audio_.insert(audio_.end(), samples, samples + n_samples);
}

StreamingUpdate
OfflineRunner::finalize() {
    if (finalized_)
        throw std::runtime_error("OfflineRunner: finalize called twice");
    finalized_ = true;
    StreamingUpdate update;
    update.is_final = true;
    update.audio_processed_sec =
        static_cast<float>(audio_.size()) / static_cast<float>(model_->sample_rate());
    if (audio_.empty())
        return update;

    std::unique_ptr<Decoder> owned_decoder;
    Decoder* decoder = nullptr;
    std::vector<int> tokens;
    if (model_->head_kind() == HeadKind::Ctc) {
        auto* ctc = static_cast<CtcModel*>(model_);
        std::vector<float> log_probs;
        int T = 0, C = 0;
        ctc->infer_ctc(audio_.data(), audio_.size(), log_probs, T, C);
        decoder = ctc_decoder_.get();
        decoder->set_compute_timestamps(opts_.enable_word_time_offsets);
        decoder->set_request_options(opts_);
        tokens = decoder->step(log_probs.data(), C, T, 0);
    } else {
        auto* transducer = static_cast<RnntModel*>(model_);
        std::vector<float> enc;
        int T = 0;
        transducer->infer_offline(audio_.data(), audio_.size(), enc, T, prompt_index_);
        owned_decoder = transducer->make_transducer_decoder();
        decoder = owned_decoder.get();
        decoder->set_compute_timestamps(opts_.enable_word_time_offsets);
        decoder->set_finalizing(true);
        const auto decode_begin = std::chrono::steady_clock::now();
        tokens = decoder->step(enc.data(), transducer->rnnt_config().joint_dim, T, 0);
        if (std::getenv("NEMO_SPEECH_TIMING")) {
            std::fprintf(
                stderr, "[timing] offline-transducer decode frames=%d = %.2f ms\n", T,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - decode_begin)
                    .count());
        }
    }

    decoder->finalize();
    update.new_token_ids = tokens;
    update.transcript_so_far = decoder->final_transcript();
    if (update.transcript_so_far.empty())
        update.transcript_so_far = detokenize_sentencepiece(tokens, decoder->vocab());
    detected_languages_ = extract_lang_tags(update.transcript_so_far);
    if (opts_.enable_word_time_offsets)
        update.words = decoder->word_timings();
    update.confidence = decoder->confidence();
    if (opts_.max_alternatives > 1)
        update.extra_alternatives = decoder->additional_hypotheses(opts_.max_alternatives);
    return update;
}

void
OfflineRunner::reset() {
    audio_.clear();
    detected_languages_.clear();
    finalized_ = false;
    if (ctc_decoder_)
        ctc_decoder_->reset();
}

CacheStreamRunner::CacheStreamRunner(
    RnntModel* model, const RecognizerConfig& cfg, std::shared_ptr<SileroVadModel> vad_model)
    : model_(model) {
    if (!model)
        throw std::runtime_error("CacheStreamRunner: null model");

    // Set the requested right context before the cache-aware session is built.
    const int right_ctx = cfg.streaming.rnnt_right_context;
    if (right_ctx >= 0) {
        model->set_cache_right_ctx(right_ctx);
    }

    // The cache-aware encoder geometry the model's CacheAwareEncoder will use -
    // we need cache_left_ctx / cache_right_ctx / cache_chunk_frames for our
    // attn_mask sizing and chunk-shift math. Same derivation the encoder uses
    // (make_cache_aware_config), so the two can't drift.
    enc_cfg_ = make_cache_aware_config(model->encoder_config(), right_ctx);

    // The per-stream K/V/conv cache is device-resident (cache_state_), allocated
    // lazily on first encode. Only attn_mask stays host-side - it's a per-call
    // Input that grows with cache_filled_frames_. The FE and the cache-aware encoder
    // Session are owned by the model; the runner just drives them.
    const int left_ctx = enc_cfg_.cache_left_ctx;
    attn_mask_.assign(left_ctx + enc_cfg_.cache_chunk_frames, 0.0f);

    head_ = model->make_transducer_decoder();

    build_vad_stack(
        model_, cfg.vad.model_path, cfg.vad.masker, cfg.endpointing, std::move(vad_model), vad_,
        vad_masker_, endpointer_);
}

// Out-of-line (not defaulted in the header) so cache_state_'s device buffers
// free against the complete runtime types.
CacheStreamRunner::~CacheStreamRunner() = default;

RnntDecodeStats
CacheStreamRunner::rnnt_decode_stats() const {
    if (const auto* rnnt = dynamic_cast<const RnntGreedyDecoder*>(head_.get()))
        return rnnt->stats();
    if (const auto* tdt = dynamic_cast<const TdtGreedyDecoder*>(head_.get()))
        return tdt->stats();
    return {};
}

void
CacheStreamRunner::set_request_options(const AsrRequestOptions& opts) {
    opts_ = opts;
    if (head_)
        head_->set_request_options(opts);
    if (endpointer_)
        endpointer_->set_stop_history_eou_ms(opts.stop_history_eou_ms);
}

void
CacheStreamRunner::force_eou() {
    if (endpointer_) {
        force_eou_pending_ = true;
        endpointer_->force();
    }
}

void
CacheStreamRunner::zero_caches() {
    // Swap in a fresh zeroed device cache for the next utterance. Only once one
    // has been allocated (first encode) - before that the next encode allocates
    // a zeroed one anyway, so leaving it invalid here preserves the lazy build.
    if (cache_state_.valid()) {
        model_->reset_cache_state(cache_state_);
    }
}

void
CacheStreamRunner::upload_attn_mask() {
    // attn_mask layout: (kv_len, 1). offset = left_ctx - cache_filled_frames:
    // positions [0, offset) are masked (-1e9), rest are 0. Same warm-up rule
    // as NeMo's `offset = cache_len - cache_last_channel_len` feeding
    // _create_masks (conformer_encoder.py), expressed as an additive mask.
    const int left_ctx = enc_cfg_.cache_left_ctx;
    const int kv_len = static_cast<int>(attn_mask_.size());
    const int offset = std::max(0, left_ctx - cache_filled_frames_);
    for (int i = 0; i < kv_len; i++) {
        attn_mask_[i] = (i < offset) ? -1e9f : 0.0f;
    }
}

void
CacheStreamRunner::feed_audio(const float* samples, size_t n_samples) {
    if (finalized_)
        return;
    audio_buf_.insert(audio_buf_.end(), samples, samples + n_samples);
}

void
CacheStreamRunner::process_one_chunk(bool is_last) {
    // NeMo formula: chunk_size_mel = pre_encode_cache_size + sub * (1 + R).
    const int sub = enc_cfg_.subsampling_factor;
    const int R = enc_cfg_.cache_right_ctx;
    const int chunk_size_mel = pre_encode_cache_size_ + sub * (1 + R);
    const int shift_size_mel = sub * (1 + R - cache_drop_size_);
    const int n_mels = model_->fe_config().n_mels;
    const int left_ctx = enc_cfg_.cache_left_ctx;

    upload_attn_mask();

    static const bool t_log = std::getenv("NEMO_SPEECH_TIMING") != nullptr;
    using _clk = std::chrono::high_resolution_clock;
    auto _e0 = _clk::now();

    // mel_buf_ must already hold >= chunk_size_mel frames. The model binds this
    // stream's device-resident cache (cache_state_), runs the encoder, and reads
    // back enc_out; the cache persists on device between chunks (in-graph
    // feedback). Allocated lazily on first use so the Session builds with final R.
    if (!cache_state_.valid()) {
        cache_state_ = model_->make_cache_state();
    }
    model_->encode_cache_aware(
        cache_state_, mel_buf_.data(), chunk_size_mel, attn_mask_.data(),
        static_cast<int>(attn_mask_.size()), last_enc_out_, last_enc_T_, prompt_index_);
    auto _e1 = _clk::now();

    cache_filled_frames_ = std::min(cache_filled_frames_ + last_enc_T_, left_ctx);

    // If a head is attached (RNNT path), drive greedy decoding on this
    // chunk's encoder output. Accumulate emitted tokens + maintain transcript.
    if (head_ && last_enc_T_ > 0) {
        // is_last marks the end-of-utterance flush so the head can apply its
        // EOU punctuation floor (commits a marginal terminal '.'/'?').
        head_->set_finalizing(is_last);
        // RnntModel returns joint.enc-projected features, not the wider raw
        // encoder representation. This projection is computed once for the
        // whole chunk (after optional prompt fusion) and reused by every symbol
        // attempt in the greedy head.
        auto ids = head_->step(
            last_enc_out_.data(), model_->rnnt_config().joint_dim, last_enc_T_,
            total_frames_emitted_);
        for (int id : ids) {
            all_tokens_.push_back(id);
            last_step_new_tokens_.push_back(id);
        }
        if (!ids.empty()) {
            transcript_ = detokenize_sentencepiece(all_tokens_, head_->vocab());
            detected_languages_ = extract_lang_tags(transcript_);
        }
    }

    if (t_log) {
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        auto _e2 = _clk::now();
        fprintf(
            stderr, "[timing] cache-chunk enc_frames=%d encode=%.2f decode=%.2f ms\n", last_enc_T_,
            ms(_e0, _e1), ms(_e1, _e2));
    }

    total_frames_emitted_ += last_enc_T_;
    chunks_processed_ += 1;

    // Advance mel buffer: drop shift_size_mel; keep overlap_mel_frames at
    // the head so the next chunk's leading frames see the same context.
    const size_t shift_floats = static_cast<size_t>(shift_size_mel) * n_mels;
    if (mel_buf_.size() > shift_floats) {
        mel_buf_.erase(mel_buf_.begin(), mel_buf_.begin() + shift_floats);
    } else {
        mel_buf_.clear();
    }
}

StreamingUpdate
CacheStreamRunner::step() {
    StreamingUpdate update;
    if (finalized_)
        return update;

    // Streaming FE - produce only the *new* mel frames since last call.
    //
    // Frame i (NeMo center=True) has its window at audio[i*hop - n_fft/2,
    // i*hop + n_fft/2]. For i >= ceil(n_fft/(2*hop)) (= 2 here), the left
    // edge is real samples; for i < 2 the left edge needs reflection (start
    // of stream - acceptable, NeMo does the same). We only release frames
    // whose RIGHT edge is also real samples (no right-reflection
    // contamination), i.e. i*hop + n_fft/2 <= audio_buf_.size().
    //
    // NOTE: We deliberately do NOT apply per_feature normalization here.
    // The streaming model accepts raw log-mel directly. Per-utterance
    // normalization is incompatible with streaming anyway - stats computed
    // from a partial utterance early on diverge enough from full-utterance
    // stats that the encoder's cache state gets poisoned by wrongly-
    // normalized initial frames.
    const int n_mels = model_->fe_config().n_mels;
    const size_t audio_end = audio_base_ + audio_buf_.size();
    const int64_t i_start = total_mel_frames_produced_;
    std::vector<float> new_mel;
    const int n = produce_new_mel_frames(model_->fe(), audio_buf_, audio_base_, i_start, new_mel);
    if (n > 0) {
        // VAD masking happens in mel space, before these frames are appended to
        // mel_buf_ and consumed by the encoder. Feed all audio not yet seen by
        // Silero (it windows it into 512-sample blocks internally); i_start is
        // the true global index of new_mel's first frame, since frames are
        // produced strictly in order.
        if (vad_) {
            const float* new_audio = audio_buf_.data() + (audio_fed_to_vad_ - audio_base_);
            const size_t n_new = audio_end - audio_fed_to_vad_;
            vad_->observe_audio(new_audio, n_new);  // inference + binarize, once
            audio_fed_to_vad_ = audio_end;
            if (vad_masker_)
                vad_masker_->apply(new_mel.data(), n, i_start);
        }
        mel_buf_.insert(mel_buf_.end(), new_mel.begin(), new_mel.end());
        total_mel_frames_produced_ = i_start + n;
    }

    const int sub = enc_cfg_.subsampling_factor;
    const int R = enc_cfg_.cache_right_ctx;
    const int chunk_size_mel = pre_encode_cache_size_ + sub * (1 + R);

    // Process every whole chunk available, polling the endpointer per chunk
    // so a silence gap interior to one large feed still fires. An EOU breaks
    // the loop; the next utterance's remaining chunks decode on the next
    // step() call.
    last_step_new_tokens_.clear();
    while (static_cast<int>(mel_buf_.size() / n_mels) >= chunk_size_mel) {
        if (!stream_zero_padded_) {
            // First-chunk zero-pad so the encoder's unconditional
            // cache_drop_extra eats zero-pad, not the leading 160 ms of real audio.
            mel_buf_.insert(
                mel_buf_.begin(), static_cast<size_t>(pre_encode_cache_size_) * n_mels, 0.0f);
            stream_zero_padded_ = true;
        }
        process_one_chunk(/*is_last=*/false);
        if (poll_endpoint(update, /*after_chunk=*/true))
            break;
    }
    update.new_token_ids = last_step_new_tokens_;
    // On EOU, fire_eou captured the final transcript + words; the
    // per-utterance buffers are already cleared.
    if (!update.is_final)
        update.transcript_so_far = transcript_;
    update.audio_processed_sec =
        static_cast<float>(audio_end) / static_cast<float>(model_->fe_config().sample_rate);
    // A pending force_eou() must fire even on a chunk-less step.
    if (!update.is_final)
        poll_endpoint(update, /*after_chunk=*/false);
    if (!update.is_final && opts_.needs_word_timings() && head_)
        update.words = head_->word_timings();
    trim_buffers();
    return update;
}

void
CacheStreamRunner::finish_endpoint(StreamingUpdate& update, bool preserve_buffered_future) {
    const int n_mels = model_->fe_config().n_mels;
    const int sub = enc_cfg_.subsampling_factor;
    const int R = enc_cfg_.cache_right_ctx;
    const int chunk_size_mel = pre_encode_cache_size_ + sub * (1 + R);
    const int shift_size_mel = sub * (1 + R - cache_drop_size_);

    // After a decoded chunk, mel_buf_ starts with the encoder overlap. Frames
    // beyond it have not been decoded and belong to the next utterance when
    // endpointing fired automatically. A forced EOU instead commits all audio
    // already supplied by the caller.
    std::vector<float> next_mel;
    if (preserve_buffered_future) {
        const int overlap_frames = chunk_size_mel - shift_size_mel;
        const size_t split = std::min(
            mel_buf_.size(), static_cast<size_t>(overlap_frames) * static_cast<size_t>(n_mels));
        next_mel.assign(mel_buf_.begin() + split, mel_buf_.end());
        mel_buf_.resize(split);
    }

    // Flush the acoustic tail through the same EOS path as finalize(). The
    // synthetic frames may commit terminal punctuation, but they must not
    // advance the stream clock used by the next utterance.
    const int64_t real_frames_emitted = total_frames_emitted_;
    const int real_chunks_processed = chunks_processed_;
    finalizing_ = true;
    if (!mel_buf_.empty()) {
        if (!stream_zero_padded_) {
            mel_buf_.insert(
                mel_buf_.begin(), static_cast<size_t>(pre_encode_cache_size_) * n_mels, 0.0f);
            stream_zero_padded_ = true;
        }
        const size_t flush_frames = static_cast<size_t>(chunk_size_mel + shift_size_mel);
        mel_buf_.resize(mel_buf_.size() + flush_frames * static_cast<size_t>(n_mels), 0.0f);
        while (static_cast<int>(mel_buf_.size() / n_mels) >= chunk_size_mel)
            process_one_chunk(/*is_last=*/true);
    }

    fire_eou(head_.get(), opts_, all_tokens_, transcript_, update);

    // An EOU is a decoder boundary, not a new audio stream. Reset model state
    // and segment-local buffers while retaining global FE/VAD cursors and the
    // absolute encoder-frame clock.
    if (head_)
        head_->reset();
    zero_caches();
    cache_filled_frames_ = 0;
    std::fill(attn_mask_.begin(), attn_mask_.end(), 0.0f);
    mel_buf_ = std::move(next_mel);
    stream_zero_padded_ = false;
    total_frames_emitted_ = real_frames_emitted;
    chunks_processed_ = real_chunks_processed;
    last_enc_out_.clear();
    last_enc_T_ = 0;
    finalizing_ = false;
}

bool
CacheStreamRunner::poll_endpoint(StreamingUpdate& update, bool after_chunk) {
    if (!endpointer_ || finalizing_)
        return false;
    const int sample_rate = model_->fe_config().sample_rate;
    const int hop = model_->fe().hop_length();
    // Both signals run on the decode clock (encoder frames emitted), never
    // the raw-audio frontier; see BufferedStreamRunner::poll_endpoint. The
    // cache-aware decode lag is only ~1 chunk, but the clocks must agree for
    // the threshold to hold its meaning.
    double now_ms = 0.0, last_speech_ms = 0.0;
    if (vad_ && endpointer_->config().vad_based) {
        // VAD-driven: scan the Silero speech bits over the newly decoded mel
        // frames (enc frame f covers mel frames [f*sub, (f+1)*sub)).
        const double mel_ms = 1000.0 * hop / sample_rate;
        const int64_t decoded_mel = total_frames_emitted_ * enc_cfg_.subsampling_factor;
        scan_vad_speech(*vad_, decoded_mel, vad_scan_frame_, vad_speech_seen_frame_);
        now_ms = static_cast<double>(decoded_mel) * mel_ms;
        last_speech_ms = (vad_speech_seen_frame_ < 0)
                             ? 0.0
                             : static_cast<double>(vad_speech_seen_frame_ + 1) * mel_ms;
    } else {
        // Token-silence: time since the decoder's last token emission frame.
        const double frame_ms = model_->ms_per_enc_frame();
        now_ms = static_cast<double>(total_frames_emitted_) * frame_ms;
        const int64_t lef = head_ ? head_->last_emit_frame() : -1;
        last_speech_ms = (lef < 0) ? 0.0 : static_cast<double>(lef + 1) * frame_ms;
    }
    if (!endpointer_->poll(now_ms, last_speech_ms))
        return false;
    const bool preserve_buffered_future = after_chunk && !force_eou_pending_;
    force_eou_pending_ = false;
    finish_endpoint(update, preserve_buffered_future);
    return true;
}

void
CacheStreamRunner::trim_buffers() {
    // Oldest sample still readable: the FE's incremental left edge and any
    // audio Silero hasn't observed yet. mel_buf_ is already front-trimmed per
    // chunk by process_one_chunk.
    const int hop = model_->fe().hop_length();
    const int n_fft = model_->fe_config().n_fft;
    const int64_t fe_from = total_mel_frames_produced_ * hop - n_fft / 2;
    size_t keep_from = (fe_from > 0) ? static_cast<size_t>(fe_from) : 0;
    if (vad_ && audio_fed_to_vad_ < keep_from)
        keep_from = audio_fed_to_vad_;
    if (keep_from > audio_base_) {
        audio_buf_.erase(audio_buf_.begin(), audio_buf_.begin() + (keep_from - audio_base_));
        audio_base_ = keep_from;
    }
    if (vad_) {
        // Silero timeline: keep a margin below the decode cursor so the
        // masker's neighbourhood scans never cross the trimmed base.
        const int64_t margin =
            std::max<int64_t>(256, vad_masker_ ? 2 * vad_masker_->max_scan() : 0);
        const int64_t decoded_mel = total_frames_emitted_ * enc_cfg_.subsampling_factor;
        const int64_t g = decoded_mel - margin;
        if (g > 0)
            vad_->discard_timeline_before(g);
    }
}

StreamingUpdate
CacheStreamRunner::finalize() {
    StreamingUpdate update;
    if (finalized_) {
        fill_final_update(
            update, head_.get(), opts_.needs_word_timings(), transcript_,
            static_cast<float>(audio_base_ + audio_buf_.size()) /
                static_cast<float>(model_->fe_config().sample_rate),
            opts_.max_alternatives);
        return update;
    }
    finalizing_ = true;

    // Flush Silero before the tail is masked + encoded, mirroring the buffered
    // runner. Masking happens at mel-production time inside step(), so the
    // trailing partial Silero window must be finalized first or tail frames
    // mask against preliminary speech/silence bits. Feed any audio Silero has
    // not yet seen, then flush; the synthetic zero pad appended below is never
    // fed to Silero (it stays unmasked, as intended).
    if (vad_) {
        const size_t audio_end = audio_base_ + audio_buf_.size();
        if (audio_end > audio_fed_to_vad_) {
            vad_->observe_audio(
                audio_buf_.data() + (audio_fed_to_vad_ - audio_base_),
                audio_end - audio_fed_to_vad_);
            audio_fed_to_vad_ = audio_end;
        }
        vad_->flush_timeline();
    }

    auto drained = step();  // process any whole chunks first.
    update.new_token_ids = std::move(drained.new_token_ids);

    const int n_mels = model_->fe_config().n_mels;
    const int sub = enc_cfg_.subsampling_factor;
    const int R = enc_cfg_.cache_right_ctx;
    const int chunk_size_mel = pre_encode_cache_size_ + sub * (1 + R);
    const int shift_size_mel = sub * (1 + R - cache_drop_size_);
    const size_t buffered = mel_buf_.size() / n_mels;
    if (buffered > 0) {
        // A stream shorter than one chunk never entered step()'s chunk
        // loop, so the start-of-stream zero-pad hasn't happened yet; apply
        // it here or the encoder's cache_drop_extra eats the leading
        // ~160 ms of real audio instead of pad.
        if (!stream_zero_padded_) {
            mel_buf_.insert(
                mel_buf_.begin(), static_cast<size_t>(pre_encode_cache_size_) * n_mels, 0.0f);
            stream_zero_padded_ = true;
        }
        // Tail flush: pad enough zero mel for the final real frame to
        // see a full right-context window AND for the RNNT predictor
        // to get at least one all-zero enc frame to commit pending tail tokens.
        const size_t flush_pad_frames = static_cast<size_t>(chunk_size_mel + shift_size_mel);
        mel_buf_.resize(mel_buf_.size() + flush_pad_frames * static_cast<size_t>(n_mels), 0.0f);
        const size_t before = all_tokens_.size();
        last_step_new_tokens_.clear();
        while (static_cast<int>(mel_buf_.size() / n_mels) >= chunk_size_mel) {
            process_one_chunk(/*is_last=*/true);
        }
        for (size_t i = before; i < all_tokens_.size(); i++) {
            update.new_token_ids.push_back(all_tokens_[i]);
        }
    }

    finalized_ = true;
    // Flush the head's trailing in-progress word into word_timings before
    // reading them: an EOS open word otherwise never reaches word_timings()
    // (mid-stream EOU flushes via fire_eou; EOS did not). Idempotent - a second
    // finalize() flushes nothing.
    if (head_)
        head_->finalize();
    fill_final_update(
        update, head_.get(), opts_.needs_word_timings(), transcript_,
        static_cast<float>(audio_base_ + audio_buf_.size()) /
            static_cast<float>(model_->fe_config().sample_rate),
        opts_.max_alternatives);
    return update;
}

void
CacheStreamRunner::reset() {
    audio_buf_.clear();
    audio_base_ = 0;
    mel_buf_.clear();
    total_mel_frames_produced_ = 0;
    if (vad_) {
        vad_->reset();
    }
    audio_fed_to_vad_ = 0;
    if (endpointer_)
        endpointer_->reset();
    force_eou_pending_ = false;
    vad_scan_frame_ = 0;
    vad_speech_seen_frame_ = -1;
    cache_filled_frames_ = 0;
    chunks_processed_ = 0;
    total_frames_emitted_ = 0;
    last_enc_out_.clear();
    last_enc_T_ = 0;
    finalized_ = false;
    finalizing_ = false;
    stream_zero_padded_ = false;
    zero_caches();
    std::fill(attn_mask_.begin(), attn_mask_.end(), 0.0f);
    if (head_)
        head_->reset();
    all_tokens_.clear();
    transcript_.clear();
    last_step_new_tokens_.clear();
    detected_languages_.clear();
}

int
CacheStreamRunner::take_encoder_frames(std::vector<float>& out, int& T_out, int& d_model) {
    out = last_enc_out_;
    T_out = last_enc_T_;
    d_model = model_->rnnt_config().joint_dim;
    return last_enc_T_;
}

}  // namespace nemo_speech::asr
