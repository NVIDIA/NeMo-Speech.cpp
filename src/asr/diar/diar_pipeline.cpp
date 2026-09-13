// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "diar_pipeline.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

using namespace nemo_speech::asr;

static MelSpecConfig
make_fe_cfg(const SortformerModelConfig& c) {
    MelSpecConfig fe;
    fe.sample_rate = c.sample_rate;
    fe.n_fft = c.n_fft;
    fe.n_mels = c.n_mels;
    fe.window_size = c.window_size;
    fe.window_stride = c.window_stride;
    fe.preemph = c.preemph;
    fe.log_zero_guard = c.log_zero_guard;
    // NeMo torch.stft parity (see MelSpecConfig): centered window, symmetric
    // hann. The ASR models keep the legacy placement they were validated with.
    fe.stft_center_window = true;
    fe.hann_periodic = false;
    return fe;
}

DiarModel::DiarModel(
    ggml_runtime::BackendManager& bm, const std::string& gguf_path, const BatchingConfig& batching)
    : model_(bm, gguf_path, batching), fe_cfg_(make_fe_cfg(model_.cfg())),
      fe_(fe_cfg_, /*bm=*/nullptr) {  // CPU FFT path, same choice as AsrModel
    const int n_bins = fe_cfg_.n_fft / 2 + 1;
    fe_.set_mel_basis(model_.mel_basis().data(), fe_cfg_.n_mels, n_bins);
}

std::vector<float>
DiarModel::diarize_offline(const float* audio, size_t n_samples, int64_t* n_frames) {
    // NeMo's offline path (streaming_mode=False, process_signal) peak-
    // normalizes the waveform before FE: x * 1/(max(x) + eps), eps=1e-3.
    // The streaming path does NOT - this is offline-only.
    float peak = n_samples ? audio[0] : 0.f;
    for (size_t i = 1; i < n_samples; i++) peak = std::max(peak, audio[i]);
    const float gain = 1.0f / (peak + 1e-3f);
    std::vector<float> scaled(n_samples);
    for (size_t i = 0; i < n_samples; i++) scaled[i] = audio[i] * gain;

    // Whole-file mel in one FE call (reflect-left at stream start; the FE
    // zero-pads the right edge internally, NeMo pad_mode="constant").
    std::vector<float> mel;
    int t_mel = 0;
    fe_.compute(scaled.data(), n_samples, mel, t_mel, /*reflect_left=*/true, /*normalize=*/false);

    const int t_enc = model_.subsampled_len(t_mel);
    const int max_enc = model_.cfg().encoder.pos_emb_max_len;
    if (t_enc > max_enc) {
        throw std::invalid_argument(
            "diarize_offline: " + std::to_string(t_enc) +
            " encoder frames exceeds the rel-pos "
            "table (" +
            std::to_string(max_enc) + " = ~" + std::to_string(max_enc * 8 / 100 / 60) +
            " min); use DiarStream for long-form audio");
    }

    // One forward with empty spkcache/fifo == NeMo streaming_mode=False.
    auto out = model_.run_chunk(mel.data(), t_mel, nullptr, 0, nullptr, 0);
    if (n_frames)
        *n_frames = out.total_frames;
    return std::move(out.preds);
}

DiarStream::DiarStream(DiarModel& model, const DiarGeometry& geometry)
    : m_(model), geo_(geometry), n_spk_(model.cfg().num_speakers),
      sub_(model.cfg().encoder.subsampling_factor),
      sec_per_frame_(
          model.cfg().encoder.subsampling_factor * static_cast<double>(model.cfg().window_stride)),
      state_(geo_, model.cfg().scoring, n_spk_, model.cfg().encoder.d_model), birth_gate_(n_spk_) {
    n_mels_ = m_.fe().n_mels();
    geo_.validate(
        n_spk_, model.cfg().scoring.sil_frames_per_spk, model.cfg().encoder.pos_emb_max_len);
}

void
DiarStream::reset() {
    state_ = AoscState(geo_, m_.cfg().scoring, n_spk_, m_.cfg().encoder.d_model);
    birth_gate_.reset();
    audio_buf_.clear();
    audio_base_ = 0;
    mel_buf_.clear();
    mel_base_ = 0;
    mel_consumed_ = 0;
    finished_ = false;
    probs_.clear();
    probs_base_ = 0;
    frozen_segs_.clear();
}

void
DiarStream::ensure_mel() {
    std::vector<float> new_mel;
    const int n = produce_new_mel_frames(m_.fe(), audio_buf_, audio_base_, mel_produced(), new_mel);
    if (n > 0)
        mel_buf_.insert(mel_buf_.end(), new_mel.begin(), new_mel.end());
}

void
DiarStream::feed_audio(const float* samples, size_t n_samples) {
    // Trailing push after finish() is a silent no-op, matching the ASR
    // runners' post-finalize contract (a capture thread racing the stop path
    // must not turn into a mode-dependent stream error).
    if (finished_)
        return;
    audio_buf_.insert(audio_buf_.end(), samples, samples + n_samples);
    ensure_mel();
    run_ready_chunks(/*end_of_stream=*/false);
}

void
DiarStream::finish() {
    if (finished_)
        return;
    finished_ = true;
    // Release the tail frames the streaming FE held back (their windows run
    // past the last real sample). NeMo's torch.stft(center=True,
    // pad_mode="constant") pads the right edge with zeros AFTER pre-emphasis;
    // our FE pre-emphasizes whatever we append, so plain zeros would leak a
    // -preemph*x[N-1] impulse into the first pad sample. Append a geometric
    // decay tail x[N-1]*a^(k+1) instead: y[k] = a^(k+1)*x[N-1] -
    // a*a^k*x[N-1] = 0, i.e. the post-preemphasis pad is exactly zero,
    // matching NeMo's tail frames bit-for-bit (a=0 degenerates to zeros).
    const int half = m_.fe().n_fft() / 2;
    if (!audio_buf_.empty() || audio_base_ > 0) {
        const float a = m_.cfg().preemph;
        float tail = audio_buf_.empty() ? 0.f : audio_buf_.back();
        for (int k = 0; k < half; k++) {
            tail *= a;
            audio_buf_.push_back(tail);
        }
        ensure_mel();
    }
    run_ready_chunks(/*end_of_stream=*/true);
}

// Chunk scheduler, the streaming counterpart of NeMo's streaming_feat_loader:
// the next chunk covers mel [mel_consumed_, mel_consumed_ + hop) plus lc/rc
// context, clamped at the stream edges (riva feeds exact-length tails - no
// pad+mask). Forced (on-demand) chunks may be shorter than a full hop but
// stay on the 80 ms encoder-frame grid so frames are labeled exactly once.
bool
DiarStream::run_one_chunk(bool force, bool final_flush) {
    const int hop_mel = geo_.chunk_len * sub_;
    const int lc_mel_max = geo_.chunk_left_context * sub_;
    const int rc_mel_max = geo_.chunk_right_context * sub_;

    const int64_t stt = mel_consumed_;
    int64_t end = stt + hop_mel;
    if (!force) {
        // Run only when the full window incl. right context is covered.
        if (end + rc_mel_max > mel_produced())
            return false;
    } else {
        end = std::min(end, mel_produced());
        if (!final_flush)
            end = stt + ((end - stt) / sub_) * sub_;  // whole encoder frames only
        if (end <= stt)
            return false;
    }
    const int64_t lc_mel = std::min<int64_t>(lc_mel_max, stt);
    const int64_t rc_mel = std::min<int64_t>(rc_mel_max, mel_produced() - end);
    const int64_t w0 = stt - lc_mel;
    const int t_mel = static_cast<int>(end + rc_mel - w0);
    if (t_mel <= 0)
        return false;

    if (w0 < mel_base_)
        throw std::runtime_error("DiarStream: mel window trimmed too aggressively");
    const float* mel = mel_buf_.data() + (w0 - mel_base_) * m_.fe().n_mels();

    auto out = m_.model().run_chunk(
        mel, t_mel, state_.spkcache_frames() ? state_.spkcache().data() : nullptr,
        state_.spkcache_frames(), state_.fifo_frames() ? state_.fifo().data() : nullptr,
        state_.fifo_frames());

    const int lc_enc = static_cast<int>(std::lround(lc_mel / static_cast<double>(sub_)));
    const int rc_enc = static_cast<int>(std::ceil(rc_mel / static_cast<double>(sub_)));
    auto emitted =
        state_.update(out.chunk_embs.data(), out.chunk_frames, out.preds.data(), lc_enc, rc_enc);
    birth_gate_.append(emitted, probs_);
    maybe_compact();
    mel_consumed_ = end;

    // Trim consumed buffers. The next window starts at mel_consumed_ -
    // lc_mel_max; its first FE sample sits n_fft/2 before that frame's hop
    // position.
    const int64_t keep_mel = std::max<int64_t>(mel_consumed_ - lc_mel_max, 0);
    if (keep_mel > mel_base_) {
        mel_buf_.erase(
            mel_buf_.begin(), mel_buf_.begin() + (keep_mel - mel_base_) * m_.fe().n_mels());
        mel_base_ = keep_mel;
    }
    const int64_t keep_sample =
        std::max<int64_t>(keep_mel * m_.fe().hop_length() - m_.fe().n_fft() / 2, 0);
    if (keep_sample > static_cast<int64_t>(audio_base_)) {
        audio_buf_.erase(audio_buf_.begin(), audio_buf_.begin() + (keep_sample - audio_base_));
        audio_base_ = static_cast<size_t>(keep_sample);
    }
    return true;
}

void
DiarStream::run_ready_chunks(bool end_of_stream) {
    while (run_one_chunk(/*force=*/end_of_stream, /*final_flush=*/end_of_stream)) {
    }
}

void
DiarStream::flush_available(int64_t target_frame) {
    while (n_frames() < target_frame && run_one_chunk(/*force=*/true, /*final_flush=*/false)) {
    }
}

namespace {

// Nearest frozen-segment speaker for a time range that starts before the
// live window. `frozen` is sorted by t0 (see DiarStream::maybe_compact()).
// Returns -1 when `frozen` is empty.
int
speaker_from_frozen(const std::vector<DiarSegment>& frozen, double t0, double t1) {
    if (frozen.empty())
        return -1;
    // First segment that could overlap [t0, t1): the first whose end is
    // after t0.
    auto it = std::lower_bound(
        frozen.begin(), frozen.end(), t0,
        [](const DiarSegment& seg, double t) { return seg.t1 <= t; });
    if (it != frozen.end() && it->t0 < t1)
        return it->speaker;
    // No direct overlap - the range sits in a gap the segmenter dropped
    // (e.g. a short silence). Use whichever neighbor is temporally closer.
    if (it == frozen.begin())
        return it->speaker;
    if (it == frozen.end())
        return std::prev(it)->speaker;
    const auto prev = std::prev(it);
    return (t0 - prev->t1 <= it->t0 - t1) ? prev->speaker : it->speaker;
}

}  // namespace

int
nemo_speech::asr::speaker_for_frame_range(
    const std::vector<float>& probs, int64_t probs_base, int n_spk, double sec_per_frame,
    const std::vector<DiarSegment>& frozen, int64_t start_frame, int64_t end_frame) {
    if (start_frame < probs_base) {
        // This range was already compacted out of the live window (see
        // DiarStream::maybe_compact()). The clamp below would silently
        // substitute whatever sits at the live window's edge; ask the
        // frozen segments, which actually cover this span, instead.
        const int frozen_speaker =
            speaker_from_frozen(frozen, start_frame * sec_per_frame, end_frame * sec_per_frame);
        if (frozen_speaker >= 0)
            return frozen_speaker;
        // Nothing frozen this early either (e.g. compaction hasn't run
        // yet) - fall through to the live-window clamp below.
    }
    const int64_t n = probs_base + static_cast<int64_t>(probs.size()) / n_spk;
    if (n == 0 || probs.empty())
        return -1;
    // riva extrapolates words past the diarized frontier with the last frame.
    start_frame = std::min(std::max<int64_t>(start_frame, probs_base), n - 1);
    end_frame = std::max(start_frame + 1, std::min(end_frame, n));
    std::vector<double> mean(n_spk, 0.0);
    for (int64_t f = start_frame; f < end_frame; f++)
        for (int s = 0; s < n_spk; s++) mean[s] += probs[(f - probs_base) * n_spk + s];
    int best = 0;
    for (int s = 1; s < n_spk; s++)
        if (mean[s] > mean[best])
            best = s;
    return best;
}

int
DiarStream::speaker_for_frames(int64_t start_frame, int64_t end_frame) const {
    return speaker_for_frame_range(
        probs_, probs_base_, n_spk_, sec_per_frame_, frozen_segs_, start_frame, end_frame);
}

int
DiarStream::speaker_for_time(double t0, double t1) const {
    const int64_t f0 = static_cast<int64_t>(t0 / sec_per_frame_);
    const int64_t f1 = static_cast<int64_t>(std::ceil(t1 / sec_per_frame_));
    return speaker_for_frames(f0, f1);
}

int
DiarStream::speaker_for_word_time(double t0, double t1) const {
    const int64_t f0 = static_cast<int64_t>(t0 / sec_per_frame_);
    const int64_t f1 = static_cast<int64_t>(std::ceil(t1 / sec_per_frame_));
    return speaker_for_frames(f0, std::min(f1, f0 + 2));
}

std::optional<DiarSpeakerChange>
nemo_speech::asr::detect_speaker_change(
    const std::vector<DiarSegment>& segments, std::optional<int> last_reported) {
    if (segments.empty())
        return std::nullopt;
    const DiarSegment& latest = segments.back();
    if (last_reported.has_value() && *last_reported == latest.speaker)
        return std::nullopt;
    return DiarSpeakerChange{latest.speaker, latest.t0};
}

std::vector<DiarSegment>
nemo_speech::asr::diar_segments_from_probs(
    const float* probs, int64_t n_frames, int n_spk, double sec_per_frame,
    const DiarSegmentationCfg& cfg) {
    std::vector<DiarSegment> out;
    const int64_t n = n_frames;
    const double total = n * sec_per_frame;
    for (int s = 0; s < n_spk; s++) {
        // 1. Onset/offset hysteresis + edge padding.
        std::vector<std::pair<double, double>> segs;
        bool active = false;
        int64_t start = 0;
        for (int64_t f = 0; f < n; f++) {
            const float p = probs[f * n_spk + s];
            if (!active && p > cfg.onset) {
                active = true;
                start = f;
            } else if (active && p < cfg.offset) {
                active = false;
                segs.emplace_back(
                    std::max(0.0, start * sec_per_frame - cfg.pad_onset),
                    std::min(total, f * sec_per_frame + cfg.pad_offset));
            }
        }
        if (active)
            segs.emplace_back(std::max(0.0, start * sec_per_frame - cfg.pad_onset), total);

        // 2. Fill short gaps, 3. drop short segments (pyannote binarize order).
        std::vector<std::pair<double, double>> merged;
        for (const auto& sg : segs) {
            if (!merged.empty() && sg.first - merged.back().second < cfg.min_duration_off)
                merged.back().second = std::max(merged.back().second, sg.second);
            else
                merged.push_back(sg);
        }
        for (const auto& sg : merged)
            if (sg.second - sg.first >= cfg.min_duration_on)
                out.push_back({sg.first, sg.second, s});
    }
    // stable_sort: two segments can share an identical t0 (padding can clamp
    // multiple speakers' segment starts to 0.0 at stream start), and their
    // relative order should stay deterministic run to run.
    std::stable_sort(out.begin(), out.end(), [](const DiarSegment& a, const DiarSegment& b) {
        return a.t0 < b.t0;
    });
    return out;
}

std::vector<DiarStream::Segment>
DiarStream::segments(const DiarSegmentationCfg& cfg) const {
    auto tail = diar_segments_from_probs(
        probs_.data(), static_cast<int64_t>(probs_.size()) / n_spk_, n_spk_, sec_per_frame_, cfg);
    if (frozen_segs_.empty())
        return tail;
    // Shift the tail onto the global clock and append after the frozen prefix.
    // The compaction cut sits inside an all-speaker-silent gap wider than the
    // postprocessing's temporal reach, so prefix and tail segments can never
    // pad or merge across it; frozen ends strictly precede tail starts.
    const double base_sec = probs_base_ * sec_per_frame_;
    std::vector<Segment> out = frozen_segs_;
    out.reserve(out.size() + tail.size());
    for (auto& sg : tail) out.push_back({sg.t0 + base_sec, sg.t1 + base_sec, sg.speaker});
    return out;
}

void
DiarStream::set_compaction(int64_t trigger_frames, int64_t retain_frames) {
    if (retain_frames <= 0 || trigger_frames <= retain_frames)
        throw std::invalid_argument("compaction: need trigger_frames > retain_frames > 0");
    compact_trigger_frames_ = trigger_frames;
    compact_retain_frames_ = retain_frames;
}

void
DiarStream::maybe_compact() {
    const int64_t retained = static_cast<int64_t>(probs_.size()) / n_spk_;
    if (retained <= compact_trigger_frames_)
        return;
    // A cut is safe only where every speaker is silent for longer than the
    // postprocessing's temporal reach: below min(onset, offset) nothing opens
    // (> onset required) and anything open closes (< offset), and a gap wider
    // than min_duration_off + both pads can never be bridged by merging or
    // edge padding. Freezing the prefix at such a cut therefore yields
    // exactly the segments a full recompute would produce there.
    const DiarSegmentationCfg cfg;  // library defaults; see header contract
    const float thr = std::min(cfg.onset, cfg.offset);
    const int64_t gap_need = static_cast<int64_t>(
        std::ceil((cfg.min_duration_off + cfg.pad_onset + cfg.pad_offset + 0.5) / sec_per_frame_));
    const int64_t cut_limit = retained - compact_retain_frames_;
    int64_t run = 0, best_cut = -1;
    for (int64_t f = 0; f < cut_limit; f++) {
        bool silent = true;
        for (int s = 0; s < n_spk_ && silent; s++) silent = probs_[f * n_spk_ + s] < thr;
        run = silent ? run + 1 : 0;
        if (run >= gap_need)
            best_cut = f - gap_need / 2;  // middle of the gap
    }
    if (best_cut <= 0)
        return;  // no safe gap yet; retry as more audio arrives
    auto prefix = diar_segments_from_probs(probs_.data(), best_cut, n_spk_, sec_per_frame_, cfg);
    const double base_sec = probs_base_ * sec_per_frame_;
    for (const auto& sg : prefix)
        frozen_segs_.push_back({sg.t0 + base_sec, sg.t1 + base_sec, sg.speaker});
    probs_.erase(probs_.begin(), probs_.begin() + best_cut * n_spk_);
    probs_base_ += best_cut;
}
