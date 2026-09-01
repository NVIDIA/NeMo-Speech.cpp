// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "fe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "nvtx_utils.h"
#include "runtime.h"

// NeMo's per_feature normalization matches this formula:
//   mean_i = mean over time of mel_i
//   var_i  = var  over time of mel_i
//   out_i  = (mel_i - mean_i) / (sqrt(var_i) + CONSTANT)  (CONSTANT=1e-5 default)
// Applied per utterance. See nemo.collections.asr.parts.preprocessing.features.

static constexpr float kPi = 3.14159265358979323846f;

static float
hz_to_mel(float f)  // slaney (HTK-like used by NeMo default)
{
    return 1127.0f * std::log(1.0f + f / 700.0f);
}

static float
mel_to_hz(float m) {
    return 700.0f * (std::exp(m / 1127.0f) - 1.0f);
}

// Iterative radix-2 FFT over real and imaginary arrays of equal length.
static void
fft_radix2_inplace(std::vector<float>& re, std::vector<float>& im) {
    const int n = static_cast<int>(re.size());
    if (n <= 1)
        return;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * kPi / len;
        const float wre = std::cos(ang);
        const float wim = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            float cur_re = 1.0f, cur_im = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                const float ur = re[i + k];
                const float ui = im[i + k];
                const float vr = re[i + k + len / 2] * cur_re - im[i + k + len / 2] * cur_im;
                const float vi = re[i + k + len / 2] * cur_im + im[i + k + len / 2] * cur_re;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                const float nre = cur_re * wre - cur_im * wim;
                const float nim = cur_re * wim + cur_im * wre;
                cur_re = nre;
                cur_im = nim;
            }
        }
    }
}

// GPU graph: im2col → window → DFT → power → mel projection → log.
class FeModule : public ggml_runtime::Module {
   public:
    FeModule(
        const MelSpecConfig& cfg, const std::vector<float>& window,
        const std::vector<float>& mel_basis)
        : cfg_(cfg), n_bins_(cfg.n_fft / 2 + 1) {
        // ggml has no FFT op, so the GPU path uses real DFT matrices shared by
        // all frames.
        const int n_fft = cfg.n_fft;
        dft_re_.assign((size_t)n_fft * (size_t)n_bins_, 0.0f);
        dft_im_.assign((size_t)n_fft * (size_t)n_bins_, 0.0f);
        const double TWO_PI = 6.283185307179586476925286766559;
        for (int k = 0; k < n_bins_; ++k) {
            for (int n = 0; n < n_fft; ++n) {
                const double ang = TWO_PI * (double)k * (double)n / (double)n_fft;
                dft_re_[(size_t)n + (size_t)k * n_fft] = static_cast<float>(std::cos(ang));
                dft_im_[(size_t)n + (size_t)k * n_fft] = static_cast<float>(-std::sin(ang));
            }
        }
        // Window placement inside the n_fft frame: torch.stft centers it
        // (stft_center_window=true, NeMo parity); legacy ASR GGUFs were
        // validated against right-aligned placement (offset 0).
        const int woff = cfg.stft_center_window ? (n_fft - static_cast<int>(window.size())) / 2 : 0;
        window_padded_.assign(n_fft, 0.0f);
        for (size_t i = 0; i < window.size() && woff + (int)i < n_fft; ++i)
            window_padded_[woff + i] = window[i];
        mel_basis_ = mel_basis;  // (n_mels, n_bins) row-major
        // ggml_im2col reads its kernel arg only for output sizing; never reads
        // the data. Allocate zeros once.
        dummy_im2col_kernel_.assign(n_fft, 0.0f);
    }
    ~FeModule() override = default;

    void define_tensors(ggml_runtime::Session* session) override {
        auto* tc = session->model_tensor_container.get();
        tc->create_tensor_2d("fe.dft_re", GGML_TYPE_F32, cfg_.n_fft, n_bins_);
        tc->create_tensor_2d("fe.dft_im", GGML_TYPE_F32, cfg_.n_fft, n_bins_);
        tc->create_tensor_1d("fe.window", GGML_TYPE_F32, cfg_.n_fft);
        tc->create_tensor_2d("fe.mel_basis", GGML_TYPE_F32, n_bins_, cfg_.n_mels);
        tc->create_tensor_3d("fe.im2col_kernel", GGML_TYPE_F32, cfg_.n_fft, 1, 1);
    }

    void set_data(ggml_runtime::Session* session) override { upload_static_weights(session); }

    void update_mel_basis(ggml_runtime::Session* session, const float* fb, int n_mels, int n_bins) {
        mel_basis_.assign(fb, fb + (size_t)n_mels * (size_t)n_bins);
        auto t = session->model_tensor_container->get_tensor_by_name("fe.mel_basis");
        ggml_backend_tensor_set(t.tensor, mel_basis_.data(), 0, mel_basis_.size() * sizeof(float));
    }

    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override {
        // Packed input shape is (n_audio_padded, refl_flag, batch).  refl_flag
        // remains a shape-key discriminator (only row zero contains audio),
        // while ne[2] is the real dynamic batch dimension.
        auto x = input_tensors.get_tensor(0);
        const int n_audio_padded = static_cast<int>(x.tensor->ne[0]);
        const int refl_flag = static_cast<int>(x.tensor->ne[1]);
        const int batch = static_cast<int>(x.tensor->ne[2]);
        // refl_flag = 2 → reflect_left=true (CPU-side pads n_fft/2 on the
        // left, so im2col pads n_fft/2 too). refl_flag = 1 → reflect_left=false
        // (caller pre-prepended real left context, no extra pad). Encoded via
        // ne[1] so Session::run_cache_ keys disambiguate the two graphs even
        // when n_audio_padded happens to coincide.
        const bool reflect_left = (refl_flag == 2);
        const int p0 = reflect_left ? cfg_.n_fft / 2 : 0;
        const int hop = static_cast<int>(cfg_.window_stride * cfg_.sample_rate + 0.5f);
        auto bf_ctx = tc->get_ctx_of_buffer_type(x.buft);
        ggml_context* g = bf_ctx.ctx;

        // Weight tensors live in the session's model_tensor_container (set up
        // by define_tensors + set_data during Session::setup()); the per-run
        // `tc` is the probe TC that holds this call's input + scratch.
        auto* mtc = session->model_tensor_container.get();
        auto kernel = mtc->get_tensor_by_name("fe.im2col_kernel");
        auto window = mtc->get_tensor_by_name("fe.window");
        auto dft_re = mtc->get_tensor_by_name("fe.dft_re");
        auto dft_im = mtc->get_tensor_by_name("fe.dft_im");
        auto mel_basis = mtc->get_tensor_by_name("fe.mel_basis");

        // `x` is [audio, reflect-shape-key, batch]. Only the first row of the
        // reflect dimension contains samples; its sole purpose is to keep the
        // two padding modes distinct in Session's shape cache. View those rows
        // as a strided [audio, 1, batch] tensor. ggml's 1-D CUDA im2col already
        // treats ne[2] as N and honors nb[2], so one invocation frames the
        // entire microbatch without allowing windows to cross item boundaries.
        //
        // Preserve B through the graph so output is directly [mel, frame, batch].
        ggml_tensor* audio_batch = ggml_view_3d(
            g, x.tensor, n_audio_padded, 1, batch, x.tensor->nb[1], x.tensor->nb[2], 0);
        ggml_tensor* frames = ggml_im2col(
            g, kernel.tensor, audio_batch,
            /*s0=*/hop, /*s1=*/0,
            /*p0=*/p0, /*p1=*/0,
            /*d0=*/1, /*d1=*/0,
            /*is_2D=*/false, GGML_TYPE_F32);
        ggml_tensor* window_2d = ggml_reshape_2d(g, window.tensor, cfg_.n_fft, 1);
        ggml_tensor* windowed = ggml_mul(g, frames, window_2d);
        ggml_tensor* re_spec = ggml_mul_mat(g, dft_re.tensor, windowed);
        ggml_tensor* im_spec = ggml_mul_mat(g, dft_im.tensor, windowed);
        ggml_tensor* power = ggml_add(g, ggml_sqr(g, re_spec), ggml_sqr(g, im_spec));
        ggml_tensor* mel = ggml_mul_mat(g, mel_basis.tensor, power);
        ggml_tensor* mel_eps = ggml_scale_bias(g, mel, 1.0f, cfg_.log_zero_guard);
        ggml_tensor* log_mel = ggml_log(g, mel_eps);
        ggml_set_name(log_mel, "log_mel");

        ggml_runtime::TensorBag out;
        out.add_tensor(ggml_runtime::ggml_bf_tensor(log_mel, x.buft));
        return out;
    }

    const MelSpecConfig& cfg() const { return cfg_; }

   private:
    void upload_static_weights(ggml_runtime::Session* session) {
        auto* tc = session->model_tensor_container.get();
        auto dft_re = tc->get_tensor_by_name("fe.dft_re");
        auto dft_im = tc->get_tensor_by_name("fe.dft_im");
        auto win = tc->get_tensor_by_name("fe.window");
        auto mb = tc->get_tensor_by_name("fe.mel_basis");
        auto k = tc->get_tensor_by_name("fe.im2col_kernel");
        ggml_backend_tensor_set(dft_re.tensor, dft_re_.data(), 0, dft_re_.size() * sizeof(float));
        ggml_backend_tensor_set(dft_im.tensor, dft_im_.data(), 0, dft_im_.size() * sizeof(float));
        ggml_backend_tensor_set(
            win.tensor, window_padded_.data(), 0, window_padded_.size() * sizeof(float));
        ggml_backend_tensor_set(mb.tensor, mel_basis_.data(), 0, mel_basis_.size() * sizeof(float));
        ggml_backend_tensor_set(
            k.tensor, dummy_im2col_kernel_.data(), 0, dummy_im2col_kernel_.size() * sizeof(float));
    }

    MelSpecConfig cfg_;
    int n_bins_;
    std::vector<float> dft_re_;
    std::vector<float> dft_im_;
    std::vector<float> window_padded_;
    std::vector<float> mel_basis_;
    std::vector<float> dummy_im2col_kernel_;
};

class MelSpectrogramExtractor::GpuBatcher {
   public:
    struct Key {
        size_t n_samples = 0;
        bool reflect_left = true;
        bool normalize = true;
        bool operator==(const Key& other) const {
            return n_samples == other.n_samples && reflect_left == other.reflect_left &&
                   normalize == other.normalize;
        }
    };

    GpuBatcher(const MelSpectrogramExtractor* owner, const nemo_speech::asr::BatchingConfig& cfg)
        : owner_(owner), queue_(cfg, [this](const Key&, std::vector<BatchRequest>&& requests) {
              const ggml_nvtx::range nvtx("asr.frontend.batch");
              return owner_->compute_gpu_batch_via_session(std::move(requests));
          }) {}

    BatchResult run(
        const float* audio, size_t n_samples, size_t valid_samples, bool reflect_left,
        bool normalize) {
        BatchRequest request;
        request.audio.assign(audio, audio + n_samples);
        request.valid_samples = valid_samples;
        request.reflect_left = reflect_left;
        request.normalize = normalize;
        return queue_.run({n_samples, reflect_left, normalize}, std::move(request));
    }

    nemo_speech::asr::BatchMetrics metrics() const { return queue_.metrics(); }

   private:
    const MelSpectrogramExtractor* owner_;
    nemo_speech::asr::MicroBatcher<Key, BatchRequest, BatchResult> queue_;
};

MelSpectrogramExtractor::MelSpectrogramExtractor(
    const MelSpecConfig& cfg, ggml_runtime::BackendManager* bm,
    const nemo_speech::asr::BatchingConfig& batching)
    : cfg_(cfg), bm_(bm) {
    if (cfg_.fmax <= 0.0f)
        cfg_.fmax = 0.5f * cfg_.sample_rate;
    int v = cfg_.n_fft;
    while (v > 1) {
        if (v & 1)
            throw std::runtime_error("n_fft must be power of 2 for this FE");
        v >>= 1;
    }
    init_window();
    init_mel_basis();

    if (bm_ != nullptr && bm_->gpu_backend_handle() != nullptr) {
        module_ = std::make_unique<FeModule>(cfg_, window_, mel_basis_);
        session_ =
            std::make_unique<ggml_runtime::Session>(*bm_, module_.get(), /*gguf_loader=*/nullptr);
        // Dynamic streaming sees both startup/tail lengths and several
        // observed batch widths. FE entries are small, so retain enough shapes
        // to avoid rebuilding them at each stream start.
        if (batching.enabled && batching.max_batch_size > 1)
            session_->set_run_cache_capacity(
                std::max(16, std::min(64, batching.max_batch_size * 2)));
        session_->setup();
        if (batching.enabled && batching.max_batch_size > 1)
            batcher_ = std::make_unique<GpuBatcher>(this, batching);
    }
}

MelSpectrogramExtractor::~MelSpectrogramExtractor() {
    // Stop the worker while session_ and compute_mu_ are still alive. Member
    // destruction is reverse declaration order, so relying on the implicit
    // destructor would otherwise destroy compute_mu_ before batcher_.
    batcher_.reset();
}

nemo_speech::asr::BatchMetrics
MelSpectrogramExtractor::batch_metrics() const {
    return batcher_ ? batcher_->metrics() : nemo_speech::asr::BatchMetrics{};
}

void
MelSpectrogramExtractor::init_window() {
    const int win = win_length();
    window_.resize(win);
    // Legacy ASR path: periodic Hann. NeMo-parity path (hann_periodic=false):
    // symmetric Hann, matching torch.hann_window(periodic=False) which is
    // what FilterbankFeatures registers.
    const float denom = cfg_.hann_periodic ? static_cast<float>(win) : static_cast<float>(win - 1);
    for (int i = 0; i < win; i++) {
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * kPi * i / denom));
    }
}

void
MelSpectrogramExtractor::init_mel_basis() {
    // Fallback Slaney-style filterbank for models without a serialized basis.
    const int n_bins = cfg_.n_fft / 2 + 1;
    const int n_mels = cfg_.n_mels;
    mel_basis_.assign(n_mels * n_bins, 0.0f);

    const float mel_min = hz_to_mel(cfg_.fmin);
    const float mel_max = hz_to_mel(cfg_.fmax);
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        const float m = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        mel_points[i] = mel_to_hz(m);
    }
    std::vector<float> bin_freqs(n_bins);
    for (int i = 0; i < n_bins; i++) {
        bin_freqs[i] = static_cast<float>(i) * cfg_.sample_rate / cfg_.n_fft;
    }
    for (int m = 0; m < n_mels; m++) {
        const float f_left = mel_points[m];
        const float f_center = mel_points[m + 1];
        const float f_right = mel_points[m + 2];
        for (int k = 0; k < n_bins; k++) {
            const float f = bin_freqs[k];
            float w = 0.0f;
            if (f >= f_left && f <= f_center)
                w = (f - f_left) / (f_center - f_left + 1e-12f);
            else if (f >= f_center && f <= f_right)
                w = (f_right - f) / (f_right - f_center + 1e-12f);
            mel_basis_[m * n_bins + k] = std::max(0.0f, w);
        }
    }
}

void
MelSpectrogramExtractor::set_mel_basis(const float* fb, int n_mels, int n_bins) {
    const int expected_bins = cfg_.n_fft / 2 + 1;
    if (n_mels != cfg_.n_mels || n_bins != expected_bins) {
        throw std::runtime_error(
            "MelSpectrogramExtractor::set_mel_basis: shape mismatch - got (" +
            std::to_string(n_mels) + ", " + std::to_string(n_bins) + "), expected (" +
            std::to_string(cfg_.n_mels) + ", " + std::to_string(expected_bins) + ")");
    }
    mel_basis_.assign(fb, fb + (size_t)n_mels * (size_t)n_bins);
    if (module_ && session_) {
        module_->update_mel_basis(session_.get(), fb, n_mels, n_bins);
    }
}

void
MelSpectrogramExtractor::compute(
    const float* audio, size_t n_samples, std::vector<float>& features, int& n_frames,
    bool reflect_left, bool normalize) const {
    compute_padded(audio, n_samples, n_samples, features, n_frames, reflect_left, normalize);
}

void
MelSpectrogramExtractor::compute_padded(
    const float* audio, size_t n_samples, size_t valid_samples, std::vector<float>& features,
    int& n_frames, bool reflect_left, bool normalize) const {
    if (valid_samples > n_samples)
        throw std::invalid_argument("valid audio length exceeds padded length");
    // `normalize` is a call-site request (streaming can explicitly suppress
    // utterance normalization); the model metadata is authoritative about
    // whether normalization exists at all.
    normalize = normalize && cfg_.normalize_per_feature;
    static const bool t_log = std::getenv("NEMO_SPEECH_TIMING") != nullptr;
    struct FeTimer {
        bool on;
        std::chrono::high_resolution_clock::time_point t0;
        size_t n_samples;
        const int& n_frames;
        bool gpu;
        ~FeTimer() {
            if (!on)
                return;
            const auto t1 = std::chrono::high_resolution_clock::now();
            fprintf(
                stderr, "[timing] fe path=%s n_samples=%zu n_frames=%d = %.2f ms\n",
                gpu ? "gpu" : "cpu", n_samples, n_frames,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    } fe_timer{
        t_log, std::chrono::high_resolution_clock::now(), n_samples, n_frames,
        n_samples > 0 && session_ != nullptr};

    // CPU-only builds fall through to the radix-2 FFT path below.
    if (n_samples > 0 && session_) {
        if (batcher_) {
            auto result = batcher_->run(audio, n_samples, valid_samples, reflect_left, normalize);
            features = std::move(result.features);
            n_frames = result.n_frames;
        } else {
            compute_gpu_via_session(
                audio, n_samples, valid_samples, features, n_frames, reflect_left, normalize);
        }
        return;
    }
    const int hop = hop_length();
    const int win = win_length();
    const int n_fft_ = cfg_.n_fft;

    // Pre-emphasis y[t] = x[t] - α·x[t-1] on the RAW signal, before STFT
    // framing. Matches NeMo's order (`preemph(audio)` then `torch.stft`);
    // applying it after padding biases the reflected/zero edges. First sample
    // kept as `y[0] = x[0]`.
    std::vector<float> pre(n_samples, 0.0f);
    std::copy_n(audio, valid_samples, pre.data());
    if (cfg_.preemph != 0.0f && valid_samples >= 2) {
        for (size_t i = valid_samples - 1; i > 0; --i) {
            pre[i] = pre[i] - cfg_.preemph * pre[i - 1];
        }
    }
    const float* preemphed = pre.data();

    // Zero-pad by n_fft/2 on each side to match NeMo's
    // `torch.stft(center=True, pad_mode='constant')` (see
    // nemo.collections.asr.parts.preprocessing.features.FilterbankFeatures.stft).
    //
    // `reflect_left=false` is the mid-stream streaming case where the caller
    // has already prepended n_fft/2 real left-context samples - we still emit
    // the right-side zero pad so frame indexing is consistent; the streaming
    // runner only consumes frames whose right edge is real, so the synthetic
    // right samples are never actually read.
    std::vector<float> padded;
    padded.reserve(n_samples + n_fft_);
    if (reflect_left) {
        for (int i = 0; i < n_fft_ / 2; i++) padded.push_back(0.0f);
    }
    for (size_t i = 0; i < n_samples; i++) padded.push_back(preemphed[i]);
    for (int i = 0; i < n_fft_ / 2; i++) padded.push_back(0.0f);

    n_frames = static_cast<int>((padded.size() - n_fft_) / hop + 1);
    if (n_frames <= 0) {
        features.clear();
        n_frames = 0;
        return;
    }

    const int n_bins = n_fft_ / 2 + 1;
    features.assign(static_cast<size_t>(cfg_.n_mels) * n_frames, 0.0f);

    std::vector<float> re(n_fft_, 0.0f);
    std::vector<float> im(n_fft_, 0.0f);

    for (int f = 0; f < n_frames; f++) {
        const int offset = f * hop;
        std::fill(re.begin(), re.end(), 0.0f);
        std::fill(im.begin(), im.end(), 0.0f);
        // Window placement: legacy ASR path right-aligns (woff=0); the
        // NeMo-parity path centers the window inside the n_fft frame the way
        // torch.stft pads it on both sides (stft_center_window=true).
        const int woff = cfg_.stft_center_window ? (n_fft_ - win) / 2 : 0;
        for (int i = 0; i < win; i++) {
            float sample = padded[offset + woff + i];
            re[woff + i] = sample * window_[i];
        }

        fft_radix2_inplace(re, im);

        // Power spectrum -> mel
        for (int m = 0; m < cfg_.n_mels; m++) {
            float acc = 0.0f;
            for (int k = 0; k < n_bins; k++) {
                const float power = re[k] * re[k] + im[k] * im[k];
                acc += mel_basis_[m * n_bins + k] * power;
            }
            features[static_cast<size_t>(m) + static_cast<size_t>(f) * cfg_.n_mels] =
                std::log(acc + cfg_.log_zero_guard);
        }
    }

    const int valid = std::min(static_cast<int>(valid_samples) / hop, n_frames);
    if (cfg_.mask_invalid_frames) {
        for (int f = valid; f < n_frames; ++f) {
            std::fill_n(features.data() + static_cast<size_t>(f) * cfg_.n_mels, cfg_.n_mels, 0.0f);
        }
    }

    // Per-feature normalization over the valid frames in this call. Streaming
    // mode skips it and maintains incremental statistics in the runner.
    if (!normalize)
        return;

    // NeMo normalizes only the valid `floor(samples / hop)` frames, uses an
    // unbiased variance, and masks the final centered/padded frame. This is
    // observably different from normalizing every STFT frame.
    for (int m = 0; m < cfg_.n_mels; m++) {
        double sum = 0.0;
        for (int f = 0; f < valid; f++)
            sum += features[static_cast<size_t>(m) + static_cast<size_t>(f) * cfg_.n_mels];
        const double mean = valid > 0 ? sum / valid : 0.0;
        double var = 0.0;
        for (int f = 0; f < valid; f++) {
            const double d =
                features[static_cast<size_t>(m) + static_cast<size_t>(f) * cfg_.n_mels] - mean;
            var += d * d;
        }
        if (valid > 1)
            var /= valid - 1;
        const double inv_std = 1.0 / (std::sqrt(var) + 1e-5);
        for (int f = 0; f < n_frames; f++) {
            const size_t idx = static_cast<size_t>(m) + static_cast<size_t>(f) * cfg_.n_mels;
            features[idx] = f < valid ? static_cast<float>((features[idx] - mean) * inv_std) : 0.0f;
        }
    }
}

bool
read_wav_mono_16k(const std::string& path, std::vector<float>& out_samples, int& out_sr) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return false;
    struct RiffHeader {
        char riff[4];
        uint32_t size;
        char wave[4];
    } rh;
    if (std::fread(&rh, 1, sizeof(rh), f) != sizeof(rh)) {
        std::fclose(f);
        return false;
    }
    if (std::memcmp(rh.riff, "RIFF", 4) || std::memcmp(rh.wave, "WAVE", 4)) {
        std::fclose(f);
        return false;
    }

    uint16_t fmt_tag = 0, channels = 0, bits = 0;
    uint32_t sr = 0, byte_rate = 0;
    uint16_t block_align = 0;
    std::vector<uint8_t> data_bytes;

    char chunk[4];
    uint32_t chunk_size;
    while (std::fread(chunk, 1, 4, f) == 4 && std::fread(&chunk_size, 4, 1, f) == 1) {
        if (!std::memcmp(chunk, "fmt ", 4)) {
            std::fread(&fmt_tag, 2, 1, f);
            std::fread(&channels, 2, 1, f);
            std::fread(&sr, 4, 1, f);
            std::fread(&byte_rate, 4, 1, f);
            std::fread(&block_align, 2, 1, f);
            std::fread(&bits, 2, 1, f);
            if (chunk_size > 16)
                std::fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (!std::memcmp(chunk, "data", 4)) {
            data_bytes.resize(chunk_size);
            std::fread(data_bytes.data(), 1, chunk_size, f);
            break;
        } else {
            std::fseek(f, chunk_size, SEEK_CUR);
        }
    }
    std::fclose(f);

    if (fmt_tag != 1 && fmt_tag != 3) {
        return false;
    }  // PCM or IEEE float
    out_sr = static_cast<int>(sr);

    const size_t n = data_bytes.size() / (bits / 8) / channels;
    out_samples.resize(n);
    if (bits == 16 && fmt_tag == 1) {
        const int16_t* p = reinterpret_cast<const int16_t*>(data_bytes.data());
        for (size_t i = 0; i < n; i++) {
            int32_t s = 0;
            for (int c = 0; c < channels; c++) s += p[i * channels + c];
            s /= channels;
            out_samples[i] = s / 32768.0f;
        }
    } else if (bits == 32 && fmt_tag == 3) {
        const float* p = reinterpret_cast<const float*>(data_bytes.data());
        for (size_t i = 0; i < n; i++) {
            float s = 0.0f;
            for (int c = 0; c < channels; c++) s += p[i * channels + c];
            out_samples[i] = s / channels;
        }
    } else {
        return false;
    }
    return true;
}

void
MelSpectrogramExtractor::compute_gpu_via_session(
    const float* audio, size_t n_samples, size_t valid_samples, std::vector<float>& features,
    int& n_frames, bool reflect_left, bool normalize) const {
    BatchRequest request;
    request.audio.assign(audio, audio + n_samples);
    request.valid_samples = valid_samples;
    request.reflect_left = reflect_left;
    request.normalize = normalize;
    std::vector<BatchRequest> requests;
    requests.push_back(std::move(request));
    auto results = compute_gpu_batch_via_session(std::move(requests));
    features = std::move(results.front().features);
    n_frames = results.front().n_frames;
}

std::vector<MelSpectrogramExtractor::BatchResult>
MelSpectrogramExtractor::compute_gpu_batch_via_session(std::vector<BatchRequest>&& requests) const {
    if (requests.empty())
        return {};

    const size_t n_samples = requests.front().audio.size();
    const bool reflect_left = requests.front().reflect_left;
    const bool normalize = requests.front().normalize && cfg_.normalize_per_feature;
    const int n_fft_ = cfg_.n_fft;
    const int hop = hop_length();
    const int n_audio_padded = reflect_left ? (int)n_samples + 1 : (int)n_samples + n_fft_ / 2 + 1;
    const int p0 = reflect_left ? n_fft_ / 2 : 0;
    const int n_frames = (n_audio_padded + 2 * p0 - n_fft_) / hop + 1;
    std::vector<BatchResult> results(requests.size());
    if (n_frames <= 0) {
        return results;
    }

    // refl_flag is encoded in ne[1] so the graph cache distinguishes the two
    // padding modes. Only the first row contains pre-emphasized audio.
    const int refl_flag = reflect_left ? 2 : 1;
    const size_t input_item = static_cast<size_t>(n_audio_padded) * refl_flag;
    std::vector<float> pre(input_item * requests.size(), 0.0f);
    for (size_t b = 0; b < requests.size(); ++b) {
        const auto& request = requests[b];
        if (request.audio.size() != n_samples || request.valid_samples > n_samples ||
            request.reflect_left != reflect_left || request.normalize != normalize)
            throw std::runtime_error("GPU frontend batch contains incompatible inputs");
        float* dst = pre.data() + b * input_item;
        if (request.valid_samples == 0)
            continue;
        dst[0] = request.audio[0];
        if (cfg_.preemph != 0.0f) {
            for (size_t i = 1; i < request.valid_samples; ++i)
                dst[i] = request.audio[i] - cfg_.preemph * request.audio[i - 1];
        } else if (request.valid_samples > 1) {
            std::memcpy(
                dst + 1, request.audio.data() + 1, (request.valid_samples - 1) * sizeof(float));
        }
    }

    // Cross-stream serialisation: session_->run is itself not re-entrant
    // (the run_cache_ probe builds a TensorContainer per call). The
    // BackendManager's compute mutex inside Session covers the actual
    // graph_compute, but two concurrent compute()s can still trip the
    // per-Session container scratch - serialise graph submission. Host-side
    // per-item normalization below does not need the lock.
    const size_t output_item = static_cast<size_t>(cfg_.n_mels) * n_frames;
    std::vector<float> packed(output_item * requests.size());
    // push_back rather than a braced std::vector init: MSVC's backend ICEs
    // (C1001) aggregate-initializing a vector of Session::Output, whose C-array
    // member (out_shape[4]) trips the bug. push_back of the same aggregate is fine.
    {
        std::lock_guard<std::mutex> lock(compute_mu_);
        std::vector<ggml_runtime::Session::Output> outputs;
        outputs.push_back({0, "", packed.data(), packed.size() * sizeof(float)});
        session_->run(
            {{"fe.in.audio",
              GGML_TYPE_F32,
              pre.data(),
              {n_audio_padded, refl_flag, static_cast<int64_t>(requests.size())}}},
            outputs);
        if (outputs[0].out_shape[2] != static_cast<int64_t>(requests.size()))
            throw std::runtime_error("GPU frontend graph lost its batch dimension");
    }

    // Split the packed [mel,frame,batch] output and normalize each utterance
    // independently. The z-score loop is byte-identical to the CPU path; only
    // the FFT/DFT arithmetic can introduce small frontend numeric differences.
    const int n_mels = cfg_.n_mels;
    for (size_t b = 0; b < requests.size(); ++b) {
        auto& features = results[b].features;
        results[b].n_frames = n_frames;
        features.assign(packed.begin() + b * output_item, packed.begin() + (b + 1) * output_item);
        const int valid = std::min(static_cast<int>(requests[b].valid_samples) / hop, n_frames);
        if (cfg_.mask_invalid_frames) {
            for (int f = valid; f < n_frames; ++f) {
                std::fill_n(features.data() + static_cast<size_t>(f) * n_mels, n_mels, 0.0f);
            }
        }
        if (!normalize)
            continue;
        for (int m = 0; m < n_mels; ++m) {
            double sum = 0.0;
            for (int f = 0; f < valid; ++f) sum += features[(size_t)m + (size_t)f * n_mels];
            const double mean = valid > 0 ? sum / valid : 0.0;
            double var = 0.0;
            for (int f = 0; f < valid; ++f) {
                const double d = features[(size_t)m + (size_t)f * n_mels] - mean;
                var += d * d;
            }
            if (valid > 1)
                var /= valid - 1;
            const double inv_std = 1.0 / (std::sqrt(var) + 1e-5);
            for (int f = 0; f < n_frames; ++f) {
                const size_t idx = (size_t)m + (size_t)f * n_mels;
                features[idx] =
                    f < valid ? static_cast<float>((features[idx] - mean) * inv_std) : 0.0f;
            }
        }
    }
    return results;
}

int
produce_new_mel_frames(
    MelSpectrogramExtractor& fe, const std::vector<float>& audio, size_t audio_base,
    int64_t i_start, std::vector<float>& out) {
    // Geometry comes from the extractor itself; callers cannot desync it from
    // the frames fe.compute() actually produces.
    const int n_mels = fe.n_mels();
    const int hop = fe.hop_length();
    const int n_fft = fe.n_fft();
    GGML_ASSERT(n_mels > 0 && hop > 0 && n_fft > 0);
    out.clear();
    const size_t audio_end = audio_base + audio.size();
    const int64_t i_max = (audio_end >= static_cast<size_t>(n_fft / 2))
                              ? static_cast<int64_t>((audio_end - n_fft / 2) / hop) + 1
                              : 0;
    if (i_max <= i_start)
        return 0;
    std::vector<float> partial;
    int n_frames = 0;
    int64_t first_global;
    if (i_start * hop < n_fft / 2) {
        fe.compute(
            audio.data(), audio.size(), partial, n_frames, /*reflect_left=*/true,
            /*normalize=*/false);
        first_global = 0;
    } else {
        const int64_t off = i_start * hop - n_fft / 2 - static_cast<int64_t>(audio_base);
        fe.compute(
            audio.data() + off, audio.size() - static_cast<size_t>(off), partial, n_frames,
            /*reflect_left=*/false, /*normalize=*/false);
        first_global = i_start;
    }
    const int64_t skip = i_start - first_global;
    const int64_t take = std::min<int64_t>(static_cast<int64_t>(n_frames) - skip, i_max - i_start);
    if (take <= 0 || skip < 0)
        return 0;
    out.assign(
        partial.data() + skip * n_mels,
        partial.data() + (skip + take) * static_cast<int64_t>(n_mels));
    return static_cast<int>(take);
}
