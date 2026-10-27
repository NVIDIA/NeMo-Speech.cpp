// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Parity of the PEE port against NeMo reference dumps (NERB arrays: scripts/asr/export_ref_bins.py
// unpacks a reference .npz written with NeMo): feeds the reference mel through PeeModel and
// compares ASR states, 80 ms speaker probabilities and the fused output. A manual check, not part
// of ctest.
//
//   usage: test_pee_parity <pee.gguf> <ref-bins-dir> [--gpu]
//
// --ab compares the device-resident streaming speaker state (GPU default) with the host reference
// path (state updated and compressed on the CPU) on the same input, element by element; with
// --wav <file> the input is any 16 kHz WAV (CPU front end, ref-bins-dir "-": no NeMo reference),
// e.g. long multi-window audio. --ctc <adapter.gguf> also compares the native 10 ms speaker
// probabilities. --host-state runs the reference comparison on the host path.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "features/fe.h"
#include "pee/pee_model.h"
#include "runtime.h"

namespace {

struct RefArray {
    std::vector<int64_t> shape;
    std::vector<float> f;
    std::vector<int64_t> i;
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

RefArray
load_ref(const std::string& dir, const std::string& key) {
    const std::string path = dir + "/" + key + ".bin";
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("missing reference array: " + path);
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "NERB", 4) != 0)
        throw std::runtime_error("bad magic: " + path);
    uint32_t code;
    int64_t n_dims;
    f.read(reinterpret_cast<char*>(&code), 4);
    f.read(reinterpret_cast<char*>(&n_dims), 8);
    RefArray a;
    a.shape.resize(n_dims);
    f.read(reinterpret_cast<char*>(a.shape.data()), 8 * n_dims);
    const int64_t n = a.numel();
    if (code == 0) {
        a.f.resize(n);
        f.read(reinterpret_cast<char*>(a.f.data()), 4 * n);
    } else {
        a.i.resize(n);
        f.read(reinterpret_cast<char*>(a.i.data()), 8 * n);
    }
    if (!f)
        throw std::runtime_error("truncated reference array: " + path);
    return a;
}

struct Stats {
    float max_abs = 0, p999_abs = 0, mean_abs = 0, ref_rms = 0;
    size_t argmax = 0;
};

Stats
compare(const std::vector<float>& got, const std::vector<float>& ref, size_t n) {
    Stats s;
    double sum = 0, rms = 0;
    std::vector<float> diffs(n);
    for (size_t i = 0; i < n; i++) {
        const float d = std::fabs(got[i] - ref[i]);
        diffs[i] = d;
        if (d > s.max_abs) {
            s.max_abs = d;
            s.argmax = i;
        }
        sum += d;
        rms += static_cast<double>(ref[i]) * ref[i];
    }
    s.mean_abs = static_cast<float>(sum / n);
    s.ref_rms = static_cast<float>(std::sqrt(rms / n));
    const size_t k = static_cast<size_t>(0.999 * (n - 1));
    std::nth_element(diffs.begin(), diffs.begin() + k, diffs.end());
    s.p999_abs = diffs[k];
    return s;
}

// Gate on the mean and the 99.9th percentile: with f16 weights a handful of
// numerically sensitive frames (peaked attention) can deviate 10-50x more than
// the rest on either backend; f32 weights bring them back (verified), so the
// single worst element is reported but not gated.
int
report(const char* what, const Stats& s, float tol_p999, float tol_mean, int cols) {
    const bool ok = s.p999_abs <= tol_p999 && s.mean_abs <= tol_mean;
    std::printf(
        "  %-14s max_abs %.4e (at frame %zu col %zu) p99.9 %.4e mean_abs %.4e ref_rms %.4e  %s\n",
        what, s.max_abs, s.argmax / cols, s.argmax % cols, s.p999_abs, s.mean_abs, s.ref_rms,
        ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

// Element-wise A/B: values that differ, the largest difference and the first differing frame.
int
ab_report(const char* what, const std::vector<float>& a, const std::vector<float>& b, int cols) {
    if (a.size() != b.size()) {
        std::printf("  %-14s size %zu vs %zu  FAIL\n", what, a.size(), b.size());
        return 1;
    }
    size_t n_diff = 0, first = a.size();
    float max_abs = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] == b[i] || (std::isnan(a[i]) && std::isnan(b[i])))
            continue;
        n_diff++;
        first = std::min(first, i);
        max_abs = std::max(max_abs, std::fabs(a[i] - b[i]));
    }
    if (n_diff == 0)
        std::printf("  %-14s %zu values identical\n", what, a.size());
    else
        std::printf(
            "  %-14s %zu / %zu values differ, max_abs %.4e, first at frame %zu\n", what, n_diff,
            a.size(), max_abs, first / cols);
    return n_diff ? 1 : 0;
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(
            stderr,
            "usage: %s <pee.gguf> <ref-bins-dir> [--gpu] [--bucket N] [--repeat N] [--bench N]\n",
            argv[0]);
        return 2;
    }
    const std::string gguf_path = argv[1];
    const std::string ref_dir = argv[2];
    bool use_gpu = false, ab = false, host_state = false;
    int bench_reps = 0, bucket = 1, repeat = 1;
    std::string wav_path, ctc_path;
    double seconds = 0;
    for (int i = 3; i < argc; i++) {
        if (std::string(argv[i]) == "--gpu")
            use_gpu = true;
        if (std::string(argv[i]) == "--ab")
            ab = true;
        if (std::string(argv[i]) == "--host-state")
            host_state = true;
        if (std::string(argv[i]) == "--wav" && i + 1 < argc)
            wav_path = argv[++i];
        if (std::string(argv[i]) == "--ctc" && i + 1 < argc)
            ctc_path = argv[++i];
        if (std::string(argv[i]) == "--seconds" && i + 1 < argc)
            seconds = std::atof(argv[++i]);  // --wav: use the first N seconds
        if (std::string(argv[i]) == "--bench" && i + 1 < argc)
            bench_reps = std::atoi(argv[++i]);
        if (std::string(argv[i]) == "--repeat" && i + 1 < argc)
            repeat = std::atoi(
                argv[++i]);  // re-encode N times; compare the last (cached-graph re-execution)
        if (std::string(argv[i]) == "--bucket" && i + 1 < argc)
            bucket = std::atoi(argv[++i]);  // mel frames; >1 pads
    }

    ggml_runtime::Params params;
    params.use_gpu = use_gpu;
    ggml_runtime::BackendManager bm(params);
    nemo_speech::speechlm::PeeModel model(bm, gguf_path, /*want_asr_states=*/true, ctc_path);
    model.set_bucket_frames(bucket);
    if (host_state)
        model.set_speaker_state_on_device(false);
    if (bucket > 1)
        std::printf("bucketed padding: windows padded to multiples of %d mel frames\n", bucket);
    const auto& cfg = model.cfg();
    std::printf(
        "pee: asr d=%d L=%d H=%d rot=%d qk_norm=%d | diar d=%d L=%d H=%d rot=%d | n_spk=%d "
        "thr=%.2f\n",
        cfg.asr.d_model, cfg.asr.n_layers, cfg.asr.n_heads, cfg.asr.rotary_dims, cfg.asr.qk_norm,
        cfg.diar.d_model, cfg.diar.n_layers, cfg.diar.n_heads, cfg.diar.rotary_dims,
        cfg.num_speakers, cfg.speaker_activity_threshold);

    RefArray mel;  // (T_mel, n_mels)
    if (!wav_path.empty()) {
        // raw log-mel from the CPU front end, configured as the SALM engine's
        std::vector<float> samples;
        int sr = 0;
        if (!read_wav_mono_16k(wav_path, samples, sr) || sr != cfg.sample_rate)
            throw std::runtime_error("--wav needs a readable WAV at the model rate: " + wav_path);
        if (seconds > 0)
            samples.resize(std::min(samples.size(), static_cast<size_t>(seconds * sr)));
        MelSpecConfig fe_cfg;
        fe_cfg.sample_rate = cfg.sample_rate;
        fe_cfg.n_fft = cfg.n_fft;
        fe_cfg.n_mels = cfg.n_mels;
        fe_cfg.window_size = cfg.window_size;
        fe_cfg.window_stride = cfg.window_stride;
        fe_cfg.normalize_per_feature = false;
        MelSpectrogramExtractor fe(fe_cfg);
        fe.set_mel_basis(model.mel_basis().data(), cfg.n_mels, cfg.n_fft / 2 + 1);
        int frames = 0;
        fe.compute(samples.data(), samples.size(), mel.f, frames, true, false);
        frames = std::min(frames, static_cast<int>(samples.size() / fe.hop_length()) + 1);
        mel.f.resize(static_cast<size_t>(frames) * cfg.n_mels);
        mel.shape = {frames, cfg.n_mels};
        std::printf(
            "wav %s: %.1f s, %d mel frames\n", wav_path.c_str(),
            samples.size() / static_cast<double>(sr), frames);
    } else {
        mel = load_ref(ref_dir, "mel");
    }
    const int t_mel = static_cast<int>(mel.shape[0]);
    const int n_mels = static_cast<int>(mel.shape[1]);
    if (n_mels != cfg.n_mels)
        throw std::runtime_error("reference n_mels mismatch");

    if (ab) {
        // Device-resident speaker state vs the host reference path, same input and batch shapes.
        if (!use_gpu || !model.stateful_speakers())
            throw std::runtime_error(
                "--ab needs --gpu and a GGUF with the speaker silence embedding");
        const auto plan = model.plan_windows(t_mel);
        std::printf("A/B: %zu windows, speaker state on the device vs on the host\n", plan.size());
        nemo_speech::speechlm::PeeModel::Output out[2];
        const char* names[2] = {"device", "host"};
        for (int path = 0; path < 2; path++) {
            model.set_speaker_state_on_device(path == 0);
            double best = 1e30;
            std::string runs;
            for (int r = 0; r < std::max(2, repeat); r++) {  // the first run builds the graphs
                const size_t miss0 = model.graph_misses(), stale0 = model.graph_stale();
                const auto t0 = std::chrono::steady_clock::now();
                out[path] = model.encode(mel.f.data(), t_mel);
                const double ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                        .count();
                char buf[96];
                std::snprintf(
                    buf, sizeof buf, " %.1f ms (%zu built, %zu stale)", ms,
                    model.graph_misses() - miss0, model.graph_stale() - stale0);
                runs += buf;
                if (r > 0)
                    best = std::min(best, ms);
            }
            std::printf("  %-8s runs:%s\n", names[path], runs.c_str());
            std::printf(
                "  %-8s encode %.1f ms (best of %d warm runs)\n", names[path], best,
                std::max(2, repeat) - 1);
        }
        int diffs = 0;
        diffs += ab_report("spk_probs80", out[0].spk_probs, out[1].spk_probs, cfg.num_speakers);
        diffs += ab_report("spk_binary", out[0].spk_binary, out[1].spk_binary, cfg.num_speakers);
        diffs += ab_report(
            "spk_probs10", out[0].spk_probs10, out[1].spk_probs10,
            cfg.num_speakers * cfg.upsample_factor);
        diffs += ab_report("fused", out[0].fused, out[1].fused, cfg.asr.d_model);
        diffs += ab_report("proj", out[0].proj, out[1].proj, std::max(cfg.proj_dim, 1));
        std::printf(
            diffs ? "speaker state A/B (device vs host): DIFFER (%d outputs)\n"
                  : "speaker state A/B (device vs host): IDENTICAL\n",
            diffs);
        return diffs ? 1 : 0;
    }
    auto ref_asr = load_ref(ref_dir, "asr_encoded");      // (T', d)
    auto ref_p80 = load_ref(ref_dir, "diar_preds_80ms");  // (T', n_spk)
    auto ref_bin = load_ref(ref_dir, "spk_binary");       // (T', n_spk)
    auto ref_fused = load_ref(ref_dir, "fused");          // (T', d)
    auto ref_norm = load_ref(ref_dir, "asr_mel_norm");    // (T_mel, n_mels)

    // FE normalization parity first (host code, exact expectations).
    std::vector<float> norm(mel.f.size());
    nemo_speech::speechlm::per_feature_normalize(mel.f.data(), t_mel, n_mels, norm.data());
    int failures = 0;
    std::printf("T_mel=%d -> T'=%d\n", t_mel, model.out_frames(t_mel));
    failures += report("mel_norm", compare(norm, ref_norm.f, norm.size()), 5e-4f, 2e-5f, n_mels);

    if (bench_reps > 0) {
        // Warm, repeated timing across audio lengths (prefixes of the reference
        // mel): per-length first call (graph build for a new shape) vs median of
        // warm calls. Single utterance, no batching.
        std::printf(
            "bench (%s): length | first-call ms | warm median ms (n=%d) | warm RTFx\n",
            use_gpu ? "gpu" : "cpu", bench_reps);
        const int lens[] = {500, 1000, 2000, 3000, t_mel};
        for (int L : lens) {
            if (L > t_mel)
                continue;
            auto tick = [&]() {
                const auto a = std::chrono::steady_clock::now();
                (void)model.encode(mel.f.data(), L);
                return std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - a)
                    .count();
            };
            const double first = tick();
            std::vector<double> ts;
            for (int r = 0; r < bench_reps; r++) ts.push_back(tick());
            std::sort(ts.begin(), ts.end());
            const double med = ts[ts.size() / 2];
            const double audio_s = L * cfg.window_stride;
            std::printf(
                "  %5.1f s | %8.1f | %8.1f | %7.1f\n", audio_s, first, med,
                audio_s / (med / 1000.0));
        }
        return 0;
    }

    for (int r = 1; r < repeat; r++) (void)model.encode(mel.f.data(), t_mel);
    if (repeat > 1)
        std::printf("re-encoding the utterance: comparing run %d of %d\n", repeat, repeat);
    const auto t0 = std::chrono::steady_clock::now();
    auto out = model.encode(mel.f.data(), t_mel);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double audio_s = t_mel * cfg.window_stride;
    std::printf(
        "encode: %.0f ms for %.1f s audio (RTFx %.1f, %s)\n", ms, audio_s, audio_s / (ms / 1000.0),
        use_gpu ? "gpu" : "cpu");
    const int T = out.n_frames;
    if (T != ref_asr.shape[0]) {
        std::printf(
            "frame count mismatch: got %d ref %lld\n", T, static_cast<long long>(ref_asr.shape[0]));
        return 1;
    }
    const int d = cfg.asr.d_model;
    const int n_spk = cfg.num_speakers;
    // Bounds are ~4x the errors measured on the two reference clips with f16
    // weights vs the fp32 PyTorch reference (CPU: asr max 4.7e-3 / mean 7.8e-5,
    // probs max 1.4e-3, fused max 2.9e-2 / mean 4.4e-4). The CUDA backend runs
    // f16 matmuls (activations included), so its deviations are ~6x larger
    // (asr max 2.8e-2 / mean 3.5e-4); scale the bounds accordingly.
    const float k = use_gpu ? 6.0f : 1.0f;
    failures += report(
        "asr_states", compare(out.asr_states, ref_asr.f, out.asr_states.size()), k * 2e-2f,
        k * 4e-4f, d);
    failures += report(
        "spk_probs80", compare(out.spk_probs, ref_p80.f, out.spk_probs.size()), k * 6e-3f,
        k * 1e-4f, n_spk);
    // Binary activity: count disagreements (threshold flips near 0.5 are expected to be rare).
    int flips = 0;
    for (size_t i = 0; i < out.spk_binary.size(); i++)
        if (out.spk_binary[i] != ref_bin.f[i])
            flips++;
    std::printf(
        "  %-14s %d / %zu frames*spk disagree  %s\n", "spk_binary", flips, out.spk_binary.size(),
        flips <= static_cast<int>(out.spk_binary.size() / 200) ? "OK" : "FAIL");
    if (flips > static_cast<int>(out.spk_binary.size() / 200))
        failures++;
    failures +=
        report("fused", compare(out.fused, ref_fused.f, out.fused.size()), k * 0.12f, k * 2e-3f, d);
    if (cfg.proj_dim > 0) {
        // SALM modality projection (decoder input); reference dumps carry proj.bin when the
        // checkpoint has one. Same relative bounds as fused, scaled by the projection's fan-in.
        if (std::ifstream(ref_dir + "/proj.bin", std::ios::binary)) {
            auto ref_proj = load_ref(ref_dir, "proj");  // (T', proj_dim)
            failures += report(
                "proj", compare(out.proj, ref_proj.f, out.proj.size()), k * 0.25f, k * 4e-3f,
                cfg.proj_dim);
        } else {
            std::printf("  %-14s reference proj.bin absent; skipped\n", "proj");
        }
    }

    std::printf(failures ? "PEE parity: FAIL (%d)\n" : "PEE parity: PASS\n", failures);
    return failures ? 1 : 0;
}
