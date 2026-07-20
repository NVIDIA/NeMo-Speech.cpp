// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Parity test for the Sortformer diarizer graph against NeMo reference dumps.
//
// Stage 1 (this file, graph parity): every chunk is run TEACHER-FORCED - the
// spkcache/fifo inputs come from the NeMo reference state, so any divergence
// is the ggml graph's alone (pre_encode + conformer + transformer + head),
// not accumulated AOSC drift.
//
// Reference data: scripts/asr/dump_sortformer_reference.py (NeMo runtime
// geometry) -> export_ref_bins.py.
//
// Usage: test_sortformer_parity <model.gguf> <ref-bins-dir> [--gpu]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "aosc_state.h"
#include "diar_pipeline.h"
#include "sortformer_model.h"

namespace {

struct RefArray {
    std::vector<int64_t> shape;
    std::vector<float> f;
    std::vector<int64_t> i;
    bool is_f32 = true;
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

RefArray
load_ref(const std::string& dir, const std::string& key) {
    std::string name = key;
    size_t pos;
    while ((pos = name.find('/')) != std::string::npos) name.replace(pos, 1, "__");
    const std::string path = dir + "/" + name + ".bin";
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
    a.is_f32 = (code == 0);
    a.shape.resize(n_dims);
    f.read(reinterpret_cast<char*>(a.shape.data()), 8 * n_dims);
    const int64_t n = a.numel();
    if (a.is_f32) {
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

bool
has_ref(const std::string& dir, const std::string& key) {
    std::string name = key;
    size_t pos;
    while ((pos = name.find('/')) != std::string::npos) name.replace(pos, 1, "__");
    return std::ifstream(dir + "/" + name + ".bin").good();
}

std::string
chunk_key(int c, const char* field) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "chunk%03d/%s", c, field);
    return buf;
}

float
max_abs_diff(const std::vector<float>& a, const std::vector<float>& b, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; i++) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > m)
            m = d;
    }
    return m;
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <ref-bins-dir> [--gpu]\n", argv[0]);
        return 2;
    }
    const std::string gguf_path = argv[1];
    const std::string ref_dir = argv[2];
    bool use_gpu = false;
    for (int i = 3; i < argc; i++)
        if (std::string(argv[i]) == "--gpu")
            use_gpu = true;

    ggml_runtime::BackendManager bm({.use_gpu = use_gpu});
    nemo_speech::asr::SortformerModel model(bm, gguf_path);

    const int n_chunks = static_cast<int>(load_ref(ref_dir, "n_chunks").i[0]);
    std::printf("[parity] %d chunks, teacher-forced graph check\n", n_chunks);

    float worst_pre = 0.f, worst_pred = 0.f;
    int worst_pre_chunk = -1, worst_pred_chunk = -1;

    for (int c = 0; c < n_chunks; c++) {
        auto mel = load_ref(ref_dir, chunk_key(c, "mel_window"));          // (T_mel, n_mels)
        auto lens = load_ref(ref_dir, chunk_key(c, "state_lens_before"));  // [L1, L2, T3]
        const int l1 = static_cast<int>(lens.i[0]);
        const int l2 = static_cast<int>(lens.i[1]);
        const int t3 = static_cast<int>(lens.i[2]);
        // NeMo pads the mel tensor beyond the true signal length and masks;
        // riva (and our pipeline) feed exact-length tails instead. For tail
        // windows (valid < window rows) run trimmed and report-only - conv
        // edge context differs by construction there.
        const int feat_len = static_cast<int>(load_ref(ref_dir, chunk_key(c, "feat_length")).i[0]);
        const bool full_window = feat_len == static_cast<int>(mel.shape[0]);
        const int t_mel = full_window ? static_cast<int>(mel.shape[0]) : feat_len;

        // Teacher-forced state from the previous chunk's reference dump.
        RefArray spk, fifo;
        const float* spk_p = nullptr;
        const float* fifo_p = nullptr;
        if (c > 0 && l1 > 0) {
            spk = load_ref(ref_dir, chunk_key(c - 1, "spkcache_after"));
            if (static_cast<int>(spk.shape[0]) != l1)
                throw std::runtime_error("spkcache length mismatch at chunk " + std::to_string(c));
            spk_p = spk.f.data();
        }
        if (c > 0 && l2 > 0) {
            fifo = load_ref(ref_dir, chunk_key(c - 1, "fifo_after"));
            if (static_cast<int>(fifo.shape[0]) != l2)
                throw std::runtime_error("fifo length mismatch at chunk " + std::to_string(c));
            fifo_p = fifo.f.data();
        }

        auto out = model.run_chunk(mel.f.data(), t_mel, spk_p, l1, fifo_p, l2);
        if (out.chunk_frames != t3) {
            std::fprintf(
                stderr, "chunk %d: subsampled len %d != reference %d\n", c, out.chunk_frames, t3);
            return 1;
        }

        auto ref_pre = load_ref(ref_dir, chunk_key(c, "pre_encode"));    // (>=T3, 512)
        auto ref_preds = load_ref(ref_dir, chunk_key(c, "preds_full"));  // (>=L1+L2+T3, n_spk)
        const size_t n_pre = static_cast<size_t>(t3) * 512;
        const size_t n_pred = static_cast<size_t>(l1 + l2 + t3) * ref_preds.shape[1];
        // pre_encode embeddings span +-150+; measure them relative to the
        // reference scale (the F16 conv stem bounds this at a few e-3).
        float pre_scale = 1.f;
        for (size_t i = 0; i < n_pre; i++) pre_scale = std::max(pre_scale, std::fabs(ref_pre.f[i]));
        const float d_pre = max_abs_diff(out.chunk_embs, ref_pre.f, n_pre) / pre_scale;
        const float d_pred = max_abs_diff(out.preds, ref_preds.f, n_pred);
        if (full_window) {
            if (d_pre > worst_pre)
                worst_pre = d_pre, worst_pre_chunk = c;
            if (d_pred > worst_pred)
                worst_pred = d_pred, worst_pred_chunk = c;
        }
        if (c < 3 || !full_window || d_pred > 1e-2f)
            std::printf(
                "[parity] chunk %3d (L1=%3d L2=%2d T3=%2d%s): pre %.3e  preds %.3e\n", c, l1, l2,
                t3, full_window ? "" : ", tail", d_pre, d_pred);
    }

    std::printf(
        "[parity] worst pre_encode %.3e (chunk %d), worst preds %.3e (chunk %d)\n", worst_pre,
        worst_pre_chunk, worst_pred, worst_pred_chunk);

    // Gates: pre_encode (relative) runs the conv stem in F16 (~4e-3 floor);
    // preds are probabilities after 35 layers. Fail loudly on gross divergence.
    bool ok = worst_pre <= 1e-2f && worst_pred <= 1e-2f;
    std::printf("[parity] graph stage: %s\n", ok ? "OK" : "FAIL");

    // ------------------------------------------------------------------
    // Stage 2: AOSC state machine, teacher-forced with NeMo's own preds and
    // pre-encode embeddings. Any divergence is the host-side update /
    // compression port, isolated from graph noise.
    // ------------------------------------------------------------------
    auto geom = load_ref(ref_dir, "geometry");  // [chunk, lc, rc, fifo, spkcache, update]
    nemo_speech::asr::DiarGeometry geo;
    geo.chunk_len = static_cast<int>(geom.i[0]);
    geo.chunk_left_context = static_cast<int>(geom.i[1]);
    geo.chunk_right_context = static_cast<int>(geom.i[2]);
    geo.fifo_len = static_cast<int>(geom.i[3]);
    geo.spkcache_len = static_cast<int>(geom.i[4]);
    geo.spkcache_update_period = static_cast<int>(geom.i[5]);

    const auto& mcfg = model.cfg();
    nemo_speech::asr::AoscState st(geo, mcfg.scoring, mcfg.num_speakers, mcfg.encoder.d_model);
    const int sub = mcfg.encoder.subsampling_factor;

    float worst_state = 0.f;
    int worst_state_chunk = -1;
    for (int c = 0; c < n_chunks; c++) {
        auto lens = load_ref(ref_dir, chunk_key(c, "state_lens_before"));
        if (st.spkcache_frames() != static_cast<int>(lens.i[0]) ||
            st.fifo_frames() != static_cast<int>(lens.i[1])) {
            std::fprintf(
                stderr, "[aosc] chunk %d: state lens (%d,%d) != reference (%ld,%ld)\n", c,
                st.spkcache_frames(), st.fifo_frames(), (long)lens.i[0], (long)lens.i[1]);
            return 1;
        }
        auto ref_pre = load_ref(ref_dir, chunk_key(c, "pre_encode"));
        auto ref_preds = load_ref(ref_dir, chunk_key(c, "preds_full"));
        auto offs = load_ref(ref_dir, chunk_key(c, "offsets"));  // mel-frame offsets
        const int lc = static_cast<int>(std::lround(offs.i[0] / static_cast<double>(sub)));
        const int rc = static_cast<int>(std::ceil(offs.i[1] / static_cast<double>(sub)));
        // NeMo's update consumes the full pre-encode tensor (tail chunks
        // include a padding row); mirror that.
        st.update(ref_pre.f.data(), static_cast<int>(ref_pre.shape[0]), ref_preds.f.data(), lc, rc);

        auto ref_spk = load_ref(ref_dir, chunk_key(c, "spkcache_after"));
        auto ref_fifo = load_ref(ref_dir, chunk_key(c, "fifo_after"));
        auto ref_sil = load_ref(ref_dir, chunk_key(c, "mean_sil_emb_after"));
        const int64_t ref_nsil = load_ref(ref_dir, chunk_key(c, "n_sil_frames_after")).i[0];
        float d = 0.f;
        if (st.spkcache_frames() != static_cast<int>(ref_spk.shape[0]) ||
            st.fifo_frames() != static_cast<int>(ref_fifo.shape[0]) ||
            st.n_sil_frames() != ref_nsil) {
            std::fprintf(
                stderr, "[aosc] chunk %d: post lens (%d,%d,%ld) != ref (%ld,%ld,%ld)\n", c,
                st.spkcache_frames(), st.fifo_frames(), (long)st.n_sil_frames(),
                (long)ref_spk.shape[0], (long)ref_fifo.shape[0], (long)ref_nsil);
            return 1;
        }
        d = std::max(d, max_abs_diff(st.spkcache(), ref_spk.f, ref_spk.f.size()));
        d = std::max(d, max_abs_diff(st.fifo(), ref_fifo.f, ref_fifo.f.size()));
        d = std::max(d, max_abs_diff(st.mean_sil_emb(), ref_sil.f, ref_sil.f.size()));
        if (has_ref(ref_dir, chunk_key(c, "spkcache_preds_after")) && st.spkcache_preds_valid()) {
            auto rp = load_ref(ref_dir, chunk_key(c, "spkcache_preds_after"));
            d = std::max(d, max_abs_diff(st.spkcache_preds(), rp.f, rp.f.size()));
        }
        if (d > worst_state)
            worst_state = d, worst_state_chunk = c;
    }
    std::printf(
        "[parity] AOSC stage (teacher-forced): worst state diff %.3e (chunk %d) %s\n", worst_state,
        worst_state_chunk, worst_state <= 1e-5f ? "OK" : "FAIL");
    ok = ok && worst_state <= 1e-5f;

    // ------------------------------------------------------------------
    // Stage 3: free-running end-to-end - ggml graph + C++ AOSC from mel only.
    // Divergence accumulates (F16 conv floor feeds the state); report the
    // emitted-probability drift vs NeMo per chunk, gate loosely.
    // ------------------------------------------------------------------
    nemo_speech::asr::AoscState st3(geo, mcfg.scoring, mcfg.num_speakers, mcfg.encoder.d_model);
    float worst_e2e = 0.f;
    int worst_e2e_chunk = -1;
    for (int c = 0; c < n_chunks; c++) {
        auto mel = load_ref(ref_dir, chunk_key(c, "mel_window"));
        const int feat_len = static_cast<int>(load_ref(ref_dir, chunk_key(c, "feat_length")).i[0]);
        const bool full_window = feat_len == static_cast<int>(mel.shape[0]);
        if (!full_window)
            break;  // tail handling deviates from NeMo by design
        auto offs = load_ref(ref_dir, chunk_key(c, "offsets"));
        const int lc = static_cast<int>(std::lround(offs.i[0] / static_cast<double>(sub)));
        const int rc = static_cast<int>(std::ceil(offs.i[1] / static_cast<double>(sub)));

        auto out = model.run_chunk(
            mel.f.data(), static_cast<int>(mel.shape[0]),
            st3.spkcache_frames() ? st3.spkcache().data() : nullptr, st3.spkcache_frames(),
            st3.fifo_frames() ? st3.fifo().data() : nullptr, st3.fifo_frames());
        auto emitted =
            st3.update(out.chunk_embs.data(), out.chunk_frames, out.preds.data(), lc, rc);

        auto ref_cp = load_ref(ref_dir, chunk_key(c, "chunk_preds"));
        const float d = max_abs_diff(emitted, ref_cp.f, std::min(emitted.size(), ref_cp.f.size()));
        if (d > worst_e2e)
            worst_e2e = d, worst_e2e_chunk = c;
    }
    std::printf(
        "[parity] e2e stage (free-running): worst emitted-prob diff %.3e (chunk %d) %s\n",
        worst_e2e, worst_e2e_chunk, worst_e2e <= 5e-2f ? "OK" : "FAIL");
    ok = ok && worst_e2e <= 5e-2f;

    // ------------------------------------------------------------------
    // Stage 4: full pipeline from RAW AUDIO (DiarStream: own FE + graph +
    // AOSC). Adds the mel front-end to the comparison - a log-floor or
    // window mismatch shows up here and nowhere above. Tail frames deviate
    // by design (riva-style exact-length tails vs NeMo pad+mask), so gate on
    // the frames before the tail chunks.
    // ------------------------------------------------------------------
    {
        auto audio = load_ref(ref_dir, "audio");
        auto ref_total = load_ref(ref_dir, "total_preds");  // (T, n_spk)
        nemo_speech::asr::DiarModel dmodel(bm, gguf_path);

        // FE diagnostic: our mel vs NeMo's (ref "mel" is (n_mels, T) with
        // n_mels the OUTER numpy dim -> flat [m*T + t]; ours is frame-major).
        {
            auto ref_mel = load_ref(ref_dir, "mel");
            const int64_t ref_t = ref_mel.shape[1];
            const int64_t mel_valid = load_ref(ref_dir, "mel_len").i[0];
            std::vector<float> mel;
            int mel_frames = 0;
            dmodel.fe().compute(
                audio.f.data(), audio.f.size(), mel, mel_frames, /*reflect_left=*/true,
                /*normalize=*/false);
            const int n_mels = dmodel.fe().n_mels();
            float dmax = 0.f;
            int64_t dmax_t = -1;
            int dmax_m = -1;
            const int64_t t_cmp = std::min<int64_t>(mel_frames, mel_valid);
            for (int64_t t = 0; t < t_cmp; t++)
                for (int m = 0; m < n_mels; m++) {
                    const float df = std::fabs(mel[t * n_mels + m] - ref_mel.f[m * ref_t + t]);
                    if (df > dmax)
                        dmax = df, dmax_t = t, dmax_m = m;
                }
            int64_t over_1 = 0, over_01 = 0;
            double sum = 0.0;
            for (int64_t t = 0; t < t_cmp; t++)
                for (int m = 0; m < n_mels; m++) {
                    const float df = std::fabs(mel[t * n_mels + m] - ref_mel.f[m * ref_t + t]);
                    sum += df;
                    if (df > 1.f)
                        over_1++;
                    if (df > 0.1f)
                        over_01++;
                }
            std::printf(
                "[parity] FE diagnostic: %d frames (NeMo valid %ld), worst mel diff %.3e "
                "(frame %ld bin %d: ours %.4f vs NeMo %.4f); mean %.2e, cells>0.1: %ld, "
                ">1.0: %ld of %ld\n",
                mel_frames, (long)mel_valid, dmax, (long)dmax_t, dmax_m,
                dmax_t >= 0 ? mel[dmax_t * n_mels + dmax_m] : 0.f,
                dmax_t >= 0 ? ref_mel.f[dmax_m * ref_t + dmax_t] : 0.f, sum / (t_cmp * n_mels),
                (long)over_01, (long)over_1, (long)(t_cmp * n_mels));
        }
        nemo_speech::asr::DiarStream stream(dmodel, geo);
        const size_t push = 160 * 16;  // 160 ms
        for (size_t off = 0; off < audio.f.size(); off += push)
            stream.feed_audio(audio.f.data() + off, std::min(push, audio.f.size() - off));
        stream.finish();

        // Compare emitted frames before the tail region.
        const int64_t n_tail_guard = 3 * geo.chunk_len;  // last few chunks
        const int64_t n_cmp =
            std::min<int64_t>(stream.n_frames(), ref_total.shape[0]) - n_tail_guard;
        float d = 0.f;
        int64_t worst_f = -1;
        int64_t over_05 = 0, argmax_flips = 0;
        for (int64_t f = 0; f < n_cmp; f++) {
            float frame_d = 0.f;
            int am_ours = 0, am_ref = 0;
            for (int s = 0; s < mcfg.num_speakers; s++) {
                const float po = stream.frame_probs()[f * mcfg.num_speakers + s];
                const float pr = ref_total.f[f * mcfg.num_speakers + s];
                frame_d = std::max(frame_d, std::fabs(po - pr));
                if (po > stream.frame_probs()[f * mcfg.num_speakers + am_ours])
                    am_ours = s;
                if (pr > ref_total.f[f * mcfg.num_speakers + am_ref])
                    am_ref = s;
            }
            if (frame_d > d)
                d = frame_d, worst_f = f;
            if (frame_d > 5e-2f)
                over_05++;
            if (am_ours != am_ref)
                argmax_flips++;
        }
        // Gate on decision-level agreement: a handful of boundary frames may
        // flicker (tiny FE diffs at speaker transitions push a sigmoid across
        // a steep region); what matters is that argmax speaker assignment
        // matches nearly everywhere.
        const bool stage_ok = argmax_flips <= n_cmp / 100 && over_05 <= n_cmp / 20;
        std::printf(
            "[parity] pipeline stage (raw audio, %ld frames vs NeMo %ld): worst prob diff "
            "%.3e (frame %ld), frames>5e-2: %ld/%ld, argmax flips: %ld %s\n",
            (long)stream.n_frames(), (long)ref_total.shape[0], d, (long)worst_f, (long)over_05,
            (long)n_cmp, (long)argmax_flips, stage_ok ? "OK" : "FAIL");
        ok = ok && stage_ok;
    }

    std::printf("[parity] %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
