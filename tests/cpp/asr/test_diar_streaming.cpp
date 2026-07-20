// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Standalone streaming diarization CLI: stream a wav through the Sortformer
// pipeline and print speaker segments (optionally as RTTM for DER scoring).
//
// Usage:
//   test_diar_streaming <sortformer.gguf> <audio.wav> [--gpu] [--offline]
//       [--rttm NAME] [--dump-probs FILE] [--push-ms MS]
//       [--preset streaming|offline]
//       [--chunk N] [--rc N] [--lc N] [--fifo N] [--spkcache N] [--update N]
//       [--onset P] [--offset P] [--pad-onset S] [--pad-offset S]
//       [--min-on S] [--min-off S]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "diar_pipeline.h"
#include "fe.h"

using namespace nemo_speech::asr;

static const char* kUsage =
    "usage: %s <sortformer.gguf> <audio.wav> [--gpu] [--offline] [--rttm NAME]\n"
    "    [--dump-probs FILE] [--push-ms MS] [--compact-frames N]\n"
    "    [--preset streaming|offline]\n"
    "    [--chunk N] [--rc N] [--lc N] [--fifo N] [--spkcache N] [--update N]\n"
    "    [--onset P] [--offset P] [--pad-onset S] [--pad-offset S] [--min-on S] [--min-off S]\n";

int
main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, kUsage, argv[0]);
        return 2;
    }
    const std::string gguf_path = argv[1];
    const std::string wav_path = argv[2];
    bool use_gpu = false;
    bool offline = false;    // full-attention single pass, no streaming state
    std::string probs_path;  // raw f32 (n_frames x n_spk) frame-probs dump
    std::string rttm_name;
    DiarGeometry geo;             // streaming preset
    DiarSegmentationCfg seg_cfg;  // NeMo v2 postprocessing defaults
    int push_ms = 160;
    int compact_frames = 0;  // 0 = library default compaction horizon
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        // Bounds-checked value fetch: a value-taking flag as the last token is
        // a usage error, not a read past argv.
        auto next_str = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        auto next = [&]() { return std::stoi(next_str()); };
        if (a == "--gpu")
            use_gpu = true;
        else if (a == "--offline")
            offline = true;
        else if (a == "--dump-probs")
            probs_path = next_str();
        else if (a == "--rttm")
            rttm_name = next_str();
        else if (a == "--preset")
            geo = DiarGeometry::preset(next_str());  // later flags override
        else if (a == "--chunk")
            geo.chunk_len = next();
        else if (a == "--rc")
            geo.chunk_right_context = next();
        else if (a == "--lc")
            geo.chunk_left_context = next();
        else if (a == "--fifo")
            geo.fifo_len = next();
        else if (a == "--spkcache")
            geo.spkcache_len = next();
        else if (a == "--update")
            geo.spkcache_update_period = next();
        else if (a == "--push-ms") {
            push_ms = next();
            // push_ms drives the feed loop's stride (push_ms * 16 samples);
            // <= 0 would make the offset never advance.
            if (push_ms <= 0) {
                std::fprintf(stderr, "--push-ms must be > 0 (got %d)\n", push_ms);
                return 2;
            }
        } else if (a == "--compact-frames")
            // Test hook: force aggressive timeline compaction (trigger after N
            // frames, retain N/2) to exercise the long-stream memory bound on
            // short clips. Output must match an uncompacted run exactly.
            compact_frames = next();
        else if (a == "--onset")
            seg_cfg.onset = std::stof(next_str());
        else if (a == "--offset")
            seg_cfg.offset = std::stof(next_str());
        else if (a == "--pad-onset")
            seg_cfg.pad_onset = std::stod(next_str());
        else if (a == "--pad-offset")
            seg_cfg.pad_offset = std::stod(next_str());
        else if (a == "--min-on")
            seg_cfg.min_duration_on = std::stod(next_str());
        else if (a == "--min-off")
            seg_cfg.min_duration_off = std::stod(next_str());
        else {
            std::fprintf(stderr, "unknown arg: %s\n%s", a.c_str(), kUsage);
            return 2;
        }
    }

    std::vector<float> audio;
    int sr = 0;
    if (!read_wav_mono_16k(wav_path, audio, sr) || sr != 16000) {
        std::fprintf(stderr, "failed to read 16 kHz mono wav: %s\n", wav_path.c_str());
        return 1;
    }

    ggml_runtime::BackendManager bm({.use_gpu = use_gpu});
    DiarModel model(bm, gguf_path);

    std::vector<float> probs;
    std::vector<DiarSegment> segs;
    int64_t n_frames = 0;
    double sec_per_frame = 0.0;
    if (offline) {
        probs = model.diarize_offline(audio.data(), audio.size(), &n_frames);
        sec_per_frame =
            model.cfg().encoder.subsampling_factor * static_cast<double>(model.cfg().window_stride);
        segs = diar_segments_from_probs(
            probs.data(), n_frames, model.cfg().num_speakers, sec_per_frame, seg_cfg);
    } else {
        DiarStream stream(model, geo);
        if (compact_frames > 0)
            stream.set_compaction(compact_frames, compact_frames / 2);
        const size_t push = static_cast<size_t>(push_ms) * 16;
        for (size_t off = 0; off < audio.size(); off += push) {
            stream.feed_audio(audio.data() + off, std::min(push, audio.size() - off));
        }
        stream.finish();
        // segments() folds any compaction-frozen prefix in; frame_probs() is
        // only the retained tail, so it must not be re-segmented with the
        // total frame count.
        segs = stream.segments(seg_cfg);
        probs = stream.frame_probs();
        n_frames = stream.n_frames();
        sec_per_frame = stream.seconds_per_frame();
    }
    if (!probs_path.empty()) {
        std::FILE* f = std::fopen(probs_path.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", probs_path.c_str());
            return 1;
        }
        std::fwrite(probs.data(), sizeof(float), probs.size(), f);
        std::fclose(f);
    }

    if (!rttm_name.empty()) {
        for (const auto& s : segs) {
            std::printf(
                "SPEAKER %s 1 %.3f %.3f <NA> <NA> speaker_%d <NA> <NA>\n", rttm_name.c_str(), s.t0,
                s.t1 - s.t0, s.speaker);
        }
        return 0;
    }

    std::printf(
        "%.1fs audio -> %ld diar frames (%.0f ms/frame)%s\n", audio.size() / 16000.0,
        (long)n_frames, sec_per_frame * 1000.0, offline ? " [offline]" : "");
    for (const auto& s : segs) {
        std::printf("  [%7.2fs - %7.2fs] speaker_%d\n", s.t0, s.t1, s.speaker);
    }
    return 0;
}
