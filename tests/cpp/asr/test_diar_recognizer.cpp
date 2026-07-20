// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// End-to-end word-level speaker tagging through the recognizer: stream a
// multi-speaker wav with enable_speaker_diarization and check the final
// words carry sane speaker tags. Runner-agnostic (the ASR GGUF's head picks
// BufferedStreamRunner or CacheStreamRunner; the diarizer sidecar rides in
// RecognitionStream either way).
//
// Usage: test_diar_recognizer <asr.gguf> <diar.gguf> <audio.wav>
//            [--gpu N] [--min-speakers N]

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "fe.h"
#include "recognizer.h"

using namespace nemo_speech::asr;

int
main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(
            stderr, "usage: %s <asr.gguf> <diar.gguf> <audio.wav> [--gpu N] [--min-speakers N]\n",
            argv[0]);
        return 2;
    }
    RecognizerConfig cfg;
    cfg.model.path = argv[1];
    cfg.diar.model_path = argv[2];
    const std::string wav_path = argv[3];
    cfg.backend.gpu = -1;
    int min_speakers = 2;
    std::string seglst_session;  // non-empty: emit machine-readable SEG lines
    for (int i = 4; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--gpu" && i + 1 < argc)
            cfg.backend.gpu = std::atoi(argv[++i]);
        else if (a == "--min-speakers" && i + 1 < argc)
            min_speakers = std::atoi(argv[++i]);
        else if (a == "--seglst" && i + 1 < argc)
            seglst_session = argv[++i];
        else if (a == "--endpointing")
            cfg.endpointing.enable = true;  // mid-stream EOU finals -> exercises on-demand diar
        else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }

    std::vector<float> audio;
    int sr = 0;
    if (!read_wav_mono_16k(wav_path, audio, sr) || sr != 16000) {
        std::fprintf(stderr, "failed to read 16 kHz mono wav: %s\n", wav_path.c_str());
        return 1;
    }

    Recognizer rec(cfg);
    AsrRequestOptions opts;
    // Deliberately do NOT set enable_word_time_offsets: diarization alone
    // must imply word timings end to end (needs_word_timings()), which is
    // exactly how riva clients commonly send it. This is the regression test
    // for the "diarization-only requests produce no tags" bug.
    opts.enable_speaker_diarization = true;
    auto stream = rec.streaming_recognize(opts, "");

    std::vector<Word> words;
    auto take = [&](const Result& r) {
        if (r.is_final && !r.alternatives.empty())
            for (const auto& w : r.alternatives[0].words) words.push_back(w);
    };

    const size_t push = 160 * 16;  // 160 ms
    for (size_t off = 0; off < audio.size(); off += push) {
        stream->push(audio.data() + off, std::min(push, audio.size() - off));
        while (auto r = stream->next()) {
            take(*r);
            if (!r->is_final)
                break;
        }
    }
    take(stream->finish());

    if (!seglst_session.empty()) {
        // Machine-readable segments for cpWER scoring: contiguous same-speaker
        // word runs, tab-separated (session, speaker, t0, t1, words).
        size_t i = 0;
        while (i < words.size()) {
            size_t j = i;
            std::string text = words[i].word;
            while (j + 1 < words.size() && words[j + 1].speaker_tag == words[i].speaker_tag) {
                j++;
                text += " " + words[j].word;
            }
            std::printf(
                "SEG\t%s\tspk%d\t%.3f\t%.3f\t%s\n", seglst_session.c_str(), words[i].speaker_tag,
                words[i].start_time / 1000.0, words[j].end_time / 1000.0, text.c_str());
            i = j + 1;
        }
        return 0;
    }

    std::map<int, int> tag_counts;
    int untagged = 0;
    for (const auto& w : words) {
        if (w.speaker_tag > 0)
            tag_counts[w.speaker_tag]++;
        else
            untagged++;
        std::printf(
            "  %7.2fs-%7.2fs spk_%d  %s\n", w.start_time / 1000.0, w.end_time / 1000.0,
            w.speaker_tag, w.word.c_str());
    }
    std::printf(
        "[diar-rec] %zu words, %zu distinct speakers, %d untagged\n", words.size(),
        tag_counts.size(), untagged);
    for (const auto& [tag, n] : tag_counts)
        std::printf("[diar-rec]   speaker %d: %d words\n", tag, n);

    if (words.empty() || untagged > 0 || static_cast<int>(tag_counts.size()) < min_speakers) {
        std::printf("[diar-rec] FAIL\n");
        return 1;
    }
    std::printf("[diar-rec] OK\n");
    return 0;
}
