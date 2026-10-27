// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gguf.h"
#include "salm_engine.h"
#include "salm_recognizer.h"

using namespace nemo_speech;
void
check(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
class FakeBackend final : public asr::OfflineRecognitionBackend {
   public:
    asr::Result recognize(
        const float* audio, size_t count, asr::AsrRequestOptions opts, const std::string& lang,
        int rate) override {
        check(
            audio && count == 2 && rate == 8000 && lang == "en-US" && opts.enable_word_time_offsets,
            "request forwarding lost fields");
        asr::Result out;
        out.audio_processed = 0.25f;
        return out;
    }
    void warmup() override { warmed = true; }
    bool supports_word_timestamps() const override { return true; }
    bool supports_speaker_diarization() const override { return true; }
    int sample_rate() const override { return 16000; }
    std::vector<std::string> supported_languages() const override { return {"auto"}; }
    asr::BatchMetrics batch_metrics() const override {
        asr::BatchMetrics m;
        m.items = 17;
        return m;
    }
    bool warmed = false;
    asr::postproc::Postprocessor pp{{}, nullptr, true};
};
// A decoder GGUF names its encoder file (nemo_speech.speech_model), found next to it, and both
// carry the checkpoint's id: the resolver fills the SALM paths from the decoder alone and refuses a
// missing encoder file or one of another checkpoint.
void
encoder_pairing() {
    namespace fs = std::filesystem;
    const fs::path dir =
        fs::temp_directory_path() /
        ("salm_pairing_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    auto write = [&](const char* name, const std::vector<std::pair<const char*, const char*>>& keys,
                     bool speech) {
        gguf_context* ctx = gguf_init_empty();
        for (const auto& [key, value] : keys) gguf_set_val_str(ctx, key, value);
        if (speech) {
            const char* parts[] = {"pee", "ctc_timestamp"};
            gguf_set_arr_str(ctx, "nemo_speech.components", parts, 2);
        }
        const std::string path = (dir / name).string();
        const bool ok = gguf_write_to_file(ctx, path.c_str(), false);
        gguf_free(ctx);
        check(ok, "cannot write a test GGUF");
        return path;
    };
    const std::string decoder = write(
        "d.gguf",
        {{"general.architecture", "nemotron_h_moe"},
         {"nemo_speech.speech_model", "s.gguf"},
         {"nemo_speech.checkpoint_id", "a1"}},
        false);
    const std::string speech = write(
        "s.gguf", {{"general.architecture", "pee"}, {"nemo_speech.checkpoint_id", "a1"}}, true);
    auto resolved = [&](const std::string& model, const std::string& encoder = "") {
        asr::RecognizerConfig cfg;
        cfg.model.path = model;
        cfg.salm.encoder_path = encoder;
        asr::resolve_salm_encoder(cfg);
        return std::make_pair(cfg.salm.encoder_path, cfg.salm.ctc_path);
    };
    auto refused = [&](const std::string& model) {
        try {
            resolved(model);
        }
        catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    check(
        resolved(decoder) == std::make_pair(speech, speech),
        "encoder file next to the decoder not found");
    check(
        resolved(decoder, speech) == std::make_pair(speech, speech),
        "an explicit encoder file lost its CTC head");
    const std::string plain = write("p.gguf", {{"general.architecture", "nemotron_h_moe"}}, false);
    check(
        resolved(plain) == std::make_pair(std::string(), std::string()),
        "a decoder without an encoder file got SALM paths");
    write("s.gguf", {{"general.architecture", "pee"}, {"nemo_speech.checkpoint_id", "b2"}}, true);
    check(refused(decoder), "encoder file of another checkpoint accepted");
    fs::remove(dir / "s.gguf");
    check(refused(decoder), "missing encoder file not reported");
    fs::remove_all(dir);
}

int
main() {
    try {
        encoder_pairing();
        asr::RecognizerConfig cfg;
        cfg.log_status = false;
        cfg.model.name = "test-salm";
        auto backend = std::make_unique<FakeBackend>();
        auto* fake = backend.get();
        asr::Recognizer rec(cfg, std::move(backend));
        check(
            rec.model() == nullptr && rec.sample_rate() == 16000 && rec.model_name() == "test-salm",
            "backend metadata requires acoustic model");
        check(
            rec.supported_languages() == std::vector<std::string>{"auto"} &&
                rec.batch_metrics().at(0).second.items == 17,
            "backend metadata not delegated");
        check(
            !rec.supports_streaming() && rec.supports_word_timestamps() &&
                rec.supports_speaker_diarization(),
            "backend capabilities not delegated");
        rec.warmup();
        check(fake->warmed, "warmup not delegated");
        asr::AsrRequestOptions opts;
        opts.enable_word_time_offsets = true;
        float samples[] = {0, 0};
        check(
            rec.recognize(samples, 2, opts, "en-US", 8000).audio_processed == 0.25f,
            "result not delegated");
        bool rejected = false;
        try {
            rec.streaming_recognize({}, "");
        }
        catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "offline backend must reject streaming before model access");

        // The alignment's words: text, speaker tag, start and end in seconds.
        auto aligned = [](const std::vector<std::tuple<std::string, int, double, double>>& words) {
            auto details = std::make_shared<asr::SpeechLmDetails>();
            for (const auto& [word, speaker, start, end] : words) {
                asr::SpeechLmWord w;
                w.word = word;
                w.speaker = speaker;
                w.start = start;
                w.end = end;
                details->words.push_back(w);
            }
            return details;
        };
        speechlm::SalmResult raw;
        raw.text = "Hello world.";
        raw.audio_seconds = 2.5;
        raw.details = aligned({{"Hello", 0, .123, .789}, {"world.", 0, .790, 1.2}});
        opts.enable_automatic_punctuation = true;
        auto result = speechlm::to_asr_result(raw, opts, "en-US", fake->pp);
        const auto& alt = result.alternatives.at(0);
        check(
            alt.transcript == raw.text && alt.words.size() == 2,
            "timestamp conversion changed text");
        check(
            alt.words[0].start_time == 123 && alt.words[0].end_time == 789 &&
                alt.words[1].end_time == 1200,
            "seconds not converted to milliseconds");
        check(
            alt.words[0].speaker_tag == 0 && alt.words[0].language_code == "en-US" &&
                result.audio_processed == 2.5f,
            "plain ASR metadata wrong");

        raw.text = "<spk:0> one <spk:2> two <spk:0> three";
        raw.details = aligned({{"one", 0, .1, .7}, {"two", 2, .3, .9}, {"three", 0, 1., 1.5}});
        opts.enable_speaker_diarization = true;
        result = speechlm::to_asr_result(raw, opts, "auto", fake->pp);
        const auto& diar = result.alternatives.at(0);
        check(
            diar.transcript == "one two three" && diar.words.size() == 3,
            "speaker tags leaked into text");
        check(
            diar.words[0].speaker_tag == 1 && diar.words[1].speaker_tag == 3 &&
                diar.words[2].speaker_tag == 1,
            "speaker IDs not preserved as one-based");
        check(
            diar.words[1].start_time < diar.words[0].end_time && diar.language_codes.empty(),
            "overlap or auto language lost");
        // overlapping speakers: the aligned words come sorted by time ("yes" starts inside "good
        // morning"); the text keeps the decoder's order, the words stay in time order
        raw.text = "<spk:0> good morning <spk:1> yes";
        raw.details = aligned({{"good", 0, .1, .4}, {"yes", 1, .3, .5}, {"morning", 0, .45, .9}});
        raw.details->words[0].word_index = 0;
        raw.details->words[1].word_index = 2;
        raw.details->words[2].word_index = 1;
        for (bool speakers : {true, false}) {
            opts.enable_speaker_diarization = speakers;
            result = speechlm::to_asr_result(raw, opts, "auto", fake->pp);
            const auto& overlap = result.alternatives.at(0);
            check(
                overlap.transcript == "good morning yes" && overlap.words.size() == 3,
                "timestamps reordered the transcript");
            check(
                overlap.words[0].word == "good" && overlap.words[1].word == "yes" &&
                    overlap.words[2].word == "morning",
                "words are not in time order");
        }
        opts.enable_speaker_diarization = true;
        raw.text = "<spk:0> one <spk:2> two <spk:0> three";
        // spk:2 without a legal alignment: its word stays in the transcript, not in the words
        raw.details = aligned({{"one", 0, .1, .7}, {"three", 0, 1., 1.5}});
        raw.details->unaligned_words = 1;
        result = speechlm::to_asr_result(raw, opts, "auto", fake->pp);
        check(
            result.alternatives[0].transcript == "one two three" &&
                result.alternatives[0].words.size() == 2,
            "an unaligned speaker's words left the transcript");
        // no alignment at all: the decoded text without its speaker tags
        raw.details = aligned({});
        result = speechlm::to_asr_result(raw, opts, "auto", fake->pp);
        check(
            result.alternatives[0].transcript == "one two three" &&
                result.alternatives[0].words.empty(),
            "speaker tags leaked without an alignment");
        opts.enable_speaker_diarization = false;
        opts.enable_word_time_offsets = false;
        raw.text = "Hello world.";
        result = speechlm::to_asr_result(raw, opts, "", fake->pp);
        check(
            result.alternatives[0].words.empty() && result.alternatives[0].transcript == raw.text,
            "text-only request returned timing data");
        std::cout << "SALM recognizer delegation and transport conversion passed\n";
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
