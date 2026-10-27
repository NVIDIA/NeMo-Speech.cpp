// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// The CTC timestamp head and the word aligner: the dynamic programs and the goldens of NeMo's
// aligner (tests/cpp/speechlm/data, generated from NeMo main 6f558408db) in ctest; by hand, with a
// head GGUF and fixtures of encoder states and NeMo's outputs, the head itself and the served path
// end to end:
//
//   test_ctc_timestamp [--tokenizer <head.model>] [<ctc.gguf> <fixtures-dir> [--gpu]]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "json.h"
#include "pee/ctc_timestamp.h"
#include "types.h"

using namespace nemo_speech::speechlm;
void
require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
std::vector<float>
load(const std::string& path, size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    if (!f)
        throw std::runtime_error("cannot read " + path);
    return v;
}
void
compare(
    const char* name, const std::vector<float>& got, const std::vector<float>& ref, double max_tol,
    double mean_tol) {
    require(got.size() == ref.size(), "parity size mismatch");
    double maximum = 0, mean = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        require(std::isfinite(got[i]), "non-finite parity output");
        double d = std::abs(got[i] - ref[i]);
        maximum = std::max(maximum, d);
        mean += d;
    }
    mean /= got.size();
    std::printf("%s: max %.8f mean %.8f\n", name, maximum, mean);
    require(maximum < max_tol && mean < mean_tol, "CTC numerical parity failure");
}
namespace ts = nemo_speech::speechlm::timestamps;
using nemo_speech::json::Value;

Value
load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot read " + path);
    std::stringstream buffer;
    buffer << f.rdbuf();
    return Value::parse(buffer.str());
}
std::vector<int>
ints(const Value& v) {
    std::vector<int> out;
    for (const auto& x : v.array()) out.push_back(int(x.number()));
    return out;
}
// float32 arrays stored as "f32:<base64>" or, when bfloat16 holds them exactly, "bf16:<base64>"
// (little-endian).
std::vector<float>
floats64(const Value& v) {
    static const std::string alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const std::string& text = v.string();
    const bool bf16 = text.rfind("bf16:", 0) == 0;
    require(bf16 || text.rfind("f32:", 0) == 0, "array encoding");
    std::string bytes;
    unsigned acc = 0;
    int bits = 0;
    for (char c : text.substr(text.find(':') + 1)) {
        if (c == '=')
            break;
        acc = (acc << 6) | unsigned(alphabet.find(c));
        if ((bits += 6) >= 8)
            bits -= 8, bytes.push_back(char((acc >> bits) & 255));
    }
    const size_t width = bf16 ? 2 : 4;
    std::vector<float> out(bytes.size() / width);
    for (size_t k = 0; k < out.size(); ++k) {
        uint32_t word = 0;
        std::memcpy(reinterpret_cast<char*>(&word) + (bf16 ? 2 : 0), &bytes[k * width], width);
        std::memcpy(&out[k], &word, 4);
    }
    return out;
}
uint32_t
bits_of(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}
std::string
fixture_path(const char* env, const char* fallback) {
    const char* value = std::getenv(env);
    return value ? value : fallback;
}

// Greedy column per frame of frames x columns log-probabilities.
std::vector<int>
argmax_rows(const std::vector<float>& lp, int frames, int columns) {
    std::vector<int> out(frames);
    for (int t = 0; t < frames; ++t)
        out[t] =
            int(std::max_element(
                    lp.begin() + size_t(t) * columns, lp.begin() + size_t(t + 1) * columns) -
                (lp.begin() + size_t(t) * columns));
    return out;
}

// ---- banded search against an exhaustive oracle (full band under 1000 frames) ----
void
dp_tests() {
    for (const std::vector<int>& labels :
         {std::vector<int>{0, 1, 0, 2, 0}, std::vector<int>{0, 1, 0, 1, 0}}) {
        for (int trial = 0; trial < 30; ++trial) {
            const int T = 6, C = 3, S = int(labels.size());
            std::vector<float> scores(T * C), prior(T);
            for (int i = 0; i < T * C; ++i) scores[i] = std::sin(float((i + 1) * (trial + 3)));
            for (int t = 0; t < T; ++t) prior[t] = -std::abs(std::cos(float(t * (trial + 1))));
            for (const bool with_prior : {false, true}) {
                const auto cell = [&](int t, int s) {
                    return double(scores[t * C + labels[s]]) +
                           (with_prior && labels[s] ? prior[t] : 0.0);
                };
                double best = -std::numeric_limits<double>::infinity();
                std::function<void(int, int, double)> visit = [&](int t, int s, double total) {
                    total += cell(t, s);
                    if (t == T - 1) {
                        if (s >= S - 2)
                            best = std::max(best, total);
                        return;
                    }
                    visit(t + 1, s, total);
                    if (s + 1 < S)
                        visit(t + 1, s + 1, total);
                    if (s + 2 < S && labels[s + 2] != 0 && labels[s + 2] != labels[s])
                        visit(t + 1, s + 2, total);
                };
                visit(0, 0, 0);
                visit(0, 1, 0);
                std::vector<int> positions(T);
                for (int t = 0; t < T; ++t) positions[t] = t;
                const auto found = ts::banded_ctc_search(
                    scores.data(), C, labels, positions, with_prior ? prior.data() : nullptr,
                    argmax_rows(scores, T, C));
                double got = 0;
                for (int t = 0; t < T; ++t) got += cell(t, found.path[t]);
                require(
                    std::abs(best - got) < 1e-5,
                    "banded search differs from the exhaustive oracle");
                require(std::abs(found.score - got) < 1e-5, "banded search score");
            }
        }
    }
    const std::vector<float> scores{1, 0, 1, 0};
    const auto impossible =
        ts::banded_ctc_search(scores.data(), 2, {0, 1, 0, 1, 0}, {0, 1}, nullptr, {0, 0});
    require(
        std::isinf(impossible.score) && impossible.path == std::vector<int>{-1, -1},
        "impossible repeated-label target accepted");
    std::puts(
        "banded CTC search: exhaustive oracle, prior, repeated labels, impossible targets: PASS");
}

// The path of a full-band search over (F x 6) log-probs of classes {0, 49, 93, 107, 281, blank}.
std::vector<int>
full_band_path(const std::vector<float>& lp, int frames) {
    std::vector<float> compact(size_t(frames) * 4);
    std::vector<int> positions(frames);
    for (int t = 0; t < frames; ++t) {
        const float* row = lp.data() + size_t(t) * 6;
        compact[size_t(t) * 4] = row[5];
        for (int c = 0; c < 3; ++c) compact[size_t(t) * 4 + 1 + c] = row[1 + c];
        positions[t] = t;
    }
    return ts::banded_ctc_search(
               compact.data(), 4, {0, 1, 0, 2, 0, 3, 0}, positions, nullptr,
               argmax_rows(compact, frames, 4))
        .path;
}

// ---- head windows and Python rounding (timestamp_golden.json) ----
void
golden_windows(const Value& g) {
    for (const auto& c : g.at("head_windows").array()) {
        const auto windows = ts::head_windows(int(c.at("frames").number()), 500, 50, 50);
        const auto& expected = c.at("windows").array();
        require(windows.size() == expected.size(), "head window count");
        for (size_t k = 0; k < windows.size(); ++k) {
            const auto e = ints(expected[k]);
            require(
                windows[k].start == e[0] && windows[k].end == e[1] &&
                    windows[k].context_start == e[2] && windows[k].context_end == e[3],
                "head window geometry");
        }
    }
    for (const auto& pair : g.at("round6").array())
        require(
            ts::round_decimals(pair.array()[0].number(), 6) == pair.array()[1].number(),
            "round(x, 6) differs from Python");
    std::puts("deferred-head windows, Python round: PASS");
}

// ---- alignment units (goldens from NeMo main 6f558408db) ----
void
golden_units(const Value& g) {
    for (const auto& c : g.at("cases").array()) {
        const auto& text = c.at("text").string();
        const auto words = ts::parse_sot_words(text);
        const auto& lexical = c.at("lexical").array();
        require(words.size() == lexical.size(), ("word count: " + text).c_str());
        for (size_t k = 0; k < words.size(); ++k) {
            std::string joined;
            for (const auto& ch : ts::lexical_characters(words[k].word)) joined += ch;
            require(joined == lexical[k].string(), ("lexical characters: " + text).c_str());
        }
        require(
            ts::timestamp_script(words) == c.at("script").string(), ("script: " + text).c_str());
        const auto units = ts::timestamp_units(words);
        const auto& expected = c.at("units").array();
        require(units.size() == expected.size(), ("unit count: " + text).c_str());
        for (size_t k = 0; k < units.size(); ++k) {
            const auto& e = expected[k].array();
            require(units[k].word == e[0].string(), ("unit text: " + text).c_str());
            require(int(units[k].kind) == int(e[1].number()), ("unit kind: " + text).c_str());
            require(units[k].display == e[2].string(), ("unit display: " + text).c_str());
            require(units[k].joins_previous == e[3].boolean(), ("unit join: " + text).c_str());
            require(units[k].index == int(k), "unit index");
        }
    }
    std::printf("alignment units: %zu golden transcripts PASS\n", g.at("cases").array().size());
}

// The units' speaker streams and CTC targets against NeMo's (_tokenize_words, _build_ctc_target),
// with the tokenizer the goldens were recorded with.
void
golden_units_tokenize(
    const Value& g, const std::function<ts::AlignmentTargets(const std::string&)>& targets,
    int blank, const char* source) {
    int checked = 0;
    for (const auto& c : g.at("cases").array()) {
        const auto& text = c.at("text").string();
        if (c.find("error")) {
            bool failed = false;
            try {
                targets(text);
            }
            catch (const std::exception& e) {
                failed = std::string(e.what()) == c.at("error").string();
            }
            require(failed, ("unit tokenizer refusal expected: " + text).c_str());
            continue;
        }
        const auto got = targets(text);
        const auto& streams = c.at("streams").array();
        if (!streams.empty() && ints(streams[0].at("labels"))[0] != blank) {
            std::printf("unit tokenisation goldens: recorded with another tokenizer, skipped\n");
            return;
        }
        require(got.streams.size() == streams.size(), ("unit streams: " + text).c_str());
        for (size_t s = 0; s < streams.size(); ++s) {
            std::vector<int> labels;
            for (int column : got.streams[s].labels) labels.push_back(got.columns[column]);
            require(
                got.streams[s].tag == int(streams[s].at("tag").number()) &&
                    labels == ints(streams[s].at("labels")) &&
                    got.streams[s].state_to_word == ints(streams[s].at("state_to_word")),
                ("unit CTC targets: " + text).c_str());
        }
        ++checked;
    }
    std::printf(
        "unit tokenisation + per-speaker CTC targets (%s): %d golden transcripts PASS\n", source,
        checked);
}

// ---- the aligner (goldens from NeMo main 6f558408db) ----
void
golden_numerics(const Value& g) {
    const auto inputs = floats64(g.at("log_inputs")), outputs = floats64(g.at("log_outputs"));
    for (size_t k = 0; k < inputs.size(); ++k)
        require(bits_of(ts::numpy_log(inputs[k])) == bits_of(outputs[k]), "numpy.log differs");
    for (const auto& c : g.at("resample").array()) {
        const auto probs = floats64(c.at("probs")), expected = floats64(c.at("expected"));
        const auto got = ts::resample_speaker_probs(
            probs.data(), int(c.at("rows").number()), int(c.at("speakers").number()),
            int(c.at("frames").number()));
        require(got.size() == expected.size(), "resample size");
        for (size_t k = 0; k < got.size(); ++k)
            require(bits_of(got[k]) == bits_of(expected[k]), "speaker resample differs from torch");
    }
    for (const auto& c : g.at("diarization").array()) {
        const auto activity = floats64(c.at("activity10"));
        const auto got = ts::diarization_segments(
            activity.data(), int(c.at("frames").number()), int(c.at("speakers").number()), 0.01,
            c.at("duration").number());
        const auto& expected = c.at("segments").array();
        require(got.size() == expected.size(), "diarization segment count");
        for (size_t k = 0; k < got.size(); ++k)
            require(
                got[k].speaker == int(expected[k].array()[0].number()) &&
                    got[k].start == expected[k].array()[1].number() &&
                    got[k].end == expected[k].array()[2].number(),
                "diarization segments differ from NeMo's");
    }
    std::printf(
        "numerics: numpy.log on %zu float32, %zu torch resamples bit-exact, %zu diarization label "
        "cases PASS\n",
        inputs.size(), g.at("resample").array().size(), g.at("diarization").array().size());
}

void
golden_lsap(const Value& g) {
    for (const auto& c : g.at("lsap").array()) {
        std::vector<double> cost;
        for (const auto& v : c.at("cost").array()) cost.push_back(v.number());
        const auto got =
            ts::linear_sum_assignment(cost, int(c.at("rows").number()), int(c.at("cols").number()));
        const auto& expected = c.at("assignment").array();
        require(got.size() == expected.size(), "assignment size");
        for (size_t k = 0; k < got.size(); ++k)
            require(
                got[k].first == int(expected[k].array()[0].number()) &&
                    got[k].second == int(expected[k].array()[1].number()),
                "linear_sum_assignment differs from SciPy");
    }
    std::printf("linear_sum_assignment: %zu SciPy cases PASS\n", g.at("lsap").array().size());
}

struct RecordCheck {
    std::string name;
    void operator()(bool ok, const std::string& what) const {
        if (!ok)
            throw std::runtime_error("record " + name + ": " + what + " differs from NeMo");
    }
};

ts::AlignmentTargets
record_targets(const Value& r) {
    ts::AlignmentTargets targets;
    for (const auto& w : r.at("words").array()) {
        ts::TranscriptWord word;
        word.word = w.array()[0].string();
        word.tag = int(w.array()[1].number());
        word.index = int(targets.words.size());
        word.kind =
            w.array()[2].string() == "character" ? ts::UnitKind::character : ts::UnitKind::word;
        targets.words.push_back(word);
    }
    for (const auto& s : r.at("streams").array()) {
        ts::SpeakerStream stream;
        stream.tag = int(s.at("tag").number());
        stream.words = ints(s.at("words"));
        stream.labels = ints(s.at("labels"));
        stream.state_to_word = ints(s.at("state_to_word"));
        targets.streams.push_back(stream);
    }
    targets.columns = ints(r.at("columns"));
    return targets;
}

std::string
python_tag(int tag) {
    return tag < 0 ? "None" : std::to_string(tag);
}

using TargetsOf = std::function<ts::AlignmentTargets(const std::string&, int frames)>;

// Every record of an aligner golden: the alignment from NeMo's prepared targets and log-probs, and,
// with `targets_of`, the targets themselves (and NeMo's rejections).
void
golden_records(const Value& g, const TargetsOf& targets_of, const char* source) {
    const int blank = int(g.at("blank").number());
    int records = 0, rejected = 0, words = 0, failed = 0, banded = 0, targets_checked = 0;
    for (const auto& r : g.at("records").array()) {
        const RecordCheck check{r.at("name").string()};
        const auto& transcript = r.at("transcript").string();
        const int frames = int(r.at("frames").number());
        if (r.find("error")) {
            if (targets_of) {
                std::string error;
                try {
                    targets_of(transcript, frames);
                }
                catch (const std::exception& e) {
                    error = e.what();
                }
                check(error == r.at("error").string(), "rejection");
                ++targets_checked;
            }
            ++rejected;
            continue;
        }
        // Transcript words and units.
        const auto units = ts::timestamp_units(ts::parse_sot_words(transcript));
        const auto& expected_words = r.at("words").array();
        check(units.size() == expected_words.size(), "unit count");
        for (size_t k = 0; k < units.size(); ++k)
            check(
                units[k].word == expected_words[k].array()[0].string() &&
                    units[k].tag == int(expected_words[k].array()[1].number()),
                "unit " + std::to_string(k));
        const auto targets = record_targets(r);
        if (targets_of) {
            const auto got = targets_of(transcript, frames);
            check(got.columns == targets.columns, "compact columns");
            check(got.streams.size() == targets.streams.size(), "stream count");
            for (size_t s = 0; s < got.streams.size(); ++s)
                check(
                    got.streams[s].tag == targets.streams[s].tag &&
                        got.streams[s].words == targets.streams[s].words &&
                        got.streams[s].labels == targets.streams[s].labels &&
                        got.streams[s].state_to_word == targets.streams[s].state_to_word,
                    "stream targets");
            ++targets_checked;
        }
        // Alignment.
        const auto log_probs = floats64(r.at("log_probs"));
        const auto speaker = r.at("speaker_probs").is_null() ? std::vector<float>{}
                                                             : floats64(r.at("speaker_probs"));
        // NeMo's 10 ms labels (runs of active frames per speaker) as activity.
        std::vector<float> activity10;
        if (!r.at("labels10").is_null()) {
            const auto& rows = r.at("labels10").array();
            activity10.assign(size_t(r.at("label_frames").number()) * rows.size(), 0.0f);
            for (size_t c = 0; c < rows.size(); ++c)
                for (const auto& run : rows[c].array())
                    for (int t = int(run.array()[0].number()); t < int(run.array()[1].number());
                         ++t)
                        activity10[size_t(t) * rows.size() + c] = 1.0f;
        }
        ts::AlignmentInputs inputs;
        inputs.log_probs = log_probs.data();
        inputs.frames = frames;
        inputs.speaker_probs = speaker.empty() ? nullptr : speaker.data();
        inputs.speaker_frames = int(r.at("speaker_frames").number());
        inputs.speakers = int(r.at("speakers").number());
        inputs.activity10 = activity10.empty() ? nullptr : activity10.data();
        inputs.label_frames = int(r.at("label_frames").number());
        inputs.label_speakers = inputs.speakers;
        inputs.duration = r.at("duration").number();
        ts::AlignmentConfig config;
        config.max_word_frames = int(r.at("max_word_frames").number());
        check(log_probs.size() == size_t(frames) * targets.columns.size(), "log-prob shape");
        const auto result = ts::align_record(targets, inputs, config);
        const auto& e = r.at("expected");
        check(result.ctc_step == e.at("ctc_step").number(), "ctc step");
        const auto& mapping = e.at("mapping").array();
        check(result.mapping.size() == mapping.size(), "mapping size");
        for (size_t k = 0; k < mapping.size(); ++k)
            check(
                result.mapping[k].first == int(mapping[k].array()[0].number()) &&
                    result.mapping[k].second == int(mapping[k].array()[1].number()),
                "speaker mapping");
        const auto& searches = e.at("searches").array();
        check(result.scores.size() == searches.size(), "search count");
        for (size_t k = 0; k < searches.size(); ++k) {
            const auto& score = searches[k].at("score");
            check(
                score.is_null() ? std::isinf(result.scores[k])
                                : double(result.scores[k]) == score.number(),
                "path score of stream " + std::to_string(k));
            banded += searches[k].at("retained").number() >= 1000;
        }
        const auto& failures = e.at("failed").array();
        check(result.failed_streams.size() == failures.size(), "failed streams");
        for (size_t k = 0; k < failures.size(); ++k) {
            const auto& f = failures[k].array();
            std::vector<std::string> names;
            for (int w : result.failed_streams[k].words) names.push_back(targets.words[w].word);
            std::vector<std::string> expected_names;
            for (const auto& w : f[2].array()) expected_names.push_back(w.string());
            check(
                result.failed_streams[k].tag == int(f[0].number()) &&
                    result.failed_streams[k].reason == f[1].string() && names == expected_names,
                "failed stream");
        }
        failed += int(failures.size());
        // Word rows (frames, column, confidence; times after refinement below).
        for (const auto& row : e.at("rows").array()) {
            const auto& x = row.array();
            const auto it = std::find_if(
                result.words.begin(), result.words.end(),
                [&](const ts::WordRow& w) { return w.word == int(x[0].number()); });
            check(it != result.words.end(), "word row");
            check(
                it->start_frame == int(x[1].number()) && it->end_frame == int(x[2].number()) &&
                    it->column == int(x[3].number()),
                "word frames of word " + std::to_string(it->word));
            check(
                x[4].is_null() ? it->speaker_confidence < 0
                               : std::abs(it->speaker_confidence - x[4].number()) < 1e-6,
                "speaker confidence");
        }
        // Refined speaker_word_timestamps, in NeMo's group and word order, exact.
        size_t at = 0;
        for (const auto& group : e.at("speaker_word_timestamps").array()) {
            const int tag = int(group.array()[0].number());
            for (const auto& w : group.array()[1].array()) {
                check(at < result.words.size(), "refined word count");
                const auto& got = result.words[at++];
                check(
                    got.tag == tag && targets.words[got.word].word == w.array()[0].string() &&
                        got.start == w.array()[1].number() && got.end == w.array()[2].number(),
                    "refined word " + std::to_string(at - 1));
            }
        }
        check(at == result.words.size(), "refined word count");
        const auto& refinement = e.at("refinement");
        check(
            result.refinement.max_seconds == refinement.at("max_seconds").number() &&
                result.refinement.words == int(refinement.at("words").number()) &&
                result.refinement.onsets_moved == int(refinement.at("onsets_moved").number()) &&
                result.refinement.unmatched_words == int(refinement.at("unmatched_words").number()),
            "refinement statistics");
        const auto& diarization = e.at("diarization").array();
        check(result.diarization.size() == diarization.size(), "diarization segment count");
        for (size_t k = 0; k < diarization.size(); ++k) {
            const auto& s = diarization[k].array();
            check(
                result.diarization[k].speaker == int(s[0].number()) &&
                    result.diarization[k].start == s[1].number() &&
                    result.diarization[k].end == s[2].number(),
                "diarization segment");
        }
        // The plugin's public result: words of all speakers by (start, end), milliseconds.
        const auto& pub = e.at("public");
        const auto ordered = ts::public_words(result.words);
        const auto& pub_words = pub.at("words").array();
        check(ordered.size() == pub_words.size(), "public word count");
        for (size_t k = 0; k < ordered.size(); ++k)
            check(
                targets.words[ordered[k].word].word == pub_words[k].at("word").string() &&
                    ordered[k].start == pub_words[k].at("start").number() &&
                    ordered[k].end == pub_words[k].at("end").number() &&
                    python_tag(ordered[k].tag) == pub_words[k].at("speaker").string(),
                "public word " + std::to_string(k));
        const auto& pub_diarization = pub.at("diarization").array();
        for (size_t k = 0; k < pub_diarization.size(); ++k)
            check(
                ts::round_decimals(result.diarization[k].start, 3) ==
                        pub_diarization[k].at("start").number() &&
                    ts::round_decimals(result.diarization[k].end, 3) ==
                        pub_diarization[k].at("end").number(),
                "public diarization");
        for (const auto& [tag, column] : result.mapping) {
            const auto& m = pub.at("speaker_tag_to_diarization_speaker").at(python_tag(tag));
            check(m.is_null() ? column < 0 : column == int(m.number()), "public speaker mapping");
        }
        words += int(ordered.size());
        ++records;
    }
    std::printf(
        "aligner vs NeMo (%s): %d records (%d words, %d failed streams, %d banded searches over "
        "1000 "
        "frames) exact, %d rejected transcripts%s PASS\n",
        source, records, words, failed, banded, rejected,
        targets_of ? (", targets of " + std::to_string(targets_checked) + " transcripts").c_str()
                   : "");
}

void
transcript_tests() {
    // clean_sot_transcript keeps speaker, event and style tags and drops other special tokens.
    require(
        ts::clean_sot_transcript(
            "<spk:0> hi</s> <ev:laughter> <|im_start|>there <sty:end:flat>",
            {"</s>", "<|im_start|>", "<spk:0>", "<ev:laughter>", "<sty:end:flat>"}) ==
            "<spk:0> hi <ev:laughter> there <sty:end:flat>",
        "clean_sot_transcript");
    // Python's \d and case folding in the speaker tag: Arabic-Indic digits, long s, Kelvin sign.
    const auto words = ts::parse_sot_words(
        "a <spk:\xD9\xA3> b <\xC5\xBFp\xE2\x84\xAA:12> c <spk:x> d\xE2\x80\x94"
        "e");
    const std::vector<std::pair<std::string, int>> expected{{"a", -1},       {"b", 3},  {"c", 12},
                                                            {"<spk:x>", 12}, {"d", 12}, {"e", 12}};
    require(words.size() == expected.size(), "speaker tag grammar");
    for (size_t k = 0; k < words.size(); ++k)
        require(
            words[k].word == expected[k].first && words[k].tag == expected[k].second,
            "speaker tag grammar");
    std::puts("transcript cleaning and speaker-tag grammar: PASS");
}

void
golden_host_tests(const sentencepiece::SentencePieceProcessor* tokenizer) {
    transcript_tests();
    const auto aligner =
        load_json(fixture_path("TIMESTAMP_ALIGNER_GOLDEN", TIMESTAMP_ALIGNER_GOLDEN_DEFAULT));
    const int blank = int(aligner.at("blank").number());
    const auto units =
        load_json(fixture_path("TIMESTAMP_UNITS_GOLDEN", TIMESTAMP_UNITS_GOLDEN_DEFAULT));
    golden_units(units);
    if (tokenizer)
        golden_units_tokenize(
            units,
            [&](const std::string& text) {
                return ts::alignment_targets(
                    *tokenizer, text, blank, std::numeric_limits<int>::max());
            },
            blank, "tokenizer file");
    golden_windows(load_json(fixture_path("TIMESTAMP_GOLDEN", TIMESTAMP_GOLDEN_DEFAULT)));
    golden_numerics(aligner.at("numerics"));
    golden_lsap(aligner);
    golden_records(
        aligner,
        tokenizer ? TargetsOf([&](const std::string& text, int frames) {
            return ts::alignment_targets(*tokenizer, text, blank, frames);
        })
                  : TargetsOf(),
        "synthetic records");
}

// Evidence of host states (frames x dimension()) the way the encoder produces it: the states in a
// device buffer, staged into the evidence rows on the producer's stream, then the head.
std::shared_ptr<const CtcEvidence>
served_evidence(
    CtcTimestampModel& model, ggml_runtime::BackendManager& bm, const float* states, int frames) {
    ggml_backend_t backend =
        bm.gpu_backend_handle() ? bm.gpu_backend_handle() : bm.get_backends().front();
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_init_params params{ggml_tensor_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.dimension(), frames);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    ggml_backend_tensor_set(rows, states, 0, ggml_nbytes(rows));
    auto evidence = model.reserve(frames);
    model.stage(backend, {rows, buft, 0}, 0, *evidence, 0, frames);
    ggml_backend_synchronize(backend);
    model.finish({evidence});
    evidence->wait();
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return evidence;
}

void
contract_model_tests(
    CtcTimestampModel& model, ggml_runtime::BackendManager& bm, const std::string& fixtures) {
    golden_units_tokenize(
        load_json(fixture_path("TIMESTAMP_UNITS_GOLDEN", TIMESTAMP_UNITS_GOLDEN_DEFAULT)),
        [&](const std::string& text) { return model.targets(text); }, model.blank_id(),
        "the head's tokenizer");
    int B, T, D;
    {
        std::ifstream shape(fixtures + "/real.shape");
        shape >> B >> T >> D;
    }
    const auto states = load(fixtures + "/real.input.f32", size_t(B) * T * D);
    const int blank = model.blank_id(), U = model.upsampling();
    // Head log-probabilities against NeMo's fp32 reference (one window: T < 500 encoder frames;
    // U CTC frames per encoder frame).
    const std::vector<int> ids{0, 49, 93, 107, 281, blank};
    TimestampOptions exact;
    exact.bf16_emissions = false;
    const auto lp = model.head_log_probs(states.data(), T, ids, exact);
    compare(
        "head log-probs vs NeMo", lp, load(fixtures + "/real.logprobs.f32", lp.size()), .08, .008);
    const auto rounded = model.head_log_probs(states.data(), T, ids);
    for (size_t k = 0; k < lp.size(); ++k) {
        float r = lp[k];
        uint32_t bits;
        std::memcpy(&bits, &r, 4);
        bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
        std::memcpy(&r, &bits, 4);
        require(rounded[k] == r, "bf16 emissions are the rounded fp32 log-probs");
    }
    // Windowed head over a long file (windows in encoder frames) equals each window run on its own
    // slice (the class projection's GEMM shape follows the request length: equal up to its
    // rounding), and NeMo's windowed head when the fixtures have it.
    const int long_frames = 1200;
    std::vector<float> tiled(size_t(long_frames) * D);
    for (int t = 0; t < long_frames; ++t)
        std::memcpy(&tiled[size_t(t) * D], &states[size_t(t % T) * D], D * sizeof(float));
    const auto whole = model.head_log_probs(tiled.data(), long_frames, ids, exact);
    require(whole.size() == size_t(U) * long_frames * ids.size(), "windowed head frames");
    TimestampOptions single = exact;
    single.head_window = 0;
    std::vector<float> windowed, alone;
    const size_t row = size_t(U) * ids.size();  // floats per encoder frame
    for (const auto& w : ts::head_windows(long_frames, 500, 50, 50)) {
        const auto part = model.head_log_probs(
            tiled.data() + size_t(w.context_start) * D, w.context_end - w.context_start, ids,
            single);
        windowed.insert(windowed.end(), whole.begin() + w.start * row, whole.begin() + w.end * row);
        alone.insert(
            alone.end(), part.begin() + (w.start - w.context_start) * row,
            part.begin() + (w.end - w.context_start) * row);
    }
    compare("windowed head vs each window alone", windowed, alone, 1e-3, 1e-4);
    if (std::ifstream(fixtures + "/long.logprobs.f32"))
        compare(
            "windowed head vs NeMo's windowed head", whole,
            load(fixtures + "/long.logprobs.f32", whole.size()), .08, .008);
    // Requests batched together (different lengths, classes and windows) match each run alone,
    // within the batch effect of the adapter's attention and GEMM kernels (as for the bridge).
    {
        std::vector<CtcTimestampModel::HeadRequest> requests{
            {tiled.data(), long_frames, ids},
            {tiled.data() + size_t(7) * D, 130, {5, 49, 300, blank}},
            {tiled.data() + size_t(31) * D, 61, {0, 93, 1024, 2000, 4095, blank}},
            {tiled.data() + size_t(3) * D, 700, {107, 281, blank}}};
        const auto batched = model.head_log_probs(requests, exact);
        std::vector<float> got, alone;
        for (size_t r = 0; r < requests.size(); ++r) {
            const auto one = model.head_log_probs(
                requests[r].states, requests[r].frames, requests[r].columns, exact);
            require(batched[r].size() == one.size(), "batched head size");
            got.insert(got.end(), batched[r].begin(), batched[r].end());
            alone.insert(alone.end(), one.begin(), one.end());
        }
        compare("batched requests vs each alone", got, alone, .05, .005);
    }
    // End to end on the real clip: its 80 ms speaker probabilities, 10 ms activity derived from
    // them (as the aligner goldens derive it), the served head.
    std::ifstream tf(fixtures + "/transcript.txt");
    std::string plain((std::istreambuf_iterator<char>(tf)), {});
    const auto p80 = load(fixtures + "/probs80.f32", size_t(T) * 8);
    std::vector<float> p10(size_t(T) * 8 * 8);
    for (size_t t = 0; t < size_t(T) * 8; ++t)
        for (int c = 0; c < 8; ++c)
            p10[t * 8 + c] =
                p80[(t / 8) * 8 + c] * (0.9f + 0.2f * float((t * 7 + c * 3) % 11) / 10.f);
    TimestampEvidence ev;
    ev.head = served_evidence(model, bm, states.data(), T);
    ev.encoder_frames = T;
    require(
        ev.head->encoder_frames() == T && ev.ctc_frames() == U * T &&
            ev.head->lse().size() == size_t(U) * T,
        "evidence frames: encoder frames and U x as many CTC frames");
    ev.activity = p80.data();
    ev.activity_frames = T;
    ev.speakers = 8;
    ev.activity10 = p10.data();
    ev.activity10_frames = T * 8 - 3;
    ev.duration_seconds = (T * 8 - 3) * 0.01;
    // The served path (evidence, class projection, bf16 emissions, alignment) against NeMo's
    // public result for the same clip with its own head (the aligner goldens' real records).
    const bool has_golden = bool(std::ifstream(fixtures + "/aligner_real.json"));
    const auto golden = has_golden ? load_json(fixtures + "/aligner_real.json")
                                   : Value(Value::Object{{"blank", Value(-1)}});
    if (int(golden.at("blank").number()) == blank)
        golden_records(
            golden,
            [&](const std::string& text, int frames) { return model.targets(text, frames); },
            "real clip and long records");
    int compared = 0;
    for (const auto& r : has_golden ? golden.at("records").array() : Value::Array{}) {
        if (r.at("origin").string() != "real" || int(golden.at("blank").number()) != blank ||
            int(r.at("upsampling").number()) != U)
            continue;
        nemo_speech::asr::SpeechLmDetails details;
        const auto error = model.word_timestamps(ev, r.at("transcript").string(), details);
        require(
            error.empty() && details.alignment_status == "aligned",
            ("end-to-end alignment failed: " + error).c_str());
        require(details.alignment_model == model.model_name(), "alignment model name");
        require(
            details.frame_stride_seconds == ev.duration_seconds / (U * T),
            "frame stride = duration / CTC frames");
        const auto& pub = r.at("expected").at("public");
        const auto& expected = pub.at("words").array();
        require(details.words.size() == expected.size(), "served words: count");
        // Words matched by speaker and occurrence (a one-frame shift may reorder neighbours).
        std::map<std::pair<std::string, std::string>, std::vector<const Value*>> by_word;
        for (const auto& w : expected)
            by_word[{w.at("speaker").string(), w.at("word").string()}].push_back(&w);
        std::map<std::pair<std::string, std::string>, size_t> seen;
        int exact_bounds = 0, total = 0;
        double max_error = 0;
        for (const auto& w : details.words) {
            const auto key = std::make_pair(
                w.speaker < 0 ? std::string("None") : std::to_string(w.speaker), w.word);
            const auto& list = by_word[key];
            const size_t k = seen[key]++;
            require(k < list.size(), "served words: words and speakers");
            for (double delta :
                 {std::abs(w.start - list[k]->at("start").number()),
                  std::abs(w.end - list[k]->at("end").number())}) {
                exact_bounds += delta < 1e-9;
                ++total;
                max_error = std::max(max_error, delta);
            }
        }
        for (const auto& w : details.words) {
            const auto& m =
                pub.at("speaker_tag_to_diarization_speaker")
                    .at(w.speaker < 0 ? std::string("None") : std::to_string(w.speaker));
            require(
                m.is_null() ? w.speaker_slot < 0 : w.speaker_slot == int(m.number()),
                "served speaker slot");
        }
        const auto& segments = pub.at("diarization").array();
        require(details.diarization.size() == segments.size(), "served diarization segments");
        for (size_t k = 0; k < segments.size(); ++k)
            require(
                details.diarization[k].speaker == int(segments[k].at("speaker").number()) &&
                    details.diarization[k].start == segments[k].at("start").number() &&
                    details.diarization[k].end == segments[k].at("end").number(),
                "served diarization segment");
        std::printf(
            "served path vs NeMo (%s): %zu words, %d/%d boundaries exact, max difference %.3f s\n",
            r.at("name").string().c_str(), details.words.size(), exact_bounds, total, max_error);
        require(
            exact_bounds >= total * .95 && max_error <= 2 * details.frame_stride_seconds + 1e-9,
            "served-path word alignment differs from NeMo's");
        ++compared;
    }
    if (!compared)
        std::puts("served path vs NeMo: goldens recorded with another head, skipped");
    // Failure semantics: never thrown, exact reasons.
    nemo_speech::asr::SpeechLmDetails d;
    model.word_timestamps(ev, "  \t ", d);
    require(
        d.alignment_status == "unavailable" && d.failure_reason == "empty_transcript" &&
            d.words.empty(),
        "empty transcript");
    TimestampOptions cleaning;
    cleaning.special_tokens = {"</s>"};
    model.word_timestamps({ev}, {" </s> "}, {&d}, cleaning);
    require(d.failure_reason == "empty_transcript", "special tokens only");
    TimestampEvidence missing = ev;
    missing.head = nullptr;
    model.word_timestamps(missing, plain, d);
    require(
        d.alignment_status == "unavailable" && d.failure_reason == "ctc_evidence_unavailable",
        "missing evidence");
    TimestampEvidence short_ev = ev;
    short_ev.head = served_evidence(model, bm, states.data(), 3);
    short_ev.encoder_frames = 3;
    require(
        !model.word_timestamps(short_ev, plain, d).empty() &&
            d.failure_reason == "ctc_alignment_failed" && d.alignment_model.empty() &&
            d.frame_stride_seconds == 0 && !d.diarization.empty(),
        "too few frames: rejected, the diarization still returned");
    model.word_timestamps(ev, "<spk:0>", d);
    require(
        d.alignment_status == "aligned" && d.words.empty() && d.speaker_words.empty() &&
            !d.diarization.empty(),
        "a transcript of tags only aligns with no words");
    // More tags than Sortformer columns: the unmapped stream is left out, the rest stay aligned.
    std::string many;
    for (int k = 0; k < 10; ++k)
        many += "<spk:" + std::to_string(k) + "> word" + std::to_string(k) + " ";
    model.word_timestamps(ev, many, d);
    require(
        d.alignment_status == "aligned" && d.words.size() < 10 && !d.warnings.empty(),
        "unmapped speaker streams are dropped with a warning");
    std::puts("alignment status / failure_reason semantics: PASS");
}

int
main(int argc, char** argv) {
    try {
        // Optional: --tokenizer <SentencePiece model of the head> checks the targets too.
        std::vector<char*> args(argv, argv + argc);
        std::shared_ptr<const sentencepiece::SentencePieceProcessor> tokenizer;
        for (size_t k = 1; k + 1 < args.size(); ++k)
            if (std::string(args[k]) == "--tokenizer") {
                tokenizer = ts::load_tokenizer(args[k + 1]);
                args.erase(args.begin() + k, args.begin() + k + 2);
                break;
            }
        argc = static_cast<int>(args.size());
        argv = args.data();
        dp_tests();
        golden_host_tests(tokenizer.get());
        if (argc < 3)
            return 0;
        ggml_runtime::Params params;
        params.use_gpu = argc > 3 && std::string(argv[3]) == "--gpu";
        ggml_runtime::BackendManager bm(params);
        CtcTimestampModel model(bm, argv[1]);
        ggml_runtime::GGUFLoader metadata(argv[1]);
        int blank_id = metadata.get_u32("ctc.vocab_size") - 1;
        // Fixtures: encoder states (B, T, D) and lengths in encoder frames; NeMo's outputs on the
        // head's CTC frames, U per encoder frame.
        const int U = model.upsampling();
        std::printf("head: %d classes, %d CTC frames per encoder frame\n", blank_id + 1, U);
        for (const std::string name : {"real", "padded"}) {
            const std::string base = std::string(argv[2]) + "/" + name;
            int B, T, D;
            std::ifstream shape(base + ".shape");
            shape >> B >> T >> D;
            require(bool(shape), "missing fixture shape");
            std::vector<int> lengths(B);
            for (auto& l : lengths) shape >> l;
            auto x = load(base + ".input.f32", size_t(B) * T * D);
            auto h = model.bridge(x.data(), T, lengths);
            compare(
                (name + " bridge").c_str(), h, load(base + ".bridge.f32", h.size()), .012, .001);
            const std::vector<int> ids{0, 49, 93, 107, 281, blank_id};
            const int F = U * T;  // CTC frames per batch item
            const auto reference = load(base + ".logits.f32", size_t(B) * F * ids.size());
            const auto logprobs = load(base + ".logprobs.f32", reference.size());
            for (int b = 0; b < B; ++b) {
                std::vector<float> sample(
                    h.begin() + size_t(b) * F * D, h.begin() + size_t(b + 1) * F * D);
                auto y = model.project(sample, F, ids);
                std::vector<float> ref(
                    reference.begin() + size_t(b) * F * ids.size(),
                    reference.begin() + size_t(b + 1) * F * ids.size());
                auto valid_y = y, valid_ref = ref;
                valid_y.resize(size_t(U) * lengths[b] * ids.size());
                valid_ref.resize(valid_y.size());
                compare((name + " valid logits").c_str(), valid_y, valid_ref, .08, .008);
                std::vector<float> lp(
                    logprobs.begin() + size_t(b) * F * ids.size(),
                    logprobs.begin() + size_t(b + 1) * F * ids.size());
                require(
                    full_band_path(ref, F) == full_band_path(lp, F),
                    "full-vocabulary normalization changed path");
            }
            if (B == 2) {
                auto single = model.bridge(x.data() + T * D, lengths[1], {lengths[1]});
                std::vector<float> valid(
                    h.begin() + size_t(F) * D, h.begin() + size_t(F + U * lengths[1]) * D);
                compare("padded batch vs unpadded single", valid, single, .012, .001);
            }
        }
        contract_model_tests(model, bm, argv[2]);
        std::puts("CTC parity: PASS");
        return 0;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
