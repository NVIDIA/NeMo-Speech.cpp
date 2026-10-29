// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cmath>
#if defined(NEMO_SPEECH_CLI_LIVE)
#include <chrono>
#include <csignal>
#endif
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "audio_file.h"
#include "cli_util.h"
#include "commands.h"
#include "engine_registry.h"
#if defined(NEMO_SPEECH_CLI_LIVE)
#include "live_terminal.h"
#include "live_transcript.h"
#include "microphone_capture.h"
#endif
#include "model_utils.h"
#include "parameter_parser.h"
#include "recognizer.h"
#include "subtitles.h"
#if defined(NEMO_SPEECH_CLI_NMT)
#include "speech_translator.h"
#endif

namespace {
namespace fs = std::filesystem;
namespace asr = nemo_speech::asr;

struct Transcript {
    std::string text;
    std::string source_text;
    std::string target_language;
    float confidence = 0.0f;
    float audio_seconds = 0.0f;
    std::vector<std::string> languages;
    std::vector<nemo_speech::subtitle::Word> words;
    // False when a speech-LLM could not align every word: `text` has words `words` lacks.
    bool words_cover_text = true;
};

enum class OutputFormat { Text, Json, Srt, Vtt };

struct Options {
    fs::path input;
    std::string model;
    std::string language;
    std::string vad_model;
    std::string diar_model;
    std::string itn_model_dir;
    std::string pnc_model;
    std::string instruction;  // speech-LLM: replaces the served prompt
    fs::path output;
    fs::path output_dir;
    OutputFormat format = OutputFormat::Text;
    asr::RecognizerConfig engine;
    asr::AsrRequestOptions request;
    std::string config_file;
    std::vector<std::string> speech_contexts;
    float speech_context_boost = 0.0f;
    int gpu = default_gpu_index();
    int concurrency = 0;
    bool recursive = false;
    bool force = false;
    bool word_times = false;
    bool punctuation = true;
    bool verbatim = false;
    bool diarize = false;
    bool live = false;
    bool stream = false;
    bool warmup = true;
    bool batching = true;
    bool device_set = false;
    std::string translate_to;
    bool speech_llm_translates = false;  // --translate-to by the speech-LLM itself (no NMT)
#if defined(NEMO_SPEECH_CLI_NMT)
    std::string nmt_model;
    nemo_speech::nmt::TranslatorConfig nmt;
#endif
};

std::string
required_value(int& i, int argc, char** argv, const std::string& option) {
    if (++i >= argc)
        throw std::invalid_argument(option + " requires a value");
    return argv[i];
}

OutputFormat
parse_format(const std::string& value) {
    if (value == "text" || value == "txt")
        return OutputFormat::Text;
    if (value == "json")
        return OutputFormat::Json;
    if (value == "srt")
        return OutputFormat::Srt;
    if (value == "vtt")
        return OutputFormat::Vtt;
    throw std::invalid_argument("--format must be text, json, srt, or vtt");
}

Options
parse_options(int argc, char** argv) {
    Options o;
    if (cli_json())
        o.format = OutputFormat::Json;
    o.engine.backend.gpu = default_gpu_index();
#if defined(NEMO_SPEECH_CLI_NMT)
    o.nmt.backend.gpu = default_gpu_index();
#endif
    nemo_speech::common::ParameterParser engine_parser;
    engine_parser.Register("asr", o.engine);
#if defined(NEMO_SPEECH_CLI_NMT)
    engine_parser.Register("nmt", o.nmt);
#endif
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--config")
            o.config_file = required_value(i, argc, argv, "--config");
#if defined(NEMO_SPEECH_CLI_LIVE)
        // Live default; YAML and later flags still override.
        else if (std::string(argv[i]) == "--live")
            o.engine.endpointing.enable = true;
#endif
    }
    if (!o.config_file.empty())
        engine_parser.ApplyYaml(o.config_file);
    engine_parser.ApplyEnv("NEMO_SPEECH");
    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config")
            ++i;
        else if (arg == "--model" || arg == "-m")
            o.model = required_value(i, argc, argv, arg);
        else if (arg == "--language" || arg == "-l")
            o.language = required_value(i, argc, argv, arg);
        else if (arg == "--device" || arg == "--backend") {
            o.gpu = parse_device(required_value(i, argc, argv, arg), arg);
            o.device_set = true;
        } else if (arg == "--gpu") {
            o.gpu = parse_int(required_value(i, argc, argv, arg), arg, -1, 1024);
            o.device_set = true;
        } else if (arg == "--concurrency" || arg == "-c")
            o.concurrency = parse_int(
                required_value(i, argc, argv, arg), arg, 1, std::numeric_limits<int>::max());
        else if (arg == "--format" || arg == "-f")
            o.format = parse_format(required_value(i, argc, argv, arg));
        else if (arg == "--output" || arg == "-o")
            o.output = required_value(i, argc, argv, arg);
        else if (arg == "--output-dir")
            o.output_dir = required_value(i, argc, argv, arg);
        else if (arg == "--vad-model")
            o.vad_model = required_value(i, argc, argv, arg);
        else if (arg == "--diar-model") {
            o.diar_model = required_value(i, argc, argv, arg);
            o.diarize = true;
        } else if (arg == "--diarize") {
            o.diarize = true;
        } else if (arg == "--itn-model-dir")
            o.itn_model_dir = required_value(i, argc, argv, arg);
        else if (arg == "--pnc-model")
            o.pnc_model = required_value(i, argc, argv, arg);
        else if (arg == "--instruction") {
            o.instruction = required_value(i, argc, argv, arg);
            if (o.instruction.find_first_not_of(" \t\r\n") == std::string::npos)
                throw std::invalid_argument("--instruction must not be empty");
        } else if (arg == "--translate-to")
            o.translate_to = required_value(i, argc, argv, arg);
#if defined(NEMO_SPEECH_CLI_NMT)
        else if (arg == "--nmt-model")
            o.nmt_model = required_value(i, argc, argv, arg);
#endif
        else if (arg == "--recursive" || arg == "-r")
            o.recursive = true;
        else if (arg == "--force")
            o.force = true;
        else if (arg == "--word-times")
            o.word_times = true;
#if defined(NEMO_SPEECH_CLI_LIVE)
        else if (arg == "--live")
            o.live = true;
#endif
        else if (arg == "--stream")
            o.stream = true;
        else if (arg == "--no-warmup")
            o.warmup = false;
        else if (arg == "--no-batching")
            o.batching = false;
        else if (arg == "--max-alternatives")
            o.request.max_alternatives = parse_int(required_value(i, argc, argv, arg), arg, 1, 100);
        else if (arg == "--max-speaker-count")
            o.request.max_speaker_count = parse_int(required_value(i, argc, argv, arg), arg, 1, 32);
        else if (arg == "--profanity-filter")
            o.request.profanity_filter = true;
        else if (arg == "--endpointing-ms")
            o.request.stop_history_eou_ms =
                static_cast<float>(parse_double(required_value(i, argc, argv, arg), arg));
        else if (arg == "--speech-context")
            o.speech_contexts.push_back(required_value(i, argc, argv, arg));
        else if (arg == "--speech-context-boost")
            o.speech_context_boost =
                static_cast<float>(parse_double(required_value(i, argc, argv, arg), arg));
        else if (arg == "--no-punctuation")
            o.punctuation = false;
        else if (arg == "--verbatim")
            o.verbatim = true;
        else if (!arg.empty() && arg[0] == '-') {
            bool consumed = false;
            const char* next = i + 1 < argc ? argv[i + 1] : nullptr;
            if (!engine_parser.ParseCliArg(arg, next, &consumed))
                throw std::invalid_argument("unknown option: " + arg);
            if (consumed)
                ++i;
        } else if (o.input.empty())
            o.input = arg;
        else
            throw std::invalid_argument("unexpected argument: " + arg);
    }
    if (o.live && !o.input.empty())
        throw std::invalid_argument("--live does not accept an input file or directory");
    if (!o.live && o.input.empty())
        throw std::invalid_argument("an input WAV file or directory is required");
    if (!o.output.empty() && !o.output_dir.empty())
        throw std::invalid_argument("use only one of --output and --output-dir");
    if (o.live && !o.output_dir.empty())
        throw std::invalid_argument("--output-dir is not valid with --live; use --output");
    if (o.live && o.recursive)
        throw std::invalid_argument("--recursive is not valid with --live");
    if (o.live && o.concurrency > 0)
        throw std::invalid_argument("--concurrency is not valid with --live");
    if (!o.translate_to.empty() && (o.format == OutputFormat::Srt || o.format == OutputFormat::Vtt))
        throw std::invalid_argument("--translate-to currently supports text and json output");
#if defined(NEMO_SPEECH_CLI_NMT)
    if (!o.nmt_model.empty() && o.translate_to.empty())
        throw std::invalid_argument("--nmt-model requires --translate-to");
#endif
    return o;
}

// JSON carries word timings when the recognizer can produce them (otherwise an empty `words`
// array, like HTTP `json`); --word-times, subtitles and diarization require them.
asr::AsrRequestOptions
make_request_options(const Options& options, bool word_timestamps_supported) {
    asr::AsrRequestOptions request = options.request;
    request.language_code = options.language;
    bool need_word_times = options.word_times ||
                           (options.format == OutputFormat::Json && word_timestamps_supported) ||
                           options.format == OutputFormat::Srt ||
                           options.format == OutputFormat::Vtt || options.diarize;
#if defined(NEMO_SPEECH_CLI_LIVE)
    need_word_times = need_word_times || options.live;
#endif
    request.enable_word_time_offsets = need_word_times;
    request.enable_automatic_punctuation = options.punctuation;
    request.verbatim_transcripts = options.verbatim;
    request.enable_speaker_diarization = options.diarize;
    request.speechlm.instructions = options.instruction;
    if (options.speech_llm_translates) {  // the translation alone: no words to time
        request.speechlm.task = "translate";
        request.speechlm.target_language = options.translate_to;
        request.enable_word_time_offsets = false;
    }
    if (!options.speech_contexts.empty())
        request.speech_contexts.push_back({options.speech_contexts, options.speech_context_boost});
    return request;
}

// Punctuation the model emitted after an endpoint closes the previous final.
void
append_late_punctuation(Transcript& transcript, const std::string& punctuation) {
    if (punctuation.empty() || transcript.text.empty())
        return;
    transcript.text += punctuation;
    if (!transcript.words.empty())
        transcript.words.back().text += punctuation;
}

void
append_result(Transcript& transcript, const asr::Result& result) {
    append_late_punctuation(transcript, result.late_punctuation);
    if (result.speechlm && result.speechlm->unaligned_words > 0)
        transcript.words_cover_text = false;
    if (result.alternatives.empty())
        return;
    const auto& alternative = result.alternatives.front();
    if (!alternative.transcript.empty()) {
        if (!transcript.text.empty() &&
            !nemo_speech::subtitle::attaches_to_previous(alternative.transcript))
            transcript.text += ' ';
        transcript.text += alternative.transcript;
    }
    transcript.confidence = alternative.confidence;
    transcript.audio_seconds = std::max(transcript.audio_seconds, result.audio_processed);
    for (const auto& language : alternative.language_codes)
        if (std::find(transcript.languages.begin(), transcript.languages.end(), language) ==
            transcript.languages.end())
            transcript.languages.push_back(language);
    for (const auto& word : alternative.words) {
        int speaker = word.speaker_tag;
        if (!transcript.words.empty() &&
            (word.joins_previous || nemo_speech::subtitle::attaches_to_previous(word.word)))
            speaker = transcript.words.back().speaker;
        transcript.words.push_back(
            {word.word, word.start_time, word.end_time, word.confidence, speaker, word.display,
             word.joins_previous});
    }
}

Transcript
transcribe_one(asr::Recognizer& recognizer, const Options& options, const fs::path& path) {
    const auto audio = nemo_speech::audio::load_wav_file(path.string());
    const asr::AsrRequestOptions request =
        make_request_options(options, recognizer.supports_word_timestamps());

    Transcript transcript;
    if (!options.stream) {
        append_result(
            transcript, recognizer.recognize(
                            audio.samples.data(), audio.samples.size(), request, options.language,
                            audio.sample_rate));
    } else {
        auto stream = recognizer.streaming_recognize(request, options.language);
        const size_t chunk = std::max<size_t>(1, audio.sample_rate * 160 / 1000);
        for (size_t offset = 0; offset < audio.samples.size(); offset += chunk) {
            const size_t count = std::min(chunk, audio.samples.size() - offset);
            stream->push(audio.samples.data() + offset, count, audio.sample_rate);
            while (auto result = stream->next()) {
                if (result->is_final) {
                    append_result(transcript, *result);
                } else {
                    append_late_punctuation(transcript, result->late_punctuation);
                    break;
                }
            }
        }
        append_result(transcript, stream->finish());
    }
    if (transcript.text.empty() && transcript.words.empty())
        transcript.audio_seconds = audio.samples.size() / static_cast<float>(audio.sample_rate);
    return transcript;
}

#if defined(NEMO_SPEECH_CLI_LIVE)
volatile std::sig_atomic_t live_running = 1;

void
stop_live_capture(int /*signal*/) {
    live_running = 0;
}

class SignalHandlerGuard {
   public:
    SignalHandlerGuard() : previous_(std::signal(SIGINT, stop_live_capture)) {}
    ~SignalHandlerGuard() { std::signal(SIGINT, previous_); }

    SignalHandlerGuard(const SignalHandlerGuard&) = delete;
    SignalHandlerGuard& operator=(const SignalHandlerGuard&) = delete;

   private:
    using Handler = void (*)(int);
    Handler previous_;
};

Transcript
transcribe_live(asr::Recognizer& recognizer, const Options& options) {
    auto stream = recognizer.streaming_recognize(
        make_request_options(options, recognizer.supports_word_timestamps()), options.language);
    // Only the diarized presenter reads interim words and speaker tags.
    stream->set_interim_words(options.diarize);
    nemo_speech::cli::MicrophoneCapture microphone;
    microphone.start();
    nemo_speech::cli::LiveTerminal terminal(!cli_quiet() && !cli_json(), options.diarize);
    terminal.start(microphone.device_name(), microphone.sample_rate());

    live_running = 1;
    SignalHandlerGuard signal_guard;
    Transcript transcript;
    std::string last_interim;
    size_t captured_samples = 0;
    nemo_speech::cli::LiveTranscriptPresenter presenter(options.diarize);
    nemo_speech::cli::LiveDiarizationBuffer diar_finals;
    asr::Result diar_interim;

    auto emit_turns = [&](const std::vector<nemo_speech::subtitle::SpeakerTurn>& turns,
                          double fallback_seconds) {
        for (const auto& turn : turns) terminal.final_turn(turn, fallback_seconds);
    };

    auto commit_final = [&](asr::Result& result) {
        if (options.diarize)
            stream->refresh_speaker_tags(result);
        append_result(transcript, result);
        last_interim.clear();
        // Even an empty non-diarized endpoint settles the interim line. For
        // diarization, render the complete preview once after all queued
        // updates; an intermediate current-turn redraw can flash an old label.
        if (!options.diarize)
            terminal.clear_interim();
        emit_turns(presenter.present_final(result), result.audio_processed);
    };

    auto update_diar_finals = [&](bool finished = false) {
        const double stable_time =
            finished ? std::numeric_limits<double>::infinity() : stream->stable_speaker_time();
        for (auto& result : diar_finals.update(
                 stable_time, [&](asr::Result& result) { stream->refresh_speaker_tags(result); }))
            commit_final(result);
        stream->refresh_speaker_tags(diar_interim);
    };

    auto show_diar_preview = [&](double audio_processed) {
        const auto* partial =
            diar_interim.alternatives.empty() ? nullptr : &diar_interim.alternatives.front();
        auto presentation = diar_finals.preview(presenter, partial, stream->stable_speaker_time());
        if (partial)
            terminal.partial_turns(presentation.turns, audio_processed);
        else
            terminal.current_turns(presentation.turns, audio_processed);
    };

    auto emit_final = [&](asr::Result result) {
        if (options.diarize) {
            const double audio_processed = result.audio_processed;
            diar_interim = {};
            diar_finals.push(std::move(result));
            update_diar_finals();
            // Words are final, labels are not. Keep the entire revisable suffix
            // visible after silence without freezing its speaker assignments.
            show_diar_preview(audio_processed);
            return;
        }
        commit_final(result);
    };

    // Attach late punctuation once: to the buffered diarized final if any, else
    // to the transcript and the presenter's open turn. Printed lines stay as is.
    auto attach_late_punctuation = [&](asr::Result& result) {
        auto punctuation = std::exchange(result.late_punctuation, {});
        if (punctuation.empty())
            return;
        if (options.diarize && diar_finals.attach_late_punctuation(punctuation))
            return;
        append_late_punctuation(transcript, punctuation);
        if (!presenter.attach_late_punctuation(punctuation))
            terminal.amend_last_final(punctuation);
    };

    auto drain_results = [&] {
        while (true) {
            auto result = stream->next();
            if (result)
                attach_late_punctuation(*result);
            if (!result) {
                if (options.diarize) {
                    update_diar_finals();
                    show_diar_preview(
                        captured_samples / static_cast<double>(microphone.sample_rate()));
                }
                break;
            }
            if (result->is_final) {
                emit_final(std::move(*result));
                continue;
            }
            if (!result->alternatives.empty()) {
                const auto& alternative = result->alternatives.front();
                if (options.diarize) {
                    diar_interim = *result;
                    update_diar_finals();
                    // Redraw label-only revisions too. These are previews, not
                    // irreversible finals; a later result owns the full text.
                    show_diar_preview(result->audio_processed);
                } else {
                    auto presentation = presenter.present_partial(alternative);
                    if (!presentation.text.empty() && presentation.text != last_interim) {
                        terminal.partial(presentation.text);
                        last_interim = std::move(presentation.text);
                    }
                }
            }
            // By contract, an interim is the last result until more audio is pushed.
            break;
        }
    };

    while (live_running) {
        auto samples = microphone.drain();
        if (samples.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        captured_samples += samples.size();
        stream->push(samples.data(), samples.size(), microphone.sample_rate());
        drain_results();
    }

    microphone.stop();
    auto tail = microphone.drain();
    if (!tail.empty()) {
        captured_samples += tail.size();
        stream->push(tail.data(), tail.size(), microphone.sample_rate());
        drain_results();
    }
    auto final_result = stream->finish();
    attach_late_punctuation(final_result);
    const double final_audio_processed = final_result.audio_processed;
    // Always render the canonical final, including corrections made by flush.
    if (options.diarize) {
        diar_finals.push(std::move(final_result));
        update_diar_finals(/*finished=*/true);
    } else {
        commit_final(final_result);
    }
    last_interim.clear();
    terminal.clear_interim();
    emit_turns(presenter.finish(), final_audio_processed);
    transcript.audio_seconds = std::max(
        transcript.audio_seconds, captured_samples / static_cast<float>(microphone.sample_rate()));
    terminal.stopped(transcript.audio_seconds);
    return transcript;
}
#endif

float
json_number(float value) {
    return std::isfinite(value) ? value : 0.0f;
}

std::string
render(const Transcript& t, OutputFormat format, const fs::path& source) {
    std::ostringstream output;
    if (format == OutputFormat::Text) {
        // A diarized transcript reads as speaker turns; a translation has no speaker words, and
        // turns built from partially aligned words would drop the rest of the transcript.
        const bool translated = !t.source_text.empty() || !t.target_language.empty();
        const bool diarized = std::any_of(
            t.words.begin(), t.words.end(), [](const auto& word) { return word.speaker > 0; });
        if (diarized && !translated && t.words_cover_text)
            output << nemo_speech::subtitle::render_speaker_turns(
                nemo_speech::subtitle::make_speaker_turns(t.words));
        else
            output << t.text << '\n';
    } else if (format == OutputFormat::Json) {
        output << "{\n  \"file\": \"" << json_escape(source.string()) << "\",\n"
               << "  \"text\": \"" << json_escape(t.text) << "\",\n"
               << "  \"confidence\": " << json_number(t.confidence) << ",\n"
               << "  \"duration\": " << json_number(t.audio_seconds) << ",\n  \"languages\": [";
        for (size_t i = 0; i < t.languages.size(); ++i)
            output << (i ? ", " : "") << '"' << json_escape(t.languages[i]) << '"';
        output << "],";
        if (!t.source_text.empty())  // NMT: the transcript it translated
            output << "\n  \"transcript\": \"" << json_escape(t.source_text) << "\",";
        if (!t.target_language.empty())
            output << "\n  \"target_language\": \"" << json_escape(t.target_language) << "\",";
        output << "\n  \"words\": [";
        for (size_t i = 0; i < t.words.size(); ++i) {
            const auto& w = t.words[i];
            output << (i ? "," : "") << "\n    {\"word\": \"" << json_escape(w.text)
                   << "\", \"start\": " << w.start_ms / 1000.0 << ", \"end\": " << w.end_ms / 1000.0
                   << ", \"confidence\": " << json_number(w.confidence);
            if (w.speaker > 0)
                output << ", \"speaker\": " << w.speaker;
            output << '}';
        }
        output << (t.words.empty() ? "" : "\n  ") << "]\n}\n";
    } else {
        output << nemo_speech::subtitle::render_cues(
            nemo_speech::subtitle::make_cues(
                t.words, t.text, static_cast<int>(t.audio_seconds * 1000)),
            format == OutputFormat::Vtt);
    }
    return output.str();
}

std::string
extension(OutputFormat format) {
    switch (format) {
        case OutputFormat::Text:
            return ".txt";
        case OutputFormat::Json:
            return ".json";
        case OutputFormat::Srt:
            return ".srt";
        case OutputFormat::Vtt:
            return ".vtt";
    }
    return ".txt";
}

}  // namespace

void
print_transcribe_help(const char* program) {
    std::printf(
        "Usage: %s transcribe INPUT [--model MODEL] [options]\n"
#if defined(NEMO_SPEECH_CLI_LIVE)
        "       %s transcribe --live [--model MODEL] [options]\n\n"
        "Transcribe a WAV file, directory, or the default microphone. Directory\n"
#else
        "\n"
        "Transcribe a WAV file or directory. Directory\n"
#endif
        "work shares one recognizer; --concurrency feeds multiple utterances to\n"
        "that recognizer so compatible inference is batched on the GPU.\n\n"
        "Options:\n"
        "  -m, --model MODEL         ASR GGUF path or indexed HF repo (speech-LLM: the\n"
        "                            decoder GGUF; its encoder file next to it is loaded)\n"
        "                            (default: nvidia/nemotron-3.5-asr-streaming-0.6b)\n"
        "  --salm-encoder FILE       Speech-LLM encoder GGUF instead of the one the\n"
        "                            decoder names\n"
        "  --asr.salm.* VALUE        SALM scheduling settings (config/env also supported)\n"
#if defined(NEMO_SPEECH_CLI_LIVE)
        "  --live                    Transcribe the default microphone until Ctrl-C;\n"
        "                            endpointing is enabled by default\n"
#endif
        "  -l, --language CODE       Language code (SALM: language hint)\n"
        "  --instruction TEXT        Speech-LLM: instruction sent with the audio instead of\n"
        "                            the default prompt, used as written\n"
        "  --device, --backend DEVICE\n"
        "                            auto, cpu, cuda[:N], metal, or vulkan[:N]\n"
        "  -c, --concurrency N       Concurrent utterances; one shared model\n"
        "  -f, --format FORMAT       text, json, srt, or vtt (default: text)\n"
        "  -o, --output PATH         Output path for one input\n"
        "  --output-dir DIR          Preserve directory layout under DIR\n"
        "  -r, --recursive           Recurse into input directories\n"
        "  --word-times              Request word timings; subtitles imply it, and JSON\n"
        "                            includes them when the model provides them\n"
        "  --vad-model PATH          Optional Silero VAD GGUF\n"
        "  --vad-masking             Mask silence features (requires --vad-model)\n"
        "  --diarize                 Label speakers: per word in JSON, per turn in text,\n"
        "                            srt and vtt (speech-LLM: needs the CTC timestamp\n"
        "                            head, part of the encoder file)\n"
        "  --diar-model MODEL        --diarize with a selected Sortformer diarizer\n"
        "  --itn-model-dir DIR       Inverse text normalization grammars\n"
        "  --pnc-model PATH          Punctuation/capitalization GGUF\n"
        "  --translate-to CODE       Translate: a speech-LLM translates the speech itself\n"
        "                            (text only); other models translate the transcript\n"
        "                            with an NMT model\n"
#if defined(NEMO_SPEECH_CLI_NMT)
        "  --nmt-model MODEL         Translation GGUF path or installed id (with a\n"
        "                            speech-LLM: translate its transcript with NMT instead)\n"
#endif
        "  --no-punctuation          Disable automatic punctuation (SALM keeps its own)\n"
        "  --verbatim                Disable ordinary ITN\n"
        "  --stream                  Stream chunks from a recorded WAV input (not SALM)\n"
        "  --endpointing[=BOOL]      Finalize streaming utterances on silence\n"
#if defined(NEMO_SPEECH_CLI_LIVE)
        "                            (default: enabled with --live)\n"
#else
        "                            (default: disabled)\n"
#endif
        "  --stop-history-eou-ms N   Endpoint silence threshold (default 800)\n"
        "  --max-alternatives N      Request N-best; current decoders return one\n"
        "  --speech-context PHRASE   Add a decoder boost phrase (repeatable)\n"
        "  --speech-context-boost N  Boost applied to speech context phrases\n"
        "  --profanity-filter        Mask words from the configured list\n"
        "  --config FILE             Load the complete ASR YAML config tree\n"
        "  --asr.SECTION.KEY VALUE   Override any C++ ASR engine setting\n"
#if defined(NEMO_SPEECH_CLI_NMT)
        "  --nmt.SECTION.KEY VALUE   Override any C++ NMT engine setting\n"
#endif
        "  --no-warmup               Skip model warmup\n"
        "  --no-batching             Disable dynamic batching (SALM always batches)\n"
        "  --force                   Replace existing output files\n",
        program
#if defined(NEMO_SPEECH_CLI_LIVE)
        ,
        program
#endif
    );
}

int
command_transcribe(int argc, char** argv) {
    try {
        if (argc > 0 && is_help_argument(argv[0])) {
            print_transcribe_help("nemo-speech");
            return 0;
        }
        Options options = parse_options(argc, argv);
        const bool encoder_given = !options.engine.salm.encoder_path.empty();  // --salm-encoder
        std::vector<fs::path> inputs;
        bool directory = false;
        if (!options.live) {
            inputs = collect_wav_inputs(options.input, options.recursive);
            directory = fs::is_directory(options.input);
        }
        if (directory && !options.output.empty())
            throw std::invalid_argument("--output is only valid for one input; use --output-dir");
        if (!directory && !options.output_dir.empty())
            throw std::invalid_argument("--output-dir is only valid for a directory input");
        // Preflight destinations before loading models or starting workers.
        // WAV and WAVE (including case variants) may share the same stem.
        fs::path output_dir = options.output_dir;
        if (directory && output_dir.empty())
            output_dir = fs::current_path() / "transcripts";
        std::vector<fs::path> destinations(inputs.size());
        if (directory) {
            std::map<fs::path, fs::path> owners;
            for (size_t i = 0; i < inputs.size(); ++i) {
                fs::path relative = relative_output_path(options.input, inputs[i]);
                relative.replace_extension(extension(options.format));
                destinations[i] = output_dir / relative;
                const auto key = fs::weakly_canonical(destinations[i]);
                const auto [owner, inserted] = owners.emplace(key, inputs[i]);
                if (!inserted)
                    throw std::invalid_argument(
                        "output collision: " + owner->second.string() + " and " +
                        inputs[i].string() + " both map to " + key.string());
            }
        }
        const int configured_gpu = options.device_set ? options.gpu : options.engine.backend.gpu;
        const int concurrency =
            options.live ? 1
                         : std::min<int>(
                               options.concurrency > 0 ? options.concurrency
                                                       : (directory && configured_gpu >= 0 ? 4 : 1),
                               inputs.size());

        asr::RecognizerConfig config = options.engine;
        config.log_status = !cli_quiet() && !cli_json();
        if (options.device_set)
            config.backend.gpu = options.gpu;
        // SALM without an LLM path reaches the backend's own error instead of resolving (and
        // possibly downloading) the default ASR model.
        const std::string model_reference =
            options.model.empty() ? config.model.path : options.model;
        if (!encoder_given || !model_reference.empty())
            config.model.path = resolve_model_file(model_reference, "asr", "ASR model").string();
        // A speech-LLM decoder names its encoder file (GGUF headers only), so SALM is known here,
        // before any model load: it recognizes complete audio only and diarizes with its own
        // speaker branch.
        asr::resolve_salm_encoder(config);
        const bool speech_llm = !config.salm.encoder_path.empty();
        if (!options.instruction.empty() && !speech_llm)
            throw std::invalid_argument("--instruction needs a speech-LLM (SALM) model");
        // --translate-to: a speech-LLM translates the speech itself unless an NMT model is named.
        if (!options.translate_to.empty()) {
#if defined(NEMO_SPEECH_CLI_NMT)
            options.speech_llm_translates =
                speech_llm && options.nmt_model.empty() && options.nmt.model.path.empty();
#else
            if (!speech_llm)
                throw std::invalid_argument(
                    "--translate-to needs a speech-LLM model, or a build with NMT");
            options.speech_llm_translates = true;
#endif
        }
        if (options.speech_llm_translates && (options.word_times || options.diarize))
            throw std::invalid_argument(
                "--translate-to with a speech-LLM returns the translation alone; drop --word-times "
                "and --diarize");
        if (options.speech_llm_translates && !options.instruction.empty())
            throw std::invalid_argument(
                "use --instruction or --translate-to: an instruction replaces the translation "
                "prompt");
        if (speech_llm && (options.stream || options.live))
            throw std::invalid_argument(
                std::string("SALM supports complete audio only; ") +
                (options.live ? "--live" : "--stream") + " streaming recognition is unavailable");
        if (!options.vad_model.empty() || !config.vad.model_path.empty())
            config.vad.model_path =
                require_model_file(
                    options.vad_model.empty() ? config.vad.model_path : options.vad_model,
                    "VAD model")
                    .string();
        if ((options.diarize && !speech_llm) || !options.diar_model.empty() ||
            !config.diar.model_path.empty())
            config.diar.model_path =
                resolve_model_file(
                    options.diar_model.empty() ? config.diar.model_path : options.diar_model,
                    "diarization", "diarization model")
                    .string();
        if (!options.itn_model_dir.empty() || !config.postproc.itn_model_dir.empty())
            config.postproc.itn_model_dir =
                require_model_directory(
                    options.itn_model_dir.empty() ? config.postproc.itn_model_dir
                                                  : options.itn_model_dir,
                    "ITN model")
                    .string();
        if (!options.pnc_model.empty() || !config.postproc.pnc_model_path.empty())
            config.postproc.pnc_model_path =
                require_model_file(
                    options.pnc_model.empty() ? config.postproc.pnc_model_path : options.pnc_model,
                    "punctuation and capitalization model")
                    .string();
        if (speech_llm && !options.batching && !cli_quiet())
            std::fprintf(
                stderr,
                "note: SALM batches concurrent requests continuously; --no-batching has "
                "no effect\n");
        config.batching.enabled = options.batching && concurrency > 1;
        config.batching.max_batch_size = std::max(config.batching.max_batch_size, concurrency);
        config.batching.max_queue_depth =
            std::max(config.batching.max_queue_depth, concurrency * 4);
        config.batching.state_arena_slots =
            std::max(config.batching.state_arena_slots, concurrency);
        if (cli_verbose()) {
            if (options.live)
                std::fprintf(
                    stderr, "transcribe: model=%s input=microphone device=%d\n",
                    config.model.path.c_str(), config.backend.gpu);
            else
                std::fprintf(
                    stderr, "transcribe: model=%s inputs=%zu concurrency=%d device=%d\n",
                    config.model.path.c_str(), inputs.size(), concurrency, config.backend.gpu);
        }
        nemo_speech::EngineRegistryConfig registry_config;
        registry_config.asr = true;
#if defined(NEMO_SPEECH_CLI_NMT)
        registry_config.nmt = !options.translate_to.empty() && !options.speech_llm_translates;
#endif
        nemo_speech::EngineRegistry engines(registry_config);
        auto recognizer = engines.load_asr(std::move(config));
#if defined(NEMO_SPEECH_CLI_NMT)
        std::shared_ptr<nemo_speech::speech::SpeechTranslator> speech_translator;
        if (registry_config.nmt) {
            options.nmt.model.path =
                require_model_file(
                    options.nmt_model.empty() ? options.nmt.model.path : options.nmt_model,
                    "translation model")
                    .string();
            if (options.device_set)
                options.nmt.backend.gpu = options.gpu;
            options.nmt.verbose = cli_verbose();
            options.nmt.pool.contexts = std::max(options.nmt.pool.contexts, concurrency);
            engines.load_nmt(std::move(options.nmt));
            speech_translator = engines.speech_translation();
        }
#endif
        if (options.warmup)
            engines.warmup();

#if defined(NEMO_SPEECH_CLI_LIVE)
        if (options.live) {
            Transcript transcript = transcribe_live(*recognizer, options);
#if defined(NEMO_SPEECH_CLI_NMT)
            if (speech_translator && !transcript.text.empty()) {
                const auto translated = speech_translator->translate_text(
                    transcript.text, options.language, options.translate_to, transcript.languages);
                transcript.source_text = translated.transcript;
                transcript.text = translated.text;
                transcript.target_language = translated.language_code;
            }
#endif
            const std::string contents =
                render(transcript, options.format, fs::path("<microphone>"));
            if (options.output.empty())
                std::fwrite(contents.data(), 1, contents.size(), stdout);
            else {
                write_text_file(options.output, contents, options.force);
                if (!cli_quiet())
                    std::fprintf(stderr, "microphone -> %s\n", options.output.string().c_str());
            }
            return 0;
        }
#endif

        // Publish each completed transcript atomically so finished files remain
        // available if another input fails or the process is interrupted.
        std::vector<Transcript> transcripts(inputs.size());
        std::vector<std::string> errors(inputs.size());
        std::vector<CliErrorKind> error_kinds(inputs.size(), CliErrorKind{"runtime_error", 1});
        std::atomic<size_t> next{0};
        size_t completed = 0;  // Guarded by progress_mutex.
        std::mutex progress_mutex;
        std::vector<std::thread> workers;
        for (int thread = 0; thread < concurrency; ++thread) {
            workers.emplace_back([&] {
                for (;;) {
                    const size_t index = next.fetch_add(1);
                    if (index >= inputs.size())
                        break;
                    try {
                        transcripts[index] = transcribe_one(*recognizer, options, inputs[index]);
                        if (options.speech_llm_translates)
                            transcripts[index].target_language = options.translate_to;
#if defined(NEMO_SPEECH_CLI_NMT)
                        if (speech_translator && !transcripts[index].text.empty()) {
                            const auto translated = speech_translator->translate_text(
                                transcripts[index].text, options.language, options.translate_to,
                                transcripts[index].languages);
                            transcripts[index].source_text = translated.transcript;
                            transcripts[index].text = translated.text;
                            transcripts[index].target_language = translated.language_code;
                        }
#endif
                        if (directory) {
                            const std::string contents =
                                render(transcripts[index], options.format, inputs[index]);
                            write_text_file(destinations[index], contents, options.force);
                            // Do not retain every transcript for the lifetime
                            // of a large directory job after it is on disk.
                            transcripts[index] = {};
                        }
                    }
                    catch (const std::exception& error) {
                        errors[index] = error.what();
                        error_kinds[index] = cli_error_kind(error);
                    }
                    if (directory && !cli_quiet()) {
                        const std::lock_guard<std::mutex> lock(progress_mutex);
                        const size_t done = ++completed;
                        if (errors[index].empty())
                            std::fprintf(
                                stderr, "[%zu/%zu] %s -> %s\n", done, inputs.size(),
                                inputs[index].string().c_str(),
                                destinations[index].string().c_str());
                        else
                            std::fprintf(
                                stderr, "[%zu/%zu] %s failed: %s\n", done, inputs.size(),
                                inputs[index].string().c_str(), errors[index].c_str());
                    }
                }
            });
        }
        for (auto& worker : workers) worker.join();

        // Failures of one kind exit with that kind's code (2 when every input was invalid);
        // mixed failures exit 1.
        int failures = 0;
        int exit_code = kCliExitSuccess;
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!errors[i].empty()) {
                const auto& kind = error_kinds[i];
                print_cli_error(
                    "transcribe", inputs[i].string() + ": " + errors[i], kind.exit_code, kind.type);
                exit_code =
                    failures == 0 || exit_code == kind.exit_code ? kind.exit_code : kCliExitRuntime;
                ++failures;
                continue;
            }
            if (directory)
                continue;  // published atomically by the worker
            const std::string contents = render(transcripts[i], options.format, inputs[i]);
            if (options.output.empty()) {
                std::fwrite(contents.data(), 1, contents.size(), stdout);
                continue;
            }
            try {
                write_text_file(options.output, contents, options.force);
                if (!cli_quiet())
                    std::fprintf(
                        stderr, "%s -> %s\n", inputs[i].string().c_str(),
                        options.output.string().c_str());
            }
            catch (const std::exception& error) {
                const auto kind = cli_error_kind(error);
                print_cli_error("transcribe", error.what(), kind.exit_code, kind.type);
                exit_code =
                    failures == 0 || exit_code == kind.exit_code ? kind.exit_code : kCliExitRuntime;
                ++failures;
            }
        }
        if (directory && !cli_quiet())
            std::fprintf(
                stderr, "transcribed %zu file%s (%d failed)\n", inputs.size(),
                inputs.size() == 1 ? "" : "s", failures);
        return exit_code;
    }
    catch (const std::invalid_argument& error) {
        return print_cli_error(
            "transcribe", std::string(error.what()) + " (run 'nemo-speech help transcribe')",
            kCliExitInvalidArgument, "invalid_argument");
    }
    catch (const std::exception& error) {
        return print_cli_exception("transcribe", error);
    }
}
