// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#if defined(NEMO_SPEECH_CLI_LIVE)
#include <chrono>
#include <csignal>
#endif
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "audio_file.h"
#include "cli_util.h"
#include "commands.h"
#include "engine_registry.h"
#if defined(NEMO_SPEECH_CLI_LIVE)
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
#if defined(NEMO_SPEECH_CLI_NMT)
    std::string nmt_model;
    std::string translate_to;
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
            o.concurrency = parse_int(required_value(i, argc, argv, arg), arg, 1, 1024);
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
#if defined(NEMO_SPEECH_CLI_NMT)
        else if (arg == "--nmt-model")
            o.nmt_model = required_value(i, argc, argv, arg);
        else if (arg == "--translate-to")
            o.translate_to = required_value(i, argc, argv, arg);
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
#if defined(NEMO_SPEECH_CLI_NMT)
    if (!o.translate_to.empty() && (o.format == OutputFormat::Srt || o.format == OutputFormat::Vtt))
        throw std::invalid_argument("--translate-to currently supports text and json output");
    if (!o.nmt_model.empty() && o.translate_to.empty())
        throw std::invalid_argument("--nmt-model requires --translate-to");
#endif
    return o;
}

asr::AsrRequestOptions
make_request_options(const Options& options) {
    asr::AsrRequestOptions request = options.request;
    request.language_code = options.language;
    request.enable_word_time_offsets = options.word_times || options.format == OutputFormat::Json ||
                                       options.format == OutputFormat::Srt ||
                                       options.format == OutputFormat::Vtt || options.diarize;
    request.enable_automatic_punctuation = options.punctuation;
    request.verbatim_transcripts = options.verbatim;
    request.enable_speaker_diarization = options.diarize;
    if (!options.speech_contexts.empty())
        request.speech_contexts.push_back({options.speech_contexts, options.speech_context_boost});
    return request;
}

void
append_result(Transcript& transcript, const asr::Result& result) {
    if (result.alternatives.empty())
        return;
    const auto& alternative = result.alternatives.front();
    if (!alternative.transcript.empty()) {
        if (!transcript.text.empty() && alternative.transcript.front() != '.' &&
            alternative.transcript.front() != ',' && alternative.transcript.front() != '!' &&
            alternative.transcript.front() != '?')
            transcript.text += ' ';
        transcript.text += alternative.transcript;
    }
    transcript.confidence = alternative.confidence;
    transcript.audio_seconds = std::max(transcript.audio_seconds, result.audio_processed);
    for (const auto& language : alternative.language_codes)
        if (std::find(transcript.languages.begin(), transcript.languages.end(), language) ==
            transcript.languages.end())
            transcript.languages.push_back(language);
    for (const auto& word : alternative.words)
        transcript.words.push_back(
            {word.word, word.start_time, word.end_time, word.confidence, word.speaker_tag});
}

Transcript
transcribe_one(asr::Recognizer& recognizer, const Options& options, const fs::path& path) {
    const auto audio = nemo_speech::audio::load_wav_file(path.string());
    const asr::AsrRequestOptions request = make_request_options(options);

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
                if (result->is_final)
                    append_result(transcript, *result);
                else
                    break;
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
    auto stream = recognizer.streaming_recognize(make_request_options(options), options.language);
    nemo_speech::cli::MicrophoneCapture microphone;
    microphone.start();

    if (!cli_quiet() && !cli_json())
        std::fprintf(
            stderr, "[live] listening on \"%s\" at %d Hz; press Ctrl-C to stop\n",
            microphone.device_name().c_str(), microphone.sample_rate());

    live_running = 1;
    SignalHandlerGuard signal_guard;
    Transcript transcript;
    std::string last_interim;
    size_t captured_samples = 0;

    auto drain_results = [&] {
        while (auto result = stream->next()) {
            if (result->is_final) {
                append_result(transcript, *result);
                last_interim.clear();
                if (!cli_quiet() && !cli_json() && !result->alternatives.empty() &&
                    !result->alternatives.front().transcript.empty())
                    std::fprintf(
                        stderr, "[live final @ %.2fs] %s\n", result->audio_processed,
                        result->alternatives.front().transcript.c_str());
                continue;
            }
            if (!result->alternatives.empty()) {
                const std::string& text = result->alternatives.front().transcript;
                if (!text.empty() && text != last_interim) {
                    if (!cli_quiet() && !cli_json())
                        std::fprintf(
                            stderr, "[live partial @ %.2fs] %s\n", result->audio_processed,
                            text.c_str());
                    last_interim = text;
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
    append_result(transcript, stream->finish());
    transcript.audio_seconds = std::max(
        transcript.audio_seconds, captured_samples / static_cast<float>(microphone.sample_rate()));
    if (!cli_quiet() && !cli_json())
        std::fprintf(
            stderr, "[live] stopped after %.2fs of captured audio\n", transcript.audio_seconds);
    return transcript;
}
#endif

std::string
timestamp(int milliseconds, bool vtt) {
    milliseconds = std::max(0, milliseconds);
    const int hours = milliseconds / 3600000;
    milliseconds %= 3600000;
    const int minutes = milliseconds / 60000;
    milliseconds %= 60000;
    const int seconds = milliseconds / 1000;
    milliseconds %= 1000;
    std::ostringstream output;
    output << std::setfill('0') << std::setw(2) << hours << ':' << std::setw(2) << minutes << ':'
           << std::setw(2) << seconds << (vtt ? '.' : ',') << std::setw(3) << milliseconds;
    return output.str();
}

std::string
render(const Transcript& t, OutputFormat format, const fs::path& source) {
    std::ostringstream output;
    if (format == OutputFormat::Text) {
        output << t.text << '\n';
    } else if (format == OutputFormat::Json) {
        output << "{\n  \"file\": \"" << json_escape(source.string()) << "\",\n"
               << "  \"text\": \"" << json_escape(t.text) << "\",\n"
               << "  \"confidence\": " << t.confidence << ",\n"
               << "  \"duration\": " << t.audio_seconds << ",\n  \"languages\": [";
        for (size_t i = 0; i < t.languages.size(); ++i)
            output << (i ? ", " : "") << '"' << json_escape(t.languages[i]) << '"';
        output << "],";
        if (!t.target_language.empty())
            output << "\n  \"transcript\": \"" << json_escape(t.source_text)
                   << "\",\n  \"target_language\": \"" << json_escape(t.target_language) << "\",";
        output << "\n  \"words\": [";
        for (size_t i = 0; i < t.words.size(); ++i) {
            const auto& w = t.words[i];
            output << (i ? "," : "") << "\n    {\"word\": \"" << json_escape(w.text)
                   << "\", \"start\": " << w.start_ms / 1000.0 << ", \"end\": " << w.end_ms / 1000.0
                   << ", \"confidence\": " << w.confidence;
            if (w.speaker > 0)
                output << ", \"speaker\": " << w.speaker;
            output << '}';
        }
        output << (t.words.empty() ? "" : "\n  ") << "]\n}\n";
    } else {
        const bool vtt = format == OutputFormat::Vtt;
        if (vtt)
            output << "WEBVTT\n\n";
        const auto cues = nemo_speech::subtitle::make_cues(
            t.words, t.text, static_cast<int>(t.audio_seconds * 1000));
        for (size_t i = 0; i < cues.size(); ++i) {
            if (!vtt)
                output << i + 1 << '\n';
            output << timestamp(cues[i].start_ms, vtt) << " --> "
                   << timestamp(std::max(cues[i].start_ms + 1, cues[i].end_ms), vtt) << '\n'
                   << cues[i].text << "\n\n";
        }
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
        "  -m, --model MODEL         ASR GGUF path or indexed HF repo\n"
        "                            (default: nvidia/nemotron-3.5-asr-streaming-0.6b)\n"
#if defined(NEMO_SPEECH_CLI_LIVE)
        "  --live                    Transcribe the default microphone until Ctrl-C\n"
#endif
        "  -l, --language CODE       Language code or prompt\n"
        "  --device, --backend DEVICE\n"
        "                            auto, cpu, cuda[:N], metal, or vulkan[:N]\n"
        "  -c, --concurrency N       Concurrent utterances; one shared model\n"
        "  -f, --format FORMAT       text, json, srt, or vtt (default: text)\n"
        "  -o, --output PATH         Output path for one input\n"
        "  --output-dir DIR          Preserve directory layout under DIR\n"
        "  -r, --recursive           Recurse into input directories\n"
        "  --word-times              Include word timestamps in diagnostics\n"
        "  --vad-model PATH          Optional Silero VAD GGUF\n"
        "  --diar-model PATH         Tag words with a Sortformer diarizer\n"
        "  --max-speaker-count N     Diarization speaker cap (default 8)\n"
        "  --itn-model-dir DIR       Inverse text normalization grammars\n"
        "  --pnc-model PATH          Punctuation/capitalization GGUF\n"
#if defined(NEMO_SPEECH_CLI_NMT)
        "  --translate-to CODE       Translate the final transcript\n"
        "  --nmt-model MODEL         Translation GGUF path or installed id\n"
#endif
        "  --no-punctuation          Disable automatic punctuation\n"
        "  --verbatim                Disable ordinary ITN\n"
        "  --stream                  Stream chunks from a recorded WAV input\n"
        "  --max-alternatives N      Request N-best hypotheses\n"
        "  --speech-context PHRASE   Add a decoder boost phrase (repeatable)\n"
        "  --speech-context-boost N  Boost applied to speech context phrases\n"
        "  --profanity-filter        Mask words from the configured list\n"
        "  --config FILE             Load the complete ASR YAML config tree\n"
        "  --asr.SECTION.KEY VALUE   Override any C++ ASR engine setting\n"
#if defined(NEMO_SPEECH_CLI_NMT)
        "  --nmt.SECTION.KEY VALUE   Override any C++ NMT engine setting\n"
#endif
        "  --no-warmup               Skip model warmup\n"
        "  --no-batching             Disable dynamic batching\n"
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
        config.model.path =
            resolve_model_file(
                options.model.empty() ? config.model.path : options.model, "asr", "ASR model")
                .string();
        if (!options.vad_model.empty() || !config.vad.model_path.empty())
            config.vad.model_path =
                require_model_file(
                    options.vad_model.empty() ? config.vad.model_path : options.vad_model,
                    "VAD model")
                    .string();
        if (options.diarize || !options.diar_model.empty() || !config.diar.model_path.empty())
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
        registry_config.nmt = !options.translate_to.empty();
#endif
        nemo_speech::EngineRegistry engines(registry_config);
        auto recognizer = engines.load_asr(std::move(config));
#if defined(NEMO_SPEECH_CLI_NMT)
        std::shared_ptr<nemo_speech::speech::SpeechTranslator> speech_translator;
        if (!options.translate_to.empty()) {
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

        std::vector<Transcript> transcripts(inputs.size());
        std::vector<std::string> errors(inputs.size());
        std::atomic<size_t> next{0};
        std::vector<std::thread> workers;
        for (int thread = 0; thread < concurrency; ++thread) {
            workers.emplace_back([&] {
                for (;;) {
                    const size_t index = next.fetch_add(1);
                    if (index >= inputs.size())
                        break;
                    try {
                        transcripts[index] = transcribe_one(*recognizer, options, inputs[index]);
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
                    }
                    catch (const std::exception& error) {
                        errors[index] = error.what();
                    }
                }
            });
        }
        for (auto& worker : workers) worker.join();

        int failures = 0;
        fs::path output_dir = options.output_dir;
        if (directory && output_dir.empty())
            output_dir = fs::current_path() / "transcripts";
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!errors[i].empty()) {
                print_cli_error(
                    "transcribe", inputs[i].string() + ": " + errors[i], 1, "runtime_error");
                ++failures;
                continue;
            }
            const std::string contents = render(transcripts[i], options.format, inputs[i]);
            if (!directory && options.output.empty()) {
                std::fwrite(contents.data(), 1, contents.size(), stdout);
                continue;
            }
            fs::path destination;
            if (!directory) {
                destination = options.output;
            } else {
                fs::path relative = relative_output_path(options.input, inputs[i]);
                relative.replace_extension(extension(options.format));
                destination = output_dir / relative;
            }
            try {
                write_text_file(destination, contents, options.force);
                if (!cli_quiet())
                    std::fprintf(
                        stderr, "%s -> %s\n", inputs[i].string().c_str(),
                        destination.string().c_str());
            }
            catch (const std::exception& error) {
                print_cli_error("transcribe", error.what(), 1, "runtime_error");
                ++failures;
            }
        }
        if (directory && !cli_quiet())
            std::fprintf(
                stderr, "transcribed %zu file%s (%d failed)\n", inputs.size(),
                inputs.size() == 1 ? "" : "s", failures);
        return failures == 0 ? 0 : 1;
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
