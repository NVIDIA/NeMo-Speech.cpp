// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
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
#include "model_utils.h"
#include "parameter_parser.h"
#include "recognizer.h"
#if defined(NEMO_SPEECH_CLI_NMT)
#include "speech_translator.h"
#endif

namespace {
namespace fs = std::filesystem;
namespace asr = nemo_speech::asr;

struct Word {
    std::string text;
    int start_ms = 0;
    int end_ms = 0;
    float confidence = 0.0f;
    int speaker = 0;
};

struct Transcript {
    std::string text;
    std::string source_text;
    std::string target_language;
    float confidence = 0.0f;
    float audio_seconds = 0.0f;
    std::vector<std::string> languages;
    std::vector<Word> words;
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
    if (o.input.empty())
        throw std::invalid_argument("an input WAV file or directory is required");
    if (!o.output.empty() && !o.output_dir.empty())
        throw std::invalid_argument("use only one of --output and --output-dir");
#if defined(NEMO_SPEECH_CLI_NMT)
    if (!o.translate_to.empty() && (o.format == OutputFormat::Srt || o.format == OutputFormat::Vtt))
        throw std::invalid_argument("--translate-to currently supports text and json output");
    if (!o.nmt_model.empty() && o.translate_to.empty())
        throw std::invalid_argument("--nmt-model requires --translate-to");
#endif
    return o;
}

std::vector<fs::path>
collect_inputs(const Options& options) {
    std::error_code error;
    if (fs::is_regular_file(options.input, error)) {
        if (!nemo_speech::audio::is_wav_path(options.input.string()))
            throw std::invalid_argument("input must be a .wav file");
        return {fs::absolute(options.input)};
    }
    if (!fs::is_directory(options.input, error))
        throw std::invalid_argument(options.input.string() + " is not a file or directory");
    std::vector<fs::path> files;
    auto add = [&](const auto& entry) {
        if (entry.is_regular_file(error) && nemo_speech::audio::is_wav_path(entry.path().string()))
            files.push_back(fs::absolute(entry.path()));
    };
    if (options.recursive) {
        for (const auto& entry : fs::recursive_directory_iterator(options.input)) add(entry);
    } else {
        for (const auto& entry : fs::directory_iterator(options.input)) add(entry);
    }
    std::sort(files.begin(), files.end());
    if (files.empty())
        throw std::invalid_argument(options.input.string() + " contains no WAV files");
    return files;
}

Transcript
transcribe_one(asr::Recognizer& recognizer, const Options& options, const fs::path& path) {
    const auto audio = nemo_speech::audio::load_wav_file(path.string());
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

    Transcript transcript;
    auto append = [&](const asr::Result& result) {
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
    };
    if (!options.stream) {
        append(recognizer.recognize(
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
                    append(*result);
                else
                    break;
            }
        }
        append(stream->finish());
    }
    if (transcript.text.empty() && transcript.words.empty())
        transcript.audio_seconds = audio.samples.size() / static_cast<float>(audio.sample_rate);
    return transcript;
}

bool
attaches_to_previous(const std::string& word) {
    static const std::string punctuation = ".,!?;:%)]}\xE2\x80\x9D\xE2\x80\x99";
    return !word.empty() && punctuation.find(word) != std::string::npos;
}

std::string
join_words(const std::vector<Word>& words, size_t begin, size_t end) {
    std::string output;
    for (size_t i = begin; i < end; ++i) {
        if (!output.empty() && !attaches_to_previous(words[i].text))
            output += ' ';
        output += words[i].text;
    }
    return output;
}

struct Cue {
    int start_ms;
    int end_ms;
    std::string text;
};

std::vector<Cue>
make_cues(const Transcript& transcript) {
    std::vector<Cue> cues;
    if (transcript.words.empty()) {
        if (!transcript.text.empty())
            cues.push_back(
                {0, std::max(1, static_cast<int>(transcript.audio_seconds * 1000)),
                 transcript.text});
        return cues;
    }
    size_t begin = 0;
    for (size_t i = 1; i <= transcript.words.size(); ++i) {
        const bool end = i == transcript.words.size();
        const int duration =
            end ? 0 : transcript.words[i].end_ms - transcript.words[begin].start_ms;
        const int pause = end ? 0 : transcript.words[i].start_ms - transcript.words[i - 1].end_ms;
        const std::string candidate =
            end ? std::string() : join_words(transcript.words, begin, i + 1);
        if (end || duration > 5000 || pause > 800 || candidate.size() > 84) {
            cues.push_back(
                {transcript.words[begin].start_ms, transcript.words[i - 1].end_ms,
                 join_words(transcript.words, begin, i)});
            begin = i;
        }
    }
    return cues;
}

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
        const auto cues = make_cues(t);
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
        "Usage: %s transcribe INPUT [--model MODEL] [options]\n\n"
        "Transcribe one WAV file or every WAV file in a directory. Directory\n"
        "work shares one recognizer; --concurrency feeds multiple utterances to\n"
        "that recognizer so compatible inference is batched on the GPU.\n\n"
        "Options:\n"
        "  -m, --model MODEL         Local ASR GGUF path\n"
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
        "  --stream                  Exercise cache-aware streaming inference\n"
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
        program);
}

int
command_transcribe(int argc, char** argv) {
    try {
        if (argc > 0 && is_help_argument(argv[0])) {
            print_transcribe_help("nemo-speech");
            return 0;
        }
        Options options = parse_options(argc, argv);
        const auto inputs = collect_inputs(options);
        const bool directory = fs::is_directory(options.input);
        if (directory && !options.output.empty())
            throw std::invalid_argument("--output is only valid for one input; use --output-dir");
        if (!directory && !options.output_dir.empty())
            throw std::invalid_argument("--output-dir is only valid for a directory input");
        const int configured_gpu = options.device_set ? options.gpu : options.engine.backend.gpu;
        const int concurrency = std::min<int>(
            options.concurrency > 0 ? options.concurrency
                                    : (directory && configured_gpu >= 0 ? 4 : 1),
            inputs.size());

        asr::RecognizerConfig config = options.engine;
        if (options.device_set)
            config.backend.gpu = options.gpu;
        config.model.path =
            require_model_file(
                options.model.empty() ? config.model.path : options.model, "ASR model")
                .string();
        if (!options.vad_model.empty() || !config.vad.model_path.empty())
            config.vad.model_path =
                require_model_file(
                    options.vad_model.empty() ? config.vad.model_path : options.vad_model,
                    "VAD model")
                    .string();
        if (!options.diar_model.empty() || !config.diar.model_path.empty())
            config.diar.model_path =
                require_model_file(
                    options.diar_model.empty() ? config.diar.model_path : options.diar_model,
                    "diarization model")
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
        if (cli_verbose())
            std::fprintf(
                stderr, "transcribe: model=%s inputs=%zu concurrency=%d device=%d\n",
                config.model.path.c_str(), inputs.size(), concurrency, config.backend.gpu);
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
