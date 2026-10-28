// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Chat API (/v1/chat/completions): exact response bodies, every validation refusal, the error
// envelope and the request limits.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "chat_api.h"
#include "errors.h"

namespace chat = nemo_speech::http::chat;
namespace asr = nemo_speech::asr;

namespace {

int failures = 0;

#define CHECK(condition, what)                                        \
    do {                                                              \
        if (!(condition)) {                                           \
            std::cerr << "FAIL " << __LINE__ << ": " << what << '\n'; \
            ++failures;                                               \
        }                                                             \
    } while (0)

const std::string kModel = "salm";

std::string
base64(const std::string& bytes) {
    static const char* alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < bytes.size(); i += 3) {
        uint32_t v = static_cast<unsigned char>(bytes[i]) << 16;
        if (i + 1 < bytes.size())
            v |= static_cast<unsigned char>(bytes[i + 1]) << 8;
        if (i + 2 < bytes.size())
            v |= static_cast<unsigned char>(bytes[i + 2]);
        out += alphabet[(v >> 18) & 63];
        out += alphabet[(v >> 12) & 63];
        out += i + 1 < bytes.size() ? alphabet[(v >> 6) & 63] : '=';
        out += i + 2 < bytes.size() ? alphabet[v & 63] : '=';
    }
    return out;
}

// PCM16 WAV with a canonical 44-byte header.
std::string
wav(size_t samples, uint32_t rate = 16000, uint16_t channels = 1) {
    const uint32_t data = static_cast<uint32_t>(samples * 2 * channels);
    std::string out = "RIFF";
    auto u32 = [&](uint32_t v) {
        for (int i = 0; i < 4; ++i) out += static_cast<char>((v >> (8 * i)) & 0xFF);
    };
    auto u16 = [&](uint16_t v) {
        out += static_cast<char>(v & 0xFF);
        out += static_cast<char>(v >> 8);
    };
    u32(36 + data);
    out += "WAVEfmt ";
    u32(16);
    u16(1);
    u16(channels);
    u32(rate);
    u32(rate * 2 * channels);
    u16(static_cast<uint16_t>(2 * channels));
    u16(16);
    out += "data";
    u32(data);
    for (size_t i = 0; i < samples * channels; ++i) {
        const int16_t s = static_cast<int16_t>((i % 7) * 100);
        out += static_cast<char>(s & 0xFF);
        out += static_cast<char>((s >> 8) & 0xFF);
    }
    return out;
}

const std::string kWav = base64(wav(400));

// FLAC: STREAMINFO and `frames` (< 128) frames of 4096 samples (16 kHz mono, 16-bit) holding one
// constant value.
std::string
flac_constant(int16_t value, int frames = 1) {
    std::string out = "fLaC";
    const uint32_t total = 4096u * static_cast<uint32_t>(frames);
    const uint8_t info[] = {
        0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00, 0, 0, 0, 0, 0, 0,
        // 16000 Hz (20 bits), 1 channel, 16 bits, `total` samples (36 bits)
        0x03, 0xE8, 0x00, 0xF0, static_cast<uint8_t>(total >> 24),
        static_cast<uint8_t>(total >> 16), static_cast<uint8_t>(total >> 8),
        static_cast<uint8_t>(total)};
    out.append(reinterpret_cast<const char*>(info), sizeof(info));
    out.append(16, '\0');  // MD5 unknown
    for (int n = 0; n < frames; ++n) {
        // frame header: fixed blocking, 4096 samples, 16 kHz, mono, 16-bit, frame n
        std::string frame = {'\xFF', '\xF8', '\xC5', '\x08', static_cast<char>(n)};
        uint8_t crc8 = 0;
        for (unsigned char c : frame) {
            crc8 ^= c;
            for (int b = 0; b < 8; ++b)
                crc8 = static_cast<uint8_t>(crc8 & 0x80 ? (crc8 << 1) ^ 0x07 : crc8 << 1);
        }
        frame += static_cast<char>(crc8);
        frame += '\0';  // CONSTANT subframe
        frame += static_cast<char>((value >> 8) & 0xFF);
        frame += static_cast<char>(value & 0xFF);
        uint16_t crc16 = 0;
        for (unsigned char c : frame) {
            crc16 ^= static_cast<uint16_t>(c << 8);
            for (int b = 0; b < 8; ++b)
                crc16 = static_cast<uint16_t>(crc16 & 0x8000 ? (crc16 << 1) ^ 0x8005 : crc16 << 1);
        }
        frame += static_cast<char>(crc16 >> 8);
        frame += static_cast<char>(crc16 & 0xFF);
        out += frame;
    }
    return out;
}

std::string
audio_part(const std::string& data = kWav, const std::string& format = "wav") {
    return R"({"type":"input_audio","input_audio":{"data":")" + data + R"(","format":")" + format +
           R"("}})";
}

// A chat request with `extra` top-level fields and the given content parts (default: one WAV).
std::string
chat_body(const std::string& extra = "", const std::string& parts = "") {
    std::string body = R"({"model":")" + kModel + R"(","messages":[{"role":"user","content":[)" +
                       (parts.empty() ? audio_part() : parts) + "]}]";
    if (!extra.empty())
        body += "," + extra;
    return body + "}";
}

const std::string kWords = R"("timestamp_granularities":["word"])";

// An aligned two-word transcript.
asr::SpeechLmDetails
fixture() {
    asr::SpeechLmDetails d;
    d.text = "Hello world.";
    d.language = "en";
    d.duration_seconds = 2.0;
    asr::SpeechLmWord hello;
    hello.word = "Hello";
    hello.word_index = 0;
    hello.start = 0.42;
    hello.end = 0.81;
    asr::SpeechLmWord world;
    world.word = "world.";
    world.word_index = 1;
    world.start = 0.82;
    world.end = 1.31;
    d.words = {hello, world};
    d.alignment_status = "aligned";
    d.alignment_model = "stt_en_fastconformer_ctc_large";
    d.frame_stride_seconds = 0.04;
    return d;
}

struct Fake {
    asr::SpeechLmDetails details = fixture();
    bool timestamps = true;
    bool instructions = true;            // a speech-LLM
    std::function<void()> on_recognize;  // throw or block
    std::vector<std::string> pieces;     // reported through on_text, then
    std::function<void()> after_text;    // throw
    asr::AsrRequestOptions seen;
    std::mutex mutex;
};

chat::Backend
backend(Fake& fake) {
    chat::Backend b;
    b.recognize = [&fake](const float*, size_t, int, const asr::AsrRequestOptions& options) {
        {
            std::lock_guard<std::mutex> lock(fake.mutex);
            fake.seen = options;
        }
        if (fake.on_recognize)
            fake.on_recognize();
        for (const auto& piece : fake.pieces)
            if (options.speechlm.on_text)
                options.speechlm.on_text(piece);
        if (fake.after_text)
            fake.after_text();
        asr::Result result;
        result.speechlm = std::make_shared<asr::SpeechLmDetails>(fake.details);
        return result;
    };
    b.ready = [] { return true; };
    b.supports_timestamps = [&fake] { return fake.timestamps; };
    b.supports_instructions = [&fake] { return fake.instructions; };
    b.model_alias = [] { return kModel; };
    b.request_id = [] { return std::string("abc"); };
    b.unix_seconds = [] { return uint64_t{1790000000}; };
    return b;
}

struct Sent {
    chat::HttpResponse response;
    std::string header(const std::string& name) const {
        for (const auto& [key, value] : response.headers)
            if (key == name)
                return value;
        return {};
    }
};

Sent
post(
    chat::Api& api, const std::string& body, std::string content_type = "application/json",
    std::string content_length = "auto", const chat::ClientGone& gone = {}) {
    if (content_length == "auto")
        content_length = std::to_string(body.size());
    const chat::HeaderLookup headers = [&](const std::string& name, std::string& value) {
        if (name == "Content-Type" && !content_type.empty()) {
            value = content_type;
            return true;
        }
        if (name == "Content-Length" && !content_length.empty()) {
            value = content_length;
            return true;
        }
        return false;
    };
    const chat::BodyReader reader = [&](const std::function<bool(const char*, size_t)>& sink) {
        for (size_t i = 0; i < body.size(); i += 4096)
            if (!sink(body.data() + i, std::min<size_t>(4096, body.size() - i)))
                return false;
        return true;
    };
    return {api.post(headers, reader, gone)};
}

std::string
error_json(
    const std::string& message, const std::string& type, const std::string& param,
    const std::string& code) {
    return R"({"error":{"message":")" + message + R"(","type":")" + type + R"(","param":)" +
           (param.empty() ? std::string("null") : "\"" + param + "\"") + R"(,"code":")" + code +
           "\"}}";
}

void
expect_error(
    const Sent& sent, int status, const std::string& code, const std::string& message_part,
    const std::string& param, int line) {
    const auto& r = sent.response;
    const bool ok =
        r.status == status && r.body.find("\"code\":\"" + code + "\"") != std::string::npos &&
        r.body.find(message_part) != std::string::npos &&
        r.body.find("\"param\":" + (param.empty() ? std::string("null") : "\"" + param + "\"")) !=
            std::string::npos &&
        sent.header("Cache-Control") == "no-store";
    if (!ok) {
        std::cerr << "FAIL " << line << ": expected " << status << " " << code << " ["
                  << message_part << "] param=" << param << ", got " << r.status << " " << r.body
                  << '\n';
        ++failures;
    }
}

#define EXPECT_ERROR(sent, status, code, part, param) \
    expect_error(sent, status, code, part, param, __LINE__)

chat::Limits
limits() {
    return {};
}

std::string
with_text(const std::string& text) {
    return R"({"type":"text","text":")" + text + R"("},)" + audio_part();
}

void
test_rendered_fixtures() {
    Fake fake;
    chat::Api api(limits(), backend(fake));
    // Plain completion with usage (10/2, no audio count), byte-exact.
    fake.details.prompt_tokens = 10;
    fake.details.completion_tokens = 2;
    auto sent = post(api, chat_body(R"("language":"en")"));
    CHECK(sent.response.status == 200, sent.response.body);
    CHECK(
        sent.response.body ==
            R"({"id":"chatcmpl-abc","object":"chat.completion","created":1790000000,"model":"salm","choices":[{"index":0,"message":{"role":"assistant","content":"Hello world.","refusal":null},"finish_reason":"stop","logprobs":null}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12},"transcription":{"schema_version":"1","task":"transcribe","transcript_origin":"llm","language":"en","duration":2.0,"text":"Hello world.","segments":[]}})",
        sent.response.body);
    CHECK(sent.header("Content-Type") == "application/json", "content type");
    CHECK(sent.header("Cache-Control").empty(), "no cache-control on 200");
    fake.details.prompt_tokens = 194;
    fake.details.completion_tokens = 35;
    fake.details.audio_tokens = 180;
    sent = post(api, chat_body());
    CHECK(
        sent.response.body.find(
            R"("usage":{"prompt_tokens":194,"completion_tokens":35,"total_tokens":229,"prompt_tokens_details":{"audio_tokens":180}})") !=
            std::string::npos,
        sent.response.body);
    fake.details.prompt_tokens = fake.details.completion_tokens = fake.details.audio_tokens = 0;
    sent = post(api, chat_body());
    CHECK(sent.response.body.find("\"usage\"") == std::string::npos, "no usage when no decode ran");

    // Timestamps. The recognizer always returns the diarization products together with an
    // alignment.
    sent = post(api, chat_body(R"("language":"en",)" + kWords));
    CHECK(
        sent.response.body.find(
            R"("transcription":{"schema_version":"1","task":"transcribe","transcript_origin":"llm","language":"en","duration":2.0,"text":"Hello world.","segments":[],"words":[{"word":"Hello","word_index":0,"start":0.42,"end":0.81},{"word":"world.","word_index":1,"start":0.82,"end":1.31}],"subtitle_cues":[{"first_word":0,"last_word":1,"start":0.42,"end":1.31,"text":"Hello world."}],"word_timestamp_alignment":{"status":"aligned","model":"stt_en_fastconformer_ctc_large","canonical_word_version":"nemo-ctc-sot-v1","frame_stride_seconds":0.04,"failure_reason":null},"diarization_timestamps":[],"speaker_word_timestamps":{}}})") !=
            std::string::npos,
        sent.response.body);
    sent = post(api, chat_body(R"("timestamp_granularities":["segment"])"));
    CHECK(
        sent.response.body.find(R"("words":[{"word":"Hello")") != std::string::npos,
        "segment granularity asks for the timestamps too");
    CHECK(
        fake.seen.language_code.empty(),
        "an omitted language stays omitted on a timestamp request");
    sent = post(api, chat_body(R"("timestamp_granularities":[])"));
    CHECK(
        sent.response.status == 200 && sent.response.body.find("\"words\"") == std::string::npos,
        "an empty granularity list asks for no timestamps");
    // word_timestamps asks for the same payload; false for none.
    const auto granular = post(api, chat_body(kWords)).response.body;
    sent = post(api, chat_body(R"("word_timestamps":true)"));
    CHECK(sent.response.body == granular, sent.response.body);
    sent = post(api, chat_body(R"("word_timestamps":false)"));
    CHECK(
        sent.response.status == 200 && sent.response.body.find("\"words\"") == std::string::npos,
        "word_timestamps false asks for no timestamps");
    // subtitle_format adds the cues as SRT or WebVTT text after the speaker products.
    sent = post(api, chat_body(R"("word_timestamps":true,"subtitle_format":"srt")"));
    CHECK(
        sent.response.body.find(
            R"("speaker_word_timestamps":{},"subtitles":{"format":"srt","text":"1\n00:00:00,420 --> 00:00:01,310\nHello world.\n\n"}}})") !=
            std::string::npos,
        sent.response.body);
    sent = post(api, chat_body(kWords + R"(,"subtitle_format":"vtt")"));
    CHECK(
        sent.response.body.find(
            R"("subtitles":{"format":"vtt","text":"WEBVTT\n\n00:00:00.420 --> 00:00:01.310\nHello world.\n\n"}}})") !=
            std::string::npos,
        sent.response.body);

    // Speaker attribution renders per word when present.
    fake.details.words[0].speaker = 1;
    fake.details.words[0].speaker_slot = 3;
    fake.details.speaker_words["spk:1"] = {fake.details.words[0]};
    fake.details.diarization = {{2, 0.4, 0.9}};
    fake.details.diarization_frame_seconds = 0.08;
    sent = post(api, chat_body(kWords));
    CHECK(
        sent.response.body.find(
            R"({"word":"Hello","word_index":0,"start":0.42,"end":0.81,"speaker":"spk:1","speaker_slot":3})") !=
            std::string::npos,
        sent.response.body);
    CHECK(
        sent.response.body.find(
            R"("diarization_timestamps":[{"speaker":2,"start":0.4,"end":0.9}],"diarization_frame_seconds":0.08,"speaker_word_timestamps":{"spk:1":[{"word":"Hello")") !=
            std::string::npos,
        sent.response.body);

    // Without an alignment the transcript is a 200 with empty words and cues.
    fake.details = fixture();
    fake.details.words.clear();
    fake.details.alignment_status = "unavailable";
    fake.details.alignment_model.clear();
    fake.details.failure_reason = "ctc_unavailable";
    sent = post(api, chat_body(kWords));
    CHECK(
        sent.response.body.find(
            R"("words":[],"subtitle_cues":[],"word_timestamp_alignment":{"status":"unavailable","model":null,"canonical_word_version":"nemo-ctc-sot-v1","frame_stride_seconds":0.04,"failure_reason":"ctc_unavailable"}})") !=
            std::string::npos,
        sent.response.body);
    // Subtitles need the alignment: the request fails.
    EXPECT_ERROR(
        post(api, chat_body(kWords + R"(,"subtitle_format":"srt")")), 422, "timestamp_unavailable",
        "word timestamps are unavailable (ctc_unavailable), so no subtitles can be rendered",
        "subtitle_format");

    // Readable cues: five words of 34 graphemes in 2.4 s are one single-line cue.
    fake.details = fixture();
    fake.details.text = "Hello world. Second sentence here.";
    fake.details.duration_seconds = 4.0;
    const char* words[] = {"Hello", "world.", "Second", "sentence", "here."};
    const double times[][2] = {
        {0.42, 0.81}, {0.82, 1.31}, {1.50, 1.90}, {1.91, 2.40}, {2.41, 2.80}};
    fake.details.words.clear();
    for (int i = 0; i < 5; ++i) {
        asr::SpeechLmWord w;
        w.word = words[i];
        w.word_index = i;
        w.start = times[i][0];
        w.end = times[i][1];
        fake.details.words.push_back(w);
    }
    sent = post(api, chat_body(kWords));
    CHECK(
        sent.response.body.find(
            R"("subtitle_cues":[{"first_word":0,"last_word":4,"start":0.42,"end":2.8,"text":"Hello world. Second sentence here."}])") !=
            std::string::npos,
        sent.response.body);
    // NeMo's aligner: words of two speakers overlap and a word can have no duration; the cue ends
    // at the latest word end.
    const char* spoken[] = {"Hello", "Yes.", "there.", "Fine."};
    const int speakers[] = {0, 1, 0, 1};
    const double spans[][2] = {{0.40, 0.90}, {0.60, 1.00}, {0.95, 0.95}, {1.10, 1.50}};
    fake.details.words.clear();
    for (int i = 0; i < 4; ++i) {
        asr::SpeechLmWord w;
        w.word = spoken[i];
        w.word_index = i;
        w.speaker = speakers[i];
        w.start = spans[i][0];
        w.end = spans[i][1];
        fake.details.words.push_back(w);
    }
    sent = post(api, chat_body(kWords));
    CHECK(sent.response.status == 200, sent.response.body);
    CHECK(
        sent.response.body.find(
            R"("subtitle_cues":[{"first_word":0,"last_word":3,"start":0.4,"end":1.5,"text":"Hello Yes. there. Fine."}])") !=
            std::string::npos,
        sent.response.body);
    // Overlapping cues (two speakers) render as they are, and a cue that rounds to no duration
    // ends 1 ms after its start.
    fake.details.words = {fake.details.words[0], fake.details.words[1]};
    fake.details.words[0].end =
        9.0;  // 0.4-9.0 and 1.0-1.0005: two cues, the second inside the first
    fake.details.words[1].start = 1.0;
    fake.details.words[1].end = 1.0004;
    sent = post(api, chat_body(kWords + R"(,"subtitle_format":"srt")"));
    CHECK(
        sent.response.body.find(
            R"("text":"1\n00:00:00,400 --> 00:00:09,000\nHello\n\n2\n00:00:01,000 --> 00:00:01,001\nYes.\n\n")") !=
            std::string::npos,
        sent.response.body);
}

void
test_validation() {
    Fake fake;
    chat::Api api(limits(), backend(fake));

    // Body and JSON (clients match substrings of these messages).
    EXPECT_ERROR(
        post(api, "{\"model\": "), 400, "invalid_request",
        "invalid request body: EOF while parsing a value at line 1 column 10", "");
    EXPECT_ERROR(
        post(api, chat_body(R"("foo":"bar")")), 400, "invalid_request",
        "unknown field `foo`, expected one of `model`, `messages`, `stream`, `stream_options`, "
        "`temperature`, `seed`, `user`, `metadata`, `modalities`, `n`, `tools`, `tool_choice`, "
        "`response_format`, `max_tokens`, `max_completion_tokens`, `audio`, `language`, "
        "`timestamp_granularities`, `word_timestamps`, `subtitle_format` at line 1 column",
        "");
    EXPECT_ERROR(
        post(api, chat_body(R"("max_output_tokens":5)")), 400, "invalid_request",
        "unknown field `max_output_tokens`", "");
    EXPECT_ERROR(
        post(api, chat_body(R"("temperature":0.2,"temperature":0.3)")), 400, "invalid_request",
        "duplicate field `temperature`", "");
    EXPECT_ERROR(
        post(api, chat_body(R"("seed":1,"seed":1)")), 400, "invalid_request",
        "duplicate field `seed`", "");
    for (const char* value : {"\"true\"", "1", "\"yes\""})
        EXPECT_ERROR(
            post(api, chat_body(std::string(R"("stream":)") + value)), 400, "invalid_request",
            "expected a boolean", "");
    for (const char* value : {R"("word")", R"(["words"])", R"([1])", R"({"word":true})"})
        EXPECT_ERROR(
            post(api, chat_body(std::string(R"("timestamp_granularities":)") + value)), 400,
            "invalid_request",
            "timestamp_granularities must be a list of \\\"word\\\" and \\\"segment\\\"",
            "timestamp_granularities");
    // word_timestamps and subtitle_format.
    EXPECT_ERROR(
        post(api, chat_body(kWords + R"(,"word_timestamps":true)")), 400, "invalid_request",
        "use timestamp_granularities or word_timestamps", "timestamp_granularities");
    EXPECT_ERROR(
        post(api, chat_body(R"("word_timestamps":"true")")), 400, "invalid_request",
        "expected a boolean", "");
    EXPECT_ERROR(
        post(api, chat_body(R"("word_timestamps":true,"subtitle_format":"ass")")), 400,
        "invalid_request", "subtitle_format must be srt or vtt", "subtitle_format");
    for (const char* timestamps : {"", R"("word_timestamps":false,)"})
        EXPECT_ERROR(
            post(api, chat_body(std::string(timestamps) + R"("subtitle_format":"vtt")")), 400,
            "invalid_request", "vtt requires word_timestamps=true", "word_timestamps");
    EXPECT_ERROR(
        post(api, chat_body(R"("stream":true,"word_timestamps":true,"subtitle_format":"srt")")),
        400, "invalid_request", "subtitle_format does not support stream=true", "subtitle_format");
    CHECK(post(api, chat_body(R"("stream":false)")).response.status == 200, "stream false");
    CHECK(post(api, chat_body(R"("temperature":null)")).response.status == 200, "null is absent");

    // Content type.
    EXPECT_ERROR(
        post(api, chat_body(), "text/plain"), 415, "unsupported_media_type",
        "Content-Type must be application/json", "");
    EXPECT_ERROR(post(api, chat_body(), ""), 415, "unsupported_media_type", "Content-Type", "");
    CHECK(
        post(api, chat_body(), "Application/JSON; charset=utf-8").response.status == 200,
        "media params");

    // Message grammar (every refusal names messages).
    EXPECT_ERROR(
        post(api, chat_body("", audio_part() + "," + audio_part())), 400, "invalid_request",
        "the message must carry exactly one input_audio part", "messages");
    EXPECT_ERROR(
        post(api, chat_body("", R"({"type":"text","text":"hi"})")), 400, "invalid_request",
        "the message must carry exactly one input_audio part", "messages");
    EXPECT_ERROR(
        post(api, chat_body("", audio_part() + R"(,{"type":"input_image","image_url":"x"})")), 400,
        "invalid_request",
        "unsupported content part type: input_image (supported: input_audio, text)", "messages");
    EXPECT_ERROR(
        post(api, chat_body("", R"({"type":"image_url","image_url":{"url":"x"}})")), 400,
        "invalid_request",
        "unsupported content part type: image_url (supported: input_audio, text)", "messages");
    EXPECT_ERROR(
        post(api, chat_body("", R"({"text":"no type"})")), 400, "invalid_request",
        "content parts need a type", "messages");
    EXPECT_ERROR(
        post(
            api, chat_body(
                     "", R"({"type":"input_audio","input_audio":{"data":")" + kWav +
                             R"(","format":"wav","sample_rate":16000}})")),
        400, "invalid_request",
        "invalid input_audio part: unknown field `sample_rate`, expected `data` or `format`",
        "messages");
    EXPECT_ERROR(
        post(
            api,
            chat_body(
                "", R"({"type":"text","text":"a"},{"type":"text","text":"b"},)" + audio_part())),
        400, "invalid_request", "the message must carry at most one text part", "messages");
    {
        std::string body =
            R"({"model":")" + kModel + R"(","messages":[{"role":"user","content":"just text"}]})";
        EXPECT_ERROR(
            post(api, body), 400, "invalid_request",
            "messages must be a list with exactly one user message whose content is a list of "
            "input_audio and text parts",
            "messages");
        body = R"({"model":")" + kModel +
               R"(","messages":[{"role":"user","tool_calls":[],"content":[)" + audio_part() +
               "]}]}";
        EXPECT_ERROR(
            post(api, body), 400, "invalid_request", "unsupported message field: tool_calls",
            "messages");
        body = R"({"model":")" + kModel +
               R"(","messages":[{"type":"message","role":"user","content":[)" + audio_part() +
               "]}]}";
        EXPECT_ERROR(
            post(api, body), 400, "invalid_request", "unsupported message field: type", "messages");
        body = R"({"model":")" + kModel + R"(","messages":[{"role":"system","content":"x"}]})";
        EXPECT_ERROR(
            post(api, body), 400, "invalid_request", "the message role must be user", "messages");
        body = R"({"model":")" + kModel +
               R"(","messages":[{"role":"user","name":"alice","content":[)" + audio_part() + "]}]}";
        CHECK(post(api, body).response.status == 200, "chat name is accepted");
        body = R"({"messages":[{"role":"user","content":[)" + audio_part() + "]}]}";
        EXPECT_ERROR(post(api, body), 400, "invalid_request", "model is required", "model");
        body = R"({"model":"","messages":[{"role":"user","content":[)" + audio_part() + "]}]}";
        EXPECT_ERROR(post(api, body), 400, "invalid_request", "model is required", "model");
        body = R"({"model":")" + kModel + R"(","messages":null})";
        EXPECT_ERROR(post(api, body), 400, "invalid_request", "messages is required", "messages");
    }
    EXPECT_ERROR(
        post(api, chat_body(R"("model":null)")), 400, "invalid_request", "duplicate field `model`",
        "");

    // Audio data and format (no param).
    EXPECT_ERROR(
        post(api, chat_body("", audio_part(""))), 400, "invalid_request",
        "input_audio.data must not be empty", "");
    EXPECT_ERROR(
        post(api, chat_body("", audio_part("not base64!"))), 400, "invalid_request",
        "input_audio.data is not valid base64", "");
    {
        std::string unpadded = kWav;
        while (!unpadded.empty() && unpadded.back() == '=') unpadded.pop_back();
        CHECK(unpadded.size() != kWav.size(), "fixture needs padding");
        EXPECT_ERROR(
            post(api, chat_body("", audio_part(unpadded))), 400, "invalid_request",
            "input_audio.data is not valid base64", "");
    }
    for (const char* format : {"", "WAV", "audio/wav", "wav; codec=pcm", "xxxxxxxxxxxxxxxxx"})
        EXPECT_ERROR(
            post(api, chat_body("", audio_part(kWav, format))), 400, "invalid_request",
            "input_audio.format must be a lowercase container name such as wav", "");

    // Sampling and timestamp admission.
    EXPECT_ERROR(post(api, chat_body(R"("seed":1.5)")), 400, "invalid_request", "expected i64", "");
    EXPECT_ERROR(
        post(api, chat_body(R"("seed":9999999999999999999999)")), 400, "invalid_request",
        "expected i64", "");
    CHECK(post(api, chat_body(R"("seed":9223372036854775807)")).response.status == 200, "max seed");
    CHECK(
        fake.seen.speechlm.has_seed && fake.seen.speechlm.seed == 9223372036854775807ULL,
        "seed forwarded exactly");
    for (const char* value : {"-0.1", "1.01", "\"0.5\"", "true"})
        CHECK(
            post(api, chat_body(std::string(R"("temperature":)") + value)).response.status == 400,
            value);
    CHECK(post(api, chat_body(R"("temperature":1.0)")).response.status == 200, "temperature 1.0");
    CHECK(fake.seen.speechlm.temperature == 1.0f, "temperature forwarded");
    for (const char* language :
         {"en", "EN", "en-US", "en_GB", "en-Latn-US", "es", "zh-CN", "ja", "th"}) {
        CHECK(
            post(api, chat_body(kWords + std::string(R"(,"language":")") + language + "\""))
                    .response.status == 200,
            language);
        CHECK(fake.seen.language_code == language, "language forwarded verbatim");
    }
    CHECK(
        post(api, chat_body(R"("language":"fr")")).response.status == 200 &&
            fake.seen.language_code == "fr" && fake.seen.speechlm.task == "transcribe",
        "a language hint without timestamps");

    // Record validation, then model, then the instruction rules.
    EXPECT_ERROR(
        post(api, chat_body(R"("language":"")")), 400, "invalid_request",
        "language must not be empty", "language");
    {
        std::string body = chat_body();
        body.replace(body.find(kModel), kModel.size(), "nope");
        EXPECT_ERROR(post(api, body), 404, "unknown_model", "requested model is not available", "");
        auto b = backend(fake);
        b.model_alias = [] { return std::string("asr-model-id"); };
        chat::Api alias_only(limits(), b);
        CHECK(post(alias_only, chat_body()).response.status == 404, "only the alias is served");
        auto named = limits();
        named.model_name = kModel;  // http.chat.model-name
        chat::Api aliased(named, b);
        body = chat_body();
        body.replace(body.find(kModel), kModel.size(), "asr-model-id");
        CHECK(
            post(aliased, body).response.status == 200, "the recognizer's /v1/models id is served");
        CHECK(post(aliased, chat_body()).response.status == 200, "the configured name also is");
    }
    EXPECT_ERROR(
        post(api, chat_body("", with_text("Transcribe <|audio|> now."))), 400, "invalid_request",
        "<|audio|> locator must be final and be preceded by exactly one space", "");
    EXPECT_ERROR(
        post(api, chat_body("", with_text("a <|audio|> b <|audio|>"))), 400, "invalid_request",
        "<|audio|> locator must appear exactly once", "");
    CHECK(
        post(api, chat_body("", with_text("  Transcribe verbatim.  "))).response.status == 200,
        "prompt");
    CHECK(
        fake.seen.speechlm.instructions == "Transcribe verbatim.", fake.seen.speechlm.instructions);
    CHECK(post(api, chat_body("", with_text("  \\n\\t "))).response.status == 200, "blank prompt");
    CHECK(fake.seen.speechlm.instructions.empty(), "a blank instruction is the served prompt");
    // Chat template turn markers would forge or close a turn; the model's own tags are text.
    for (const char* token : {"<|im_start|>", "<|im_end|>", "<think>", "</think>"})
        EXPECT_ERROR(
            post(api, chat_body("", with_text(std::string("Transcribe.") + token + "assistant"))),
            400, "invalid_request",
            "the final messages text must not contain chat template control tokens", "messages");
    CHECK(
        post(api, chat_body("", with_text("Start every speaker turn with <spk:0>.")))
                .response.status == 200,
        "speaker tags in an instruction");

    // Refusals name their field.
    const std::pair<std::string, std::pair<std::string, std::string>> refusals[] = {
        {R"("stream_options":{"include_usage":true})",
         {"stream_options requires stream", "stream_options"}},
        {R"("n":2)", {"n must be 1: one transcript per upload", "n"}},
        {R"("tools":[])", {"tools are not supported", "tools"}},
        {R"("tool_choice":"auto")", {"tools are not supported", "tool_choice"}},
        {R"("response_format":{"type":"json_object"})",
         {"response_format must be {\\\"type\\\": \\\"text\\\"}", "response_format"}},
        {R"("max_tokens":10)",
         {"max_tokens is not supported: the decoder budget follows the audio duration",
          "max_tokens"}},
        {R"("max_completion_tokens":10)",
         {"max_completion_tokens is not supported", "max_completion_tokens"}},
        {R"("audio":{"voice":"alloy","format":"wav"})", {"audio output is not supported", "audio"}},
        {R"("modalities":["text","audio"])", {"modalities must be [\\\"text\\\"]", "modalities"}},
        {R"("seed":-1)", {"seed must be a non-negative integer", "seed"}},
        {R"("temperature":2)",
         {"temperature must be a finite number between 0 and 1", "temperature"}},
    };
    for (const auto& [extra, expected] : refusals)
        EXPECT_ERROR(
            post(api, chat_body(extra)), 400, "invalid_request", expected.first, expected.second);
    EXPECT_ERROR(
        post(api, chat_body(R"("logprobs":true)")), 400, "invalid_request",
        "unknown field `logprobs`", "");
    CHECK(
        post(
            api,
            chat_body(
                R"("n":1.0,"tools":null,"max_tokens":null,"stream":null,"tool_choice":"none","response_format":{"type":"text"},"modalities":["text"],"user":"u","metadata":{"a":1})"))
                .response.status == 200,
        "fields that change nothing");
}

void
test_audio_and_backend_errors() {
    Fake fake;
    chat::Api api(limits(), backend(fake));
    EXPECT_ERROR(
        post(api, chat_body("", audio_part(base64(wav(319))))), 400, "invalid_request",
        "audio is shorter than one feature frame (320 samples at 16 kHz)", "");
    CHECK(post(api, chat_body("", audio_part(base64(wav(320))))).response.status == 200, "320");
    {
        std::string lies = wav(400);
        lies[4] = '\xF0';
        lies[5] = lies[6] = lies[7] = '\xFF';
        EXPECT_ERROR(
            post(api, chat_body("", audio_part(base64(lies)))), 400, "invalid_request",
            "malformed WAV: truncated RIFF payload", "");
        std::string streamed = wav(400) + std::string(1, '\x07');  // plus a partial frame
        for (size_t at : {4u, 40u}) streamed.replace(at, 4, std::string(4, '\xFF'));
        CHECK(
            post(api, chat_body("", audio_part(base64(streamed)))).response.status == 200,
            "a streamed WAV (sizes 0xFFFFFFFF) reads to the end");
    }
    {
        chat::Limits small;
        small.max_audio_seconds = 1;
        chat::Api short_api(small, backend(fake));
        EXPECT_ERROR(
            post(short_api, chat_body("", audio_part(base64(wav(16002))))), 400, "invalid_request",
            "audio exceeds configured limit", "");
        CHECK(
            post(short_api, chat_body("", audio_part(base64(wav(16001))))).response.status == 200,
            "max_samples + 1 frames pass the WAV length check");
        // a FLAC over the chat limit: the limit's 400, not a decoding failure
        EXPECT_ERROR(
            post(short_api, chat_body("", audio_part(base64(flac_constant(1000, 5)), "flac"))), 400,
            "invalid_request", "audio is 1.3 s, the limit is 1 s", "");
    }
    CHECK(
        post(api, chat_body("", audio_part(base64(flac_constant(1000)), "flac"))).response.status ==
            200,
        "FLAC uploads are decoded");
    // Uploads that are not PCM16 / G.711 WAV report decoding errors after the model check.
    EXPECT_ERROR(
        post(api, chat_body("", audio_part(base64("fLaC not really flac"), "flac"))), 422,
        "invalid_request", "invalid or unsupported audio file", "");
    {
        std::string body = chat_body("", audio_part(base64("fLaC not really flac"), "flac"));
        body.replace(body.find(kModel), kModel.size(), "nope");
        EXPECT_ERROR(post(api, body), 404, "unknown_model", "requested model is not available", "");
    }
    // Backend exceptions.
    fake.on_recognize = [] {
        throw std::invalid_argument("audio is shorter than one feature frame");
    };
    EXPECT_ERROR(
        post(api, chat_body()), 400, "invalid_request", "audio is shorter than one feature frame",
        "");
    fake.on_recognize = [] { throw asr::ResourceExhausted("SALM: queue is full"); };
    EXPECT_ERROR(post(api, chat_body()), 429, "overloaded", "SALM: queue is full", "");
    CHECK(
        post(api, chat_body()).response.body.find("rate_limit_error") != std::string::npos,
        "429 type");
    fake.on_recognize = [] { throw asr::Unavailable("SALM: backend is shutting down"); };
    {
        const auto sent = post(api, chat_body());
        CHECK(
            sent.response.status == 503 &&
                sent.response.body ==
                    error_json("backend request failed", "server_error", "", "engine_error"),
            sent.response.body);
    }
    fake.on_recognize = [] { throw std::runtime_error("CUDA error at /opt/x.cu"); };
    {
        const auto sent = post(api, chat_body());
        CHECK(
            sent.response.status == 503 && sent.response.body.find("CUDA") == std::string::npos,
            sent.response.body);
    }
    fake.on_recognize = nullptr;
    fake.timestamps = false;
    EXPECT_ERROR(
        post(api, chat_body(kWords)), 422, "timestamp_unavailable",
        "word timestamps are unavailable", "");
    fake.timestamps = true;
    // Other models refuse what they would ignore; a seed and greedy decoding stay accepted.
    fake.instructions = false;
    EXPECT_ERROR(
        post(api, chat_body("", with_text("Translate the speech into English."))), 400,
        "invalid_request", "an instruction needs a speech-LLM model", "messages");
    EXPECT_ERROR(
        post(api, chat_body(R"("temperature":0.5)")), 400, "invalid_request",
        "a temperature above 0 needs a speech-LLM model", "temperature");
    CHECK(
        post(api, chat_body(R"("temperature":0,"seed":7)")).response.status == 200 &&
            post(api, chat_body("", with_text("  "))).response.status == 200,
        "greedy, seeded and blank-instruction requests");
}

void
test_limits() {
    Fake fake;
    {  // 413: declared or discovered while reading.
        chat::Limits l;
        l.max_request_bytes = 4096;
        chat::Api api(l, backend(fake));
        const std::string big = chat_body("", audio_part(base64(wav(4000))));
        EXPECT_ERROR(
            post(api, big), 413, "payload_too_large",
            "request body exceeds the 4096-byte upload limit", "");
        EXPECT_ERROR(
            post(api, big, "application/json", ""), 413, "payload_too_large", "4096-byte", "");
        EXPECT_ERROR(
            post(api, chat_body(), "text/plain", "5000"), 413, "payload_too_large", "4096-byte",
            "");
    }
}

void
test_text_and_numbers() {
    Fake fake;
    chat::Api api(limits(), backend(fake));
    // JSON escaping (control characters as \u00xx, UTF-8 kept, invalid bytes as U+FFFD) and float
    // format.
    fake.details.text = std::string("tab\there \"q\" \\ \x01 caf\xC3\xA9 \xFF end");
    fake.details.duration_seconds = 1e-7;
    auto sent = post(api, chat_body());
    CHECK(
        sent.response.body.find(R"("content":"tab\there \"q\" \\ \u0001 café )"
                                "\xEF\xBF\xBD"
                                R"( end")") != std::string::npos,
        sent.response.body);
    CHECK(sent.response.body.find(R"("duration":1e-7)") != std::string::npos, sent.response.body);
    fake.details.duration_seconds = 12345678901234567.0;
    sent = post(api, chat_body());
    CHECK(
        sent.response.body.find(R"("duration":1.2345678901234568e16)") != std::string::npos,
        sent.response.body);
    fake.details.duration_seconds = 10.885;
    sent = post(api, chat_body());
    CHECK(sent.response.body.find(R"("duration":10.885)") != std::string::npos, sent.response.body);

    // Cue bounds: 7 s, two lines of at most 42 graphemes, every word in exactly one cue, in order.
    fake.details = fixture();
    fake.details.words.clear();
    fake.details.text.clear();
    fake.details.duration_seconds = 30.0;
    const char* vocabulary[] = {
        "internationalization",
        "is",
        "a",
        "rather",
        "long",
        "word.",
        "Another",
        "sentence",
        "follows",
        "here",
        "with",
        "more",
        "words",
        "than",
        "fit"};
    for (int i = 0; i < 40; ++i) {
        asr::SpeechLmWord w;
        w.word = vocabulary[i % 15];
        w.word_index = i;
        w.start = i * 0.5;
        w.end = i * 0.5 + 0.4;
        fake.details.text += (i ? " " : "") + w.word;
        fake.details.words.push_back(w);
    }
    sent = post(api, chat_body(kWords));
    CHECK(sent.response.status == 200, sent.response.body);
    const auto& body = sent.response.body;
    size_t expected_first = 0, cues = 0;
    for (size_t at = body.find("{\"first_word\":"); at != std::string::npos;
         at = body.find("{\"first_word\":", at + 1)) {
        const size_t first = std::stoul(body.substr(at + 14));
        const size_t last = std::stoul(body.substr(body.find("\"last_word\":", at) + 12));
        const double start = std::stod(body.substr(body.find("\"start\":", at) + 8));
        const double end = std::stod(body.substr(body.find("\"end\":", at) + 6));
        const size_t text_at = body.find("\"text\":\"", at) + 8;
        const std::string text = body.substr(text_at, body.find('"', text_at) - text_at);
        CHECK(first == expected_first && last >= first, "cues cover the words in order");
        CHECK(end - start <= 7.0 + 1e-9 || first == last, "cue duration");
        const size_t newline = text.find("\\n");
        const std::string line1 = text.substr(0, newline),
                          line2 = newline == std::string::npos ? "" : text.substr(newline + 2);
        CHECK(line1.size() <= 42 && line2.size() <= 42, text);
        expected_first = last + 1;
        ++cues;
    }
    CHECK(expected_first == 40 && cues > 1, "every word is in a cue");
}

// The events of a streamed response; `budget` writes succeed, later ones report a gone client.
struct Streamed {
    bool complete = false;
    std::vector<std::string> events;  // the data of each event
    std::string content;              // the delta contents joined
};

Streamed
consume(const Sent& sent, size_t budget = 1000) {
    Streamed out;
    std::string body;
    out.complete = sent.response.stream([&](std::string_view data) {
        if (budget == 0)
            return false;
        --budget;
        body += data;
        return true;
    });
    for (size_t at = 0; at < body.size();) {
        const size_t end = body.find("\n\n", at);
        CHECK(body.compare(at, 6, "data: ") == 0 && end != std::string::npos, body);
        if (end == std::string::npos)
            break;
        out.events.push_back(body.substr(at + 6, end - at - 6));
        at = end + 2;
    }
    for (const auto& event : out.events) {
        const size_t at = event.find(R"("delta":{)");
        const size_t content = event.find(R"("content":")", at);
        if (at == std::string::npos || content == std::string::npos)
            continue;
        const size_t first = content + 11;
        out.content += event.substr(first, event.find('"', first) - first);
    }
    return out;
}

void
test_streaming() {
    Fake fake;
    chat::Api api(limits(), backend(fake));
    const std::string head =
        R"({"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1790000000,"model":")" +
        kModel + R"(","choices":[{"index":0,"delta":)";

    // The text as the backend reports it, then the stop chunk with the payload and [DONE].
    fake.pieces = {"Hello", " world."};
    auto sent = post(api, chat_body(R"("stream":true)"));
    CHECK(sent.response.status == 200 && sent.response.body.empty(), sent.response.body);
    CHECK(sent.header("Content-Type") == "text/event-stream", "event stream type");
    CHECK(sent.header("Cache-Control") == "no-cache", "event stream caching");
    CHECK(static_cast<bool>(fake.seen.speechlm.on_text), "the backend gets a text sink");
    auto events = consume(sent);
    CHECK(events.complete && events.events.size() >= 4, "stream events");
    CHECK(
        events.events.front() ==
            head + R"({"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]})",
        events.events.front());
    CHECK(events.content == "Hello world.", events.content);
    const std::string& stop = events.events[events.events.size() - 2];
    CHECK(
        stop.rfind(
            head +
                R"({},"logprobs":null,"finish_reason":"stop"}],"transcription":{"schema_version":"1")",
            0) == 0,
        stop);
    CHECK(stop.find(R"("text":"Hello world.")") != std::string::npos, "payload text");
    CHECK(events.events.back() == "[DONE]", events.events.back());

    // A backend that reports nothing while decoding: the whole content arrives at the end.
    fake.pieces.clear();
    events = consume(post(api, chat_body(R"("stream":true)")));
    CHECK(events.complete && events.content == "Hello world.", events.content);

    // include_usage: a final chunk without choices.
    fake.details.prompt_tokens = 30;
    fake.details.audio_tokens = 25;
    fake.details.completion_tokens = 4;
    events =
        consume(post(api, chat_body(R"("stream":true,"stream_options":{"include_usage":true})")));
    CHECK(events.events.size() >= 3, "usage events");
    CHECK(
        events.events[events.events.size() - 2] ==
            R"({"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1790000000,"model":")" +
                kModel +
                R"(","choices":[],"usage":{"prompt_tokens":30,"completion_tokens":4,"total_tokens":34,"prompt_tokens_details":{"audio_tokens":25}}})",
        events.events[events.events.size() - 2]);
    for (const auto& event : events.events)
        CHECK(
            event == events.events[events.events.size() - 2] ||
                event.find("\"usage\"") == std::string::npos,
            "usage only in its own chunk");
    EXPECT_ERROR(
        post(api, chat_body(R"("stream":true,"stream_options":{"include_usage":1})")), 400,
        "invalid_request", "expected a boolean", "");
    EXPECT_ERROR(
        post(api, chat_body(R"("stream":true,"stream_options":{"chunk":1})")), 400,
        "invalid_request", "unknown field `chunk`", "");
    EXPECT_ERROR(
        post(api, chat_body(R"("stream":false,"stream_options":{"include_usage":true})")), 400,
        "invalid_request", "stream_options requires stream", "stream_options");

    // Refusals before any text keep their status; failures after it are an error event.
    fake.on_recognize = [] { throw std::invalid_argument("SALM: audio is too long"); };
    EXPECT_ERROR(
        post(api, chat_body(R"("stream":true)")), 400, "invalid_request", "audio is too long", "");
    fake.on_recognize = [] { throw asr::ResourceExhausted("SALM: queue is full"); };
    EXPECT_ERROR(post(api, chat_body(R"("stream":true)")), 429, "overloaded", "queue is full", "");
    fake.on_recognize = nullptr;
    fake.pieces = {"Hel"};
    fake.after_text = [] { throw std::runtime_error("decoder failed"); };
    events = consume(post(api, chat_body(R"("stream":true)")));
    CHECK(events.complete && events.content == "Hel", events.content);
    CHECK(
        events.events.size() >= 3 &&
            events.events[events.events.size() - 2] ==
                error_json("backend request failed", "server_error", "", "engine_error") &&
            events.events.back() == "[DONE]",
        events.content);
    fake.after_text = nullptr;

    // A client that goes away: the stream stops and the decode still completes.
    {
        auto gone = post(api, chat_body(R"("stream":true)"));
        CHECK(!consume(gone, 1).complete, "a gone client ends the stream");
    }
    CHECK(post(api, chat_body()).response.status == 200, "the next request is served");
}

// Waits up to 5 s for the backend to see its client gone (AsrRequestOptions::cancelled).
bool
told_cancelled(Fake& fake) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < until) {
        {
            std::lock_guard<std::mutex> lock(fake.mutex);
            if (fake.seen.cancelled && fake.seen.cancelled())
                return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

// A client that disconnects: its decode is told through AsrRequestOptions::cancelled (499),
// streamed or not; later requests are served as usual.
void
test_cancellation() {
    Fake fake;
    std::atomic<bool> left{false};
    const chat::ClientGone gone = [&left] { return left.load(); };
    const auto post_gone = [&](chat::Api& api, const std::string& body) {
        return post(api, body, "application/json", "auto", gone);
    };
    chat::Api api(limits(), backend(fake));

    // Not streamed: the backend gets the probe and drops the request (asr::Cancelled -> 499).
    fake.on_recognize = [&] {
        left = true;
        if (told_cancelled(fake))
            throw asr::Cancelled("SALM: request cancelled");
    };
    EXPECT_ERROR(
        post_gone(api, chat_body()), 499, "client_closed_request", "client closed the request", "");
    left = false;
    fake.on_recognize = nullptr;
    CHECK(post(api, chat_body()).response.status == 200, "served after a cancel");

    // Streamed, before any text: the response is refused and the decode is cancelled.
    fake.on_recognize = [&] {
        left = true;
        if (told_cancelled(fake))
            throw asr::Cancelled("SALM: request cancelled");
    };
    EXPECT_ERROR(
        post_gone(api, chat_body(R"("stream":true)")), 499, "client_closed_request",
        "client closed the request", "");
    left = false;
    fake.on_recognize = nullptr;

    // Streamed, after the first text: the writer notices while waiting and stops; the decode is
    // cancelled.
    std::atomic<bool> decode_cancelled{false};
    fake.pieces = {"Hel"};
    fake.after_text = [&] {
        left = true;
        decode_cancelled = told_cancelled(fake);
        if (decode_cancelled)
            throw asr::Cancelled("SALM: request cancelled");
    };
    {
        auto sent = post_gone(api, chat_body(R"("stream":true)"));
        CHECK(sent.response.status == 200, sent.response.body);
        const auto streamed = consume(sent);
        CHECK(!streamed.complete && streamed.content == "Hel", "a gone client ends the stream");
    }
    CHECK(decode_cancelled.load(), "the streamed decode is cancelled");
    left = false;
    fake.pieces.clear();
    fake.after_text = nullptr;
    CHECK(post(api, chat_body()).response.status == 200, "the slot is free again");
}

}  // namespace

int
main() {
    test_rendered_fixtures();
    test_validation();
    test_audio_and_backend_errors();
    test_limits();
    test_text_and_numbers();
    test_streaming();
    test_cancellation();
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "chat_api: all checks passed\n";
    return 0;
}
