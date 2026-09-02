// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "s2s_realtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "audio_resampler.h"
#include "http_server.h"
#include "json.h"
#include "s2s/voicechat.h"

namespace nemo_speech::http {
namespace {
using json::Value;

constexpr int kDefaultClientRate = 24000;
constexpr int kOutputRate = 24000;
constexpr int kMinInputRate = 16000;
constexpr int kMaxInputRate = 48000;
constexpr int kOutputChunkSamples = kOutputRate * 80 / 1000;
constexpr size_t kMaxInstructionsLength = 32000;

constexpr const char* kDefaultInstructions =
    "You are an AI voice assistant developed by NVIDIA. Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one. Do not repeat the "
    "same sentence over and over again. Start the conversation by greeting the user.";

constexpr const char* kDefaultToolInstructions =
    "You are an AI voice assistant developed by NVIDIA. Your name is NVIDIA Voice Chat. Your "
    "job is to be helpful and harmless and have engaging conversations in English. Maintain a "
    "warm and friendly tone. Keep the dialogue open and ongoing. Be clear and direct, especially "
    "when answering yes or no questions and multiple-choice questions. Avoid long answers unless "
    "the user asks you to provide details or context. You must provide diverse responses and "
    "rephrase answers if the user asks the same question. DO NOT interrupt the user when they are "
    "speaking, let them finish their turn before answering.";

constexpr const char* kToolsPrefix = R"PROMPT(
When you receive a request, follow this decision process:
1. Does the request match one of your available tools below? If yes, you MUST call that tool - never answer it directly from your own knowledge, even if you think you know the answer.
2. Is it a general knowledge question (history, science, geography, math, facts, etc.)? If yes, answer directly from your own knowledge - do not call any tool.
3. Does it require an external action or live data that none of your tools cover (e.g. ordering food, sending email)? If yes, politely say you don't have that capability.

NEVER say "I don't have a tool for that" for general knowledge questions you can answer yourself.

DO NOT use any tools when not needed to answer the user's requests, under no circumstance.

You are an expert across history, geography, science, math, literature, biographies, languages, recipes, programming, current affairs, and general knowledge. When the user asks about any of these, answer directly and conversationally from your own knowledge — no <TOOLCALL>.

Call a tool ONLY when the user's request matches one of the tools listed in <AVAILABLE_TOOLS> below. For every other request, do not call any tool — just answer from your knowledge. Never invent or call a tool name that is not literally in <AVAILABLE_TOOLS>.

Tool-call arguments must be values the user spoke. If a required argument is missing, ask the user; never guess.

If a tool call fails or returns an error, do not retry the tool call for the same request. Tell the user that the API has an issue.

You can use the following tools to assist the user if required:
<AVAILABLE_TOOLS>)PROMPT";

constexpr const char* kToolsSuffix = R"PROMPT(</AVAILABLE_TOOLS>

If you decide to call any tool(s), use the following format:
<TOOLCALL>[{"name": "tool_name1", "arguments": "tool_args1"}, {"name": "tool_name2", "arguments": "tool_args2"}]</TOOLCALL>

The user will execute tool-calls and return responses from tool(s) in this format:
<TOOL_RESPONSE>[{"tool_response1"}, {"tool_response2"}]</TOOL_RESPONSE>

Based on the tool responses, you can call additional tools if needed, correct tool calls if any errors are found, or just respond to the user.
)PROMPT";

std::string
decode_base64_audio(const std::string& input) {
    static constexpr int8_t invalid = -1;
    static const auto table = [] {
        std::array<int8_t, 256> result{};
        result.fill(invalid);
        const std::string alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (size_t i = 0; i < alphabet.size(); ++i)
            result[static_cast<unsigned char>(alphabet[i])] = static_cast<int8_t>(i);
        return result;
    }();
    std::string output;
    output.reserve(input.size() * 3 / 4);
    uint32_t accumulator = 0;
    int bits = 0;
    size_t sextets = 0;
    size_t padding = 0;
    for (const unsigned char c : input) {
        if (c == '=') {
            if (++padding > 2)
                throw std::invalid_argument("audio is not valid base64");
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
            continue;
        if (padding != 0 || table[c] == invalid)
            throw std::invalid_argument("audio is not valid base64");
        accumulator = (accumulator << 6) | static_cast<uint32_t>(table[c]);
        ++sextets;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            output.push_back(static_cast<char>((accumulator >> bits) & 0xff));
        }
    }
    const size_t remainder = sextets % 4;
    const size_t expected_padding = remainder == 2 ? 2 : remainder == 3 ? 1 : 0;
    if (remainder == 1 || (padding != 0 && padding != expected_padding) ||
        (bits != 0 && (accumulator & ((uint32_t{1} << bits) - 1)) != 0))
        throw std::invalid_argument("audio is not valid base64");
    return output;
}

std::string
encode_base64(const uint8_t* data, size_t size) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    output.reserve((size + 2) / 3 * 4);
    for (size_t i = 0; i < size; i += 3) {
        const uint32_t a = data[i];
        const uint32_t b = i + 1 < size ? data[i + 1] : 0;
        const uint32_t c = i + 2 < size ? data[i + 2] : 0;
        const uint32_t value = (a << 16) | (b << 8) | c;
        output.push_back(alphabet[(value >> 18) & 0x3f]);
        output.push_back(alphabet[(value >> 12) & 0x3f]);
        output.push_back(i + 1 < size ? alphabet[(value >> 6) & 0x3f] : '=');
        output.push_back(i + 2 < size ? alphabet[value & 0x3f] : '=');
    }
    return output;
}

std::vector<float>
decode_pcm16(const std::string& bytes) {
    if ((bytes.size() & 1u) != 0)
        throw std::invalid_argument("PCM16 audio must contain complete samples");
    std::vector<float> samples(bytes.size() / 2);
    for (size_t i = 0; i < samples.size(); ++i) {
        const uint16_t bits = static_cast<uint8_t>(bytes[i * 2]) |
                              (static_cast<uint16_t>(static_cast<uint8_t>(bytes[i * 2 + 1])) << 8);
        samples[i] = static_cast<int16_t>(bits) / 32768.0f;
    }
    return samples;
}

int16_t
float_to_pcm16(float sample) {
    if (!std::isfinite(sample))
        return 0;
    constexpr double scale = 32768.0;
    const double scaled = std::clamp(sample * scale, -32768.0, 32767.0);
    return static_cast<int16_t>(std::trunc(scaled));
}

std::vector<uint8_t>
encode_pcm16(const float* samples, size_t count) {
    std::vector<uint8_t> output(count * 2);
    for (size_t i = 0; i < count; ++i) {
        const int16_t value = float_to_pcm16(samples[i]);
        const uint16_t bits = static_cast<uint16_t>(value);
        output[i * 2] = static_cast<uint8_t>(bits & 0xff);
        output[i * 2 + 1] = static_cast<uint8_t>((bits >> 8) & 0xff);
    }
    return output;
}

void
erase_all(std::string& value, const std::string& needle) {
    size_t position = 0;
    while ((position = value.find(needle, position)) != std::string::npos)
        value.erase(position, needle.size());
}

bool
has_configured_tools(const Value& tools) {
    return (tools.is_array() && !tools.array().empty()) ||
           (tools.is_string() && tools.string() != "[]" && !tools.string().empty());
}

Value
parse_tools(const Value* tools) {
    if (!tools || tools->is_null())
        return Value::Array{};
    if (tools->is_array())
        return *tools;
    if (tools->is_string()) {
        try {
            Value parsed = Value::parse(tools->string());
            return parsed.is_array() ? parsed : Value(Value::Array{});
        }
        catch (...) {
            return Value::Array{};
        }
    }
    return Value::Array{};
}

std::string
build_prompt(const std::string& instructions, const Value& tools) {
    std::string prompt = instructions;
    if (!tools.is_array() || tools.array().empty())
        return prompt;

    Value::Array clean_tools;
    Value::Array acknowledgements;
    for (const auto& tool : tools.array()) {
        if (!tool.is_object())
            continue;
        const Value* definition = tool.find("function");
        if (!definition || !definition->is_object())
            definition = &tool;
        Value::Object clean = definition->object();
        clean.erase("type");
        clean.erase("ack_messages");
        clean_tools.emplace_back(std::move(clean));

        const Value* messages = definition->find("ack_messages");
        if (messages && messages->is_array() && !messages->array().empty()) {
            Value ack(Value::Object{});
            ack["name"] = definition->string_or("name");
            ack["ack_messages"] = *messages;
            acknowledgements.emplace_back(std::move(ack));
        }
    }
    if (clean_tools.empty())
        return prompt;
    if (!prompt.empty())
        prompt += "\n\n";
    prompt += kToolsPrefix;
    prompt += Value(std::move(clean_tools)).dump();
    prompt += kToolsSuffix;
    if (!acknowledgements.empty()) {
        prompt += "\n<TOOL_ACK_MESSAGES>";
        prompt += Value(std::move(acknowledgements)).dump();
        prompt += "</TOOL_ACK_MESSAGES>";
    }
    return prompt;
}

Value
structured_pcm_format(int rate) {
    Value format(Value::Object{});
    format["type"] = "audio/pcm";
    format["rate"] = rate;
    return format;
}

class RealtimeSession {
   public:
    RealtimeSession(
        httplib::ws::WebSocket& socket, std::shared_ptr<s2s::VoiceChat> voicechat,
        const ServerConfig& config, std::atomic<uint64_t>& ids)
        : socket_(socket), voicechat_(std::move(voicechat)), config_(config), ids_(ids),
          session_id_(id("session")),
          input_resampler_(kDefaultClientRate, voicechat_->input_sample_rate()),
          output_resampler_(voicechat_->output_sample_rate(), kOutputRate) {}

    ~RealtimeSession() {
        if (stream_) {
            try {
                voicechat_->end_stream(*stream_);
            }
            catch (...) {
            }
        }
    }

    bool start() {
        Value event(Value::Object{});
        event["type"] = "session.created";
        Value session(Value::Object{});
        session["type"] = "realtime";
        session["id"] = session_id_;
        session["model"] = "nemotron-voicechat";
        session["modalities"] = Value::Array{Value("audio")};
        Value audio(Value::Object{});
        Value input(Value::Object{});
        input["format"] = structured_pcm_format(input_rate_);
        Value output(Value::Object{});
        output["format"] = structured_pcm_format(kOutputRate);
        audio["input"] = std::move(input);
        audio["output"] = std::move(output);
        session["audio"] = std::move(audio);
        session["instructions"] = kDefaultInstructions;
        event["session"] = std::move(session);
        return send(std::move(event));
    }

    bool handle_text(const std::string& message, bool& graceful_close) {
        Value event;
        try {
            event = Value::parse(message);
        }
        catch (const std::exception& error) {
            throw std::invalid_argument(error.what());
        }
        const std::string type = event.string_or("type");
        if (type == "session.update") {
            handle_session_update(event);
        } else if (type == "input_audio_buffer.append") {
            append_audio(decode_base64_audio(event.string_or("audio")));
        } else if (type == "conversation.item.create") {
            handle_conversation_item(event);
        } else if (type == "session.close") {
            graceful_close = true;
            return false;
        }
        return true;
    }

    bool handle_binary(const std::string& pcm) {
        append_audio(pcm);
        return true;
    }

    void emit_error(
        const std::string& message, const std::string& code = "invalid_event",
        const std::string& type = "invalid_request_error") {
        Value event(Value::Object{});
        event["type"] = "error";
        Value error(Value::Object{});
        error["type"] = type;
        error["code"] = code;
        error["message"] = message;
        error["param"] = nullptr;
        event["error"] = std::move(error);
        send(std::move(event));
    }

    bool duration_exceeded() const {
        return config_.realtime_s2s_max_session_seconds > 0 &&
               input_samples_received_ >
                   static_cast<uint64_t>(config_.realtime_s2s_max_session_seconds) * input_rate_;
    }

    void finish(bool graceful) {
        if (graceful && chunks_received_ > 0) {
            std::vector<float> flushed;
            input_resampler_.finish(&flushed);
            append_model_samples(flushed);
            drain_model_input(true);
        }

        if (stream_) {
            if (graceful) {
                const auto final_result = voicechat_->finish_stream(*stream_);
                dispatch_asr_text(final_result.asr_text);
            } else {
                voicechat_->end_stream(*stream_);
            }
            stream_.reset();
        }
        if (graceful)
            flush_open_user_turn();

        std::vector<float> output_tail;
        output_resampler_.finish(&output_tail);
        output_buffer_.insert(output_buffer_.end(), output_tail.begin(), output_tail.end());
        emit_output_chunks();
        emit_response_done();

        // Completion means the conversation state is available for reuse.
        // Release it before notifying the client so a following connection
        // cannot transiently exceed the configured stream limit.
        send_session_end();
    }

   private:
    std::string id(const char* prefix) {
        return std::string(prefix) + "_" + std::to_string(ids_.fetch_add(1));
    }

    bool send(Value event) {
        if (!event.find("event_id"))
            event["event_id"] = id("event");
        return socket_.send(event.dump());
    }

    void ensure_stream() {
        if (stream_)
            return;
        stream_ = voicechat_->create_stream();
        const std::string prompt = build_prompt(instructions_, tools_);
        if (!prompt.empty())
            voicechat_->prefill_system_prompt(*stream_, prompt);
        prompt_applied_ = true;
    }

    void handle_session_update(const Value& event) {
        const Value* session = event.find("session");
        if (!session || !session->is_object())
            throw std::invalid_argument("session.update requires a session object");
        if (chunks_received_ != 0)
            throw std::invalid_argument("session configuration cannot change after audio starts");

        int requested_input_rate = input_rate_;
        if (const Value* audio = session->find("audio")) {
            if (!audio->is_object())
                throw std::invalid_argument("session.audio must be an object");
            if (const Value* input = audio->find("input")) {
                if (!input->is_object())
                    throw std::invalid_argument("session.audio.input must be an object");
                if (const Value* format = input->find("format"))
                    requested_input_rate = validate_pcm_format(*format, input_rate_, true);
            }
            if (const Value* output = audio->find("output")) {
                if (!output->is_object())
                    throw std::invalid_argument("session.audio.output must be an object");
                if (const Value* format = output->find("format"))
                    (void)validate_pcm_format(*format, kOutputRate, false);
            }
        }
        Value requested_tools = parse_tools(session->find("tools"));
        const bool tools_present = has_configured_tools(requested_tools);
        std::string requested_instructions;
        if (const Value* instructions = session->find("instructions");
            instructions && instructions->is_string()) {
            requested_instructions = instructions->string();
        } else {
            requested_instructions =
                tools_present ? kDefaultToolInstructions : kDefaultInstructions;
        }
        const std::string prompt = build_prompt(requested_instructions, requested_tools);
        if (prompt.size() > kMaxInstructionsLength) {
            emit_error("Instructions too long (max 32000 characters)", "instructions_too_long");
            return;
        }
        const bool prompt_changed = prompt != build_prompt(instructions_, tools_);
        if (requested_input_rate != input_rate_) {
            input_rate_ = requested_input_rate;
            input_resampler_ = audio::AudioResampler(input_rate_, voicechat_->input_sample_rate());
        }
        tools_ = std::move(requested_tools);
        instructions_ = std::move(requested_instructions);
        if (prompt_changed && stream_) {
            voicechat_->end_stream(*stream_);
            stream_.reset();
            prompt_applied_ = false;
        }
        if (!prompt_applied_) {
            ensure_stream();
        }

        Value effective = *session;
        Value audio(Value::Object{});
        Value input(Value::Object{});
        input["format"] = structured_pcm_format(input_rate_);
        Value output(Value::Object{});
        output["format"] = structured_pcm_format(kOutputRate);
        audio["input"] = std::move(input);
        audio["output"] = std::move(output);
        effective["audio"] = std::move(audio);
        effective["instructions"] = instructions_;
        effective["tools"] = tools_;
        Value response(Value::Object{});
        response["type"] = "session.updated";
        response["session"] = std::move(effective);
        send(std::move(response));
    }

    int validate_pcm_format(
        const Value& format, int fallback_rate, bool validate_input_rate) const {
        int rate = fallback_rate;
        bool pcm = false;
        if (format.is_string()) {
            pcm = format.string().find("pcm") != std::string::npos;
        } else if (format.is_object()) {
            pcm = format.string_or("type").find("pcm") != std::string::npos;
            if (const Value* requested_rate = format.find("rate")) {
                if (!requested_rate->is_number() ||
                    requested_rate->number() != std::floor(requested_rate->number()))
                    throw std::invalid_argument("audio sample rate must be an integer");
                rate = static_cast<int>(requested_rate->number());
            }
        }
        if (!pcm)
            throw std::invalid_argument("only PCM16 audio is supported");
        if (validate_input_rate && (rate < kMinInputRate || rate > kMaxInputRate))
            throw std::invalid_argument("input sample rate must be between 16000 and 48000 Hz");
        if (!validate_input_rate && rate != kOutputRate)
            throw std::invalid_argument("output sample rate must be 24000 Hz");
        return rate;
    }

    void append_audio(const std::string& bytes) {
        if (bytes.empty())
            return;
        if (bytes.size() >
            config_.max_upload_bytes - std::min(audio_bytes_received_, config_.max_upload_bytes))
            throw std::invalid_argument("realtime audio exceeds the configured upload limit");
        const auto input = decode_pcm16(bytes);
        audio_bytes_received_ += bytes.size();
        input_samples_received_ += input.size();
        ++chunks_received_;
        std::vector<float> converted;
        input_resampler_.process(input.data(), input.size(), &converted);
        append_model_samples(converted);
        drain_model_input(false);
    }

    void append_model_samples(std::vector<float>& samples) {
        model_input_.reserve(model_input_.size() + samples.size());
        constexpr float scale = 32768.0f;
        for (const float sample : samples)
            model_input_.push_back(static_cast<float>(float_to_pcm16(sample)) / scale);
    }

    void drain_model_input(bool pad_residual) {
        const size_t step = static_cast<size_t>(voicechat_->samples_per_step());
        size_t offset = 0;
        while (model_input_.size() - offset >= step) {
            process_step(model_input_.data() + offset);
            offset += step;
        }
        if (offset != 0)
            model_input_.erase(model_input_.begin(), model_input_.begin() + offset);
        if (pad_residual && !model_input_.empty()) {
            model_input_.resize(step, 0.0f);
            process_step(model_input_.data());
            model_input_.clear();
        }
    }

    void process_step(const float* audio) {
        ensure_stream();
        std::string function_response;
        if (!function_responses_.empty()) {
            function_response = std::move(function_responses_.front());
            function_responses_.pop_front();
        }
        const auto result = voicechat_->process_chunk(
            *stream_, audio, voicechat_->samples_per_step(), function_response);
        ++inferences_;
        if (!result.audio.empty()) {
            ensure_response_created();
            std::vector<float> converted;
            output_resampler_.process(result.audio.data(), result.audio.size(), &converted);
            output_buffer_.insert(output_buffer_.end(), converted.begin(), converted.end());
            emit_output_chunks();
        }
        dispatch_agent_text(result.text);
        dispatch_asr_text(result.asr_text);
        dispatch_function_text(result.function_text);
    }

    void ensure_response_created() {
        if (response_active_)
            return;
        response_active_ = true;
        response_id_ = id("response");
        output_item_id_ = id("item");

        Value created(Value::Object{});
        created["type"] = "response.created";
        Value response(Value::Object{});
        response["id"] = response_id_;
        response["object"] = "realtime.response";
        response["status"] = "in_progress";
        response["status_details"] = nullptr;
        response["output"] = Value::Array{};
        created["response"] = std::move(response);
        send(std::move(created));

        Value item_added(Value::Object{});
        item_added["type"] = "response.output_item.added";
        item_added["response_id"] = response_id_;
        item_added["output_index"] = 0;
        Value item(Value::Object{});
        item["id"] = output_item_id_;
        item["object"] = "realtime.item";
        item["type"] = "message";
        item["role"] = "assistant";
        item_added["item"] = std::move(item);
        send(std::move(item_added));

        Value part_added(Value::Object{});
        part_added["type"] = "response.content_part.added";
        add_response_coordinates(part_added);
        Value part(Value::Object{});
        part["type"] = "audio";
        part_added["part"] = std::move(part);
        send(std::move(part_added));
    }

    void add_response_coordinates(Value& event) const {
        event["response_id"] = response_id_;
        event["item_id"] = output_item_id_;
        event["output_index"] = 0;
        event["content_index"] = 0;
    }

    void emit_output_chunks() {
        size_t offset = 0;
        while (output_buffer_.size() - offset >= static_cast<size_t>(kOutputChunkSamples)) {
            const auto pcm = encode_pcm16(output_buffer_.data() + offset, kOutputChunkSamples);
            Value event(Value::Object{});
            event["type"] = "response.output_audio.delta";
            add_response_coordinates(event);
            event["delta"] = encode_base64(pcm.data(), pcm.size());
            if (!send(std::move(event)))
                return;
            ++chunks_sent_;
            offset += kOutputChunkSamples;
        }
        if (offset != 0)
            output_buffer_.erase(output_buffer_.begin(), output_buffer_.begin() + offset);
    }

    void dispatch_agent_text(std::string text) {
        if (text.empty())
            return;
        ensure_response_created();
        const bool has_start = text.find("<s>") != std::string::npos;
        const bool has_done = text.find("</s>") != std::string::npos;
        if (has_start)
            agent_separator_pending_ = previous_agent_turn_nonempty_;
        erase_all(text, "^");
        erase_all(text, "<s>");
        erase_all(text, "</s>");
        if (agent_separator_pending_) {
            const size_t first = text.find_first_not_of(" \t\r\n");
            text = first == std::string::npos ? "" : text.substr(first);
            if (!text.empty()) {
                text.insert(text.begin(), ' ');
                agent_separator_pending_ = false;
            }
        }
        if (!text.empty()) {
            Value delta(Value::Object{});
            delta["type"] = "response.output_audio_transcript.delta";
            add_response_coordinates(delta);
            delta["delta"] = text;
            send(std::move(delta));
            if (config_.realtime_s2s_output_text_events) {
                Value legacy(Value::Object{});
                legacy["type"] = "response.output_text.delta";
                add_response_coordinates(legacy);
                legacy["delta"] = text;
                send(std::move(legacy));
            }
            agent_transcript_ += text;
        }
        if (has_done) {
            Value done(Value::Object{});
            done["type"] = "response.output_audio_transcript.done";
            add_response_coordinates(done);
            done["transcript"] = agent_transcript_;
            send(std::move(done));
            if (config_.realtime_s2s_output_text_events) {
                Value legacy(Value::Object{});
                legacy["type"] = "response.output_text.done";
                add_response_coordinates(legacy);
                legacy["text"] = agent_transcript_;
                send(std::move(legacy));
            }
            previous_agent_turn_nonempty_ = !agent_transcript_.empty();
            agent_transcript_.clear();
        }
    }

    void dispatch_asr_text(std::string text) {
        if (text.empty())
            return;
        const bool has_start = text.find("<s>") != std::string::npos;
        const bool has_done = text.find("</s>") != std::string::npos;
        erase_all(text, "<s>");
        erase_all(text, "</s>");
        if (has_start && !speech_active_) {
            speech_active_ = true;
            input_item_id_ = id("item");
            Value started(Value::Object{});
            started["type"] = "input_audio_buffer.speech_started";
            started["audio_start_ms"] = 0;
            started["item_id"] = input_item_id_;
            send(std::move(started));
        }
        if (!text.empty()) {
            Value delta(Value::Object{});
            delta["type"] = "conversation.item.input_audio_transcription.delta";
            delta["item_id"] = input_item_id_.empty() ? id("item") : input_item_id_;
            delta["content_index"] = 0;
            delta["delta"] = text;
            send(std::move(delta));
            user_transcript_ += text;
        }
        if (has_done && speech_active_)
            flush_open_user_turn();
    }

    void flush_open_user_turn() {
        if (!speech_active_)
            return;
        Value completed(Value::Object{});
        completed["type"] = "conversation.item.input_audio_transcription.completed";
        completed["item_id"] = input_item_id_.empty() ? id("item") : input_item_id_;
        completed["content_index"] = 0;
        completed["transcript"] = user_transcript_;
        send(std::move(completed));

        Value stopped(Value::Object{});
        stopped["type"] = "input_audio_buffer.speech_stopped";
        stopped["audio_end_ms"] = 0;
        stopped["item_id"] = input_item_id_.empty() ? id("item") : input_item_id_;
        send(std::move(stopped));
        speech_active_ = false;
        input_item_id_.clear();
        user_transcript_.clear();
    }

    void dispatch_function_text(const std::string& text) {
        if (text.empty())
            return;
        Value parsed;
        try {
            parsed = Value::parse(text);
        }
        catch (...) {
            return;
        }
        Value::Array calls;
        if (parsed.is_array())
            calls = parsed.array();
        else if (parsed.is_object())
            calls.push_back(std::move(parsed));
        else
            return;
        const std::string tool_response_id = id("response");
        int output_index = 0;
        for (const auto& call : calls) {
            if (!call.is_object())
                continue;
            Value event(Value::Object{});
            event["type"] = "response.function_call_arguments.done";
            event["response_id"] = tool_response_id;
            event["item_id"] = id("item");
            event["output_index"] = output_index++;
            event["call_id"] = id("call");
            event["name"] = call.string_or("name");
            const Value* arguments = call.find("arguments");
            if (!arguments)
                arguments = call.find("parameters");
            event["arguments"] = arguments ? arguments->dump() : "{}";
            send(std::move(event));
        }
    }

    void handle_conversation_item(const Value& event) {
        const Value* item = event.find("item");
        if (!item || !item->is_object() || item->string_or("type") != "function_call_output")
            return;
        if (!has_configured_tools(tools_)) {
            emit_error("Tools are not set for this session", "tools_not_set");
            return;
        }
        if (function_responses_.size() >= config_.realtime_s2s_max_pending_function_responses) {
            emit_error("Too many pending function responses", "function_response_limit_exceeded");
            return;
        }
        const Value* output = item->find("output");
        if (!output)
            function_responses_.emplace_back();
        else if (output->is_string())
            function_responses_.push_back(output->string());
        else
            function_responses_.push_back(output->dump());
    }

    void emit_response_done() {
        if (!response_active_)
            return;
        Value audio_done(Value::Object{});
        audio_done["type"] = "response.output_audio.done";
        add_response_coordinates(audio_done);
        send(std::move(audio_done));

        Value part_done(Value::Object{});
        part_done["type"] = "response.content_part.done";
        add_response_coordinates(part_done);
        Value part(Value::Object{});
        part["type"] = "audio";
        part_done["part"] = std::move(part);
        send(std::move(part_done));

        Value item_done(Value::Object{});
        item_done["type"] = "response.output_item.done";
        item_done["response_id"] = response_id_;
        item_done["output_index"] = 0;
        Value item(Value::Object{});
        item["id"] = output_item_id_;
        item["object"] = "realtime.item";
        item["type"] = "message";
        item["role"] = "assistant";
        item_done["item"] = std::move(item);
        send(std::move(item_done));

        Value done(Value::Object{});
        done["type"] = "response.done";
        Value response(Value::Object{});
        response["id"] = response_id_;
        response["object"] = "realtime.response";
        response["status"] = "completed";
        response["status_details"] = nullptr;
        response["output"] = Value::Array{};
        Value usage(Value::Object{});
        usage["input_tokens"] = 0;
        usage["output_tokens"] = 0;
        usage["total_tokens"] = 0;
        Value input_details(Value::Object{});
        input_details["cached_tokens"] = 0;
        usage["input_token_details"] = std::move(input_details);
        Value output_details(Value::Object{});
        output_details["text_tokens"] = 0;
        output_details["audio_tokens"] = 0;
        usage["output_token_details"] = std::move(output_details);
        response["usage"] = std::move(usage);
        done["response"] = std::move(response);
        send(std::move(done));
        response_active_ = false;
        response_id_.clear();
        output_item_id_.clear();
    }

    void send_session_end() {
        Value event(Value::Object{});
        event["type"] = "session.end";
        Value stats(Value::Object{});
        stats["chunks_received"] = static_cast<double>(chunks_received_);
        stats["chunks_sent"] = static_cast<double>(chunks_sent_);
        stats["chunks_dropped"] = 0;
        stats["triton_inferences"] = static_cast<double>(inferences_);
        const double seconds =
            static_cast<double>(input_samples_received_) / static_cast<double>(input_rate_);
        stats["audio_duration_received_s"] = std::round(seconds * 100.0) / 100.0;
        event["stats"] = std::move(stats);
        send(std::move(event));
    }

    httplib::ws::WebSocket& socket_;
    std::shared_ptr<s2s::VoiceChat> voicechat_;
    const ServerConfig& config_;
    std::atomic<uint64_t>& ids_;
    std::string session_id_;
    int input_rate_ = kDefaultClientRate;
    audio::AudioResampler input_resampler_;
    audio::AudioResampler output_resampler_;
    std::unique_ptr<s2s::S2SStream> stream_;
    bool prompt_applied_ = false;
    Value tools_ = Value::Array{};
    std::string instructions_ = kDefaultInstructions;
    std::deque<std::string> function_responses_;
    std::vector<float> model_input_;
    std::vector<float> output_buffer_;
    bool response_active_ = false;
    bool speech_active_ = false;
    bool previous_agent_turn_nonempty_ = false;
    bool agent_separator_pending_ = false;
    std::string response_id_;
    std::string output_item_id_;
    std::string input_item_id_;
    std::string user_transcript_;
    std::string agent_transcript_;
    size_t audio_bytes_received_ = 0;
    uint64_t input_samples_received_ = 0;
    uint64_t chunks_received_ = 0;
    uint64_t chunks_sent_ = 0;
    uint64_t inferences_ = 0;
};

}  // namespace

std::string
select_s2s_subprotocol(const std::vector<std::string>& protocols) {
    for (const auto& protocol : protocols)
        if (protocol.rfind("function-id", 0) == 0)
            return protocol;
    return {};
}

void
handle_s2s_realtime(
    const httplib::Request& request, httplib::ws::WebSocket& socket,
    std::shared_ptr<s2s::VoiceChat> voicechat, const ServerConfig& config,
    std::atomic<uint64_t>& event_ids) {
    if (!config.api_key.empty()) {
        const std::string expected = "Bearer " + config.api_key;
        const bool authorized =
            request.get_header_value("Authorization") == expected ||
            (request.has_param("api_key") && request.get_param_value("api_key") == config.api_key);
        if (!authorized) {
            socket.close(httplib::ws::CloseStatus::PolicyViolation, "invalid bearer token");
            return;
        }
    }

    RealtimeSession session(socket, std::move(voicechat), config, event_ids);
    if (!session.start())
        return;
    bool graceful_close = false;
    bool internal_failure = false;
    std::string message;
    for (;;) {
        const auto kind = socket.read(message);
        if (kind == httplib::ws::ReadResult::Fail)
            break;
        try {
            const bool keep_running = kind == httplib::ws::ReadResult::Binary
                                          ? session.handle_binary(message)
                                          : session.handle_text(message, graceful_close);
            if (!keep_running)
                break;
            if (session.duration_exceeded()) {
                session.emit_error(
                    "Session duration limit reached", "session_timeout", "invalid_request_error");
                break;
            }
        }
        catch (const std::invalid_argument& error) {
            session.emit_error(error.what());
        }
        catch (const std::exception& error) {
            std::fprintf(stderr, "realtime VoiceChat inference failed: %s\n", error.what());
            session.emit_error(error.what(), "inference_error", "internal_error");
            internal_failure = true;
            break;
        }
    }
    try {
        session.finish(graceful_close && !internal_failure);
    }
    catch (const std::exception& error) {
        std::fprintf(stderr, "realtime VoiceChat finalization failed: %s\n", error.what());
        session.emit_error(error.what(), "inference_error", "internal_error");
        internal_failure = true;
    }
    if (socket.is_open())
        socket.close(
            internal_failure ? httplib::ws::CloseStatus::InternalError
                             : httplib::ws::CloseStatus::Normal,
            internal_failure ? "VoiceChat inference failed" : "");
}

}  // namespace nemo_speech::http
