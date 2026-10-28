// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Speech-LLM chat API: POST /v1/chat/completions (one user message with an input_audio part, the
// text optionally streamed as server-sent events), with the OpenAI error envelope. Requests queue
// in the recognizer (batching.max_queue_depth; a full queue is 429). The routing-independent core
// (Api) is testable without sockets.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "parameter_parser.h"

#if defined(NEMO_SPEECH_REGISTRY_ASR)
#include "types.h"
#endif

namespace httplib {
class Server;
}

namespace nemo_speech {
class EngineRegistry;
namespace http::chat {

struct Limits {
    // Also accepted as `model` (the recognizer's id, which GET /v1/models lists, always is).
    std::string model_name;
    int max_request_bytes = 268435456;  // whole JSON body; over it is 413
    int max_audio_seconds = 3600;

    void Register(common::ParameterParser& p) {
        p.Register("model-name", &model_name, "chat API: another accepted model id");
        p.Register("max-request-bytes", &max_request_bytes, "chat API: request body limit");
        p.Register("max-audio-seconds", &max_audio_seconds, "chat API: longest upload");
    }
};

struct ApiConfig {
    Limits limits;

    void Register(common::ParameterParser& p) { p.Register("chat", limits); }
};

// Register the chat API routes on `server`.
void install(httplib::Server& server, EngineRegistry& engines, const ApiConfig& config);

#if defined(NEMO_SPEECH_REGISTRY_ASR)
// Writes response bytes; false once the client is gone.
using StreamWriter = std::function<bool(std::string_view)>;

struct HttpResponse {
    int status = 200;
    std::vector<std::pair<std::string, std::string>> headers;  // content-type included
    std::string body;
    // A server-sent event stream (stream: true) instead of `body`: writes the events as they come
    // and returns false when the client went away.
    std::function<bool(const StreamWriter&)> stream;
};

// Request headers, looked up case-insensitively.
using HeaderLookup = std::function<bool(const std::string& name, std::string& value)>;
// Streams the request body into `sink` (false from the sink stops reading); false when reading
// failed.
using BodyReader = std::function<bool(const std::function<bool(const char*, size_t)>& sink)>;
// True once the client has disconnected. Polled (a few times a second) while the request waits for
// admission or its decode, and while a streamed response waits for text; a disconnect drops the
// request. It must stay callable until the response, including its stream, is done.
using ClientGone = std::function<bool()>;

// The recognizer and process facts the routes need; tests substitute fakes.
struct Backend {
    std::function<asr::Result(const float*, size_t, int, const asr::AsrRequestOptions&)> recognize;
    std::function<bool()> ready;                // models loaded and warm
    std::function<bool()> supports_timestamps;  // a CTC timestamp model is loaded
    // A speech-LLM is loaded: instructions and temperature apply (other models decode greedily).
    std::function<bool()> supports_instructions;
    std::function<std::string()> request_id;  // default "asr_<pid>_<millis>_<counter>"
    std::function<uint64_t()> unix_seconds;   // default the wall clock
    // The `model` accepted: the recognizer's id, which GET /v1/models lists.
    std::function<std::string()> model_alias;
};

class Api {
   public:
    Api(Limits limits, Backend backend);
    ~Api();
    Api(const Api&) = delete;
    Api& operator=(const Api&) = delete;

    // POST /v1/chat/completions.
    HttpResponse post(
        const HeaderLookup& headers, const BodyReader& body, const ClientGone& gone = {});

   private:
    struct State;
    std::unique_ptr<State> state_;
};
#endif

}  // namespace http::chat
}  // namespace nemo_speech
