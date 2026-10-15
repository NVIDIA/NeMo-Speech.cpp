// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "native_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "json.h"
#include "pcm_conversion.h"

#ifndef _WIN32
#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace nemo_speech::tts::nemotron {
namespace {

using Clock = std::chrono::steady_clock;
using Value = json::Value;
constexpr size_t kMaxFrameBytes = 16 * 1024 * 1024;
constexpr size_t kMaxQueuedBytes = 4 * 1024 * 1024;
constexpr size_t kMaxQueuedChunks = 256;
constexpr size_t kMaxPendingRequests = 1024;
constexpr uint64_t kMaxJsonInteger = 9007199254740991ULL;

uint64_t
integer(const Value& value, const std::string& key, uint64_t maximum = kMaxJsonInteger) {
    const auto& field = value.at(key);
    if (!field.is_number() || !std::isfinite(field.number()) || field.number() < 0 ||
        std::floor(field.number()) != field.number() || field.number() > maximum)
        throw std::runtime_error("invalid native worker integer: " + key);
    return static_cast<uint64_t>(field.number());
}

Value
control(const char* kind, uint64_t request_id) {
    Value value(Value::Object{});
    value["kind"] = kind;
    value["request_id"] = static_cast<double>(request_id);
    return value;
}

uint64_t
read_be(const char* bytes, size_t count) {
    uint64_t value = 0;
    for (size_t i = 0; i < count; ++i) value = (value << 8) | static_cast<unsigned char>(bytes[i]);
    return value;
}

double
milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

#ifndef _WIN32
class TextPreparation {
   public:
    explicit TextPreparation(const Value& metadata) {
        if (!metadata.is_object() || metadata.object().size() != 3)
            throw std::runtime_error("invalid native text preparation metadata");
        const auto library = metadata.at("library").string();
        const auto tokenizer = metadata.at("tokenizer").string();
        const auto config = metadata.at("normalization_config").string();
        for (const auto* path : {&library, &tokenizer}) {
            if (path->empty() || path->front() != '/' || path->find('\0') != std::string::npos)
                throw std::runtime_error("native preparation paths must be absolute");
        }
        if (config.empty() || config.size() > 1024 * 1024 || config.find('\0') != std::string::npos)
            throw std::runtime_error("invalid native preparation configuration");
        library_ = ::dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!library_)
            throw std::runtime_error("cannot load native text preparation library");
        try {
            auto create = symbol<Create>("nemotron_text_create");
            prepare_ = symbol<Prepare>("nemotron_text_prepare");
            free_ = symbol<Free>("nemotron_text_free");
            destroy_ = symbol<Destroy>("nemotron_text_destroy");
            char* error = nullptr;
            const int status = create(config.c_str(), tokenizer.c_str(), &handle_, &error);
            std::unique_ptr<char, Free> owned_error(error, free_);
            check(status, error);
            if (!handle_)
                throw std::runtime_error("native preparation returned an empty handle");
        }
        catch (...) {
            release();
            throw;
        }
    }

    ~TextPreparation() { release(); }
    TextPreparation(const TextPreparation&) = delete;
    TextPreparation& operator=(const TextPreparation&) = delete;

    Value prepare(const std::string& input, bool normalize) const {
        char* output = nullptr;
        char* error = nullptr;
        const int status = prepare_(
            handle_, reinterpret_cast<const uint8_t*>(input.data()), input.size(), normalize,
            &output, &error);
        std::unique_ptr<char, Free> owned_output(output, free_);
        std::unique_ptr<char, Free> owned_error(error, free_);
        check(status, error);
        if (!output || ::strnlen(output, kMaxFrameBytes) >= kMaxFrameBytes)
            throw std::runtime_error("native preparation returned an invalid result");
        auto result = Value::parse(output);
        const auto& text = result.at("text").string();
        const auto& tokens = result.at("tokens").array();
        if (text.empty() || text.size() > kMaxFrameBytes || tokens.empty() || tokens.size() > 512)
            throw std::runtime_error("native preparation returned invalid text or token count");
        for (const auto& token : tokens) {
            if (!token.is_number() || !std::isfinite(token.number()) || token.number() < 0 ||
                token.number() >= 131075 || token.number() != std::floor(token.number()))
                throw std::runtime_error("native preparation returned an invalid token ID");
        }
        if (tokens.back().number() != 131073)
            throw std::runtime_error("native preparation omitted the text EOS token");
        return result;
    }

   private:
    using Create = int (*)(const char*, const char*, void**, char**);
    using Prepare = int (*)(void*, const uint8_t*, size_t, uint8_t, char**, char**);
    using Free = void (*)(char*);
    using Destroy = void (*)(void*);

    template <class Function>
    Function symbol(const char* name) {
        auto address = ::dlsym(library_, name);
        if (!address)
            throw std::runtime_error(std::string("native preparation missing symbol: ") + name);
        return reinterpret_cast<Function>(address);
    }

    static void check(int status, const char* error) {
        if (status == 0)
            return;
        const std::string message = error ? error : "native text preparation failed";
        if (status == 1)
            throw std::invalid_argument(message);
        throw std::runtime_error(message);
    }

    void release() {
        if (handle_ && destroy_)
            destroy_(handle_);
        handle_ = nullptr;
        if (library_)
            ::dlclose(library_);
        library_ = nullptr;
    }

    void* library_ = nullptr;
    void* handle_ = nullptr;
    Prepare prepare_ = nullptr;
    Free free_ = nullptr;
    Destroy destroy_ = nullptr;
};
#endif

}  // namespace

#ifndef _WIN32

struct NativeRuntime::Impl {
    struct Request {
        std::mutex mutex;
        std::condition_variable changed;
        std::deque<std::string> chunks;
        size_t queued_bytes = 0;
        uint64_t received_samples = 0;
        uint64_t received_chunks = 0;
        bool accepted = false;
        bool complete = false;
        bool discard = false;
        std::string error;
        std::string processed_text;
        size_t token_count = 0;
    };

    explicit Impl(const std::string& socket_path, int seconds) : timeout_seconds(seconds) {
        if (seconds <= 0)
            throw std::invalid_argument("native_timeout_seconds must be positive");
        if (socket_path.empty())
            throw std::invalid_argument("Nemotron backend requires native_socket");
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        if (socket_path.size() >= sizeof(address.sun_path))
            throw std::invalid_argument("native_socket path is too long");
        std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
        fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0)
            throw std::runtime_error("cannot create native worker socket");
        try {
            if (::fcntl(fd, F_SETFD, FD_CLOEXEC) < 0 || ::fcntl(fd, F_SETFL, O_NONBLOCK) < 0)
                throw std::runtime_error("cannot configure native worker socket");
#ifdef SO_NOSIGPIPE
            int enabled = 1;
            if (::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)) < 0)
                throw std::runtime_error("cannot configure native worker signal handling");
#endif
            const auto deadline = Clock::now() + std::chrono::seconds(timeout_seconds);
            if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
                if (errno != EINPROGRESS && errno != EAGAIN)
                    throw std::runtime_error("cannot connect to native worker: " + socket_path);
                wait_socket(POLLOUT, deadline);
                int error = 0;
                socklen_t size = sizeof(error);
                if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) < 0 || error != 0)
                    throw std::runtime_error("cannot connect to native worker: " + socket_path);
            }
            Value hello(Value::Object{});
            hello["kind"] = "hello";
            hello["protocol_version"] = 1;
            send(hello, deadline);
            const auto frame = read_frame(deadline);
            if (frame.front() != 'J')
                throw std::runtime_error("native worker did not send ready metadata");
            const auto ready = Value::parse(frame.substr(1));
            if (ready.string_or("kind") != "ready")
                throw std::runtime_error(
                    "native worker handshake failed: " + ready.string_or("message", "not ready"));
            name = ready.at("model_name").string();
            if (const auto* version = ready.find("protocol_version");
                version && integer(ready, "protocol_version") != 1)
                throw std::runtime_error("native worker protocol version is unsupported");
            if (name != "nemotron-tts" || integer(ready, "sample_rate") != 22050)
                throw std::runtime_error("native worker model or sample rate is unsupported");
            for (const auto& voice : ready.at("voices").array()) voices.push_back(voice.string());
            if (voices != std::vector<std::string>{"english_sample"})
                throw std::runtime_error("native worker voices are unsupported");
            width = static_cast<int>(integer(ready, "batch_size", 4096));
            if (width < 1)
                throw std::runtime_error("native worker batch_size must be positive");
            execution_mode = ready.at("mode").string();
            if (execution_mode.empty())
                throw std::runtime_error("native worker mode is empty");
            normalizes = ready.at("text_normalization_enabled").boolean();
            if (const auto* preparation = ready.find("text_preparation"))
                text_preparation = std::make_unique<TextPreparation>(*preparation);
            reader = std::thread([this] { read_loop(); });
        }
        catch (...) {
            ::close(fd);
            fd = -1;
            throw;
        }
    }

    ~Impl() {
        fail_connection("native worker connection closed");
        if (reader.joinable())
            reader.join();
        if (fd >= 0)
            ::close(fd);
    }

    void wait_socket(short events, Clock::time_point deadline) {
        while (true) {
            int timeout = -1;
            if (deadline != Clock::time_point::max()) {
                const auto remaining =
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now())
                        .count();
                if (remaining <= 0)
                    throw std::runtime_error("native worker socket timed out");
                timeout = static_cast<int>(std::min<int64_t>(remaining, 2147483647));
            }
            pollfd descriptor{fd, events, 0};
            const int result = ::poll(&descriptor, 1, timeout);
            if (result < 0 && errno == EINTR)
                continue;
            if (result == 0)
                throw std::runtime_error("native worker socket timed out");
            if (result < 0 || descriptor.revents & (POLLERR | POLLNVAL))
                throw std::runtime_error("native worker socket failed");
            if (descriptor.revents & (events | POLLHUP))
                return;
        }
    }

    void read_exact(char* output, size_t bytes, Clock::time_point deadline) {
        while (bytes > 0) {
            wait_socket(POLLIN, deadline);
            const auto received = ::recv(fd, output, bytes, 0);
            if (received < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
                continue;
            if (received <= 0)
                throw std::runtime_error("native worker disconnected");
            output += received;
            bytes -= static_cast<size_t>(received);
        }
    }

    std::string read_frame(Clock::time_point deadline) {
        char header[4];
        read_exact(header, sizeof(header), deadline);
        const auto length = read_be(header, sizeof(header));
        if (length < 2 || length > kMaxFrameBytes)
            throw std::runtime_error("invalid native worker frame length");
        std::string frame(static_cast<size_t>(length), '\0');
        read_exact(frame.data(), frame.size(), deadline);
        return frame;
    }

    void send(const Value& value, Clock::time_point deadline) {
        const std::string payload = "J" + value.dump();
        if (payload.size() > kMaxFrameBytes)
            throw std::invalid_argument("native worker request is too large");
        std::string frame(4, '\0');
        const auto length = static_cast<uint32_t>(payload.size());
        for (size_t i = 0; i < 4; ++i) frame[i] = static_cast<char>(length >> (24 - 8 * i));
        frame += payload;
        std::unique_lock<std::timed_mutex> lock(writes, std::defer_lock);
        if (!lock.try_lock_until(deadline))
            throw std::runtime_error("native worker write timed out");
        size_t offset = 0;
        while (offset < frame.size()) {
            wait_socket(POLLOUT, deadline);
#ifdef MSG_NOSIGNAL
            constexpr int flags = MSG_NOSIGNAL;
#else
            constexpr int flags = 0;
#endif
            const auto written = ::send(fd, frame.data() + offset, frame.size() - offset, flags);
            if (written < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
                continue;
            if (written <= 0)
                throw std::runtime_error("native worker write failed");
            offset += static_cast<size_t>(written);
        }
    }

    void fail_connection(const std::string& reason) {
        std::lock_guard<std::mutex> lock(pending_mutex);
        if (connection_error.empty())
            connection_error = reason;
        ::shutdown(fd, SHUT_RDWR);
        for (const auto& entry : pending) {
            auto& request = *entry.second;
            std::lock_guard<std::mutex> request_lock(request.mutex);
            request.error = connection_error;
            request.changed.notify_all();
        }
        pending.clear();
    }

    std::shared_ptr<Request> lookup(uint64_t id) {
        std::lock_guard<std::mutex> lock(pending_mutex);
        const auto found = pending.find(id);
        if (found == pending.end())
            throw std::runtime_error("native worker returned an unknown request ID");
        return found->second;
    }

    void retire(uint64_t id) {
        std::lock_guard<std::mutex> lock(pending_mutex);
        pending.erase(id);
    }

    void receive_audio(const std::string& frame) {
        const size_t sample_bytes = frame.front() == 'F' ? 4 : 2;
        if (frame.size() < 13 + sample_bytes || (frame.size() - 13) % sample_bytes != 0)
            throw std::runtime_error("native worker returned malformed audio");
        const uint64_t id = read_be(frame.data() + 1, 8);
        const uint64_t sequence = read_be(frame.data() + 9, 4);
        auto request = lookup(id);
        std::lock_guard<std::mutex> lock(request->mutex);
        if (!request->accepted || request->complete || sequence != request->received_chunks)
            throw std::runtime_error("native worker audio sequence is invalid");
        const size_t samples = (frame.size() - 13) / sample_bytes;
        const size_t bytes = samples * 2;
        ++request->received_chunks;
        request->received_samples += samples;
        if (request->discard)
            return;
        if (bytes > kMaxQueuedBytes - request->queued_bytes ||
            request->chunks.size() >= kMaxQueuedChunks) {
            request->error = "native worker audio consumer exceeded its bounded queue";
            request->discard = true;
            request->chunks.clear();
            request->queued_bytes = 0;
        } else {
            request->chunks.emplace_back(
                sample_bytes == 4 ? pcm_f32_to_s16(frame.data() + 13, frame.size() - 13)
                                  : frame.substr(13));
            request->queued_bytes += bytes;
        }
        request->changed.notify_all();
    }

    void receive_control(const Value& value) {
        if (const auto* version = value.find("protocol_version");
            version && integer(value, "protocol_version") != 1)
            throw std::runtime_error("native worker protocol version is unsupported");
        const uint64_t id = integer(value, "request_id");
        const auto kind = value.at("kind").string();
        auto request = lookup(id);
        bool terminal = false;
        {
            std::lock_guard<std::mutex> lock(request->mutex);
            if (kind == "accepted") {
                if (request->accepted || request->complete)
                    throw std::runtime_error("native worker accepted a request twice");
                request->processed_text = value.at("processed_text").string();
                request->token_count = static_cast<size_t>(integer(value, "token_count", 4096));
                if (request->processed_text.empty() || request->token_count == 0)
                    throw std::runtime_error("native worker accepted empty text or tokens");
                request->accepted = true;
            } else if (kind == "complete") {
                if (!request->discard &&
                    (!request->accepted || request->received_samples == 0 ||
                     integer(value, "total_samples") != request->received_samples ||
                     integer(value, "chunk_count") != request->received_chunks))
                    throw std::runtime_error("native worker completion does not match audio");
                request->complete = true;
                terminal = true;
            } else if (kind == "failed") {
                request->error = value.at("message").string();
                if (request->error.empty())
                    request->error = "native worker synthesis failed";
                request->complete = true;
                terminal = true;
            } else if (kind == "aborted") {
                request->error = "native worker synthesis aborted";
                request->complete = true;
                terminal = true;
            } else {
                throw std::runtime_error("unsupported native worker message: " + kind);
            }
            request->changed.notify_all();
        }
        if (terminal)
            retire(id);
    }

    void read_loop() {
        try {
            while (true) {
                wait_socket(POLLIN, Clock::time_point::max());
                const auto frame = read_frame(Clock::now() + std::chrono::seconds(timeout_seconds));
                if (frame.front() == 'A' || frame.front() == 'F')
                    receive_audio(frame);
                else if (frame.front() == 'J')
                    receive_control(Value::parse(frame.substr(1)));
                else
                    throw std::runtime_error("unsupported native worker frame type");
            }
        }
        catch (const std::exception& error) {
            fail_connection(error.what());
        }
    }

    void abort(uint64_t id, const std::shared_ptr<Request>& request) {
        {
            std::lock_guard<std::mutex> lock(request->mutex);
            request->discard = true;
            request->chunks.clear();
            request->queued_bytes = 0;
            if (request->complete)
                return;
        }
        try {
            send(control("abort", id), Clock::now() + std::chrono::seconds(timeout_seconds));
        }
        catch (const std::exception& error) {
            fail_connection(error.what());
        }
    }

    SynthesisResult synthesize(
        const PreparedSynthesis& prepared, const Synthesizer::PcmCallback& callback) {
        const auto started = Clock::now();
        const auto deadline = started + std::chrono::seconds(timeout_seconds);
        Value prepared_text;
        if (text_preparation) {
            prepared_text = text_preparation->prepare(
                prepared.metadata.original_text, prepared.apply_normalize != 0);
            if (Clock::now() >= deadline)
                throw std::runtime_error("native text preparation timed out");
        }
        std::shared_ptr<Request> request;
        uint64_t id;
        {
            std::lock_guard<std::mutex> lock(pending_mutex);
            if (!connection_error.empty())
                throw std::runtime_error(connection_error);
            if (pending.size() >= std::max(kMaxPendingRequests, static_cast<size_t>(width) * 2) ||
                next_id >= kMaxJsonInteger)
                throw std::runtime_error("native worker request capacity exhausted");
            id = next_id++;
            request = std::make_shared<Request>();
            pending.emplace(id, request);
        }
        Value submit = control(text_preparation ? "submit_prepared" : "submit", id);
        submit["text"] =
            text_preparation ? prepared_text.at("text") : Value(prepared.metadata.original_text);
        if (text_preparation)
            submit["token_ids"] = std::move(prepared_text["tokens"]);
        submit["voice"] = "english_sample";
        submit["language"] = prepared.metadata.language_code;
        submit["seed"] = prepared.options.seed < 0 ? 0 : prepared.options.seed;
        if (prepared.apply_normalize >= 0)
            submit["apply_normalize"] = prepared.apply_normalize != 0;
        try {
            send(submit, deadline);
        }
        catch (const std::exception& error) {
            fail_connection(error.what());
            throw;
        }
        SynthesisResult result;
        result.metadata = prepared.metadata;
        auto& stats = result.stats;
        stats.sample_rate = 22050;
        Clock::time_point last_audio{};
        std::vector<double> gaps;
        try {
            while (true) {
                std::string pcm;
                {
                    std::unique_lock<std::mutex> lock(request->mutex);
                    if (!request->changed.wait_until(lock, deadline, [&] {
                            return !request->error.empty() || !request->chunks.empty() ||
                                   request->complete;
                        }))
                        throw std::runtime_error("native worker synthesis timed out");
                    if (!request->error.empty())
                        throw std::runtime_error(request->error);
                    result.metadata.processed_text = request->processed_text;
                    result.metadata.token_count = request->token_count;
                    if (request->chunks.empty())
                        break;
                    pcm = std::move(request->chunks.front());
                    request->chunks.pop_front();
                    request->queued_bytes -= pcm.size();
                }
                const auto now = Clock::now();
                if (last_audio == Clock::time_point{})
                    stats.ttfa_ms = stats.e2e_ttfa_ms = milliseconds(now - started);
                else
                    gaps.push_back(milliseconds(now - last_audio));
                last_audio = now;
                result.output_samples += pcm.size() / 2;
                ++stats.chunks;
                if (callback && !callback(result.metadata, pcm)) {
                    result.cancelled = true;
                    abort(id, request);
                    break;
                }
            }
        }
        catch (...) {
            abort(id, request);
            throw;
        }
        stats.e2e_chunks = stats.chunks;
        stats.samples_written = result.output_samples;
        stats.audio_s = static_cast<double>(result.output_samples) / 22050;
        stats.elapsed_s = std::chrono::duration<double>(Clock::now() - started).count();
        stats.rtf = stats.audio_s > 0 ? stats.elapsed_s / stats.audio_s : 0;
        stats.rtfx = stats.e2e_rtfx = stats.elapsed_s > 0 ? stats.audio_s / stats.elapsed_s : 0;
        if (!gaps.empty()) {
            double sum = 0;
            for (const double gap : gaps) sum += gap;
            std::sort(gaps.begin(), gaps.end());
            stats.icl_avg_ms = stats.e2e_icl_avg_ms = sum / gaps.size();
            stats.icl_min_ms = stats.e2e_icl_min_ms = gaps.front();
            stats.icl_max_ms = stats.e2e_icl_max_ms = gaps.back();
            stats.e2e_icl_p95_ms = gaps[static_cast<size_t>(std::ceil(gaps.size() * 0.95)) - 1];
            stats.e2e_icl_p99_ms = gaps[static_cast<size_t>(std::ceil(gaps.size() * 0.99)) - 1];
        }
        return result;
    }

    int fd = -1;
    int timeout_seconds;
    std::string name;
    std::vector<std::string> voices;
    bool normalizes = false;
    int width = 1;
    std::string execution_mode;
    std::unique_ptr<TextPreparation> text_preparation;
    std::thread reader;
    std::timed_mutex writes;
    std::mutex pending_mutex;
    std::map<uint64_t, std::shared_ptr<Request>> pending;
    std::string connection_error;
    uint64_t next_id = 1;
};

#else

struct NativeRuntime::Impl {
    Impl(const std::string&, int) {
        throw std::runtime_error("Nemotron native worker sockets require a Unix platform");
    }
    SynthesisResult synthesize(const PreparedSynthesis&, const Synthesizer::PcmCallback&) {
        throw std::runtime_error("Nemotron native worker sockets require a Unix platform");
    }
    std::string name;
    std::vector<std::string> voices;
    bool normalizes = false;
    int width = 1;
    std::string execution_mode;
};

#endif

NativeRuntime::NativeRuntime(const std::string& socket_path, int timeout_seconds)
    : impl_(std::make_unique<Impl>(socket_path, timeout_seconds)) {}

NativeRuntime::~NativeRuntime() = default;

SynthesisResult
NativeRuntime::synthesize(
    const PreparedSynthesis& request, const Synthesizer::PcmCallback& callback) {
    return impl_->synthesize(request, callback);
}

const std::string&
NativeRuntime::model_name() const {
    return impl_->name;
}

const std::vector<std::string>&
NativeRuntime::speaker_names() const {
    return impl_->voices;
}

bool
NativeRuntime::text_normalization_enabled() const {
    return impl_->normalizes;
}

int
NativeRuntime::batch_size() const {
    return impl_->width;
}

const std::string&
NativeRuntime::mode() const {
    return impl_->execution_mode;
}

}  // namespace nemo_speech::tts::nemotron
