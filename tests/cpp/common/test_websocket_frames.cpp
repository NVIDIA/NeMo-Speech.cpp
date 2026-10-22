// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// WebSocket frames that queue up faster than the server reads them. The server
// reads its socket through a 4 KiB buffer, so once frames queue up a frame
// header field can straddle the end of what the buffer holds and one read
// returns part of it. cpp-httplib before 0.59 rejected such a frame and dropped
// the connection without a close frame; realtime clients saw it as a transport
// drop under load.
//
// A real server and a raw TCP client. After the upgrade the client writes, in
// one burst, a lead frame whose length shifts where the 4 KiB boundaries fall,
// then the masked 640-byte binary frames the realtime endpoints receive (20 ms
// of 16 kHz s16le). The handler reads only after the whole burst was written,
// as a server that fell behind does, and must receive every frame, at every
// alignment.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"

namespace {

constexpr size_t kPayload = 640;
constexpr int kFrames = 64;
constexpr size_t kFrameSize = 2 + 2 + 4 + kPayload;  // header, 16-bit length, mask, payload

void
append_frame(std::string& out, size_t payload_len) {
    const char mask[4] = {0x11, 0x22, 0x33, 0x44};
    out.push_back(static_cast<char>(0x82));        // FIN + binary
    out.push_back(static_cast<char>(0x80 | 126));  // masked, 16-bit length follows
    out.push_back(static_cast<char>((payload_len >> 8) & 0xFF));
    out.push_back(static_cast<char>(payload_len & 0xFF));
    out.append(mask, 4);
    for (size_t i = 0; i < payload_len; ++i)
        out.push_back(static_cast<char>((i & 0xFF) ^ mask[i % 4]));
}

bool
write_all(int fd, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
        const ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, 0);
        if (n <= 0)
            return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

// Connects, upgrades, and returns the socket, or -1.
int
open_websocket(int port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (fd < 0 || ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        if (fd >= 0)
            ::close(fd);
        return -1;
    }
    const std::string upgrade =
        "GET /ws HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n";
    std::string response;
    char c;
    if (!write_all(fd, upgrade))
        response.clear();
    else
        while (response.find("\r\n\r\n") == std::string::npos && ::recv(fd, &c, 1, 0) == 1)
            response.push_back(c);
    if (response.rfind("HTTP/1.1 101", 0) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

}  // namespace

int
main() {
    std::atomic<bool> burst_written{false};
    std::atomic<int> received{-1};

    httplib::Server server;
    server.WebSocket("/ws", [&](const httplib::Request&, httplib::ws::WebSocket& socket) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!burst_written.load() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        int frames = 0;
        std::string message;
        while (frames < kFrames + 1 && socket.read(message) == httplib::ws::ReadResult::Binary)
            ++frames;
        received.store(frames);
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        std::cerr << "FAIL: could not bind a port\n";
        return 1;
    }
    std::thread listener([&] { server.listen_after_bind(); });
    server.wait_until_ready();

    // Every phase of the frame period against the 4 KiB buffer.
    int failed = 0;
    bool upgraded = true;
    for (size_t lead = 8; lead < 8 + kFrameSize; ++lead) {
        burst_written.store(false);
        received.store(-1);
        const int fd = open_websocket(port);
        if (fd < 0) {
            upgraded = false;
            break;
        }
        std::string burst;
        append_frame(burst, lead - 8);
        for (int f = 0; f < kFrames; ++f) append_frame(burst, kPayload);
        const bool written = write_all(fd, burst);
        burst_written.store(true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (received.load() < 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ::close(fd);
        if (!written || received.load() != kFrames + 1) {
            if (failed < 3)
                std::cerr << "lead " << lead << ": received " << received.load() << " of "
                          << kFrames + 1 << " frames\n";
            ++failed;
        }
    }

    server.stop();
    listener.join();
    if (!upgraded) {
        std::cerr << "FAIL: WebSocket upgrade failed\n";
        return 1;
    }
    if (failed != 0) {
        std::cerr << "FAIL: cpp-httplib " << CPPHTTPLIB_VERSION << " dropped frames at " << failed
                  << " of " << kFrameSize << " alignments\n";
        return 1;
    }
    std::cout << "PASS: cpp-httplib " << CPPHTTPLIB_VERSION << ", " << kFrameSize
              << " alignments, every frame received\n";
    return 0;
}
