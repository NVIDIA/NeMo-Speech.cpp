// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_terminal.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <utility>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace nemo_speech::cli {
namespace {

std::string
format_live_duration(double seconds) {
    seconds = std::max(0.0, seconds);
    std::ostringstream out;
    if (seconds < 60.0) {
        out << std::fixed << std::setprecision(1) << seconds << 's';
    } else if (seconds < 3600.0) {
        const int minutes = static_cast<int>(seconds) / 60;
        out << minutes << "m " << std::fixed << std::setprecision(1) << seconds - minutes * 60
            << 's';
    } else {
        const int hours = static_cast<int>(seconds) / 3600;
        const int minutes = (static_cast<int>(seconds) / 60) % 60;
        out << hours << "h " << minutes << 'm';
    }
    return out.str();
}

std::string
format_live_timestamp(int milliseconds) {
    milliseconds = std::max(0, milliseconds);
    const int hours = milliseconds / 3600000;
    const int minutes = (milliseconds / 60000) % 60;
    const double seconds = (milliseconds % 60000) / 1000.0;
    std::ostringstream out;
    out << std::setfill('0');
    if (hours > 0)
        out << hours << ':' << std::setw(2) << minutes << ':';
    else
        out << minutes << ':';
    out << std::fixed << std::setprecision(1) << std::setw(4) << seconds;
    return out.str();
}

bool
stderr_is_terminal() {
#if defined(_WIN32)
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(fileno(stderr)) != 0;
#endif
}

bool
enable_terminal_controls() {
    if (!stderr_is_terminal())
        return false;
#if defined(_WIN32)
    const HANDLE handle = GetStdHandle(STD_ERROR_HANDLE);
    DWORD mode = 0;
    return handle != INVALID_HANDLE_VALUE && GetConsoleMode(handle, &mode) &&
           SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#else
    const char* term = std::getenv("TERM");
    return term == nullptr || std::string(term) != "dumb";
#endif
}

int
terminal_columns() {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &info))
        return info.srWindow.Right - info.srWindow.Left + 1;
#else
    winsize size{};
    if (ioctl(fileno(stderr), TIOCGWINSZ, &size) == 0 && size.ws_col > 0)
        return size.ws_col;
#endif
    return 100;
}

int
terminal_rows() {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &info))
        return info.srWindow.Bottom - info.srWindow.Top + 1;
#else
    winsize size{};
    if (ioctl(fileno(stderr), TIOCGWINSZ, &size) == 0 && size.ws_row > 0)
        return size.ws_row;
#endif
    return 24;
}

std::string
utf8_tail(const std::string& text, size_t max_bytes) {
    if (text.size() <= max_bytes)
        return text;
    if (max_bytes <= 3)
        return "...";
    size_t begin = text.size() - (max_bytes - 3);
    while (begin < text.size() && (static_cast<unsigned char>(text[begin]) & 0xc0) == 0x80) ++begin;
    return "..." + text.substr(begin);
}

std::vector<std::string>
wrap_live_text(const std::string& text, size_t max_bytes) {
    std::vector<std::string> lines;
    size_t begin = 0;
    while (begin < text.size()) {
        while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])))
            ++begin;
        if (begin == text.size())
            break;

        size_t end = std::min(text.size(), begin + std::max<size_t>(1, max_bytes));
        while (end > begin && end < text.size() &&
               (static_cast<unsigned char>(text[end]) & 0xc0) == 0x80)
            --end;
        if (end == begin)
            end = std::min(text.size(), begin + std::max<size_t>(1, max_bytes));
        if (end < text.size()) {
            const size_t space = text.rfind(' ', end - 1);
            if (space != std::string::npos && space > begin)
                end = space;
        }
        size_t line_end = end;
        while (line_end > begin && std::isspace(static_cast<unsigned char>(text[line_end - 1])))
            --line_end;
        lines.push_back(text.substr(begin, line_end - begin));
        begin = end;
    }
    if (lines.empty() && !text.empty())
        lines.push_back(text);
    return lines;
}

}  // namespace

LiveTerminal::LiveTerminal(bool visible, bool diarize)
    : visible_(visible), diarize_(diarize), interactive_(visible && enable_terminal_controls()),
      color_(interactive_ && std::getenv("NO_COLOR") == nullptr) {}

void
LiveTerminal::start(const std::string& device, int sample_rate) {
    if (!visible_)
        return;
    if (!interactive_) {
        std::fprintf(
            stderr, "[live] listening on \"%s\" at %d Hz; press Ctrl-C to stop\n", device.c_str(),
            sample_rate);
        return;
    }
    std::fprintf(
        stderr, "\n  %sLIVE%s  %s\n  %s%d kHz microphone - Ctrl-C to stop%s\n\n",
        style("\x1b[1;32m"), style("\x1b[0m"), device.c_str(), style("\x1b[2m"), sample_rate / 1000,
        style("\x1b[0m"));
}

void
LiveTerminal::partial(const std::string& text) {
    if (!interactive_ || text.empty())
        return;
    clear_current_turn();
    const int width = std::max(20, terminal_columns());
    const std::string shown = utf8_tail(text, static_cast<size_t>(std::max(12, width - 6)));
    if (partial_visible_ && shown == partial_text_)
        return;
    std::fprintf(stderr, "\r\x1b[2K%s  > %s%s", style("\x1b[2m"), shown.c_str(), style("\x1b[0m"));
    std::fflush(stderr);
    partial_visible_ = true;
    partial_text_ = shown;
}

void
LiveTerminal::current_turn(const subtitle::SpeakerTurn& turn, double fallback_seconds) {
    preview_turns({turn}, fallback_seconds, false);
}

void
LiveTerminal::current_turns(
    const std::vector<subtitle::SpeakerTurn>& turns, double fallback_seconds) {
    preview_turns(turns, fallback_seconds, false);
}

void
LiveTerminal::partial_turns(
    const std::vector<subtitle::SpeakerTurn>& turns, double fallback_seconds) {
    preview_turns(turns, fallback_seconds, true);
}

void
LiveTerminal::preview_turns(
    const std::vector<subtitle::SpeakerTurn>& turns, double fallback_seconds, bool provisional) {
    if (!interactive_)
        return;
    std::vector<subtitle::SpeakerTurn> shown;
    for (auto turn : turns) {
        if (turn.text.empty())
            continue;
        if (diarize_) {
            if (turn.speaker > 0)
                current_speaker_ = turn.speaker;
            else
                turn.speaker = current_speaker_;
            if (!shown.empty() && shown.back().speaker == turn.speaker) {
                auto& previous = shown.back();
                if (!subtitle::attaches_to_previous(turn.text) &&
                    !std::isspace(static_cast<unsigned char>(previous.text.back())) &&
                    !std::isspace(static_cast<unsigned char>(turn.text.front())))
                    previous.text += ' ';
                previous.text += turn.text;
                previous.start_ms = std::min(previous.start_ms, turn.start_ms);
                previous.end_ms = std::max(previous.end_ms, turn.end_ms);
                continue;
            }
        }
        shown.push_back(std::move(turn));
    }
    if (diarize_ && shown.size() == 1 && shown.front().speaker <= 0) {
        // Before the first identified speaker, use the ordinary partial line.
        partial(shown.front().text);
        return;
    }
    std::vector<std::string> lines;
    for (const auto& turn : shown) {
        if (turn.text.empty())
            continue;
        auto next = turn_lines(turn, fallback_seconds, provisional);
        lines.insert(lines.end(), next.begin(), next.end());
    }
    // A replaceable preview must fit on screen; cursor-up cannot erase
    // scrollback. Final output always prints the complete canonical text.
    const size_t limit = static_cast<size_t>(std::max(1, terminal_rows() - 4));
    if (lines.size() > limit) {
        lines.erase(lines.begin(), lines.end() - static_cast<std::ptrdiff_t>(limit));
        lines.front() = "  " + utf8_tail(
                                   "... (earlier text hidden while updating)",
                                   static_cast<size_t>(std::max(1, terminal_columns() - 2)));
    }
    if (!partial_visible_ && lines == current_preview_)
        return;
    clear_interim();
    for (const auto& line : lines) std::fprintf(stderr, "%s\n", line.c_str());
    std::fflush(stderr);
    current_turn_lines_ = lines.size();
    current_preview_ = std::move(lines);
}

void
LiveTerminal::final_turn(const subtitle::SpeakerTurn& turn, double fallback_seconds) {
    if (!visible_ || turn.text.empty())
        return;
    clear_interim();
    if (turn.speaker > 0)
        current_speaker_ = turn.speaker;
    if (!interactive_) {
        if (turn.speaker > 0)
            std::fprintf(
                stderr, "[live final @ %.2fs, speaker %d] %s\n", fallback_seconds, turn.speaker,
                turn.text.c_str());
        else
            std::fprintf(stderr, "[live final @ %.2fs] %s\n", fallback_seconds, turn.text.c_str());
        return;
    }
    const auto lines = turn_lines(turn, fallback_seconds, false);
    for (const auto& line : lines) std::fprintf(stderr, "%s\n", line.c_str());
    std::fflush(stderr);
    last_final_open_ = true;
    last_final_ = turn;
    last_final_seconds_ = fallback_seconds;
    last_final_lines_ = lines.size();
}

bool
LiveTerminal::amend_last_final(const std::string& punctuation) {
    if (!interactive_ || !last_final_open_ || punctuation.empty() ||
        last_final_lines_ + 2 > static_cast<size_t>(terminal_rows()))
        return false;
    clear_interim();
    for (size_t i = 0; i < last_final_lines_; ++i) std::fputs("\x1b[1A\r\x1b[2K", stderr);
    last_final_.text += punctuation;
    const auto lines = turn_lines(last_final_, last_final_seconds_, false);
    for (const auto& line : lines) std::fprintf(stderr, "%s\n", line.c_str());
    std::fflush(stderr);
    last_final_lines_ = lines.size();
    return true;
}

void
LiveTerminal::stopped(double audio_seconds) {
    if (!visible_)
        return;
    last_final_open_ = false;
    clear_partial();
    if (interactive_)
        std::fprintf(
            stderr, "\n  %sStopped%s after %s of captured audio\n", style("\x1b[1m"),
            style("\x1b[0m"), format_live_duration(audio_seconds).c_str());
    else
        std::fprintf(
            stderr, "[live] stopped after %s of captured audio\n",
            format_live_duration(audio_seconds).c_str());
}

const char*
LiveTerminal::speaker_style(int speaker) const {
    if (!color_)
        return "";
    static const char* colors[]{"\x1b[1;36m", "\x1b[1;32m", "\x1b[1;35m", "\x1b[1;33m"};
    return colors[static_cast<size_t>(std::max(1, speaker) - 1) % 4];
}

std::vector<std::string>
LiveTerminal::turn_lines(
    const subtitle::SpeakerTurn& turn, double fallback_seconds, bool provisional) const {
    const int fallback_ms = static_cast<int>(fallback_seconds * 1000.0 + 0.5);
    const int start_ms = turn.end_ms > turn.start_ms ? turn.start_ms : fallback_ms;
    const int end_ms = turn.end_ms > turn.start_ms ? turn.end_ms : fallback_ms;
    // Unknown final tags stay unknown; only the replaceable preview inherits
    // the active display speaker. Transcript status is not a speaker identity.
    const std::string label = turn.speaker > 0 ? "Speaker " + std::to_string(turn.speaker)
                              : provisional    ? "Partial"
                              : diarize_       ? "Transcript"
                                               : "Final";
    const int width = std::max(4, terminal_columns());
    const auto lines = wrap_live_text(turn.text, static_cast<size_t>(width - 2));
    std::ostringstream header;
    header << std::left << std::setw(10) << label << "  " << format_live_timestamp(start_ms)
           << " - " << format_live_timestamp(end_ms) << (provisional ? " (partial)" : "");
    std::vector<std::string> result{""};
    for (const auto& line : wrap_live_text(header.str(), static_cast<size_t>(width - 2)))
        result.push_back(
            "  " + std::string(speaker_style(turn.speaker)) + style("\x1b[1m") + line +
            style("\x1b[0m"));
    for (const auto& line : lines) result.push_back("  " + line);
    return result;
}

void
LiveTerminal::clear_current_turn() {
    if (current_turn_lines_ == 0)
        return;
    for (size_t i = 0; i < current_turn_lines_; ++i) std::fputs("\x1b[1A\r\x1b[2K", stderr);
    std::fflush(stderr);
    current_turn_lines_ = 0;
    current_preview_.clear();
}

void
LiveTerminal::clear_partial() {
    if (!partial_visible_)
        return;
    std::fputs("\r\x1b[2K", stderr);
    partial_visible_ = false;
    partial_text_.clear();
}

}  // namespace nemo_speech::cli
