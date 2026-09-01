// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "model_store.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cli_util.h"
#include "json.h"

#if defined(_WIN32)
#include <io.h>
#include <process.h>
#include <windows.h>
#elif defined(__APPLE__)
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
namespace fs = std::filesystem;
using nemo_speech::json::Value;

struct ArchiveMember {
    std::string name;
    std::string sha256;
    uint64_t size = 0;
};

struct Artifact {
    std::string role;
    std::string type;
    std::string filename;
    std::string revision;
    std::string directory;
    std::string sha256;
    uint64_t size = 0;
    uint64_t range_end = 0;
    std::string stop_before;
    std::vector<ArchiveMember> members;
};

struct Model {
    std::string repo;
    std::vector<std::string> aliases;
    std::string revision;
    std::string license;
    std::string license_url;
    std::vector<std::string> companions;
    std::vector<Artifact> artifacts;
};

struct Index {
    std::map<std::string, std::string> defaults;
    std::vector<Model> models;
};

std::string
path_utf8(const fs::path& path) {
    return path.u8string();
}

uint32_t
rotate_right(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32 - count));
}

class Sha256 {
   public:
    void update(const unsigned char* data, size_t size) {
        total_size_ += size;
        while (size > 0) {
            const size_t count = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, count);
            block_size_ += count;
            data += count;
            size -= count;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::string finish() {
        const uint64_t bits = total_size_ * 8;
        block_[block_size_++] = 0x80;
        if (block_size_ > 56) {
            std::fill(block_.begin() + static_cast<ptrdiff_t>(block_size_), block_.end(), 0);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + static_cast<ptrdiff_t>(block_size_), block_.begin() + 56, 0);
        for (size_t i = 0; i < 8; ++i) block_[63 - i] = static_cast<unsigned char>(bits >> (i * 8));
        transform(block_.data());
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const uint32_t value : state_) output << std::setw(8) << value;
        return output.str();
    }

   private:
    void transform(const unsigned char* block) {
        static constexpr std::array<uint32_t, 64> constants = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        std::array<uint32_t, 64> words{};
        for (size_t i = 0; i < 16; ++i) {
            words[i] = static_cast<uint32_t>(block[i * 4]) << 24 |
                       static_cast<uint32_t>(block[i * 4 + 1]) << 16 |
                       static_cast<uint32_t>(block[i * 4 + 2]) << 8 |
                       static_cast<uint32_t>(block[i * 4 + 3]);
        }
        for (size_t i = 16; i < words.size(); ++i) {
            const uint32_t s0 = rotate_right(words[i - 15], 7) ^ rotate_right(words[i - 15], 18) ^
                                (words[i - 15] >> 3);
            const uint32_t s1 = rotate_right(words[i - 2], 17) ^ rotate_right(words[i - 2], 19) ^
                                (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (size_t i = 0; i < words.size(); ++i) {
            const uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + choice + constants[i] + words[i];
            const uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<unsigned char, 64> block_{};
    size_t block_size_ = 0;
    uint64_t total_size_ = 0;
};

std::string
sha256_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read downloaded artifact " + path_utf8(path));
    Sha256 digest;
    std::vector<unsigned char> buffer(1024 * 1024);
    while (input) {
        input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        const auto count = input.gcount();
        if (count > 0)
            digest.update(buffer.data(), static_cast<size_t>(count));
    }
    if (!input.eof())
        throw std::runtime_error("failed while reading downloaded artifact " + path_utf8(path));
    return digest.finish();
}

fs::path
executable_path() {
#if defined(_WIN32)
    std::wstring buffer(32768, L'\0');
    const DWORD size =
        GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (size == 0 || size == buffer.size())
        return {};
    buffer.resize(size);
    return fs::path(buffer);
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) != 0)
        return {};
    buffer.resize(std::strlen(buffer.c_str()));
    return fs::weakly_canonical(buffer);
#else
    std::array<char, 4096> buffer{};
    const ssize_t size = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    return size > 0 ? fs::path(std::string(buffer.data(), static_cast<size_t>(size))) : fs::path{};
#endif
}

fs::path
index_path() {
#if defined(_WIN32)
    if (const wchar_t* override_path = _wgetenv(L"NEMO_SPEECH_MODEL_INDEX")) {
        if (*override_path)
            return override_path;
    }
#else
    if (const char* override_path = std::getenv("NEMO_SPEECH_MODEL_INDEX")) {
        if (*override_path)
            return override_path;
    }
#endif
    const fs::path executable = executable_path();
    if (!executable.empty()) {
        const fs::path installed =
            executable.parent_path().parent_path() / "share" / "nemo-speech" / "model-index.json";
        if (fs::is_regular_file(installed))
            return installed;
    }
    throw std::runtime_error(
        "model index is missing; reinstall NeMo-Speech.cpp or set NEMO_SPEECH_MODEL_INDEX");
}

uint64_t
integer(const Value& object, const std::string& key) {
    const double value = object.at(key).number();
    if (value < 0 || value > static_cast<double>(UINT64_MAX) || value != std::floor(value))
        throw std::runtime_error("invalid model index integer: " + key);
    return static_cast<uint64_t>(value);
}

void
validate_component(const std::string& value, const std::string& field) {
    if (value.empty() || value == "." || value == ".." ||
        value.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") !=
            std::string::npos)
        throw std::runtime_error("unsafe " + field + " in model index: " + value);
}

void
validate_repo(const std::string& repo) {
    const size_t slash = repo.find('/');
    if (slash == std::string::npos || slash == 0 || slash + 1 == repo.size() ||
        repo.find('/', slash + 1) != std::string::npos)
        throw std::runtime_error("invalid repository in model index: " + repo);
    validate_component(repo.substr(0, slash), "repository owner");
    validate_component(repo.substr(slash + 1), "repository name");
}

void
validate_revision(const std::string& revision, const std::string& description) {
    if (revision.size() != 40 ||
        revision.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::runtime_error(
            "model index revision must be a full commit SHA: " + description);
}

bool
looks_like_repo(const std::string& reference) {
    try {
        validate_repo(reference);
        return true;
    }
    catch (const std::exception&) {
        return false;
    }
}

Index
load_index() {
    Value root = Value::parse(read_text_file(index_path()));
    if (integer(root, "schema_version") != 1)
        throw std::runtime_error("unsupported model index schema");
    Index index;
    for (const auto& item : root.at("defaults").object())
        index.defaults.emplace(item.first, item.second.string());
    for (const auto& model_value : root.at("models").array()) {
        Model model;
        model.repo = model_value.at("repo").string();
        model.revision = model_value.at("revision").string();
        model.license = model_value.at("license").string();
        model.license_url = model_value.at("license_url").string();
        validate_repo(model.repo);
        if (model.license.empty() || model.license_url.rfind("https://", 0) != 0)
            throw std::runtime_error("invalid license metadata in model index: " + model.repo);
        if (const Value* aliases = model_value.find("aliases")) {
            for (const auto& alias : aliases->array()) {
                validate_component(alias.string(), "model alias");
                model.aliases.push_back(alias.string());
            }
        }
        validate_revision(model.revision, model.repo);
        if (const Value* companions = model_value.find("companions")) {
            for (const auto& companion : companions->array()) {
                validate_repo(companion.string());
                model.companions.push_back(companion.string());
            }
        }
        std::set<std::string> artifact_roles;
        for (const auto& artifact_value : model_value.at("artifacts").array()) {
            Artifact artifact;
            artifact.role = artifact_value.at("role").string();
            artifact.type = artifact_value.at("type").string();
            artifact.filename = artifact_value.at("filename").string();
            artifact.revision = artifact_value.string_or("revision");
            artifact.directory = artifact_value.string_or("directory");
            artifact.sha256 = artifact_value.at("sha256").string();
            artifact.size = integer(artifact_value, "size");
            artifact.range_end =
                artifact_value.find("range_end") ? integer(artifact_value, "range_end") : 0;
            artifact.stop_before = artifact_value.string_or("stop_before");
            static const std::set<std::string> allowed_roles = {
                "asr", "diarization", "tts", "codec", "tokenizer"};
            if (allowed_roles.find(artifact.role) == allowed_roles.end())
                throw std::runtime_error("unsupported artifact role in model index");
            if (!artifact.revision.empty())
                validate_revision(
                    artifact.revision, model.repo + "/" + artifact.role);
            if (!artifact_roles.insert(artifact.role).second)
                throw std::runtime_error(
                    "duplicate artifact role in model index: " + model.repo + "/" + artifact.role);
            validate_component(artifact.filename, "artifact filename");
            if (!artifact.directory.empty())
                validate_component(artifact.directory, "artifact directory");
            if (artifact.sha256.size() != 64 ||
                artifact.sha256.find_first_not_of("0123456789abcdef") != std::string::npos)
                throw std::runtime_error("invalid SHA-256 in model index");
            if (const Value* members = artifact_value.find("members")) {
                for (const auto& member_value : members->array()) {
                    ArchiveMember member;
                    member.name = member_value.at("name").string();
                    member.size = integer(member_value, "size");
                    member.sha256 = member_value.at("sha256").string();
                    validate_component(member.name, "archive member");
                    if (member.sha256.size() != 64 ||
                        member.sha256.find_first_not_of("0123456789abcdef") != std::string::npos)
                        throw std::runtime_error("invalid archive member SHA-256 in model index");
                    artifact.members.push_back(std::move(member));
                }
            }
            if (artifact.type != "file" && artifact.type != "tar-prefix")
                throw std::runtime_error("unsupported artifact type in model index");
            if (artifact.size == 0)
                throw std::runtime_error("model index artifact size must be positive");
            if (artifact.type == "file" &&
                (!artifact.directory.empty() || !artifact.stop_before.empty() ||
                 !artifact.members.empty()))
                throw std::runtime_error("regular model artifact contains archive-only fields");
            if (artifact.type == "tar-prefix" &&
                (artifact.directory.empty() || artifact.stop_before.empty() ||
                 artifact.members.empty() || artifact.range_end != artifact.size - 1))
                throw std::runtime_error("invalid tokenizer archive metadata in model index");
            model.artifacts.push_back(std::move(artifact));
        }
        if (model.artifacts.empty())
            throw std::runtime_error("model has no artifacts in model index: " + model.repo);
        index.models.push_back(std::move(model));
    }
    std::set<std::string> identifiers;
    for (const auto& model : index.models) {
        if (!identifiers.insert(model.repo).second)
            throw std::runtime_error("duplicate repository in model index: " + model.repo);
        for (const auto& alias : model.aliases)
            if (!identifiers.insert(alias).second)
                throw std::runtime_error("duplicate alias in model index: " + alias);
    }
    for (const auto& item : index.defaults) {
        if (identifiers.find(item.second) == identifiers.end())
            throw std::runtime_error(
                "model index default references an unknown model: " + item.second);
        const Model* model = nullptr;
        for (const auto& candidate : index.models)
            if (candidate.repo == item.second ||
                std::find(candidate.aliases.begin(), candidate.aliases.end(), item.second) !=
                    candidate.aliases.end())
                model = &candidate;
        const bool provides_role = std::any_of(
            model->artifacts.begin(), model->artifacts.end(),
            [&](const Artifact& artifact) { return artifact.role == item.first; });
        if (!provides_role)
            throw std::runtime_error("model index default does not provide role: " + item.first);
    }
    for (const auto& model : index.models)
        for (const auto& companion : model.companions)
            if (identifiers.find(companion) == identifiers.end())
                throw std::runtime_error(
                    "model index companion references an unknown model: " + companion);
    return index;
}

const Model&
find_model(const Index& index, const std::string& repo) {
    for (const auto& model : index.models) {
        if (model.repo == repo ||
            std::find(model.aliases.begin(), model.aliases.end(), repo) != model.aliases.end())
            return model;
    }
    throw MissingModelError(
        "unknown model repository: " + repo + " (run 'nemo-speech model list')");
}

const Artifact&
find_artifact(const Model& model, const std::string& role, bool directory) {
    for (const auto& artifact : model.artifacts) {
        if (artifact.role == role && (artifact.type == "tar-prefix") == directory)
            return artifact;
    }
    throw MissingModelError(
        model.repo + " does not provide the indexed " + role +
        (directory ? " directory" : " model"));
}

std::vector<std::string>
commands_for(const Model& model) {
    std::vector<std::string> result;
    auto add = [&](const std::string& command) {
        if (std::find(result.begin(), result.end(), command) == result.end())
            result.push_back(command);
    };
    for (const auto& artifact : model.artifacts) {
        if (artifact.role == "asr") {
            add("transcribe");
            add("bench");
            add("serve");
        } else if (artifact.role == "diarization") {
            add("diarize");
            add("transcribe --diarize");
            add("serve");
        } else if (
            artifact.role == "tts" || artifact.role == "tokenizer" || artifact.role == "codec") {
            add("synthesize");
            add("serve");
        }
    }
    return result;
}

std::vector<std::string>
defaults_for(const Index& index, const Model& model) {
    std::vector<std::string> result;
    for (const auto& item : index.defaults)
        if (item.second == model.repo)
            result.push_back(item.first);
    return result;
}

fs::path
cache_root() {
#if defined(_WIN32)
    if (const wchar_t* override_path = _wgetenv(L"NEMO_SPEECH_MODEL_DIR")) {
        if (*override_path)
            return fs::path(override_path);
    }
#else
    if (const char* override_path = std::getenv("NEMO_SPEECH_MODEL_DIR")) {
        if (*override_path)
            return override_path;
    }
#endif
#if defined(_WIN32)
    if (const wchar_t* local = _wgetenv(L"LOCALAPPDATA"))
        return fs::path(local) / "NeMoSpeech" / "models";
#elif defined(__APPLE__)
    if (const char* home = std::getenv("HOME"))
        return fs::path(home) / "Library" / "Caches" / "NeMoSpeech" / "models";
#else
    if (const char* cache = std::getenv("XDG_CACHE_HOME"))
        return fs::path(cache) / "nemo-speech" / "models";
    if (const char* home = std::getenv("HOME"))
        return fs::path(home) / ".cache" / "nemo-speech" / "models";
#endif
    throw std::runtime_error(
        "cannot determine the model cache directory; set NEMO_SPEECH_MODEL_DIR");
}

fs::path
model_directory(const Model& model) {
    const size_t slash = model.repo.find('/');
    return cache_root() / model.repo.substr(0, slash) / model.repo.substr(slash + 1) /
           model.revision;
}

class ArtifactLock {
   public:
    explicit ArtifactLock(const fs::path& path) {
        fs::create_directories(path.parent_path());
#if defined(_WIN32)
        for (;;) {
            handle_ = CreateFileW(
                path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                FILE_ATTRIBUTE_NORMAL, nullptr);
            if (handle_ != INVALID_HANDLE_VALUE)
                break;
            if (GetLastError() != ERROR_SHARING_VIOLATION)
                throw std::runtime_error("cannot lock model cache " + path_utf8(path));
            Sleep(100);
        }
#else
        descriptor_ = open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (descriptor_ < 0 || flock(descriptor_, LOCK_EX) != 0) {
            if (descriptor_ >= 0)
                close(descriptor_);
            throw std::runtime_error("cannot lock model cache " + path_utf8(path));
        }
#endif
    }

    ~ArtifactLock() {
#if defined(_WIN32)
        if (handle_ != INVALID_HANDLE_VALUE)
            CloseHandle(handle_);
#else
        if (descriptor_ >= 0) {
            flock(descriptor_, LOCK_UN);
            close(descriptor_);
        }
#endif
    }

    ArtifactLock(const ArtifactLock&) = delete;
    ArtifactLock& operator=(const ArtifactLock&) = delete;

   private:
#if defined(_WIN32)
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int descriptor_ = -1;
#endif
};

std::string
base_url() {
    std::string result = "https://huggingface.co";
    if (const char* override_url = std::getenv("NEMO_SPEECH_HF_BASE_URL"))
        if (*override_url)
            result = override_url;
    while (!result.empty() && result.back() == '/') result.pop_back();
    const bool https = result.rfind("https://", 0) == 0;
    auto loopback_authority = [](const std::string& url) {
        if (url.rfind("http://", 0) != 0)
            return false;
        const size_t end = url.find_first_of("/?#", 7);
        const std::string authority = url.substr(7, end == std::string::npos ? end : end - 7);
        auto matches = [&](const std::string& host) {
            if (authority == host)
                return true;
            if (authority.rfind(host + ":", 0) != 0)
                return false;
            const std::string port = authority.substr(host.size() + 1);
            if (port.empty() || port.find_first_not_of("0123456789") != std::string::npos)
                return false;
            try {
                const unsigned long value = std::stoul(port);
                return value > 0 && value <= 65535;
            }
            catch (const std::exception&) {
                return false;
            }
        };
        return matches("127.0.0.1") || matches("localhost") || matches("[::1]");
    };
    const bool loopback = loopback_authority(result);
    if (!https && !loopback)
        throw std::runtime_error(
            "NEMO_SPEECH_HF_BASE_URL must use HTTPS (HTTP is allowed only for loopback tests)");
    return result;
}

std::string
artifact_revision(const Model& model, const Artifact& artifact) {
    return artifact.revision.empty() ? model.revision : artifact.revision;
}

std::string
download_url(const Model& model, const Artifact& artifact) {
    return base_url() + "/" + model.repo + "/resolve/" + artifact_revision(model, artifact) + "/" +
           artifact.filename + "?download=true";
}

int
run_curl(const std::vector<std::string>& arguments) {
    const fs::path executable = model_downloader_executable();
    if (executable.empty())
        return 127;
#if defined(_WIN32)
    auto widen_utf8 = [](const std::string& value) {
        if (value.empty())
            return std::wstring();
        const int size = MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr,
            0);
        if (size <= 0)
            throw std::runtime_error("could not encode a curl argument for Windows");
        std::wstring result(static_cast<size_t>(size), L'\0');
        if (MultiByteToWideChar(
                CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()),
                result.data(), size) != size)
            throw std::runtime_error("could not encode a curl argument for Windows");
        return result;
    };
    std::vector<std::wstring> wide;
    wide.reserve(arguments.size() + 1);
    wide.emplace_back(executable.wstring());
    for (const auto& argument : arguments) wide.emplace_back(widen_utf8(argument));
    std::vector<const wchar_t*> argv;
    argv.reserve(wide.size() + 1);
    for (const auto& argument : wide) argv.push_back(argument.c_str());
    argv.push_back(nullptr);
    const intptr_t status = _wspawnv(_P_WAIT, executable.c_str(), argv.data());
    return status < 0 ? 127 : static_cast<int>(status);
#else
    const std::string executable_string = executable.string();
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 2);
    argv.push_back(const_cast<char*>(executable_string.c_str()));
    for (const auto& argument : arguments) argv.push_back(const_cast<char*>(argument.c_str()));
    argv.push_back(nullptr);
    const pid_t child = fork();
    if (child < 0)
        throw std::runtime_error("could not start curl");
    if (child == 0) {
        execv(executable_string.c_str(), argv.data());
        _exit(127);
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child)
        throw std::runtime_error("could not wait for curl");
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
#endif
}

std::string
curl_missing_message() {
    std::string message =
        "automatic model download requires the curl executable, but it was not found on PATH. "
        "Local model paths and models already in NEMO_SPEECH_MODEL_DIR still work. ";
#if defined(_WIN32)
    message +=
        "Windows 10 and 11 include curl.exe; ensure %SystemRoot%\\System32 is on PATH or repair "
        "the Windows curl installation.";
#elif defined(__APPLE__)
    message += "macOS includes /usr/bin/curl; ensure /usr/bin is on PATH.";
#else
    message +=
        "Install it with your package manager (for example, 'sudo apt install curl', "
        "'sudo dnf install curl', or 'sudo pacman -S curl').";
#endif
    return message;
}

bool
stderr_is_terminal() {
#if defined(_WIN32)
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(STDERR_FILENO) != 0;
#endif
}

void
download(const Model& model, const Artifact& artifact, const fs::path& output) {
    const bool loopback = base_url().rfind("http://", 0) == 0;
    const std::string protocols = loopback ? "=http,https" : "=https";
    fs::path curl_errors = output;
    curl_errors += ".curl-errors";
    auto invoke = [&](bool resume) {
        std::error_code remove_error;
        fs::remove(curl_errors, remove_error);
        std::vector<std::string> arguments = {
            "--fail",
            "--location",
            "--max-redirs",
            "5",
            "--proto",
            protocols,
            "--proto-redir",
            protocols,
            "--retry",
            "3",
            "--retry-delay",
            "1",
            "--connect-timeout",
            "20",
            "--speed-limit",
            "1024",
            "--speed-time",
            "30",
            "--output",
            output.u8string()};
        if (artifact.type == "tar-prefix") {
            arguments.push_back("--range");
            arguments.push_back("0-" + std::to_string(artifact.range_end));
        } else if (resume && fs::exists(output)) {
            arguments.push_back("--continue-at");
            arguments.push_back("-");
        }
        if (cli_quiet() || cli_json() || !stderr_is_terminal()) {
            arguments.push_back("--silent");
            arguments.push_back("--show-error");
            arguments.push_back("--stderr");
            arguments.push_back(curl_errors.u8string());
        } else {
            arguments.push_back("--progress-bar");
        }
        arguments.push_back(download_url(model, artifact));
        return run_curl(arguments);
    };
    int status = invoke(true);
    if (status == 33 && artifact.type == "file") {
        std::error_code error;
        fs::remove(output, error);
        status = invoke(false);
    }
    if (status == 127)
        throw std::runtime_error(curl_missing_message());
    if (status != 0) {
        std::string detail;
        if (fs::is_regular_file(curl_errors)) {
            detail = read_text_file(curl_errors);
            while (!detail.empty() && (detail.back() == '\n' || detail.back() == '\r'))
                detail.pop_back();
        }
        std::error_code error;
        fs::remove(curl_errors, error);
        throw std::runtime_error(
            "curl failed while downloading " + model.repo + " (exit code " +
            std::to_string(status) + ")" + (detail.empty() ? "" : ": " + detail));
    }
    std::error_code error;
    fs::remove(curl_errors, error);
}

struct FileState {
    uint64_t size;
    fs::file_time_type modified;
};

bool
file_state(const fs::path& path, uint64_t expected_size, FileState& state) {
    std::error_code error;
    if (!fs::is_regular_file(path, error) || error)
        return false;
    const uintmax_t size = fs::file_size(path, error);
    if (error || size != expected_size)
        return false;
    const fs::file_time_type modified = fs::last_write_time(path, error);
    if (error)
        return false;
    state = {static_cast<uint64_t>(size), modified};
    return true;
}

fs::path
verification_marker_path(const fs::path& path) {
    fs::path marker = path;
    marker += ".verified";
    return marker;
}

std::string
file_time_string(fs::file_time_type value) {
    const auto nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(value.time_since_epoch());
    return std::to_string(nanoseconds.count());
}

bool
verification_marker_matches(
    const fs::path& path, const Artifact& artifact, const FileState& state) {
    const fs::path marker = verification_marker_path(path);
    std::error_code error;
    if (!fs::is_regular_file(marker, error) || error || fs::file_size(marker, error) > 256 || error)
        return false;
    std::ifstream input(marker, std::ios::binary);
    std::array<std::string, 3> lines;
    for (auto& line : lines)
        if (!std::getline(input, line))
            return false;
    std::string extra;
    if (std::getline(input, extra))
        return false;
    return lines[0] == "sha256=" + artifact.sha256 &&
           lines[1] == "size=" + std::to_string(state.size) &&
           lines[2] == "mtime=" + file_time_string(state.modified);
}

void
write_verification_marker(const fs::path& path, const Artifact& artifact, const FileState& state) {
    const fs::path marker = verification_marker_path(path);
    fs::path temporary = marker;
    temporary += ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output << "sha256=" << artifact.sha256 << '\n'
               << "size=" << state.size << '\n'
               << "mtime=" << file_time_string(state.modified) << '\n';
        if (!output) {
            std::error_code error;
            fs::remove(temporary, error);
            return;
        }
    }
    std::error_code error;
    fs::remove(marker, error);
    error.clear();
    fs::rename(temporary, marker, error);
    if (error)
        fs::remove(temporary, error);
}

bool
valid_file(const fs::path& path, const Artifact& artifact, bool cache_hit = false) {
    FileState before{};
    if (!file_state(path, artifact.size, before))
        return false;
    if (cache_hit && verification_marker_matches(path, artifact, before))
        return true;
    if (sha256_file(path) != artifact.sha256)
        return false;
    FileState after{};
    if (!file_state(path, artifact.size, after) || before.size != after.size ||
        before.modified != after.modified)
        return false;
    if (cache_hit)
        write_verification_marker(path, artifact, after);
    return true;
}

bool
valid_member(const fs::path& path, const ArchiveMember& member) {
    std::error_code error;
    return fs::is_regular_file(path, error) && !error &&
           fs::file_size(path, error) == member.size && !error &&
           sha256_file(path) == member.sha256;
}

uint64_t
tar_octal(const char* value, size_t size) {
    uint64_t result = 0;
    size_t index = 0;
    while (index < size && (value[index] == ' ' || value[index] == '\0')) ++index;
    for (; index < size && value[index] != '\0' && value[index] != ' '; ++index) {
        if (value[index] < '0' || value[index] > '7')
            throw std::runtime_error("invalid TAR numeric field in tokenizer artifact");
        result = result * 8 + static_cast<unsigned>(value[index] - '0');
    }
    return result;
}

std::string
tar_string(const char* value, size_t size) {
    size_t length = 0;
    while (length < size && value[length] != '\0') ++length;
    return std::string(value, length);
}

void
validate_pax_metadata(const std::string& data) {
    size_t offset = 0;
    while (offset < data.size()) {
        const size_t space = data.find(' ', offset);
        if (space == std::string::npos || space == offset)
            throw std::runtime_error("invalid PAX metadata in tokenizer artifact");
        size_t length = 0;
        for (size_t i = offset; i < space; ++i) {
            if (data[i] < '0' || data[i] > '9')
                throw std::runtime_error("invalid PAX metadata in tokenizer artifact");
            length = length * 10 + static_cast<size_t>(data[i] - '0');
        }
        if (length == 0 || length > data.size() - offset || data[offset + length - 1] != '\n')
            throw std::runtime_error("invalid PAX metadata in tokenizer artifact");
        const size_t equals = data.find('=', space + 1);
        if (equals == std::string::npos || equals >= offset + length)
            throw std::runtime_error("invalid PAX metadata in tokenizer artifact");
        const std::string key = data.substr(space + 1, equals - space - 1);
        if (key != "mtime" && key != "atime" && key != "ctime")
            throw std::runtime_error("unsupported PAX field in tokenizer artifact: " + key);
        offset += length;
    }
}

void
extract_tar_prefix(const fs::path& archive, const fs::path& destination, const Artifact& artifact) {
    std::ifstream input(archive, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read tokenizer archive " + path_utf8(archive));
    fs::create_directories(destination);
    std::array<char, 512> header{};
    bool reached_stop = false;
    while (input.read(header.data(), header.size())) {
        if (std::all_of(header.begin(), header.end(), [](char c) { return c == '\0'; }))
            break;
        uint64_t checksum = 0;
        for (size_t i = 0; i < header.size(); ++i)
            checksum += static_cast<unsigned char>(i >= 148 && i < 156 ? ' ' : header[i]);
        if (checksum != tar_octal(header.data() + 148, 8))
            throw std::runtime_error("invalid TAR checksum in tokenizer artifact");
        std::string name = tar_string(header.data(), 100);
        const std::string prefix = tar_string(header.data() + 345, 155);
        if (!prefix.empty())
            name = prefix + "/" + name;
        const fs::path relative(name);
        if (relative.empty() || relative.is_absolute())
            throw std::runtime_error("unsafe path in tokenizer artifact");
        for (const auto& component : relative)
            if (component == "..")
                throw std::runtime_error("unsafe path in tokenizer artifact");
        if (relative.filename() == artifact.stop_before) {
            reached_stop = true;
            break;
        }
        const uint64_t size = tar_octal(header.data() + 124, 12);
        const char type = header[156];
        const fs::path output = destination / relative;
        if (type == '5') {
            fs::create_directories(output);
        } else if (type == '\0' || type == '0') {
            fs::create_directories(output.parent_path());
            std::ofstream file(output, std::ios::binary | std::ios::trunc);
            if (!file)
                throw std::runtime_error("cannot extract " + path_utf8(output));
            uint64_t remaining = size;
            std::array<char, 64 * 1024> buffer{};
            while (remaining > 0) {
                const size_t count =
                    static_cast<size_t>(std::min<uint64_t>(remaining, buffer.size()));
                if (!input.read(buffer.data(), static_cast<std::streamsize>(count)))
                    throw std::runtime_error("truncated tokenizer artifact");
                file.write(buffer.data(), static_cast<std::streamsize>(count));
                remaining -= count;
            }
            if (!file)
                throw std::runtime_error("cannot extract " + path_utf8(output));
        } else if ((type == 'x' || type == 'g') && size <= 64 * 1024) {
            std::string metadata(static_cast<size_t>(size), '\0');
            if (!input.read(metadata.data(), static_cast<std::streamsize>(metadata.size())))
                throw std::runtime_error("truncated tokenizer artifact");
            validate_pax_metadata(metadata);
        } else {
            throw std::runtime_error("unsupported TAR entry in tokenizer artifact");
        }
        const uint64_t padding = (512 - (size % 512)) % 512;
        if (type == '5') {
            if (size != 0)
                input.seekg(static_cast<std::streamoff>(size + padding), std::ios::cur);
        } else if (padding != 0) {
            input.seekg(static_cast<std::streamoff>(padding), std::ios::cur);
        }
        if (!input)
            throw std::runtime_error("truncated tokenizer artifact");
    }
    if (!reached_stop)
        throw std::runtime_error(
            "tokenizer archive prefix did not reach the expected model weights");
    for (const auto& member : artifact.members) {
        if (!valid_member(destination / member.name, member))
            throw std::runtime_error(
                "tokenizer artifact member failed verification: " + member.name);
    }
}

PulledModelArtifact
materialize(const Model& model, const Artifact& artifact) {
    const fs::path directory = model_directory(model);
    fs::create_directories(directory);
    const fs::path destination =
        artifact.type == "file" ? directory / artifact.filename : directory / artifact.directory;
    fs::path lock_path = destination;
    lock_path += ".lock";
    ArtifactLock lock(lock_path);
    if (artifact.type == "file" && valid_file(destination, artifact, true)) {
        if (cli_verbose())
            std::fprintf(stderr, "[model] cached: %s\n", path_utf8(destination).c_str());
        return {model.repo, artifact.role, destination, true};
    }
    if (artifact.type == "tar-prefix") {
        bool valid = fs::is_directory(destination);
        for (const auto& member : artifact.members)
            valid = valid && valid_member(destination / member.name, member);
        if (valid) {
            if (cli_verbose())
                std::fprintf(stderr, "[model] cached: %s\n", path_utf8(destination).c_str());
            return {model.repo, artifact.role, destination, true};
        }
    }

    if (!cli_quiet() && !cli_json()) {
        const std::string revision = artifact_revision(model, artifact);
        std::fprintf(
            stderr,
            "[model] downloading %s@%.12s (%s, %.1f MiB)\n"
            "[model] license: %s — %s\n",
            model.repo.c_str(), revision.c_str(), artifact.role.c_str(),
            artifact.size / 1048576.0, model.license.c_str(), model.license_url.c_str());
    }
    const fs::path partial = directory / (artifact.filename + ".partial");
    if (!valid_file(partial, artifact)) {
        if (artifact.type == "tar-prefix") {
            std::error_code error;
            fs::remove(partial, error);
        } else {
            std::error_code error;
            if (fs::is_regular_file(partial, error) &&
                fs::file_size(partial, error) > artifact.size)
                fs::remove(partial, error);
        }
        download(model, artifact, partial);
    }
    if (!cli_quiet() && !cli_json())
        std::fprintf(stderr, "[model] verifying size and SHA-256...\n");
    if (!valid_file(partial, artifact)) {
        std::error_code error;
        fs::remove(partial, error);
        throw std::runtime_error(
            "downloaded artifact failed size or SHA-256 verification: " + model.repo + "/" +
            artifact.filename);
    }

    std::error_code error;
    if (artifact.type == "file") {
        fs::remove(destination, error);
        error.clear();
        fs::rename(partial, destination, error);
        if (error)
            throw std::runtime_error("cannot install model artifact: " + error.message());
        FileState state{};
        if (file_state(destination, artifact.size, state))
            write_verification_marker(destination, artifact, state);
    } else {
        const fs::path extracting = directory / (artifact.directory + ".extracting");
        fs::remove_all(extracting, error);
        extract_tar_prefix(partial, extracting, artifact);
        fs::remove_all(destination, error);
        error.clear();
        fs::rename(extracting, destination, error);
        if (error) {
            fs::remove_all(extracting);
            throw std::runtime_error("cannot install tokenizer artifact: " + error.message());
        }
        fs::remove(partial, error);
    }
    if (!cli_quiet() && !cli_json())
        std::fprintf(stderr, "[model] ready: %s\n", path_utf8(destination).c_str());
    return {model.repo, artifact.role, destination, false};
}

fs::path
resolve(
    const std::string& reference, const std::string& role, const std::string& description,
    bool directory) {
    std::error_code error;
    if (!reference.empty()) {
        const fs::path local(reference);
        const bool exists =
            directory ? fs::is_directory(local, error) : fs::is_regular_file(local, error);
        if (exists && !error)
            return fs::absolute(local);
    }
    const Index index = load_index();
    std::string repo = reference;
    if (repo.empty()) {
        const auto found = index.defaults.find(role);
        if (found == index.defaults.end())
            throw MissingModelError(description + " path is required");
        repo = found->second;
    }
    const Model* model = nullptr;
    for (const auto& candidate : index.models)
        if (candidate.repo == repo ||
            std::find(candidate.aliases.begin(), candidate.aliases.end(), repo) !=
                candidate.aliases.end())
            model = &candidate;
    if (!model) {
        if (reference.empty() || looks_like_repo(reference))
            throw MissingModelError(
                "unknown " + description + " repository: " + repo +
                " (run 'nemo-speech model list')");
        throw MissingModelError(
            description + (directory ? " directory does not exist: " : " file does not exist: ") +
            reference);
    }
    return materialize(*model, find_artifact(*model, role, directory)).path;
}

}  // namespace

fs::path
resolve_indexed_model_file(
    const std::string& reference, const std::string& role, const std::string& description) {
    return resolve(reference, role, description, false);
}

fs::path
resolve_indexed_model_directory(
    const std::string& reference, const std::string& role, const std::string& description) {
    return resolve(reference, role, description, true);
}

std::vector<PulledModelArtifact>
pull_indexed_model(const std::string& repo) {
    const Index index = load_index();
    std::vector<PulledModelArtifact> result;
    std::set<std::string> visited;
    std::function<void(const std::string&)> pull = [&](const std::string& current_repo) {
        const Model& model = find_model(index, current_repo);
        if (!visited.insert(model.repo).second)
            return;
        for (const auto& artifact : model.artifacts) result.push_back(materialize(model, artifact));
        for (const auto& companion : model.companions) pull(companion);
    };
    pull(repo);
    return result;
}

std::string
indexed_models_json() {
    const Index index = load_index();
    Value output(Value::Object{});
    Value::Object defaults;
    for (const auto& item : index.defaults) defaults.emplace(item.first, item.second);
    output["defaults"] = std::move(defaults);
    Value::Array models;
    for (const auto& model : index.models) {
        Value item(Value::Object{});
        item["repo"] = model.repo;
        Value::Array aliases;
        for (const auto& alias : model.aliases) aliases.emplace_back(alias);
        item["aliases"] = std::move(aliases);
        item["revision"] = model.revision;
        item["license"] = model.license;
        item["license_url"] = model.license_url;
        Value::Array companions;
        for (const auto& companion : model.companions) companions.emplace_back(companion);
        item["companions"] = std::move(companions);
        Value::Array roles;
        for (const auto& artifact : model.artifacts) roles.emplace_back(artifact.role);
        item["roles"] = std::move(roles);
        Value::Array commands;
        for (const auto& command : commands_for(model)) commands.emplace_back(command);
        item["commands"] = std::move(commands);
        Value::Array default_roles;
        for (const auto& role : defaults_for(index, model)) default_roles.emplace_back(role);
        item["default_for"] = std::move(default_roles);
        models.emplace_back(std::move(item));
    }
    output["models"] = std::move(models);
    return output.dump(2) + "\n";
}

std::string
indexed_models_text() {
    const Index index = load_index();
    std::ostringstream output;
    auto has_role = [](const Model& model, const std::string& role) {
        return std::any_of(
            model.artifacts.begin(), model.artifacts.end(),
            [&](const Artifact& artifact) { return artifact.role == role; });
    };
    auto is_default = [&](const Model& model) {
        return std::any_of(index.defaults.begin(), index.defaults.end(), [&](const auto& item) {
            return item.second == model.repo;
        });
    };
    auto print_model = [&](const Model& model, const std::string& components) {
        const std::string name = model.aliases.empty() ? model.repo : model.aliases.front();
        output << "  " << (is_default(model) ? "* " : "  ") << name;
        if (!components.empty())
            output << " (" << components << ')';
        output << "\n      repo: " << model.repo << '\n';
        if (model.aliases.size() > 1) {
            output << "      also: ";
            for (size_t i = 1; i < model.aliases.size(); ++i) {
                if (i > 1)
                    output << ", ";
                output << model.aliases[i];
            }
            output << '\n';
        }
        if (!model.companions.empty()) {
            output << "      pulls: ";
            for (size_t i = 0; i < model.companions.size(); ++i) {
                if (i)
                    output << ", ";
                const Model& companion = find_model(index, model.companions[i]);
                output << (companion.aliases.empty() ? companion.repo : companion.aliases.front());
            }
            output << '\n';
        }
    };

    output << "Available models (* = command default)\n\n"
           << "ASR — transcribe, bench, serve\n";
    for (const auto& model : index.models)
        if (has_role(model, "asr"))
            print_model(model, "");

    output << "\nDiarization — diarize, transcribe --diarize, serve\n";
    for (const auto& model : index.models)
        if (has_role(model, "diarization"))
            print_model(model, "");

    output << "\nTTS — synthesize, serve\n";
    for (const auto& model : index.models) {
        const bool tts = has_role(model, "tts");
        const bool tokenizer = has_role(model, "tokenizer");
        const bool codec = has_role(model, "codec");
        if (tts || tokenizer || codec) {
            std::string components;
            if (tts)
                components = "speech model";
            if (tokenizer)
                components += (components.empty() ? "" : " + ") + std::string("tokenizer");
            if (codec)
                components += (components.empty() ? "" : " + ") + std::string("codec");
            print_model(model, components);
        }
    }
    output << "\nUse a short name or full repository ID wherever MODEL is accepted.\n"
           << "Download ahead of time with: nemo-speech pull <name>\n";
    return output.str();
}

fs::path
model_downloader_executable() {
#if defined(_WIN32)
    std::wstring buffer(32768, L'\0');
    const DWORD size = SearchPathW(
        nullptr, L"curl.exe", nullptr, static_cast<DWORD>(buffer.size()), buffer.data(), nullptr);
    if (size == 0 || size >= buffer.size())
        return {};
    buffer.resize(size);
    return fs::path(buffer);
#else
    std::istringstream paths(std::getenv("PATH") ? std::getenv("PATH") : "");
    std::string directory;
    while (std::getline(paths, directory, ':')) {
        const fs::path candidate = fs::path(directory.empty() ? "." : directory) / "curl";
        if (access(candidate.c_str(), X_OK) == 0)
            return fs::absolute(candidate);
    }
    return {};
#endif
}
