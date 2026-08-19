// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "microphone_capture.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <stdexcept>

// miniaudio is already vendored by the repository. Compiling its capture-only
// device layer here gives the CLI one implementation across CoreAudio, WASAPI,
// ALSA, PulseAudio, and the other supported host APIs without a runtime audio
// library dependency.
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

namespace nemo_speech::cli {

struct MicrophoneCapture::Impl {
    static constexpr int kSampleRate = 16000;
    static constexpr size_t kQueueCapacity = kSampleRate * 2;

    ma_device device{};
    bool initialized = false;
    bool running = false;
    std::array<float, kQueueCapacity> pending{};
    std::atomic<size_t> read_position{0};
    std::atomic<size_t> write_position{0};
    std::string name = "default microphone";

    static void data_callback(
        ma_device* device, void* /*output*/, const void* input, ma_uint32 frame_count) {
        if (input == nullptr || frame_count == 0)
            return;
        auto* self = static_cast<Impl*>(device->pUserData);
        const auto* samples = static_cast<const float*>(input);
        const size_t write = self->write_position.load(std::memory_order_relaxed);
        const size_t read = self->read_position.load(std::memory_order_acquire);
        const size_t count = std::min<size_t>(frame_count, kQueueCapacity - (write - read));

        // Preserve queued audio and drop newest samples on overflow. The callback
        // never waits for space or allocates memory.
        const size_t offset = write % kQueueCapacity;
        const size_t first = std::min(count, kQueueCapacity - offset);
        std::memcpy(self->pending.data() + offset, samples, first * sizeof(float));
        std::memcpy(self->pending.data(), samples + first, (count - first) * sizeof(float));
        self->write_position.store(write + count, std::memory_order_release);
    }

    void dispose() noexcept {
        if (running) {
            (void)ma_device_stop(&device);
            running = false;
        }
        if (initialized) {
            ma_device_uninit(&device);
            initialized = false;
        }
    }
};

MicrophoneCapture::MicrophoneCapture() : impl_(std::make_unique<Impl>()) {}

MicrophoneCapture::~MicrophoneCapture() {
    impl_->dispose();
}

void
MicrophoneCapture::start() {
    if (impl_->initialized)
        throw std::logic_error("microphone capture is already started");

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate = Impl::kSampleRate;
    config.periodSizeInMilliseconds = 40;
    config.dataCallback = Impl::data_callback;
    config.pUserData = impl_.get();
    impl_->read_position.store(0, std::memory_order_relaxed);
    impl_->write_position.store(0, std::memory_order_relaxed);

    ma_result result = ma_device_init(nullptr, &config, &impl_->device);
    if (result != MA_SUCCESS)
        throw std::runtime_error(
            "could not open the default microphone: " + std::string(ma_result_description(result)));
    impl_->initialized = true;
    impl_->name =
        impl_->device.capture.name[0] != '\0' ? impl_->device.capture.name : "default microphone";

    result = ma_device_start(&impl_->device);
    if (result != MA_SUCCESS) {
        impl_->dispose();
        throw std::runtime_error(
            "could not start microphone capture: " + std::string(ma_result_description(result)) +
            ". Check the operating system's microphone permission for this terminal");
    }
    impl_->running = true;
}

void
MicrophoneCapture::stop() {
    impl_->dispose();
}

std::vector<float>
MicrophoneCapture::drain() {
    const size_t read = impl_->read_position.load(std::memory_order_relaxed);
    const size_t write = impl_->write_position.load(std::memory_order_acquire);
    const size_t count = write - read;
    if (count == 0)
        return {};
    std::vector<float> samples(count);

    const size_t offset = read % Impl::kQueueCapacity;
    const size_t first = std::min(count, Impl::kQueueCapacity - offset);
    std::memcpy(samples.data(), impl_->pending.data() + offset, first * sizeof(float));
    std::memcpy(samples.data() + first, impl_->pending.data(), (count - first) * sizeof(float));
    impl_->read_position.store(read + count, std::memory_order_release);
    return samples;
}

int
MicrophoneCapture::sample_rate() const {
    return Impl::kSampleRate;
}

const std::string&
MicrophoneCapture::device_name() const {
    return impl_->name;
}

}  // namespace nemo_speech::cli
