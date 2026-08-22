// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from parakeet.cpp:
// Copyright (c) 2025 Jason Ni
// Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
#pragma once

#include <utility>

#include "runtime.h"

namespace ggml_runtime {

ggml_tensor* cached_q8_input(
    Session* session, TensorContainer* session_tensor_container, const ggml_bf_tensor& weight,
    ggml_tensor* input);

class Conv1D : public Module {
   public:
    Conv1D(
        const std::string& name, int in_channels, int out_channels, int kernel_size, int stride = 1,
        int padding = 0, int dilation = 1, bool use_bias = true, bool is_dw = false)
        : name(name), in_channels(in_channels), out_channels(out_channels),
          kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation),
          use_bias(use_bias), is_dw(is_dw) {
        weight_name = this->name + ".weight";
        bias_name = this->name + ".bias";
    }
    ~Conv1D() = default;


    void define_tensors(Session* session) override;

    TensorBag build_graph(
        Session* session, TensorBag input_tensors,
        TensorContainer* session_tensor_container) override;

    void set_data(Session* session) override;

   private:
    std::string name;
    std::string weight_name;
    std::string bias_name;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    bool use_bias;
    bool is_dw;

    // Set in define_tensors() when the stored weight is 2D (pointwise,
    // squeezed k=1 in the converter). 2D weights run through
    // ggml_mul_mat instead of ggml_conv_1d, which lets quantized
    // (Q8_0/k-quant) pointwise weights compute natively.
    bool is_pointwise_2d = false;

    ggml_bf_tensor weight = ggml_bf_tensor(nullptr, nullptr);
    ggml_bf_tensor bias = ggml_bf_tensor(nullptr, nullptr);
};

class Conv2D : public Module {
   public:
    Conv2D(
        const std::string& name, int in_channels, int out_channels, int kernel_size, int stride = 1,
        int padding = 0, int dilation = 1)
        : name(name), in_channels(in_channels), out_channels(out_channels),
          kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation) {
        weight_name = this->name + ".weight";
        bias_name = this->name + ".bias";
    }
    ~Conv2D() = default;


    void define_tensors(Session* session) override;

    TensorBag build_graph(
        Session* session, TensorBag input_tensors,
        TensorContainer* session_tensor_container) override;

    void set_data(Session* session) override;

   private:
    std::string name;
    std::string weight_name;
    std::string bias_name;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int dilation;

    ggml_bf_tensor weight = ggml_bf_tensor(nullptr, nullptr);
    ggml_bf_tensor bias = ggml_bf_tensor(nullptr, nullptr);
};

class ReLU : public Module {
   public:
    ReLU(const std::string& name) : name(name) {}
    ~ReLU() = default;


    void define_tensors(Session* session) override;

    TensorBag build_graph(
        Session* session, TensorBag input_tensors,
        TensorContainer* session_tensor_container) override;

    void set_data(Session* session) override;

   private:
    std::string name;
};


class Conv2DDW : public Module {
   public:
    Conv2DDW(
        const std::string& name, int in_channels, int out_channels, int kernel_size, int stride = 1,
        int padding = 0, int dilation = 1)
        : name(name), in_channels(in_channels), out_channels(out_channels),
          kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation) {
        weight_name = this->name + ".weight";
        bias_name = this->name + ".bias";
    }
    ~Conv2DDW() = default;


    void define_tensors(Session* session) override;

    TensorBag build_graph(
        Session* session, TensorBag input_tensors,
        TensorContainer* session_tensor_container) override;

    void set_data(Session* session) override;

   private:
    std::string name;
    std::string weight_name;
    std::string bias_name;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int dilation;

    ggml_bf_tensor weight = ggml_bf_tensor(nullptr, nullptr);
    ggml_bf_tensor bias = ggml_bf_tensor(nullptr, nullptr);
};

class SequenceModule : public Module {
   public:
    SequenceModule(const std::string& name) : name(name) { modules = std::vector<Module*>(); };
    ~SequenceModule() {
        for (auto module : modules) {
            delete module;
        }
    }

    std::vector<Module*> modules;


    void define_tensors(Session* session) override;
    TensorBag build_graph(
        Session* session, TensorBag input_tensors,
        TensorContainer* session_tensor_container) override;

    void set_data(Session* session) override;

   private:
    std::string name;
};

class Linear : public Module {
   public:
    Linear(const std::string& name, int in_features, int out_features, bool use_bias = true)
        : name(name), in_features(in_features), out_features(out_features), use_bias(use_bias) {
        weight_name = this->name + ".weight";
        bias_name = this->name + ".bias";
    };
    ~Linear() = default;


    void define_tensors(Session* session) override;

    TensorBag build_graph(
        Session* session, TensorBag input_tensors,
        TensorContainer* session_tensor_container) override;

    void set_data(Session* session) override;

   private:
    std::string name;
    std::string weight_name;
    std::string bias_name;
    int in_features;
    int out_features;
    bool use_bias = true;

    ggml_bf_tensor weight = ggml_bf_tensor(nullptr, nullptr);
    ggml_bf_tensor bias = ggml_bf_tensor(nullptr, nullptr);
};

class LayerNorm : public Module {
   public:
    LayerNorm(const std::string& name, const int64_t (&input_shape)[GGML_MAX_DIMS]) : name(name) {
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            this->input_shape[i] = input_shape[i];
        }
        weight_name = this->name + ".weight";
        bias_name = this->name + ".bias";
    };
    ~LayerNorm() = default;


    void define_tensors(Session* session) override;

    TensorBag build_graph(
        Session* session, TensorBag input_tensors,
        TensorContainer* session_tensor_container) override;

    void set_data(Session* session) override;

   private:
    std::string name;
    std::string weight_name;
    std::string bias_name;
    int64_t input_shape[GGML_MAX_DIMS] = {0};
};

class BatchNorm1d : public Module {
   public:
    BatchNorm1d(const std::string& name, int num_features, bool affine = true)
        : name(name), num_features(num_features), affine(affine) {
        weight_name = this->name + ".weight";
        bias_name = this->name + ".bias";
        running_mean_name = this->name + ".running_mean";
        running_var_name = this->name + ".running_var";
        eps_name = this->name + ".eps";
    };
    ~BatchNorm1d() = default;


    void define_tensors(Session* session) override;

    TensorBag build_graph(
        Session* session, TensorBag input_tensors,
        TensorContainer* session_tensor_container) override;

    void set_data(Session* session) override;

   private:
    std::string name;
    std::string weight_name;
    std::string bias_name;
    std::string running_mean_name;
    std::string running_var_name;
    std::string eps_name;
    int num_features;
    bool affine;
};

}  // namespace ggml_runtime
