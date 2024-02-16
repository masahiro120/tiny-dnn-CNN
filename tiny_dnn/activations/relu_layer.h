/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <string>
#include <utility>

#include "../activations/activation_layer.h"
#include "../layers/layer.h"


// #define RELU_F_HALF 1
// #define RELU_B_HALF 1

#include "../half.hpp"
#include "../half_define.h"
using namespace half_float;

std::vector<half> one_vector_to_half(const tiny_dnn::vec_t& array);
void one_half_to_vector(tiny_dnn::vec_t& array, std::vector<half> array_half);

namespace tiny_dnn {

class relu_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  std::string layer_type() const override { return "relu-activation"; }

  void forward_activation(const vec_t &x, vec_t &y) override {
#if RELU_F_HALF == 0

    for (size_t j = 0; j < x.size(); j++) {
      y[j] = std::max(float_t(0), x[j]);
    }

#else
    std::vector<half> x_half = one_vector_to_half(x);
    std::vector<half> y_half = one_vector_to_half(y);

    for (size_t j = 0; j < x.size(); j++) {
      y_half[j] = std::max(half(0), x_half[j]);
    }

    one_half_to_vector(y, y_half);
#endif
  }

  void forward_activation16(const vec16_t &x, vec16_t &y) override {

    for (size_t j = 0; j < x.size(); j++) {
      y[j] = std::max(half(0), x[j]);
    }

  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) override {
#if RELU_B_HALF == 0
    for (size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of relu)
      dx[j] = dy[j] * (y[j] > float_t(0) ? float_t(1) : float_t(0));
    }
#else
    std::vector<half> x_half = one_vector_to_half(x);
    std::vector<half> y_half = one_vector_to_half(y);
    std::vector<half> dx_half = one_vector_to_half(dx);
    std::vector<half> dy_half = one_vector_to_half(dy);

    for (size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of relu)
      dx_half[j] = dy_half[j] * (y_half[j] > half(0) ? half(1) : half(0));
    }

    one_half_to_vector(dx, dx_half);
#endif
  }

  void backward_activation16(const vec16_t &x,
                             const vec16_t &y,
                             vec16_t &dx,
                             const vec16_t &dy) override {

    for (size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of relu)
      dx[j] = dy[j] * (y[j] > half(0) ? half(1) : half(0));
    }

  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }

  friend struct serialization_buddy;
};

}  // namespace tiny_dnn
