/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>
#include <utility>

#include "../activations/activation_layer.h"
#include "../layers/layer.h"

#include "../half.hpp"
using namespace half_float;

namespace tiny_dnn {

class tanh_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  std::string layer_type() const override { return "tanh-activation"; }

  void forward_activation(const vec_t &x, vec_t &y) override {
    for (size_t j = 0; j < x.size(); j++) {
      y[j] = std::tanh(x[j]);
    }
  }

  void forward_activation16(const vec16_t &x, vec16_t &y) override {
    for (size_t j = 0; j < x.size(); j++) {
      y[j] = std::tanh(x[j]);
    }
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) override {
    for (size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of tanh)
      dx[j] = dy[j] * (float_t(1) - sqr(y[j]));
    }
  }

  void backward_activation16(const vec16_t &x,
                             const vec16_t &y,
                             vec16_t &dx,
                             const vec16_t &dy) override {
    for (size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of tanh)
      dx[j] = dy[j] * (half(1) - sqr(y[j]));
    }
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(-0.8), float_t(0.8));
  }

  friend struct serialization_buddy;
};

}  // namespace tiny_dnn
