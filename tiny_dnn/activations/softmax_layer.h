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

// #define SOFTMAX_F_HALF 1
// #define SOFTMAX_B_HALF 1

#include "../half.hpp"
#include "../half_define.h"
using namespace half_float;

std::vector<half> one_vector_to_half(const tiny_dnn::vec_t& array);
void one_half_to_vector(tiny_dnn::vec_t& array, std::vector<half> array_half);

namespace tiny_dnn {

class softmax_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  std::string layer_type() const override { return "softmax-activation"; }

  void forward_activation(const vec_t &x, vec_t &y) override {
#if SOFTMAX_F_HALF == 0
    const float_t alpha = *std::max_element(x.begin(), x.end());
    float_t denominator(0);
    for (size_t j = 0; j < x.size(); j++) {
      y[j] = std::exp(x[j] - alpha);
      denominator += y[j];
    }
    for (size_t j = 0; j < x.size(); j++) {
      y[j] /= denominator;
    }
#else
    std::vector<half> x_half = one_vector_to_half(x);
    std::vector<half> y_half = one_vector_to_half(y);

    const half alpha = *std::max_element(x_half.begin(), x_half.end());
    half denominator(0);
    for (size_t j = 0; j < x.size(); j++) {
      y_half[j] = std::exp(x_half[j] - alpha);
      denominator += y_half[j];
    }
    for (size_t j = 0; j < x.size(); j++) {
      y_half[j] /= denominator;
    }

    one_half_to_vector(y, y_half);
#endif
  }

  void forward_activation16(const vec16_t &x, vec16_t &y) override {

    const half alpha = *std::max_element(x.begin(), x.end());
    half denominator(0);
    for (size_t j = 0; j < x.size(); j++) {
      y[j] = std::exp(x[j] - alpha);
      denominator += y[j];
    }
    for (size_t j = 0; j < x.size(); j++) {
      y[j] /= denominator;
    }
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) override {
#if SOFTMAX_B_HALF == 0
    const size_t len = dy.size();

// auxilliary vector to store element wise softmax gradients of all elements

#if HAS_CXX11_THREAD_LOCAL
    thread_local
#endif
      vec_t df(len, 0);
    for (size_t j = 0; j < x.size(); j++) {
      for (size_t k = 0; k < x.size(); k++) {
        df[k] = (k == j) ? y[j] * (float_t(1) - y[j]) : -y[k] * y[j];
      }
      // dx = dy * (gradient of softmax)
      dx[j] = vectorize::dot(&dy[0], &df[0], len);
    }
#else
#if HAS_CXX11_THREAD_LOCAL
    thread_local
#endif

    std::vector<half> x_half = one_vector_to_half(x);
    std::vector<half> y_half = one_vector_to_half(y);
    std::vector<half> dx_half = one_vector_to_half(dx);
    std::vector<half> dy_half = one_vector_to_half(dy);

    const size_t len = dy.size();
    std::vector<float> dx_test(len, 0.0f); // 中間計算用の高精度ベクター

    for (size_t i = 0; i < len; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < len; ++j) {
            if (i == j) {
                sum += static_cast<float>(dy_half[j]) * static_cast<float>(y_half[i]) * (1.0f - static_cast<float>(y_half[i]));
            } else {
                sum -= static_cast<float>(dy_half[j]) * static_cast<float>(y_half[i]) * static_cast<float>(y_half[j]);
            }
        }
        dx_test[i] = sum; // 直接floatで計算
    }

    // 結果をhalfに変換
    for (size_t i = 0; i < len; ++i) {
        dx_half[i] = static_cast<half>(dx_test[i]);
    }

    // for (size_t i = 0; i < len; ++i) {
    //     printf("dx[%d] = %f\n", i, float(dx[i]));
    // }



    // const size_t len = dy.size();

    // // auxilliary vector to store element wise softmax gradients of all elements
    // std::vector<half> df_half(len, half(0));
    // for (size_t j = 0; j < x.size(); j++) {
    //   // for (size_t k = 0; k < x.size(); k++) {
    //   //   df_half[k] = (k == j) ? y_half[j] * (half(1) - y_half[j]) : -y_half[k] * y_half[j];
    //   // }
    //   // // dx = dy * (gradient of softmax)
    //   // dx_half[j] = vectorize::dot(&dy_half[0], &df_half[0], len);


    //   dx_half[j] = dy_half[j] * (y_half[j] * (half(1) - y_half[j]));

    //   for (size_t k = 0; k < len; k++) {
    //     if (k != j) {
    //       dx_half[j] -= dy_half[k] * y_half[j] * y_half[k];
    //     }
    //   }
    // }

    one_half_to_vector(dx, dx_half);
    
#endif
    
  }

  void backward_activation16(const vec16_t &x,
                             const vec16_t &y,
                             vec16_t &dx,
                             const vec16_t &dy) override {
#if HAS_CXX11_THREAD_LOCAL
    thread_local
#endif

    const size_t len = dy.size();
    std::vector<float> dx_test(len, 0.0f); // 中間計算用の高精度ベクター

    for (size_t i = 0; i < len; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < len; ++j) {
            if (i == j) {
                sum += static_cast<float>(dy[j]) * static_cast<float>(y[i]) * (1.0f - static_cast<float>(y[i]));
            } else {
                sum -= static_cast<float>(dy[j]) * static_cast<float>(y[i]) * static_cast<float>(y[j]);
            }
        }
        dx_test[i] = sum; // 直接floatで計算
    }

    // 結果をhalfに変換
    for (size_t i = 0; i < len; ++i) {
        dx[i] = static_cast<half>(dx_test[i]);
    }
        
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0), float_t(1));
  }

  friend struct serialization_buddy;
};

}  // namespace tiny_dnn
