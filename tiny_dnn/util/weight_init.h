/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "../util/util.h"

namespace tiny_dnn {
namespace weight_init {

class function {
 public:
  virtual void fill(vec_t *weight, size_t fan_in, size_t fan_out) = 0;
  virtual void fill16(vec16_t *weight, size_t fan_in, size_t fan_out) = 0;
  virtual float_t scale() const = 0;
};

class scalable : public function {
 public:
  explicit scalable(float_t value) : scale_(value) {}

  void scale(float_t value) { scale_ = value; }

  float_t scale() const override { return scale_; }  // 実装

 protected:
  float_t scale_;
};

/**
 * Use fan-in and fan-out for scaling
 *
 * X Glorot, Y Bengio,
 * Understanding the difficulty of training deep feedforward neural networks
 * Proc. AISTATS 10, May 2010, vol.9, pp249-256
 **/
class xavier : public scalable {
 public:
  xavier() : scalable(float_t(6)) {}
  explicit xavier(float_t value) : scalable(value) {}

  void fill(vec_t *weight, size_t fan_in, size_t fan_out) override {
    std::cout << "fan_in: " << fan_in << ", fan_out: " << fan_out << std::endl;
    const float_t weight_base = std::sqrt(scale_ / (fan_in + fan_out));
    
    std::cout << "weight_base: " << weight_base << std::endl;

    uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);

    std::cout << "1 weight[0]: " << (*weight)[0] << ", " << (void*)(weight) <<  std::endl;
  }

  void fill16(vec16_t *weight, size_t fan_in, size_t fan_out) override {
    std::cout << "fan_in: " << fan_in << ", fan_out: " << fan_out << std::endl;
    const half weight_base = half(std::sqrt(scale_ / (fan_in + fan_out)));

    std::cout << "weight_base: " << weight_base << std::endl;

    uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);

    std::cout << "2 weight[0]: " << (*weight)[0] << ", " << (void*)(weight) <<  std::endl;
  }
};

class xavier_half : public scalable {
 public:
  xavier_half() : scalable(half(6)) {}
  explicit xavier_half(half value) : scalable(value) {}

  void fill(vec_t *weight, size_t fan_in, size_t fan_out) override {
    // const half weight_base = std::sqrt(scale_ / (fan_in + fan_out));

    // uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
    // ここに来たらプログラムが終了する
    throw nn_error("xavier_half::fill");
  }

  void fill16(vec16_t *weight, size_t fan_in, size_t fan_out) override {
    const half weight_base = half(std::sqrt(scale_ / (fan_in + fan_out)));

    uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
  }
};

/**
 * Use fan-in(number of input weight for each neuron) for scaling
 *
 * Y LeCun, L Bottou, G B Orr, and K Muller,
 * Efficient backprop
 * Neural Networks, Tricks of the Trade, Springer, 1998
 **/
class lecun : public scalable {
 public:
  lecun() : scalable(float_t{1}) {}
  explicit lecun(float_t value) : scalable(value) {}

  void fill(vec_t *weight, size_t fan_in, size_t fan_out) override {
    CNN_UNREFERENCED_PARAMETER(fan_out);

    const float_t weight_base = scale_ / std::sqrt(float_t(fan_in));

    uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
  }

  void fill16(vec16_t *weight, size_t fan_in, size_t fan_out) override {
    CNN_UNREFERENCED_PARAMETER(fan_out);

    const half weight_base = half(scale_) / half(std::sqrt(float_t(fan_in)));

    uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
  }
};

class gaussian : public scalable {
 public:
  gaussian() : scalable(float_t{1}) {}
  explicit gaussian(float_t sigma) : scalable(sigma) {}

  void fill(vec_t *weight, size_t fan_in, size_t fan_out) override {
    CNN_UNREFERENCED_PARAMETER(fan_in);
    CNN_UNREFERENCED_PARAMETER(fan_out);

    gaussian_rand(weight->begin(), weight->end(), float_t{0}, scale_);
  }

  void fill16(vec16_t *weight, size_t fan_in, size_t fan_out) override {
    CNN_UNREFERENCED_PARAMETER(fan_in);
    CNN_UNREFERENCED_PARAMETER(fan_out);

    gaussian_rand(weight->begin(), weight->end(), half{0}, half(scale_));
  }
};

class constant : public scalable {
 public:
  constant() : scalable(float_t{0}) {}
  explicit constant(float_t value) : scalable(value) {}

  void fill(vec_t *weight, size_t fan_in, size_t fan_out) override {
    CNN_UNREFERENCED_PARAMETER(fan_in);
    CNN_UNREFERENCED_PARAMETER(fan_out);

    vectorize::fill(&(*weight)[0], weight->size(), scale_);
  }

  void fill16(vec16_t *weight, size_t fan_in, size_t fan_out) override {
    CNN_UNREFERENCED_PARAMETER(fan_in);
    CNN_UNREFERENCED_PARAMETER(fan_out);

    vectorize::fill(&(*weight)[0], weight->size(), half(scale_));
  }
};

class he : public scalable {
 public:
  he() : scalable(float_t{2}) {}
  explicit he(float_t value) : scalable(value) {}

  void fill(vec_t *weight, size_t fan_in, size_t fan_out) override {
    CNN_UNREFERENCED_PARAMETER(fan_out);

    const float_t sigma = std::sqrt(scale_ / fan_in);

    gaussian_rand(weight->begin(), weight->end(), float_t{0}, sigma);
  }

  void fill16(vec16_t *weight, size_t fan_in, size_t fan_out) override {
    CNN_UNREFERENCED_PARAMETER(fan_out);

    const half sigma = half(std::sqrt(scale_ / fan_in));

    gaussian_rand(weight->begin(), weight->end(), half{0}, sigma);
  }
};

// class custom_init : public scalable {
//  public:
//   custom_init(const vec_t& weights) : weights_(weights) {}

//   void fill(vec_t *weight, size_t fan_in, size_t fan_out) override {
//     CNN_UNREFERENCED_PARAMETER(fan_in);
//     CNN_UNREFERENCED_PARAMETER(fan_out);
//     *weight = weights;

//   }

//  private:
//   vec_t weights_;
// };

}  // namespace weight_init
}  // namespace tiny_dnn
