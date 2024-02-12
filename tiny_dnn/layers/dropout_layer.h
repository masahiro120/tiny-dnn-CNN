/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "../layers/layer.h"
#include "../util/util.h"

#include "../half.hpp"
#include "../half_define.h"

// #define DROP_OUT_F_HALF 1
// #define DROP_OUT_B_HALF 1

std::vector<half> one_vector_to_half(const tiny_dnn::vec_t& array);
std::vector<std::vector<half>> two_vector_to_half(const tiny_dnn::tensor_t& array);
std::vector<std::vector<std::vector<half>>> three_vector_to_half(const std::vector<tiny_dnn::tensor_t>& array);
void one_half_to_vector(tiny_dnn::vec_t& array, std::vector<half> array_half);
void three_half_to_vector(std::vector<tiny_dnn::tensor_t>& array, std::vector<std::vector<std::vector<half>>> array_half);
bool bernoulli_half(half p);

namespace tiny_dnn {

/**
 * applies dropout to the input
 **/
class dropout_layer : public layer {
 public:
  /**
   * @param in_dim       [in] number of elements of the input
   * @param dropout_rate [in] (0-1) fraction of the input units to be dropped
   * @param phase        [in] initial state of the dropout
   **/
  dropout_layer(size_t in_dim,
                float_t dropout_rate,
                net_phase phase = net_phase::train)
    : layer({vector_type::data}, {vector_type::data}),
      phase_(phase),
      dropout_rate_(dropout_rate),
      scale_(float_t(1) / (float_t(1) - dropout_rate_)),
      in_size_(in_dim) {
    mask_.resize(1, std::vector<uint8_t>(in_dim));
    clear_mask();
  }

  dropout_layer(const dropout_layer &obj) = default;
  virtual ~dropout_layer() {}

  dropout_layer(dropout_layer &&obj) = default;
  dropout_layer &operator=(const dropout_layer &obj) = default;
  dropout_layer &operator=(dropout_layer &&obj) = default;

  void set_dropout_rate(float_t rate) {
    dropout_rate_ = rate;
    scale_        = float_t(1) / (float_t(1) - dropout_rate_);
  }

  float_t dropout_rate() const { return dropout_rate_; }

  ///< number of incoming connections for each output unit
  size_t fan_in_size() const override { return 1; }

  ///< number of outgoing connections for each input unit
  size_t fan_out_size() const override { return 1; }

  std::vector<index3d<size_t>> in_shape() const override {
    return {index3d<size_t>(in_size_, 1, 1)};
  }

  std::vector<index3d<size_t>> out_shape() const override {
    return {index3d<size_t>(in_size_, 1, 1)};
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
#if DROP_OUT_B_HALF == 0
    tensor_t &prev_delta       = *in_grad[0];
    const tensor_t &curr_delta = *out_grad[0];

    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);

    for_i(prev_delta.size(), [&](size_t sample) {
      // assert(prev_delta[sample].size() == curr_delta[sample].size());
      // assert(mask_[sample].size() == prev_delta[sample].size());
      size_t sz = prev_delta[sample].size();
      for (size_t i = 0; i < sz; ++i) {
        prev_delta[sample][i] = mask_[sample][i] * curr_delta[sample][i];
      }
    });
#else

    std::vector<tiny_dnn::tensor_t> in_data_val(in_data.size());
    std::vector<tiny_dnn::tensor_t> out_data_val(out_data.size());
    std::vector<tiny_dnn::tensor_t> in_grad_val(in_grad.size());
    std::vector<tiny_dnn::tensor_t> out_grad_val(out_grad.size());

    for (size_t i = 0; i < in_data.size(); ++i) {
        in_data_val[i] = *(in_data[i]); // ポインタのデリファレンス
    }

    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data_val[i] = *(out_data[i]); // ポインタのデリファレンス
    }

    for (size_t i = 0; i < in_grad.size(); ++i) {
        in_grad_val[i] = *(in_grad[i]); // ポインタのデリファレンス
    }

    for (size_t i = 0; i < out_grad.size(); ++i) {
        out_grad_val[i] = *(out_grad[i]); // ポインタのデリファレンス
    }

    std::vector<std::vector<std::vector<half>>> in_data_half = three_vector_to_half(in_data_val);
    std::vector<std::vector<std::vector<half>>> out_data_half = three_vector_to_half(out_data_val);
    std::vector<std::vector<std::vector<half>>> in_grad_half = three_vector_to_half(in_grad_val);
    std::vector<std::vector<std::vector<half>>> out_grad_half = three_vector_to_half(out_grad_val);

    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);

    for_i(in_grad_val[0].size(), [&](size_t sample) {
      // assert(prev_delta[sample].size() == curr_delta[sample].size());
      // assert(mask_[sample].size() == prev_delta[sample].size());
      size_t sz = in_grad_val[0][sample].size();
      for (size_t i = 0; i < sz; ++i) {
        in_grad_val[0][sample][i] = mask_[sample][i] * out_grad_val[0][sample][i];
      }
    });

    three_half_to_vector(in_grad_val, in_grad_half);

    for (size_t i = 0; i < in_grad.size(); ++i) {
        *(in_grad[i]) = in_grad_val[i]; // ポインタのデリファレンス
    }
#endif
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    tiny_dnn::set_random_seed(123);
#if DROP_OUT_F_HALF == 0

#if 0

    const tensor_t &in = *in_data[0];
    tensor_t &out      = *out_data[0];

    const size_t sample_count = in.size();

    if (mask_.size() < sample_count) {
      mask_.resize(sample_count, mask_[0]);
    }

    for_i(sample_count, [&](size_t sample) {
      std::vector<uint8_t> &mask = mask_[sample];

      const vec_t &in_vec = in[sample];
      vec_t &out_vec      = out[sample];

      if (phase_ == net_phase::train) {
        for (size_t i = 0; i < in_vec.size(); i++)
          mask[i]     = bernoulli(dropout_rate_);

        for (size_t i = 0; i < in_vec.size(); i++)
          out_vec[i]  = mask[i] * scale_ * in_vec[i];
      } else {
        for (size_t i = 0, end = in_vec.size(); i < end; i++)
          out_vec[i] = in_vec[i];
      }
    });
#else
    std::vector<tiny_dnn::tensor_t> in_data_val(in_data.size());
    std::vector<tiny_dnn::tensor_t> out_data_val(out_data.size());

    for (size_t i = 0; i < in_data.size(); ++i) {
        in_data_val[i] = *(in_data[i]); // ポインタのデリファレンス
    }

    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data_val[i] = *(out_data[i]); // ポインタのデリファレンス
    }

    const size_t sample_count = in_data_val[0].size();

    if (mask_.size() < sample_count) {
      mask_.resize(sample_count, mask_[0]);
    }

    for_i(sample_count, [&](size_t sample) {
      std::vector<uint8_t> &mask = mask_[sample];

      const vec_t &in_vec = in_data_val[0][sample];
      vec_t &out_vec      = out_data_val[0][sample];

      if (phase_ == net_phase::train) {
        for (size_t i = 0; i < in_vec.size(); i++)
          mask[i]     = bernoulli(dropout_rate_);

        for (size_t i = 0; i < in_vec.size(); i++)
          out_vec[i]  = mask[i] * scale_ * in_vec[i];
      } else {
        for (size_t i = 0, end = in_vec.size(); i < end; i++)
          out_vec[i] = in_vec[i];
      }
    });

    for (size_t i = 0; i < out_data.size(); ++i) {
        *(out_data[i]) = out_data_val[i]; // ポインタのデリファレンス
    }

#endif
#else
    std::vector<tiny_dnn::tensor_t> in_data_val(in_data.size());
    std::vector<tiny_dnn::tensor_t> out_data_val(out_data.size());

    for (size_t i = 0; i < in_data.size(); ++i) {
        in_data_val[i] = *(in_data[i]); // ポインタのデリファレンス
    }

    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data_val[i] = *(out_data[i]); // ポインタのデリファレンス
    }

    // 変換関数を呼び出します。
    std::vector<std::vector<std::vector<half>>> in_data_half = three_vector_to_half(in_data_val);
    std::vector<std::vector<std::vector<half>>> out_data_half = three_vector_to_half(out_data_val);

    const size_t sample_count = in_data_half[0].size();

    if (mask_.size() < sample_count) {
      mask_.resize(sample_count, mask_[0]);
    }

    for_i(sample_count, [&](size_t sample) {
      std::vector<uint8_t> &mask = mask_[sample];

      const std::vector<half> &in_vec = in_data_half[0][sample];
      std::vector<half> &out_vec      = out_data_half[0][sample];

      if (phase_ == net_phase::train) {
        for (size_t i = 0; i < in_vec.size(); i++)
          mask[i]     = bernoulli_half(half(dropout_rate_));

        for (size_t i = 0; i < in_vec.size(); i++)
          out_vec[i]  = mask[i] * half(scale_) * in_vec[i];
      } else {
        for (size_t i = 0, end = in_vec.size(); i < end; i++)
          out_vec[i] = in_vec[i];
      }
    });

    three_half_to_vector(out_data_val, out_data_half);

    for (size_t i = 0; i < out_data.size(); ++i) {
        *(out_data[i]) = out_data_val[i]; // ポインタのデリファレンス
    }

#endif
  }

  /**
   * set dropout-context (training-phase or test-phase)
   **/
  void set_context(net_phase ctx) override { phase_ = ctx; }

  std::string layer_type() const override { return "dropout"; }

  // currently used by tests only
  const std::vector<uint8_t> &get_mask(size_t sample_index) const {
    return mask_[sample_index];
  }

  void clear_mask() {
    for (auto &sample : mask_) {
      std::fill(sample.begin(), sample.end(), 0);
    }
  }

  friend struct serialization_buddy;

 private:
  net_phase phase_;
  float_t dropout_rate_;
  float_t scale_;
  size_t in_size_;
  std::vector<std::vector<uint8_t>> mask_;
};

}  // namespace tiny_dnn
