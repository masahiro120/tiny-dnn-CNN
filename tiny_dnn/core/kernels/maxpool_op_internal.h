/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <limits>
#include <vector>

extern int MAX_POOLING_F_HALF;
extern int MAX_POOLING_B_HALF;

#include "../../half.hpp"
#include "../../half_define.h"
using namespace half_float;

std::vector<half> one_vector_to_half(const std::vector<size_t>& array);
std::vector<std::vector<half>> two_vector_to_half(const tiny_dnn::tensor_t& array);
std::vector<std::vector<half>> two_vector_to_half(const std::vector<std::vector<size_t>>& array);
void two_half_to_vector(tiny_dnn::tensor_t& array, std::vector<std::vector<half>> array_half);
void two_half_to_vector(std::vector<std::vector<size_t>>& array, std::vector<std::vector<half>> array_half);


namespace tiny_dnn {
namespace kernels {

inline void maxpool_op_internal(const tensor_t &in_data,
                                tensor_t &out_data,
                                std::vector<std::vector<size_t>> &max_idx,
                                const std::vector<std::vector<size_t>> &out2in,
                                const bool layer_parallelize) {
#if MAX_POOLING_F_HALF == 0

  for_i(layer_parallelize, in_data.size(), [&](size_t sample) {
    const vec_t &in          = in_data[sample];
    vec_t &out               = out_data[sample];
    std::vector<size_t> &max = max_idx[sample];

    for (size_t i = 0; i < out2in.size(); i++) {
      const auto &in_index = out2in[i];
      float_t max_value    = std::numeric_limits<float_t>::lowest();
      size_t idx           = 0;
      for (auto j : in_index) {
        if (in[j] > max_value) {
          max_value = in[j];
          idx       = j;
        }
      }
      max[i] = idx;
      out[i] = max_value;
    }
  });

#else
  std::vector<std::vector<half>> in_data_half = two_vector_to_half(in_data);
  std::vector<std::vector<half>> out_data_half = two_vector_to_half(out_data);
  std::vector<std::vector<half>> max_idx_half = two_vector_to_half(max_idx);
  std::vector<std::vector<half>> out2in_half = two_vector_to_half(out2in);

  for_i(layer_parallelize, in_data_half.size(), [&](size_t sample) {
    const std::vector<half> &in          = in_data_half[sample];
    std::vector<half> &out               = out_data_half[sample];
    std::vector<half> &max = max_idx_half[sample];

    for (size_t i = 0; i < out2in.size(); i++) {
      const auto &in_index = out2in_half[i];
      half max_value    = std::numeric_limits<half>::lowest();
      size_t idx           = 0;
      for (auto j : in_index) {
        if (in[j] > max_value) {
          max_value = in[j];
          idx       = j;
        }
      }
      max[i] = idx;
      out[i] = max_value;
    }
  });

  two_half_to_vector(out_data, out_data_half);
  two_half_to_vector(max_idx, max_idx_half);

#endif
}

inline void maxpool_grad_op_internal(tensor_t &prev_delta,
                                     const tensor_t &curr_delta,
                                     std::vector<std::vector<size_t>> &max_idx,
                                     const std::vector<size_t> &in2out,
                                     const bool layer_parallelize) {
#if MAX_POOLING_B_HALF == 0

  for_i(layer_parallelize, prev_delta.size(), [&](size_t sample) {
    vec_t &prev                    = prev_delta[sample];
    const vec_t &curr              = curr_delta[sample];
    const std::vector<size_t> &max = max_idx[sample];

    for (size_t i = 0; i < in2out.size(); i++) {
      size_t outi = in2out[i];
      prev[i]     = (max[outi] == i) ? curr[outi] : float_t{0};
    }
  });

#else
  std::vector<std::vector<half>> prev_delta_half = two_vector_to_half(prev_delta);
  std::vector<std::vector<half>> curr_delta_half = two_vector_to_half(curr_delta);
  std::vector<std::vector<half>> max_idx_half = two_vector_to_half(max_idx);
  std::vector<half> in2out_half = one_vector_to_half(in2out);

  for_i(layer_parallelize, prev_delta_half.size(), [&](size_t sample) {
    std::vector<half> &prev                    = prev_delta_half[sample];
    const std::vector<half> &curr              = curr_delta_half[sample];
    const std::vector<half> &max = max_idx_half[sample];

    for (size_t i = 0; i < in2out_half.size(); i++) {
      size_t outi = in2out_half[i];
      prev[i] = (max[outi] == i) ? curr[outi] : half{0};
    }
  });

  two_half_to_vector(prev_delta, prev_delta_half);

#endif
}

}  // namespace kernels
}  // namespace tiny_dnn
