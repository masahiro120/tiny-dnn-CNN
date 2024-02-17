/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "../layers/layer.h"
#include "../util/math_functions.h"
#include "../util/util.h"

#include "../half.hpp"
#include "../half_define.h"

// #define BATCH_NORM_F_HALF 1
// #define BATCH_NORM_B_HALF 1

std::vector<half> one_vector_to_half(const tiny_dnn::vec_t& array);
// vec16_t one_vector_to_half16(const tiny_dnn::vec_t& array);
std::vector<std::vector<half>> two_vector_to_half(const tiny_dnn::tensor_t& array);
std::vector<std::vector<std::vector<half>>> three_vector_to_half(const std::vector<tiny_dnn::tensor_t>& array);
std::vector<tiny_dnn::tensor16_t> three_vector_to_half16(const std::vector<tiny_dnn::tensor_t>& array);
void moments_half(const std::vector<std::vector<half>> &in, size_t spatial_dim, size_t channels, std::vector<half> &mean);
void moments_half(const std::vector<std::vector<half>> &in, size_t spatial_dim, size_t channels, std::vector<half> &mean, std::vector<half> &variance);
void moments_half(const std::vector<std::vector<half>> &in, size_t spatial_dim, size_t channels, tiny_dnn::vec_t &mean, tiny_dnn::vec_t &variance);
void one_half_to_vector(tiny_dnn::vec_t& array, std::vector<half> array_half);
void three_half_to_vector(std::vector<tiny_dnn::tensor_t>& array, std::vector<std::vector<std::vector<half>>> array_half);

extern int batch_count;

namespace tiny_dnn {

/**
 * Batch Normalization
 *
 * Normalize the activations of the previous layer at each batch
 **/
class batch_normalization_layer : public layer {
 public:
  typedef layer Base;

  /**
   * @param prev_layer      [in] previous layer to be connected with this layer
   * @param epsilon         [in] small positive value to avoid zero-division
   * @param momentum        [in] momentum in the computation of the exponential
   *average of the mean/stddev of the data
   * @param phase           [in] specify the current context (train/test)
   **/
  batch_normalization_layer(const layer &prev_layer,
                            float_t epsilon  = 1e-5,
                            float_t momentum = 0.5,
                            net_phase phase  = net_phase::train)
    : Base({vector_type::data}, {vector_type::data}),
      in_channels_(prev_layer.out_shape()[0].depth_),
      in_spatial_size_(prev_layer.out_shape()[0].area()),
      phase_(phase),
      momentum_(momentum),
      eps_(epsilon),
      update_immidiately_(false) {
    init();
  }

  /**
   * @param in_spatial_size [in] spatial size (WxH) of the input data
   * @param in_channels     [in] channels of the input data
   * @param epsilon         [in] small positive value to avoid zero-division
   * @param momentum        [in] momentum in the computation of the exponential
   *average of the mean/stddev of the data
   * @param phase           [in] specify the current context (train/test)
   **/
  batch_normalization_layer(size_t in_spatial_size,
                            size_t in_channels,
                            float_t epsilon  = 1e-5,
                            float_t momentum = 0.5,
                            net_phase phase  = net_phase::train)
    : Base({vector_type::data}, {vector_type::data}),
      in_channels_(in_channels),
      in_spatial_size_(in_spatial_size),
      phase_(phase),
      momentum_(momentum),
      eps_(epsilon),
      update_immidiately_(false) {
    init();
  }

  virtual ~batch_normalization_layer() {}

  ///< number of incoming connections for each output unit
  size_t fan_in_size() const override { return 1; }

  ///< number of outgoing connections for each input unit
  size_t fan_out_size() const override { return 1; }

  std::vector<index3d<size_t>> in_shape() const override {
    return {index3d<size_t>(in_spatial_size_, 1, in_channels_)};
  }

  std::vector<index3d<size_t>> out_shape() const override {
    return {index3d<size_t>(in_spatial_size_, 1, in_channels_)};
  }

//   void back_propagation(const std::vector<tensor_t *> &in_data,
//                         const std::vector<tensor_t *> &out_data,
//                         std::vector<tensor_t *> &out_grad,
//                         std::vector<tensor_t *> &in_grad) override {
// #if BATCH_NORM_B_HALF == 0

// #if 1
//     tensor_t &prev_delta     = *in_grad[0];
//     tensor_t &curr_delta     = *out_grad[0];
//     const tensor_t &curr_out = *out_data[0];
//     const size_t num_samples = curr_out.size();

//     CNN_UNREFERENCED_PARAMETER(in_data);

//     tensor_t delta_dot_y = curr_out;
//     vec_t mean_delta_dot_y, mean_delta, mean_Y;

//     for (size_t i = 0; i < num_samples; i++) {
//       for (size_t j = 0; j < curr_out[0].size(); j++) {
//         delta_dot_y[i][j] *= curr_delta[i][j];
//       }
//     }

//     moments(delta_dot_y, in_spatial_size_, in_channels_, mean_delta_dot_y);
//     moments(curr_delta, in_spatial_size_, in_channels_, mean_delta);
//     // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
//     //
//     // dE(Y)/dX =
//     //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
//     //     ./ sqrt(var(X) + eps)
//     //
//     for_i(num_samples, [&](size_t i) {
//       for (size_t j = 0; j < in_channels_; j++) {
//         for (size_t k = 0; k < in_spatial_size_; k++) {
//           size_t index = j * in_spatial_size_ + k;

//           prev_delta[i][index] = curr_delta[i][index] - mean_delta[j] -
//                                  mean_delta_dot_y[j] * curr_out[i][index];

//           // stddev_ is calculated in the forward pass
//           prev_delta[i][index] /= stddev_[j];
//         }
//       }
//     });
// #else
//     std::vector<tiny_dnn::tensor_t> in_data_val(in_data.size());
//     std::vector<tiny_dnn::tensor_t> out_data_val(out_data.size());
//     std::vector<tiny_dnn::tensor_t> in_grad_val(in_grad.size());
//     std::vector<tiny_dnn::tensor_t> out_grad_val(out_grad.size());

//     for (size_t i = 0; i < in_data.size(); ++i) {
//         in_data_val[i] = *(in_data[i]); // ポインタのデリファレンス
//     }

//     for (size_t i = 0; i < out_data.size(); ++i) {
//         out_data_val[i] = *(out_data[i]); // ポインタのデリファレンス
//     }

//     for (size_t i = 0; i < in_grad.size(); ++i) {
//         in_grad_val[i] = *(in_grad[i]); // ポインタのデリファレンス
//     }

//     for (size_t i = 0; i < out_grad.size(); ++i) {
//         out_grad_val[i] = *(out_grad[i]); // ポインタのデリファレンス
//     }
    
//     const size_t num_samples = out_data_val[0].size();

//     CNN_UNREFERENCED_PARAMETER(in_data);

//     tensor_t delta_dot_y = out_data_val[0];
//     vec_t mean_delta_dot_y, mean_delta, mean_Y;

//     for (size_t i = 0; i < num_samples; i++) {
//       for (size_t j = 0; j < out_data_val[0][0].size(); j++) {
//         delta_dot_y[i][j] *= out_grad_val[0][i][j];
//       }
//     }

//     moments(delta_dot_y, in_spatial_size_, in_channels_, mean_delta_dot_y);
//     moments(out_grad_val[0], in_spatial_size_, in_channels_, mean_delta);

//     for_i(num_samples, [&](size_t i) {
//       for (size_t j = 0; j < in_channels_; j++) {
//         for (size_t k = 0; k < in_spatial_size_; k++) {
//           size_t index = j * in_spatial_size_ + k;

//           in_grad_val[0][i][index] = out_grad_val[0][i][index] - mean_delta[j] -
//                                  mean_delta_dot_y[j] * out_data_val[0][i][index];

//           // stddev_ is calculated in the forward pass
//           in_grad_val[0][i][index] /= stddev_[j];
//         }
//       }
//     });

//     for (size_t i = 0; i < in_grad.size(); ++i) {
//         *(in_grad[i]) = in_grad_val[i]; // ポインタのデリファレンス
//     }

// #endif
// #else

//     std::vector<tiny_dnn::tensor_t> in_data_val(in_data.size());
//     std::vector<tiny_dnn::tensor_t> out_data_val(out_data.size());
//     std::vector<tiny_dnn::tensor_t> in_grad_val(in_grad.size());
//     std::vector<tiny_dnn::tensor_t> out_grad_val(out_grad.size());

//     for (size_t i = 0; i < in_data.size(); ++i) {
//         in_data_val[i] = *(in_data[i]); // ポインタのデリファレンス
//     }

//     for (size_t i = 0; i < out_data.size(); ++i) {
//         out_data_val[i] = *(out_data[i]); // ポインタのデリファレンス
//     }

//     for (size_t i = 0; i < in_grad.size(); ++i) {
//         in_grad_val[i] = *(in_grad[i]); // ポインタのデリファレンス
//     }

//     for (size_t i = 0; i < out_grad.size(); ++i) {
//         out_grad_val[i] = *(out_grad[i]); // ポインタのデリファレンス
//     }

//     std::vector<std::vector<std::vector<half>>> in_data_half = three_vector_to_half(in_data_val);
//     std::vector<std::vector<std::vector<half>>> out_data_half = three_vector_to_half(out_data_val);
//     std::vector<std::vector<std::vector<half>>> in_grad_half = three_vector_to_half(in_grad_val);
//     std::vector<std::vector<std::vector<half>>> out_grad_half = three_vector_to_half(out_grad_val);

//     const size_t num_samples = out_data_half[0].size();

//     CNN_UNREFERENCED_PARAMETER(in_data);

//     std::vector<std::vector<half>> delta_dot_y_half = out_data_half[0];
//     std::vector<half> mean_delta_dot_y_half, mean_delta_half, mean_Y_half;

//     for (size_t i = 0; i < num_samples; i++) {
//       for (size_t j = 0; j < out_data_half[0][0].size(); j++) {
//         delta_dot_y_half[i][j] *= out_grad_half[0][i][j];
//       }
//     }

//     moments_half(delta_dot_y_half, in_spatial_size_, in_channels_, mean_delta_dot_y_half);
//     moments_half(out_grad_half[0], in_spatial_size_, in_channels_, mean_delta_half);

//     std::vector<half> stddev_half = one_vector_to_half(stddev_);

//     for_i(num_samples, [&](size_t i) {
//       for (size_t j = 0; j < in_channels_; j++) {
//         for (size_t k = 0; k < in_spatial_size_; k++) {
//           size_t index = j * in_spatial_size_ + k;

//           in_grad_half[0][i][index] = out_grad_half[0][i][index] - mean_delta_half[j] -
//                                  mean_delta_dot_y_half[j] * out_data_half[0][i][index];

//           // stddev_ is calculated in the forward pass
//           in_grad_half[0][i][index] /= stddev_half[j];
//         }
//       }
//     });

//     three_half_to_vector(in_grad_val, in_grad_half);

//     for (size_t i = 0; i < in_grad.size(); ++i) {
//         *(in_grad[i]) = in_grad_val[i]; // ポインタのデリファレンス
//     }
// #endif
//   }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {}

  // void back_propagation16(const std::vector<tensor16_t *> &in_data,
  //                         const std::vector<tensor16_t *> &out_data,
  //                         std::vector<tensor16_t *> &out_grad,
  //                         std::vector<tensor16_t *> &in_grad) override {
    
  //   std::vector<tiny_dnn::tensor16_t> in_data_val(in_data.size());
  //   std::vector<tiny_dnn::tensor16_t> out_data_val(out_data.size());
  //   std::vector<tiny_dnn::tensor16_t> in_grad_val(in_grad.size());
  //   std::vector<tiny_dnn::tensor16_t> out_grad_val(out_grad.size());

  //   for (size_t i = 0; i < in_data.size(); ++i) {
  //       in_data_val[i] = *(in_data[i]); // ポインタのデリファレンス
  //   }

  //   for (size_t i = 0; i < out_data.size(); ++i) {
  //       out_data_val[i] = *(out_data[i]); // ポインタのデリファレンス
  //   }

  //   for (size_t i = 0; i < in_grad.size(); ++i) {
  //       in_grad_val[i] = *(in_grad[i]); // ポインタのデリファレンス
  //   }

  //   for (size_t i = 0; i < out_grad.size(); ++i) {
  //       out_grad_val[i] = *(out_grad[i]); // ポインタのデリファレンス
  //   }

  //   const size_t num_samples = out_data_val[0].size();

  //   CNN_UNREFERENCED_PARAMETER(in_data);

  //   tensor16_t delta_dot_y_half = out_data_val[0];
  //   vec16_t mean_delta_dot_y_half, mean_delta_half, mean_Y_half;

  //   for (size_t i = 0; i < num_samples; i++) {
  //     for (size_t j = 0; j < out_data_val[0][0].size(); j++) {
  //       delta_dot_y_half[i][j] *= out_grad_val[0][i][j];
  //     }
  //   }

  //   // moments_half(delta_dot_y_half, in_spatial_size_, in_channels_, mean_delta_dot_y_half);
  //   // moments_half(out_grad_val[0], in_spatial_size_, in_channels_, mean_delta_half);

  //   // vec16_t stddev_half = one_vector_to_half16(stddev_);
  //   vec16_t stddev_half;

  //   for_i(num_samples, [&](size_t i) {
  //     for (size_t j = 0; j < in_channels_; j++) {
  //       for (size_t k = 0; k < in_spatial_size_; k++) {
  //         size_t index = j * in_spatial_size_ + k;

  //         in_grad_val[0][i][index] = out_grad_val[0][i][index] - mean_delta_half[j] -
  //                                mean_delta_dot_y_half[j] * out_data_val[0][i][index];

  //         // stddev_ is calculated in the forward pass
  //         in_grad_val[0][i][index] /= stddev_half[j];
  //       }
  //     }
  //   });

  //   for (size_t i = 0; i < in_grad.size(); ++i) {
  //       *(in_grad[i]) = in_grad_val[i]; // ポインタのデリファレンス
  //   }
  // }

  void back_propagation16(const std::vector<tensor16_t *> &in_data,
                          const std::vector<tensor16_t *> &out_data,
                          std::vector<tensor16_t *> &out_grad,
                          std::vector<tensor16_t *> &in_grad) override {}

//   void forward_propagation(const std::vector<tensor_t *> &in_data,
//                            std::vector<tensor_t *> &out_data) override {
// #if BATCH_NORM_F_HALF == 0
// #if 0
//     vec_t &mean = (phase_ == net_phase::train) ? mean_current_ : mean_;
//     vec_t &variance =
//       (phase_ == net_phase::train) ? variance_current_ : variance_;
//     tensor_t &in  = *in_data[0];
//     tensor_t &out = *out_data[0];

//     if (phase_ == net_phase::train) {
//       // calculate mean/variance from this batch in train phase
//       moments(*in_data[0], in_spatial_size_, in_channels_, mean, variance);
//     }

//     // y = (x - mean) ./ sqrt(variance + eps)
//     calc_stddev(variance);

//     for_i(in_data[0]->size(), [&](size_t i) {
//       const float_t *inptr = &in[i][0];
//       float_t *outptr      = &out[i][0];

//       for (size_t j = 0; j < in_channels_; j++) {
//         float_t m = mean[j];

//         for (size_t k = 0; k < in_spatial_size_; k++) {
//           *outptr++ = (*inptr++ - m) / stddev_[j];
//         }
//       }
//     });

//     if (phase_ == net_phase::train && update_immidiately_) {
//       mean_     = mean_current_;
//       variance_ = variance_current_;
//     }

// #else

//     std::vector<tiny_dnn::tensor_t> in_data_val(in_data.size());
//     std::vector<tiny_dnn::tensor_t> out_data_val(out_data.size());

//     for (size_t i = 0; i < in_data.size(); ++i) {
//         in_data_val[i] = *(in_data[i]); // ポインタのデリファレンス
//     }

//     for (size_t i = 0; i < out_data.size(); ++i) {
//         out_data_val[i] = *(out_data[i]); // ポインタのデリファレンス
//     }

//     // batch_count++;
//     // if (batch_count == 5) {
//     //   batch_count = 1;
//     // }
//     // std::cout << batch_count << " Before Batch Normalization" << std::endl;
//     // for (size_t i = 0; i < 10; ++i) {
//     //   for (size_t j = 0; j < 2; ++j) {
//     //     std::cout << in_data_val[0][i][j] << std::endl;
//     //   }
//     // }
//     // std::cout << "mean_ = " << mean_[0] << std::endl;
//     // std::cout << "variance_ = " << variance_[0] << std::endl;
//     // std::cout << "stddev_ = " << stddev_[0] << std::endl << std::endl;
//     // std::cout << "mean_current_ = " << mean_current_[0] << std::endl;
//     // std::cout << "variance_current_ = " << variance_current_[0] << std::endl;
//     // std::cout << "tmp_mean_ = " << tmp_mean_[0] << std::endl;
//     // std::cout << std::endl;

//     vec_t mean;
//     vec_t variance;

//     if (phase_ == net_phase::train) {
//       // calculate mean/variance from this batch in train phase
//       mean = mean_current_;
//       variance = variance_current_;
//       moments(in_data_val[0], in_spatial_size_, in_channels_, mean, variance);
//     } else {
//       mean = mean_;
//       variance = variance_;
//     }

//     vec_t stddev = stddev_;

//     for (size_t i = 0; i < in_channels_; i++) {
//       stddev[i] = sqrt(variance[i] + float(eps_));
//     }

//     for_i(in_data[0]->size(), [&](size_t i) {
//       for (size_t j = 0; j < in_channels_; j++) {
//         float m = mean[j];

//         for (size_t k = 0; k < in_spatial_size_; k++) {
//           out_data_val[0][i][j * in_spatial_size_ + k] = (in_data_val[0][i][j * in_spatial_size_ + k] - m) / stddev[j];
//         }
//       }
//     });


//     if (phase_ == net_phase::train && update_immidiately_) {
//       mean_     = mean;
//       // one_half_to_vector(mean_, mean_half);
//       variance_ = variance;
//       // one_half_to_vector(variance_, variance_half);
//     } else {
//       mean_current_ = mean;
//       variance_current_ = variance;
//     }

//     // one_half_to_vector(stddev_, stddev_half);
//     stddev_ = stddev;
    
//     // three_half_to_vector(out_data_val, out_data_half);
    

//     for (size_t i = 0; i < out_data.size(); ++i) {
//         *(out_data[i]) = out_data_val[i]; // ポインタのデリファレンス
//     }


//     // std::cout << "After Batch Normalization" << std::endl;
//     // for (size_t i = 0; i < 10; i++) {
//     //   for (size_t j = 0; j < 2; j++) {
//     //     std::cout << out_data_val[0][i][j] << std::endl;
//     //   }
//     // }

//     // std::cout << "mean_ = " << mean_[0] << std::endl;
//     // std::cout << "variance_ = " << variance_[0] << std::endl;
//     // std::cout << "stddev_ = " << stddev_[0] << std::endl << std::endl;
//     // std::cout << "mean_current_ = " << mean_current_[0] << std::endl;
//     // std::cout << "variance_current_ = " << variance_current_[0] << std::endl;
//     // std::cout << "tmp_mean_ = " << tmp_mean_[0] << std::endl;
//     // std::cout << std::endl;
//     // std::cout << std::endl;

//     // out check
//     // std::cout << "After forward_propagation" << std::endl;
//     // std::cout << "out_data_val[0][0][0] = " << out_data_val[0][0][0] << std::endl;

// #endif
//     // std::cout << "*out_data[0][0][0] = " << (*out_data[0])[0][0] << std::endl;
//     // std::cout << "mean_ = " << mean_[0] << std::endl;
//     // std::cout << "variance_ = " << variance_[0] << std::endl;
//     // std::cout << "stddev_ = " << stddev_[0] << std::endl << std::endl;
//     // std::cout << "mean_current_ = " << mean_current_[0] << std::endl;
//     // std::cout << "variance_current_ = " << variance_current_[0] << std::endl;
//     // std::cout << "tmp_mean_ = " << tmp_mean_[0] << std::endl;
//     // std::cout << std::endl;


// #else

// #if 0
//     // std::vector<std::vector<std::vector<half>>> in_data_half = three_vector_to_half(*in_data);
//     // std::vector<std::vector<std::vector<half>>> out_data_half = three_vector_to_half(*out_data);


//     // in_dataとout_dataから値のベクターを作成します。
//     std::vector<tiny_dnn::tensor_t> in_data_val(in_data.size());
//     std::vector<tiny_dnn::tensor_t> out_data_val(out_data.size());

//     for (size_t i = 0; i < in_data.size(); ++i) {
//         in_data_val[i] = *(in_data[i]); // ポインタのデリファレンス
//     }

//     for (size_t i = 0; i < out_data.size(); ++i) {
//         out_data_val[i] = *(out_data[i]); // ポインタのデリファレンス
//     }

//     // batch_count++;
//     // if (batch_count == 5) {
//     //   batch_count = 1;
//     // }
//     // std::cout << batch_count << " Before Batch Normalization" << std::endl;
//     // for (size_t i = 0; i < 10; ++i) {
//     //   for (size_t j = 0; j < 2; ++j) {
//     //     std::cout << in_data_val[0][i][j] << std::endl;
//     //   }
//     // }
//     // std::cout << "mean_ = " << mean_[0] << std::endl;
//     // std::cout << "variance_ = " << variance_[0] << std::endl;
//     // std::cout << "stddev_ = " << stddev_[0] << std::endl << std::endl;
//     // std::cout << "mean_current_ = " << mean_current_[0] << std::endl;
//     // std::cout << "variance_current_ = " << variance_current_[0] << std::endl;
//     // std::cout << "tmp_mean_ = " << tmp_mean_[0] << std::endl;
//     // std::cout << std::endl;

//     // 変換関数を呼び出します。
//     std::vector<std::vector<std::vector<half>>> in_data_half = three_vector_to_half(in_data_val);
//     std::vector<std::vector<std::vector<half>>> out_data_half = three_vector_to_half(out_data_val);
    
//     std::vector<half> mean_half;
//     std::vector<half> variance_half;

//     if (phase_ == net_phase::train) {
//       // calculate mean/variance from this batch in train phase
//       mean_half = one_vector_to_half(mean_current_);
//       variance_half = one_vector_to_half(variance_current_);
//       moments_half(in_data_half[0], in_spatial_size_, in_channels_, mean_half, variance_half);
//     } else {
//       mean_half = one_vector_to_half(mean_);
//       variance_half = one_vector_to_half(variance_);
//     }

//     std::vector<half> stddev_half = one_vector_to_half(stddev_);

//     for (size_t i = 0; i < in_channels_; i++) {
//       stddev_half[i] = sqrt(variance_half[i] + half(eps_));
//       if (stddev_half[i] <= half(0) || std::isnan(stddev_half[i])) {
//         printf("stddev_half[%d] = %f, variance_half[%d] = %f, eps_ = %f\n", i, (float)stddev_half[i], i, (float)variance_half[i], (float)eps_);
//       }
//     }

//     for_i(in_data[0]->size(), [&](size_t i) {
//       for (size_t j = 0; j < in_channels_; j++) {
//         half m = mean_half[j];

//         for (size_t k = 0; k < in_spatial_size_; k++) {
//           out_data_half[0][i][j * in_spatial_size_ + k] = (in_data_half[0][i][j * in_spatial_size_ + k] - m) / stddev_half[j];
//         }
//       }
//     });


//     if (phase_ == net_phase::train && update_immidiately_) {
//       // mean_     = mean_current_;;
//       one_half_to_vector(mean_, mean_half);
//       // variance_ = variance_current_;
//       one_half_to_vector(variance_, variance_half);
//     } else {
//       one_half_to_vector(mean_current_, mean_half);
//       one_half_to_vector(variance_current_, variance_half);
//     }

//     one_half_to_vector(stddev_, stddev_half);
    
//     three_half_to_vector(out_data_val, out_data_half);

//     for (size_t i = 0; i < out_data.size(); ++i) {
//         *(out_data[i]) = out_data_val[i]; // ポインタのデリファレンス
//     }

//     // std::cout << "After Batch Normalization" << std::endl;
//     // for (size_t i = 0; i < 10; i++) {
//     //   for (size_t j = 0; j < 2; j++) {
//     //     std::cout << out_data_val[0][i][j] << std::endl;
//     //   }
//     // }

//     // std::cout << "mean_ = " << mean_[0] << std::endl;
//     // std::cout << "variance_ = " << variance_[0] << std::endl;
//     // std::cout << "stddev_ = " << stddev_[0] << std::endl << std::endl;
//     // std::cout << "mean_current_ = " << mean_current_[0] << std::endl;
//     // std::cout << "variance_current_ = " << variance_current_[0] << std::endl;
//     // std::cout << "tmp_mean_ = " << tmp_mean_[0] << std::endl;
//     // std::cout << std::endl;
//     // std::cout << std::endl;
// #else


//     // in_dataとout_dataから値のベクターを作成します。
//     std::vector<tiny_dnn::tensor_t> in_data_val(in_data.size());
//     std::vector<tiny_dnn::tensor_t> out_data_val(out_data.size());

//     for (size_t i = 0; i < in_data.size(); ++i) {
//         in_data_val[i] = *(in_data[i]); // ポインタのデリファレンス
//     }

//     for (size_t i = 0; i < out_data.size(); ++i) {
//         out_data_val[i] = *(out_data[i]); // ポインタのデリファレンス
//     }

//     // batch_count++;
//     // if (batch_count == 5) {
//     //   batch_count = 1;
//     // }
//     // std::cout << batch_count << " Before Batch Normalization" << std::endl;
//     // for (size_t i = 0; i < 10; ++i) {
//     //   for (size_t j = 0; j < 2; ++j) {
//     //     std::cout << in_data_val[0][i][j] << std::endl;
//     //   }
//     // }
//     // std::cout << "mean_ = " << mean_[0] << std::endl;
//     // std::cout << "variance_ = " << variance_[0] << std::endl;
//     // std::cout << "stddev_ = " << stddev_[0] << std::endl << std::endl;
//     // std::cout << "mean_current_ = " << mean_current_[0] << std::endl;
//     // std::cout << "variance_current_ = " << variance_current_[0] << std::endl;
//     // std::cout << "tmp_mean_ = " << tmp_mean_[0] << std::endl;
//     // std::cout << std::endl;

//     // 変換関数を呼び出します。
//     std::vector<std::vector<std::vector<half>>> in_data_half = three_vector_to_half(in_data_val);
//     std::vector<std::vector<std::vector<half>>> out_data_half = three_vector_to_half(out_data_val);

//     std::vector<tensor_t> in_data_test(in_data.size());
//     std::vector<tensor_t> out_data_test(out_data.size());

//     for (size_t i = 0; i < in_data.size(); ++i) {
//       in_data_test[i].resize(in_data_val[i].size());
//       for (size_t j = 0; j < in_data_val[i].size(); ++j) {
//         in_data_test[i][j].resize(in_data_val[i][j].size());
//         for (size_t k = 0; k < in_data_val[i][j].size(); ++k) {
//           in_data_test[i][j][k] = (float)in_data_half[i][j][k];
//         }
//       }
//     }

//     for (size_t i = 0; i < out_data.size(); ++i) {
//       out_data_test[i].resize(out_data_val[i].size());
//       for (size_t j = 0; j < out_data_val[i].size(); ++j) {
//         out_data_test[i][j].resize(out_data_val[i][j].size());
//         for (size_t k = 0; k < out_data_val[i][j].size(); ++k) {
//           out_data_test[i][j][k] = (float)out_data_half[i][j][k];
//         }
//       }
//     }
    
//     // std::vector<half> mean_half;
//     // std::vector<half> variance_half;

//     vec_t mean;
//     vec_t variance;

//     if (phase_ == net_phase::train) {
//       // calculate mean/variance from this batch in train phase
//       // mean_half = one_vector_to_half(mean_current_);
//       // variance_half = one_vector_to_half(variance_current_);
//       mean = mean_current_;
//       variance = variance_current_;
//       // moments_half(in_data_half[0], in_spatial_size_, in_channels_, mean, variance);
//       moments(in_data_test[0], in_spatial_size_, in_channels_, mean, variance);
//     } else {
//       // mean_half = one_vector_to_half(mean_);
//       // variance_half = one_vector_to_half(variance_);
//       mean = mean_;
//       variance = variance_;
//     }

//     // std::vector<half> stddev_half = one_vector_to_half(stddev_);
//     vec_t stddev = stddev_;

//     for (size_t i = 0; i < in_channels_; i++) {
//       // stddev_half[i] = sqrt(variance_half[i] + half(eps_));
//       stddev[i] = sqrt(variance[i] + float(eps_));
//       if (stddev[i] <= half(0) || std::isnan(stddev[i])) {
//         printf("stddev[%d] = %f, variance[%d] = %f, eps_ = %f\n", i, (float)stddev[i], i, (float)variance[i], (float)eps_);
//       }
//     }

//     for_i(in_data[0]->size(), [&](size_t i) {
//       for (size_t j = 0; j < in_channels_; j++) {
//         // half m = (half)(mean[j]);
//         float m = mean[j];

//         for (size_t k = 0; k < in_spatial_size_; k++) {
//           // out_data_half[0][i][j * in_spatial_size_ + k] = (in_data_half[0][i][j * in_spatial_size_ + k] - m) / stddev[j];
//           out_data_test[0][i][j * in_spatial_size_ + k] = (in_data_test[0][i][j * in_spatial_size_ + k] - m) / stddev[j];
//         }
//       }
//     });


//     if (phase_ == net_phase::train && update_immidiately_) {
//       // one_half_to_vector(mean_, mean_half);
//       // one_half_to_vector(variance_, variance_half);
//       mean_     = mean;
//       variance_ = variance;
//     } else {
//       // one_half_to_vector(mean_current_, mean_half);
//       // one_half_to_vector(variance_current_, variance_half);
//       mean_current_ = mean;
//       variance_current_ = variance;
//     }

//     // one_half_to_vector(stddev_, stddev_half);
//     stddev_ = stddev;

//     out_data_half = three_vector_to_half(out_data_test);

//     // out_dataにnanが含まれているかどうかを確認します。
//     for (size_t i = 0; i < out_data_half[0].size(); i++) {
//       for (size_t j = 0; j < out_data_half[0][0].size(); j++) {
//         if (std::isnan(out_data_half[0][i][j])) {
//           printf("out_data_half[%d][%d] = %f\n", i, j, (float)out_data_half[0][i][j]);
//         }
//       }
//     }
    
//     three_half_to_vector(out_data_val, out_data_half);

//     for (size_t i = 0; i < out_data.size(); ++i) {
//         *(out_data[i]) = out_data_val[i]; // ポインタのデリファレンス
//     }

// #endif

// #endif
//   }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {}

  // void forward_propagation16(const std::vector<tensor16_t *> &in_data,
  //                              std::vector<tensor16_t *> &out_data) override {
  //   std::vector<tiny_dnn::tensor16_t> in_data_val(in_data.size());
  //   std::vector<tiny_dnn::tensor16_t> out_data_val(out_data.size());

  //   for (size_t i = 0; i < in_data.size(); ++i) {
  //       in_data_val[i] = *(in_data[i]); // ポインタのデリファレンス
  //   }

  //   for (size_t i = 0; i < out_data.size(); ++i) {
  //       out_data_val[i] = *(out_data[i]); // ポインタのデリファレンス
  //   }

  //   std::vector<tensor_t> in_data_test(in_data.size());
  //   std::vector<tensor_t> out_data_test(out_data.size());

  //   for (size_t i = 0; i < in_data_val.size(); ++i) {
  //     in_data_test[i].resize(in_data_val[i].size());
  //     for (size_t j = 0; j < in_data_val[i].size(); ++j) {
  //       in_data_test[i][j].resize(in_data_val[i][j].size());
  //       for (size_t k = 0; k < in_data_val[i][j].size(); ++k) {
  //         in_data_test[i][j][k] = (float)in_data_val[i][j][k];
  //       }
  //     }
  //   }

  //   for (size_t i = 0; i < out_data_val.size(); ++i) {
  //     out_data_test[i].resize(out_data_val[i].size());
  //     for (size_t j = 0; j < out_data_val[i].size(); ++j) {
  //       out_data_test[i][j].resize(out_data_val[i][j].size());
  //       for (size_t k = 0; k < out_data_val[i][j].size(); ++k) {
  //         out_data_test[i][j][k] = (float)out_data_val[i][j][k];
  //       }
  //     }
  //   }
    
  //   // std::vector<half> mean_half;
  //   // std::vector<half> variance_half;

  //   vec_t mean;
  //   vec_t variance;

  //   if (phase_ == net_phase::train) {
  //     // calculate mean/variance from this batch in train phase
  //     // mean_half = one_vector_to_half(mean_current_);
  //     // variance_half = one_vector_to_half(variance_current_);
  //     mean = mean_current_;
  //     variance = variance_current_;
  //     // moments_half(in_data_half[0], in_spatial_size_, in_channels_, mean, variance);
  //     moments(in_data_test[0], in_spatial_size_, in_channels_, mean, variance);
  //   } else {
  //     // mean_half = one_vector_to_half(mean_);
  //     // variance_half = one_vector_to_half(variance_);
  //     mean = mean_;
  //     variance = variance_;
  //   }

  //   // std::vector<half> stddev_half = one_vector_to_half(stddev_);
  //   vec_t stddev = stddev_;

  //   for (size_t i = 0; i < in_channels_; i++) {
  //     // stddev_half[i] = sqrt(variance_half[i] + half(eps_));
  //     stddev[i] = sqrt(variance[i] + float(eps_));
  //     if (stddev[i] <= half(0) || std::isnan(stddev[i])) {
  //       printf("stddev[%d] = %f, variance[%d] = %f, eps_ = %f\n", i, (float)stddev[i], i, (float)variance[i], (float)eps_);
  //     }
  //   }

  //   for_i(in_data[0]->size(), [&](size_t i) {
  //     for (size_t j = 0; j < in_channels_; j++) {
  //       // half m = (half)(mean[j]);
  //       float m = mean[j];

  //       for (size_t k = 0; k < in_spatial_size_; k++) {
  //         // out_data_half[0][i][j * in_spatial_size_ + k] = (in_data_half[0][i][j * in_spatial_size_ + k] - m) / stddev[j];
  //         out_data_test[0][i][j * in_spatial_size_ + k] = (in_data_test[0][i][j * in_spatial_size_ + k] - m) / stddev[j];
  //       }
  //     }
  //   });


  //   if (phase_ == net_phase::train && update_immidiately_) {
  //     // one_half_to_vector(mean_, mean_half);
  //     // one_half_to_vector(variance_, variance_half);
  //     mean_     = mean;
  //     variance_ = variance;
  //   } else {
  //     // one_half_to_vector(mean_current_, mean_half);
  //     // one_half_to_vector(variance_current_, variance_half);
  //     mean_current_ = mean;
  //     variance_current_ = variance;
  //   }

  //   // one_half_to_vector(stddev_, stddev_half);
  //   stddev_ = stddev;

  //   out_data_val = three_vector_to_half16(out_data_test);

  //   // out_dataにnanが含まれているかどうかを確認します。
  //   for (size_t i = 0; i < out_data_val[0].size(); i++) {
  //     for (size_t j = 0; j < out_data_val[0][0].size(); j++) {
  //       if (std::isnan(out_data_val[0][i][j])) {
  //         printf("out_data_val[%d][%d] = %f\n", i, j, (float)out_data_val[0][i][j]);
  //       }
  //     }
  //   }
    
  //   for (size_t i = 0; i < out_data.size(); ++i) {
  //       *(out_data[i]) = out_data_val[i]; // ポインタのデリファレンス
  //   }

  // }

  void forward_propagation16(const std::vector<tensor16_t *> &in_data,
                               std::vector<tensor16_t *> &out_data) override {}

  void set_context(net_phase ctx) override { phase_ = ctx; }

  std::string layer_type() const override { return "batch-norm"; }

  void post_update() override {
    for (size_t i = 0; i < mean_.size(); i++) {
      mean_[i] = (mean_[i] == 0) ? mean_current_[i]
          : momentum_ * mean_[i] + (1 - momentum_) * mean_current_[i];
      
      variance_[i] = (variance_[i] == 0) ? variance_current_[i]
          : momentum_ * variance_[i] + (1 - momentum_) * variance_current_[i];
    }
  }

  void post_update16() override {
    for (size_t i = 0; i < mean_16_.size(); i++) {
      mean_16_[i] = (mean_16_[i] == half(0)) ? mean_current_16_[i]
          : momentum_16_ * mean_16_[i] + (half(1) - momentum_16_) * mean_current_16_[i];
      
      variance_16_[i] = (variance_16_[i] == half(0)) ? variance_current_16_[i]
          : momentum_16_ * variance_16_[i] + (half(1) - momentum_16_) * variance_current_16_[i];
    }
  }

  void save(
    std::ostream &os,
    const int precision = std::numeric_limits<float_t>::digits10 + 2
    /*by default, we want there to be enough precision*/) const override {
    Base::save(os, precision);
    for (auto m : mean_) os << m << " ";
    for (auto v : variance_) os << v << " ";
  }

  void load(std::istream &is,
            const int precision = std::numeric_limits<float_t>::digits10 + 2
            /*by default, we want there to be enough precision*/) override {
    Base::load(is, precision);
    for (auto &m : mean_) is >> m;
    for (auto &v : variance_) is >> v;
  }

  void load(const std::vector<float_t> &src, int &idx) override {
    Base::load(src, idx);
    for (auto &m : mean_) m     = src[idx++];
    for (auto &v : variance_) v = src[idx++];
  }

  void update_immidiately(bool update) { update_immidiately_ = update; }

  void set_stddev(const vec_t &stddev) { stddev_ = stddev; }

  void set_mean(const vec_t &mean) { mean_ = mean; }

  void set_variance(const vec_t &variance) {
    variance_ = variance;
    calc_stddev(variance);
  }

  // vec_t get_mean() override {return mean_;}

  // vec_t get_variance() override {return variance_;}

  float_t epsilon() const { return eps_; }

  float_t momentum() const { return momentum_; }

  friend struct serialization_buddy;

 private:
  void calc_stddev(const vec_t &variance) {
    for (size_t i = 0; i < in_channels_; i++) {
      stddev_[i] = sqrt(variance[i] + eps_);
    }
  }

  void init() {
    mean_current_.resize(in_channels_);
    mean_.resize(in_channels_);
    variance_current_.resize(in_channels_);
    variance_.resize(in_channels_);
    tmp_mean_.resize(in_channels_);
    stddev_.resize(in_channels_);

    mean_current_16_.resize(in_channels_);
    mean_16_.resize(in_channels_);
    variance_current_16_.resize(in_channels_);
    variance_16_.resize(in_channels_);
    tmp_mean_16_.resize(in_channels_);
    stddev_16_.resize(in_channels_);
  }

  size_t in_channels_;
  size_t in_spatial_size_;

  net_phase phase_;
  float_t momentum_;
  float_t eps_;

  half momentum_16_;
  half eps_16_;

  // mean/variance for this mini-batch
  vec_t mean_current_;
  vec_t variance_current_;

  vec_t tmp_mean_;

  vec16_t mean_current_16_;
  vec16_t variance_current_16_;

  vec16_t tmp_mean_16_;

  // moving average of mean/variance
  vec_t mean_;
  vec_t variance_;
  vec_t stddev_;

  vec16_t mean_16_;
  vec16_t variance_16_;
  vec16_t stddev_16_;

  // for test
  bool update_immidiately_;
};

}  // namespace tiny_dnn
