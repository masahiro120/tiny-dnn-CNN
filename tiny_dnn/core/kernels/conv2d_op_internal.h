/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

// #include "tiny_dnn/half.hpp"
#include "../../half.hpp"
#include "../../half_define.h"

using half_float::half;

#define F_CHECK 0
#define B_CHECK 0

// #define CONV_F_HALF 1
// #define CONV_B_HALF 1

std::vector<half> one_vector_to_half(const tiny_dnn::vec_t& array);
tiny_dnn::vec16_t one_vector_to_half16(const tiny_dnn::vec_t& array);
std::vector<std::vector<half>> two_vector_to_half(const tiny_dnn::tensor_t& array);
tiny_dnn::tensor16_t two_vector_to_half16(const tiny_dnn::tensor_t& array);
void two_half_to_vector(tiny_dnn::tensor_t& array, std::vector<std::vector<half>> array_half);
void one_half_to_vector(tiny_dnn::vec_t& array, tiny_dnn::vec16_t array_half);
void two_half_to_vector(tiny_dnn::tensor_t& array, tiny_dnn::tensor16_t array_half);
void nan_check(const tiny_dnn::vec16_t &array);
void nan_check(const tiny_dnn::tensor16_t &array);

namespace tiny_dnn {
namespace kernels {

inline void conv2d_op_internal(const tensor_t &in_data,
                               const vec_t &W,
                               const vec_t &bias,
                               tensor_t &out_data,
                               const core::conv_params &params,
                               const bool parallelize) {

  // printf("conv forward\n");
#if CONV_F_HALF == 0
  // std::cout << std::endl;
  // for (size_t i = 0; i < 10; i++) {
  //   std::cout << "in_data[" << i << "] = " << in_data[0][i] << std::endl;
  // }
  // std::cout << "W[0] = " << W[0] << std::endl;
  // std::cout << "bias[0] = " << bias[0] << std::endl;
  // std::cout << std::endl;
#if F_CHECK == 1
  tensor_t in_data_check(in_data.size(), vec_t(params.in.depth_ * params.in_padded.area()));
  for (size_t sample = 0; sample < in_data.size(); sample++) {
    for (size_t i = 0; i < params.in.depth_ * params.in_padded.area(); i++) {
      in_data_check[sample][i] = in_data[sample][i];
    }
  }

  vec_t W_check(W.size());
  for (size_t i = 0; i < W.size(); i++) {
    W_check[i] = W[i];
  }

  vec_t bias_check(bias.size());
  for (size_t i = 0; i < bias.size(); i++) {
    bias_check[i] = bias[i];
  }

  tensor_t out_data_check(out_data.size(), vec_t(params.out.depth_ * params.out.area()));
  for (size_t sample = 0; sample < out_data.size(); sample++) {
    for (size_t i = 0; i < params.out.depth_ * params.out.area(); i++) {
      out_data_check[sample][i] = out_data[sample][i];
    }
  }
#endif


  for_(parallelize, 0u, in_data.size(),
       [&](const blocked_range &r) {
         size_t out_area    = params.out.area();
         size_t iw          = params.in_padded.width_;
         size_t id          = params.in.depth_;
         size_t ow          = params.out.width_;
         size_t oh          = params.out.height_;
         size_t od          = params.out.depth_;
         size_t kw          = params.weight.width_;
         size_t kh          = params.weight.height_;
         size_t w_dilation  = params.w_dilation;
         size_t h_dilation  = params.h_dilation;
         size_t elem_stride = params.w_stride;
         size_t line_stride = iw * params.h_stride;
         for (size_t sample = r.begin(); sample < r.end(); sample++) {
           const vec_t &in = in_data[sample];
           vec_t &a        = out_data[sample];
           for (size_t o = 0; o < od; o++) {
             float_t *pa = &a[params.out.get_index(0, 0, o)];
             for (size_t inc = 0; inc < id; inc++) {
               if (!params.tbl.is_connected(o, inc)) continue;
               size_t idx;
               idx                = params.weight.get_index(0, 0, id * o + inc);
               const float_t *pw  = &W[idx];
               idx                = params.in_padded.get_index(0, 0, inc);
               const float_t *pin = &in[idx];
               float_t *pout      = pa;
               for (size_t y = 0; y < oh; y++) {
                 const float_t *pin_line = pin;
                 for (size_t x = 0; x < ow; x++) {
                   const float_t *pin_element = pin_line;
                   const float_t *pw_element  = pw;
                   float_t sum{0};
                   // should be optimized for small kernel(3x3,5x5)
                   for (size_t wy = 0; wy < kh; wy++) {    // NOLINT
                     for (size_t wx = 0; wx < kw; wx++) {  // NOLINT
                       sum += pw_element[wx] * pin_element[wx * w_dilation];
                     }
                     pw_element += kw;
                     pin_element += iw * h_dilation;
                   }
                   pout[x] += sum;
                   pin_line += elem_stride;
                 }
                 pout += ow;
                 pin += line_stride;
               }
             }
             if (params.has_bias) {
               vectorize::add(bias[o], out_area, pa);
             }
           }
         }
       },
       0u);

#if F_CHECK == 1

  
  for_(parallelize, 0u, in_data.size(),
       [&](const blocked_range &r) {
         size_t out_area    = params.out.area();
         size_t iw          = params.in_padded.width_;
         size_t id          = params.in.depth_;
         size_t ow          = params.out.width_;
         size_t oh          = params.out.height_;
         size_t od          = params.out.depth_;
         size_t kw          = params.weight.width_;
         size_t kh          = params.weight.height_;
         size_t w_dilation  = params.w_dilation;
         size_t h_dilation  = params.h_dilation;
         size_t elem_stride = params.w_stride;
         size_t line_stride = iw * params.h_stride;
         for (size_t sample = r.begin(); sample < r.end(); sample++) {
           const vec_t &in = in_data_check[sample];
           vec_t &a        = out_data_check[sample];
           for (size_t o = 0; o < od; o++) {
             float_t *pa = &a[params.out.get_index(0, 0, o)];
             for (size_t inc = 0; inc < id; inc++) {
               if (!params.tbl.is_connected(o, inc)) continue;
               size_t idx;
               idx                = params.weight.get_index(0, 0, id * o + inc);
               const float_t *pw  = &W_check[idx];
               idx                = params.in_padded.get_index(0, 0, inc);
               const float_t *pin = &in[idx];
               float_t *pout      = pa;
               for (size_t y = 0; y < oh; y++) {
                 const float_t *pin_line = pin;
                 for (size_t x = 0; x < ow; x++) {
                   const float_t *pin_element = pin_line;
                   const float_t *pw_element  = pw;
                   float_t sum{0};
                   // should be optimized for small kernel(3x3,5x5)
                   for (size_t wy = 0; wy < kh; wy++) {    // NOLINT
                     for (size_t wx = 0; wx < kw; wx++) {  // NOLINT
                       sum += pw_element[wx] * pin_element[wx * w_dilation];
                     }
                     pw_element += kw;
                     pin_element += iw * h_dilation;
                   }
                   pout[x] += sum;
                   pin_line += elem_stride;
                 }
                 pout += ow;
                 pin += line_stride;
               }
             }
             if (params.has_bias) {
              //  vectorize::add(bias_check[o], out_area, pa);
              for (size_t i = 0; i < out_area; i++) {
                pa[i] += bias_check[o];
              }
             }
           }
         }
       },
       0u);


  int flag = 0;
  for (size_t sample = 0; sample < out_data.size(); sample++) {
    for (size_t i = 0; i < params.out.depth_ * params.out.area(); i++) {
      if (out_data_check[sample][i] != out_data[sample][i]) {
        std::cout << "out_data_check[" << sample << "][" << i << "] = " << out_data_check[sample][i] << std::endl;
        std::cout << "out_data[" << sample << "][" << i << "] = " << out_data[sample][i] << std::endl;
        flag = 1;
      }
    }
  }

  if (flag == 0) {
    std::cout << "out_data is OK" << std::endl;
  } else {
    std::cout << "out_data is NG" << std::endl;
  }

  // 実行停止
  std::exit(0);
#endif
  // std::cout << std::endl;
  // for (size_t i = 0; i < 10; i++) {
  //   std::cout << "out_data[" << i << "] = " << out_data[0][i] << std::endl;
  // }
  // std::cout << std::endl;
#else
  std::vector<std::vector<half>> in_data_half = two_vector_to_half(in_data);
  std::vector<half> W_half = one_vector_to_half(W);
  std::vector<half> bias_half = one_vector_to_half(bias);
  std::vector<std::vector<half>> out_data_half = two_vector_to_half(out_data);

  // for (size_t i = 0; i < in_data_half.size(); i++) {
  //   for (size_t j = 0; j < 10; j++) {
  //     std::cout << "in_data_half[" << i << "][" << j << "] = " << in_data_half[i][j] << std::endl;
  //   }
  // }

  // for (size_t i = 0; i < 10; i++) {
  //   std::cout << "W_half[" << i << "] = " << W_half[i] << std::endl;
  // }

  // for (size_t i = 0; i < 10; i++) {
  //   std::cout << "bias_half[" << i << "] = " << bias_half[i] << std::endl;
  // }

  // std::exit(0);

  for_(parallelize, 0u, in_data_half.size(),
       [&](const blocked_range &r) {
         size_t out_area    = params.out.area();
         size_t iw          = params.in_padded.width_;
         size_t id          = params.in.depth_;
         size_t ow          = params.out.width_;
         size_t oh          = params.out.height_;
         size_t od          = params.out.depth_;
         size_t kw          = params.weight.width_;
         size_t kh          = params.weight.height_;
         size_t w_dilation  = params.w_dilation;
         size_t h_dilation  = params.h_dilation;
         size_t elem_stride = params.w_stride;
         size_t line_stride = iw * params.h_stride;
         for (size_t sample = r.begin(); sample < r.end(); sample++) {
           const std::vector<half> &in = in_data_half[sample];
           std::vector<half> &a        = out_data_half[sample];
           for (size_t o = 0; o < od; o++) {
             half *pa = &a[params.out.get_index(0, 0, o)];
             for (size_t inc = 0; inc < id; inc++) {
               if (!params.tbl.is_connected(o, inc)) continue;
               size_t idx;
               idx                = params.weight.get_index(0, 0, id * o + inc);
               const half *pw  = &W_half[idx];
               idx                = params.in_padded.get_index(0, 0, inc);
               const half *pin = &in[idx];
               half *pout      = pa;
               for (size_t y = 0; y < oh; y++) {
                 const half *pin_line = pin;
                 for (size_t x = 0; x < ow; x++) {
                   const half *pin_element = pin_line;
                   const half *pw_element  = pw;
                   half sum{0};
                   // should be optimized for small kernel(3x3,5x5)
                   for (size_t wy = 0; wy < kh; wy++) {    // NOLINT
                     for (size_t wx = 0; wx < kw; wx++) {  // NOLINT
                       sum += pw_element[wx] * pin_element[wx * w_dilation];
                     }
                     pw_element += kw;
                     pin_element += iw * h_dilation;
                   }
                   pout[x] += sum;
                   pin_line += elem_stride;
                 }
                 pout += ow;
                 pin += line_stride;
               }
             }
             if (params.has_bias) {
              //  vectorize::add(bias_half[o], out_area, pa);
              for (size_t i = 0; i < out_area; i++) {
                pa[i] += bias_half[o];
              }
             }
           }
         }
       },
       0u);

  two_half_to_vector(out_data, out_data_half);

#endif
}

inline void conv2d_op_internal(const tensor16_t &in_data,
                               const vec16_t &W,
                               const vec16_t &bias,
                               tensor16_t &out_data,
                               const core::conv_params &params,
                               const bool parallelize) {
  // for (size_t i = 0; i < in_data.size(); i++) {
  //   for (size_t j = 0; j < 10; j++) {
  //     std::cout << "in_data[" << i << "][" << j << "] = " << in_data[i][j] << std::endl;
  //   }
  // }

  // for (size_t i = 0; i < 10; i++) {
  //   std::cout << "W[" << i << "] = " << W[i] << std::endl;
  // }

  // for (size_t i = 0; i < 10; i++) {
  //   std::cout << "bias[" << i << "] = " << bias[i] << std::endl;
  // }
  // std::exit(0);
#if CONV_F_HALF == 1
  for_(parallelize, 0u, in_data.size(),
       [&](const blocked_range &r) {
         size_t out_area    = params.out.area();
         size_t iw          = params.in_padded.width_;
         size_t id          = params.in.depth_;
         size_t ow          = params.out.width_;
         size_t oh          = params.out.height_;
         size_t od          = params.out.depth_;
         size_t kw          = params.weight.width_;
         size_t kh          = params.weight.height_;
         size_t w_dilation  = params.w_dilation;
         size_t h_dilation  = params.h_dilation;
         size_t elem_stride = params.w_stride;
         size_t line_stride = iw * params.h_stride;
         for (size_t sample = r.begin(); sample < r.end(); sample++) {
           const vec16_t &in = in_data[sample];
           vec16_t &a        = out_data[sample];
           for (size_t o = 0; o < od; o++) {
             half *pa = &a[params.out.get_index(0, 0, o)];
             for (size_t inc = 0; inc < id; inc++) {
               if (!params.tbl.is_connected(o, inc)) continue;
               size_t idx;
               idx                = params.weight.get_index(0, 0, id * o + inc);
               const half *pw  = &W[idx];
               idx                = params.in_padded.get_index(0, 0, inc);
               const half *pin = &in[idx];
               half *pout      = pa;
               for (size_t y = 0; y < oh; y++) {
                 const half *pin_line = pin;
                 for (size_t x = 0; x < ow; x++) {
                   const half *pin_element = pin_line;
                   const half *pw_element  = pw;
                   half sum{0};
                   // should be optimized for small kernel(3x3,5x5)
                   for (size_t wy = 0; wy < kh; wy++) {    // NOLINT
                     for (size_t wx = 0; wx < kw; wx++) {  // NOLINT
                       sum += pw_element[wx] * pin_element[wx * w_dilation];
                     }
                     pw_element += kw;
                     pin_element += iw * h_dilation;
                   }
                   pout[x] += sum;
                   pin_line += elem_stride;
                 }
                 pout += ow;
                 pin += line_stride;
               }
             }
             if (params.has_bias) {
              //  vectorize::add(bias[o], out_area, pa);
              for (size_t i = 0; i < out_area; i++) {
                pa[i] += bias[o];
              }
             }
           }
         }
       },
       0u);
#else
  tensor_t in_data_float;
  two_half_to_vector(in_data_float, in_data);
  vec_t W_float;
  one_half_to_vector(W_float, W);
  vec_t bias_float;
  one_half_to_vector(bias_float, bias);
  tensor_t out_data_float;
  two_half_to_vector(out_data_float, out_data);

  for_(parallelize, 0u, in_data_float.size(),
       [&](const blocked_range &r) {
         size_t out_area    = params.out.area();
         size_t iw          = params.in_padded.width_;
         size_t id          = params.in.depth_;
         size_t ow          = params.out.width_;
         size_t oh          = params.out.height_;
         size_t od          = params.out.depth_;
         size_t kw          = params.weight.width_;
         size_t kh          = params.weight.height_;
         size_t w_dilation  = params.w_dilation;
         size_t h_dilation  = params.h_dilation;
         size_t elem_stride = params.w_stride;
         size_t line_stride = iw * params.h_stride;
         for (size_t sample = r.begin(); sample < r.end(); sample++) {
           const vec_t &in = in_data_float[sample];
           vec_t &a        = out_data_float[sample];
           for (size_t o = 0; o < od; o++) {
             float_t *pa = &a[params.out.get_index(0, 0, o)];
             for (size_t inc = 0; inc < id; inc++) {
               if (!params.tbl.is_connected(o, inc)) continue;
               size_t idx;
               idx                = params.weight.get_index(0, 0, id * o + inc);
               const float_t *pw  = &W_float[idx];
               idx                = params.in_padded.get_index(0, 0, inc);
               const float_t *pin = &in[idx];
               float_t *pout      = pa;
               for (size_t y = 0; y < oh; y++) {
                 const float_t *pin_line = pin;
                 for (size_t x = 0; x < ow; x++) {
                   const float_t *pin_element = pin_line;
                   const float_t *pw_element  = pw;
                   float_t sum{0};
                   // should be optimized for small kernel(3x3,5x5)
                   for (size_t wy = 0; wy < kh; wy++) {    // NOLINT
                     for (size_t wx = 0; wx < kw; wx++) {  // NOLINT
                       sum += pw_element[wx] * pin_element[wx * w_dilation];
                     }
                     pw_element += kw;
                     pin_element += iw * h_dilation;
                   }
                   pout[x] += sum;
                   pin_line += elem_stride;
                 }
                 pout += ow;
                 pin += line_stride;
               }
             }
             if (params.has_bias) {
               vectorize::add(bias_float[o], out_area, pa);
             }
           }
         }
       },
       0u);
  
  out_data = two_vector_to_half16(out_data_float);
#endif
}

/******************************************************************/

template <typename tensor_t, typename vec_t>
void conv2d_op_internal(const tensor_t &prev_out,
                        const vec_t &W,
                        tensor_t &dW,
                        tensor_t &db,
                        tensor_t &curr_delta,
                        tensor_t &prev_delta,
                        const core::conv_params &params,
                        const bool parallelize) {
  printf("conv backward\n");
  typedef typename vec_t::value_type float_t;

#if CONV_B_HALF == 0

#if B_CHECK == 1
  tensor_t prev_out_check(prev_out.size(), vec_t(params.in.depth_ * params.in_padded.area()));
  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    for (size_t i = 0; i < params.in.depth_ * params.in_padded.area(); i++) {
      prev_out_check[sample][i] = prev_out[sample][i];
    }
  }

  vec_t W_check(W.size());
  for (size_t i = 0; i < W.size(); i++) {
    W_check[i] = W[i];
  }

  tensor_t dW_check(dW.size(), vec_t(W.size()));
  for (size_t sample = 0; sample < dW.size(); sample++) {
    for (size_t i = 0; i < W.size(); i++) {
      dW_check[sample][i] = dW[sample][i];
    }
  }

  tensor_t db_check(db.size(), vec_t(db[0].size()));
  for (size_t sample = 0; sample < db.size(); sample++) {
    for (size_t i = 0; i < db[0].size(); i++) {
      db_check[sample][i] = db[sample][i];
    }
  }

  tensor_t curr_delta_check(curr_delta.size(), vec_t(params.out.depth_ * params.out.area()));
  for (size_t sample = 0; sample < curr_delta.size(); sample++) {
    for (size_t i = 0; i < params.out.depth_ * params.out.area(); i++) {
      curr_delta_check[sample][i] = curr_delta[sample][i];
    }
  }

  tensor_t prev_delta_check(prev_delta.size(), vec_t(params.in.depth_ * params.in_padded.area()));
  for (size_t sample = 0; sample < prev_delta.size(); sample++) {
    for (size_t i = 0; i < params.in.depth_ * params.in_padded.area(); i++) {
      prev_delta_check[sample][i] = prev_delta[sample][i];
    }
  }
#endif

  for_i(parallelize, prev_out.size(), [&](size_t sample) {
    // propagate delta to previous layer
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        size_t idx        = 0;
        idx               = params.in.depth_ * outc + inc;
        idx               = params.weight.get_index(0, 0, idx);
        const float_t *pw = &W[idx];

        idx                       = params.out.get_index(0, 0, outc);
        const float_t *pdelta_src = &curr_delta[sample][idx];

        idx = params.in_padded.get_index(0, 0, inc);
        // float_t* pdelta_dst = &(*prev_delta)[sample][idx];
        float_t *pdelta_dst = &prev_delta[sample][idx];

        for (size_t y = 0; y < params.out.height_; y++) {
          for (size_t x = 0; x < params.out.width_; x++) {
            const float_t *ppw = pw;

            idx                       = y * params.out.width_ + x;
            const float_t ppdelta_src = pdelta_src[idx];

            float_t *ppdelta_dst =
              pdelta_dst + y * params.h_stride * params.in_padded.width_ +
              x * params.w_stride;

            for (size_t wy = 0; wy < params.weight.height_; wy++) {   // NOLINT
              for (size_t wx = 0; wx < params.weight.width_; wx++) {  // NOLINT
                idx = wy * params.in_padded.width_ + wx;
                ppdelta_dst[idx] += *ppw++ * ppdelta_src;
              }
            }
          }
        }
      }
    }

    // accumulate dw
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        for (size_t wy = 0; wy < params.weight.height_; wy++) {
          for (size_t wx = 0; wx < params.weight.width_; wx++) {
            float_t dst{0};

            size_t idx           = 0;
            idx                  = params.in_padded.get_index(wx, wy, inc);
            const float_t *prevo = &prev_out[sample][idx];

            idx                  = params.out.get_index(0, 0, outc);
            const float_t *delta = &curr_delta[sample][idx];

            if (params.w_stride > 1) {
              for (size_t y = 0; y < params.out.height_; y++) {
                size_t prevo_idx =
                  y * params.in_padded.width_ * params.h_stride;
                size_t delta_idx = y * params.out.width_;

                for (size_t x = 0; x < params.out.width_; x++) {
                  dst += prevo[prevo_idx + x * params.w_stride] *
                         delta[delta_idx + x];
                }
              }
            } else {
              for (size_t y = 0; y < params.out.height_; y++) {
                dst += vectorize::dot(
                  prevo + y * params.in_padded.width_ * params.h_stride,
                  delta + y * params.out.width_, params.out.width_);
              }
            }

            idx = params.in.depth_ * outc + inc;
            dW[sample][params.weight.get_index(wx, wy, idx)] += dst;
          }
        }
      }
    }

    // accumulate db
    if (params.has_bias) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        size_t idx            = params.out.get_index(0, 0, outc);
        const float_t *delta  = &curr_delta[sample][idx];
        const float_t *deltaa = delta + params.out.width_ * params.out.height_;
        db[sample][outc] += std::accumulate(delta, deltaa, float_t{0});
      }
    }
  });

#if B_CHECK == 1

    for_i(parallelize, prev_out_check.size(), [&](size_t sample) {
    // propagate delta to previous layer
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        size_t idx        = 0;
        idx               = params.in.depth_ * outc + inc;
        idx               = params.weight.get_index(0, 0, idx);
        const float_t *pw = &W_check[idx];

        idx                       = params.out.get_index(0, 0, outc);
        const float_t *pdelta_src = &curr_delta_check[sample][idx];

        idx = params.in_padded.get_index(0, 0, inc);
        // float_t* pdelta_dst = &(*prev_delta_check)[sample][idx];
        float_t *pdelta_dst = &prev_delta_check[sample][idx];

        for (size_t y = 0; y < params.out.height_; y++) {
          for (size_t x = 0; x < params.out.width_; x++) {
            const float_t *ppw = pw;

            idx                       = y * params.out.width_ + x;
            const float_t ppdelta_src = pdelta_src[idx];

            float_t *ppdelta_dst =
              pdelta_dst + y * params.h_stride * params.in_padded.width_ +
              x * params.w_stride;

            for (size_t wy = 0; wy < params.weight.height_; wy++) {   // NOLINT
              for (size_t wx = 0; wx < params.weight.width_; wx++) {  // NOLINT
                idx = wy * params.in_padded.width_ + wx;
                ppdelta_dst[idx] += *ppw++ * ppdelta_src;
              }
            }
          }
        }
      }
    }

    // accumulate dw_chedW_check
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        for (size_t wy = 0; wy < params.weight.height_; wy++) {
          for (size_t wx = 0; wx < params.weight.width_; wx++) {
            float_t dst{0};

            size_t idx           = 0;
            idx                  = params.in_padded.get_index(wx, wy, inc);
            const float_t *prevo = &prev_out_check[sample][idx];

            idx                  = params.out.get_index(0, 0, outc);
            const float_t *delta = &curr_delta_check[sample][idx];

            if (params.w_stride > 1) {
              for (size_t y = 0; y < params.out.height_; y++) {
                size_t prevo_idx =
                  y * params.in_padded.width_ * params.h_stride;
                size_t delta_idx = y * params.out.width_;

                for (size_t x = 0; x < params.out.width_; x++) {
                  dst += prevo[prevo_idx + x * params.w_stride] *
                         delta[delta_idx + x];
                }
              }
            } else {
              // for (size_t y = 0; y < params.out.height_; y++) {
              //   dst += vectorize::dot(
              //     prevo + y * params.in_padded.width_ * params.h_stride,
              //     delta + y * params.out.width_, params.out.width_);
              // }
              for (size_t y = 0; y < params.out.height_; y++) {
                size_t prevo_idx =
                  y * params.in_padded.width_ * params.h_stride;
                size_t delta_idx = y * params.out.width_;
                for (size_t x = 0; x < params.out.width_; x++) {
                  dst += prevo[prevo_idx + x] * delta[delta_idx + x];
                }
              }
            }

            idx = params.in.depth_ * outc + inc;
            dW_check[sample][params.weight.get_index(wx, wy, idx)] += dst;
          }
        }
      }
    }

    // accumulate db_check
    if (params.has_bias) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        size_t idx            = params.out.get_index(0, 0, outc);
        const float_t *delta  = &curr_delta_check[sample][idx];
        const float_t *deltaa = delta + params.out.width_ * params.out.height_;
        db_check[sample][outc] += std::accumulate(delta, deltaa, float_t{0});
      }
    }
  });

  int flag = 0;
  for (size_t sample = 0; sample < prev_delta.size(); sample++) {
    for (size_t i = 0; i < params.in.depth_ * params.in_padded.area(); i++) {
      if (prev_delta_check[sample][i] != prev_delta[sample][i]) {
        std::cout << "prev_delta_check[" << sample << "][" << i << "] = " << prev_delta_check[sample][i] << std::endl;
        std::cout << "prev_delta[" << sample << "][" << i << "] = " << prev_delta[sample][i] << std::endl;
        flag = 1;
      }
    }
  }

  if (flag == 0) {
    std::cout << "prev_delta is OK" << std::endl;
  } else {
    std::cout << "prev_delta is NG" << std::endl;
  }

  flag = 0;
  for (size_t sample = 0; sample < dW.size(); sample++) {
    for (size_t i = 0; i < W.size(); i++) {
      float_t diff = dW_check[sample][i] - dW[sample][i];
      if (diff > 0.0001 || diff < -0.0001) {
        std::cout << "dW_check[" << sample << "][" << i << "] = " << dW_check[sample][i] << std::endl;
        std::cout << "dW[" << sample << "][" << i << "] = " << dW[sample][i] << std::endl;
        flag = 1;
      }
    }
  }

  if (flag == 0) {
    std::cout << "dW is OK" << std::endl;
  } else {
    std::cout << "dW is NG" << std::endl;
  }

  flag = 0;
  for (size_t sample = 0; sample < db.size(); sample++) {
    for (size_t i = 0; i < db[0].size(); i++) {
      if (db_check[sample][i] != db[sample][i]) {
        std::cout << "db_check[" << sample << "][" << i << "] = " << db_check[sample][i] << std::endl;
        std::cout << "db[" << sample << "][" << i << "] = " << db[sample][i] << std::endl;
        flag = 1;
      }
    }
  }

  if (flag == 0) {
    std::cout << "db is OK" << std::endl;
  } else {
    std::cout << "db is NG" << std::endl;
  }

  // 実行停止
  std::exit(0);
#endif

#else
  std::vector<std::vector<half>> prev_out_half = two_vector_to_half(prev_out);
  std::vector<half> W_half = one_vector_to_half(W);
  std::vector<std::vector<half>> curr_delta_half = two_vector_to_half(curr_delta);
  std::vector<std::vector<half>> prev_delta_half = two_vector_to_half(prev_delta);
  std::vector<std::vector<half>> dW_half = two_vector_to_half(dW);
  std::vector<std::vector<half>> db_half = two_vector_to_half(db);

  for_i(parallelize, prev_out_half.size(), [&](size_t sample) {
    // propagate delta to previous layer
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        size_t idx        = 0;
        idx               = params.in.depth_ * outc + inc;
        idx               = params.weight.get_index(0, 0, idx);
        const half *pw = &W_half[idx];

        idx                       = params.out.get_index(0, 0, outc);
        const half *pdelta_src = &curr_delta_half[sample][idx];

        idx = params.in_padded.get_index(0, 0, inc);
        // half* pdelta_dst = &(*prev_delta_half)[sample][idx];
        half *pdelta_dst = &prev_delta_half[sample][idx];

        for (size_t y = 0; y < params.out.height_; y++) {
          for (size_t x = 0; x < params.out.width_; x++) {
            const half *ppw = pw;

            idx                       = y * params.out.width_ + x;
            const half ppdelta_src = pdelta_src[idx];

            half *ppdelta_dst =
              pdelta_dst + y * params.h_stride * params.in_padded.width_ +
              x * params.w_stride;

            for (size_t wy = 0; wy < params.weight.height_; wy++) {   // NOLINT
              for (size_t wx = 0; wx < params.weight.width_; wx++) {  // NOLINT
                idx = wy * params.in_padded.width_ + wx;
                ppdelta_dst[idx] += *ppw++ * ppdelta_src;
              }
            }
          }
        }
      }
    }

    // accumulate dw_chedW_half
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        for (size_t wy = 0; wy < params.weight.height_; wy++) {
          for (size_t wx = 0; wx < params.weight.width_; wx++) {
            half dst{0};

            size_t idx           = 0;
            idx                  = params.in_padded.get_index(wx, wy, inc);
            const half *prevo = &prev_out_half[sample][idx];

            idx                  = params.out.get_index(0, 0, outc);
            const half *delta = &curr_delta_half[sample][idx];

            if (params.w_stride > 1) {
              for (size_t y = 0; y < params.out.height_; y++) {
                size_t prevo_idx =
                  y * params.in_padded.width_ * params.h_stride;
                size_t delta_idx = y * params.out.width_;

                for (size_t x = 0; x < params.out.width_; x++) {
                  dst += prevo[prevo_idx + x * params.w_stride] *
                         delta[delta_idx + x];
                }
              }
            } else {
              // for (size_t y = 0; y < params.out.height_; y++) {
              //   dst += vectorize::dot(
              //     prevo + y * params.in_padded.width_ * params.h_stride,
              //     delta + y * params.out.width_, params.out.width_);
              // }
              for (size_t y = 0; y < params.out.height_; y++) {
                size_t prevo_idx =
                  y * params.in_padded.width_ * params.h_stride;
                size_t delta_idx = y * params.out.width_;
                for (size_t x = 0; x < params.out.width_; x++) {
                  dst += prevo[prevo_idx + x] * delta[delta_idx + x];
                }
              }
            }

            idx = params.in.depth_ * outc + inc;
            dW_half[sample][params.weight.get_index(wx, wy, idx)] += dst;
          }
        }
      }
    }

    // accumulate db_half
    if (params.has_bias) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        size_t idx            = params.out.get_index(0, 0, outc);
        const half *delta  = &curr_delta_half[sample][idx];
        const half *deltaa = delta + params.out.width_ * params.out.height_;
        db_half[sample][outc] += std::accumulate(delta, deltaa, half{0});
      }
    }
  });

  two_half_to_vector(prev_delta, prev_delta_half);
  two_half_to_vector(dW, dW_half);
  two_half_to_vector(db, db_half);

#endif
  printf("conv backward end\n");
}

template <typename tensor16_t, typename vec16_t>
void conv2d_op_internal16(const tensor16_t &prev_out,
                        const vec16_t &W,
                        tensor16_t &dW,
                        tensor16_t &db,
                        tensor16_t &curr_delta,
                        tensor16_t &prev_delta,
                        const core::conv_params &params,
                        const bool parallelize) {
#if CONV_B_HALF == 1
  // printf("conv backward\n");

  // // nan check
  // std::cout << "prev_out nan check" << std::endl;
  // nan_check(prev_out);
  // std::cout << "W nan check" << std::endl;
  // nan_check(W);
  // std::cout << "dW nan check" << std::endl;
  // nan_check(dW);
  // std::cout << "db nan check" << std::endl;
  // nan_check(db);
  // std::cout << "curr_delta nan check" << std::endl;
  // nan_check(curr_delta);
  // std::cout << "prev_delta nan check" << std::endl;
  // nan_check(prev_delta);

  // std::cout << __FILE__ << ":" << __LINE__ << std::endl;

  typedef typename vec_t::value_type float_t;

  for_i(parallelize, prev_out.size(), [&](size_t sample) {
    // propagate delta to previous layer
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        size_t idx        = 0;
        idx               = params.in.depth_ * outc + inc;
        idx               = params.weight.get_index(0, 0, idx);
        const half *pw = &W[idx];

        idx                       = params.out.get_index(0, 0, outc);
        const half *pdelta_src = &curr_delta[sample][idx];

        idx = params.in_padded.get_index(0, 0, inc);
        // half* pdelta_dst = &(*prev_delta)[sample][idx];
        half *pdelta_dst = &prev_delta[sample][idx];

        for (size_t y = 0; y < params.out.height_; y++) {
          for (size_t x = 0; x < params.out.width_; x++) {
            const half *ppw = pw;

            idx                       = y * params.out.width_ + x;
            const half ppdelta_src = pdelta_src[idx];

            half *ppdelta_dst =
              pdelta_dst + y * params.h_stride * params.in_padded.width_ +
              x * params.w_stride;

            for (size_t wy = 0; wy < params.weight.height_; wy++) {   // NOLINT
              for (size_t wx = 0; wx < params.weight.width_; wx++) {  // NOLINT
                idx = wy * params.in_padded.width_ + wx;
                ppdelta_dst[idx] += *ppw++ * ppdelta_src;
              }
            }
          }
        }
      }
    }

    // accumulate dw_chedW
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        for (size_t wy = 0; wy < params.weight.height_; wy++) {
          for (size_t wx = 0; wx < params.weight.width_; wx++) {
            half dst{0};

            size_t idx           = 0;
            idx                  = params.in_padded.get_index(wx, wy, inc);
            const half *prevo = &prev_out[sample][idx];

            idx                  = params.out.get_index(0, 0, outc);
            const half *delta = &curr_delta[sample][idx];

            if (params.w_stride > 1) {
              for (size_t y = 0; y < params.out.height_; y++) {
                size_t prevo_idx =
                  y * params.in_padded.width_ * params.h_stride;
                size_t delta_idx = y * params.out.width_;

                for (size_t x = 0; x < params.out.width_; x++) {
                  dst += prevo[prevo_idx + x * params.w_stride] *
                         delta[delta_idx + x];
                }
              }
            } else {
              // for (size_t y = 0; y < params.out.height_; y++) {
              //   dst += vectorize::dot(
              //     prevo + y * params.in_padded.width_ * params.h_stride,
              //     delta + y * params.out.width_, params.out.width_);
              // }
              for (size_t y = 0; y < params.out.height_; y++) {
                size_t prevo_idx =
                  y * params.in_padded.width_ * params.h_stride;
                size_t delta_idx = y * params.out.width_;
                for (size_t x = 0; x < params.out.width_; x++) {
                  dst += prevo[prevo_idx + x] * delta[delta_idx + x];
                }
              }
            }

            idx = params.in.depth_ * outc + inc;
            dW[sample][params.weight.get_index(wx, wy, idx)] += dst;
          }
        }
      }
    }

    // accumulate db
    if (params.has_bias) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        size_t idx            = params.out.get_index(0, 0, outc);
        const half *delta  = &curr_delta[sample][idx];
        const half *deltaa = delta + params.out.width_ * params.out.height_;
        db[sample][outc] += std::accumulate(delta, deltaa, half{0});
      }
    }
  });

  // // nan check
  // std::cout << "prev_out nan check" << std::endl;
  // nan_check(prev_out);
  // std::cout << "W nan check" << std::endl;
  // nan_check(W);
  // std::cout << "dW nan check" << std::endl;
  // nan_check(dW);
  // std::cout << "db nan check" << std::endl;
  // nan_check(db);
  // std::cout << "curr_delta nan check" << std::endl;
  // nan_check(curr_delta);
  // std::cout << "prev_delta nan check" << std::endl;
  // nan_check(prev_delta);

  // printf("conv backward end\n");

#else

  tensor_t prev_out_float;
  two_half_to_vector(prev_out_float, prev_out);
  vec_t W_float;
  one_half_to_vector(W_float, W);
  tensor_t curr_delta_float;
  two_half_to_vector(curr_delta_float, curr_delta);
  tensor_t prev_delta_float;
  two_half_to_vector(prev_delta_float, prev_delta);
  tensor_t dW_float;
  two_half_to_vector(dW_float, dW);
  tensor_t db_float;
  two_half_to_vector(db_float, db);

  for_i(parallelize, prev_out_float.size(), [&](size_t sample) {
    // propagate delta to previous layer
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        size_t idx        = 0;
        idx               = params.in.depth_ * outc + inc;
        idx               = params.weight.get_index(0, 0, idx);
        const float_t *pw = &W_float[idx];

        idx                       = params.out.get_index(0, 0, outc);
        const float_t *pdelta_src = &curr_delta_float[sample][idx];

        idx = params.in_padded.get_index(0, 0, inc);
        // float_t* pdelta_dst = &(*prev_delta_float)[sample][idx];
        float_t *pdelta_dst = &prev_delta_float[sample][idx];

        for (size_t y = 0; y < params.out.height_; y++) {
          for (size_t x = 0; x < params.out.width_; x++) {
            const float_t *ppw = pw;

            idx                       = y * params.out.width_ + x;
            const float_t ppdelta_src = pdelta_src[idx];

            float_t *ppdelta_dst =
              pdelta_dst + y * params.h_stride * params.in_padded.width_ +
              x * params.w_stride;

            for (size_t wy = 0; wy < params.weight.height_; wy++) {   // NOLINT
              for (size_t wx = 0; wx < params.weight.width_; wx++) {  // NOLINT
                idx = wy * params.in_padded.width_ + wx;
                ppdelta_dst[idx] += *ppw++ * ppdelta_src;
              }
            }
          }
        }
      }
    }

    // accumulate dw
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        for (size_t wy = 0; wy < params.weight.height_; wy++) {
          for (size_t wx = 0; wx < params.weight.width_; wx++) {
            float_t dst{0};

            size_t idx           = 0;
            idx                  = params.in_padded.get_index(wx, wy, inc);
            const float_t *prevo = &prev_out_float[sample][idx];

            idx                  = params.out.get_index(0, 0, outc);
            const float_t *delta = &curr_delta_float[sample][idx];

            if (params.w_stride > 1) {
              for (size_t y = 0; y < params.out.height_; y++) {
                size_t prevo_idx =
                  y * params.in_padded.width_ * params.h_stride;
                size_t delta_idx = y * params.out.width_;

                for (size_t x = 0; x < params.out.width_; x++) {
                  dst += prevo[prevo_idx + x * params.w_stride] *
                          delta[delta_idx + x];
                }
              }
            } else {
              for (size_t y = 0; y < params.out.height_; y++) {
                dst += vectorize::dot(
                  prevo + y * params.in_padded.width_ * params.h_stride,
                  delta + y * params.out.width_, params.out.width_);
              }
            }

            idx = params.in.depth_ * outc + inc;
            dW_float[sample][params.weight.get_index(wx, wy, idx)] += dst;
          }
        }
      }
    }

    // accumulate db
    if (params.has_bias) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        size_t idx            = params.out.get_index(0, 0, outc);
        const float_t *delta  = &curr_delta_float[sample][idx];
        const float_t *deltaa = delta + params.out.width_ * params.out.height_;
        db_float[sample][outc] += std::accumulate(delta, deltaa, float_t{0});
      }
    }
  });

  prev_delta = two_vector_to_half16(prev_delta_float);
  dW = two_vector_to_half16(dW_float);
  db = two_vector_to_half16(db_float);
#endif
}

}  // namespace kernels
}  // namespace tiny_dnn
