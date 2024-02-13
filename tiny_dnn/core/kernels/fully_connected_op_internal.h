/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

// #include "tiny_dnn/core/params/fully_params.h"
// #include "tiny_dnn/half.hpp"

#include "../../core/params/fully_params.h"
#include "../../half.hpp"
#include "../../half_define.h"

using namespace half_float;

#define F_CHECK 0
#define B_CHECK 0

extern int FC_F_HALF;
extern int FC_B_HALF;

std::vector<half> one_vector_to_half(const tiny_dnn::vec_t& array);
std::vector<std::vector<half>> two_vector_to_half(const tiny_dnn::tensor_t& array);
void two_half_to_vector(tiny_dnn::tensor_t& array, std::vector<std::vector<half>> array_half);

namespace tiny_dnn {
namespace kernels {

inline void fully_connected_op_internal(const tensor_t &in_data,
                                        const vec_t &W,
                                        const vec_t &bias,
                                        tensor_t &out_data,
                                        const core::fully_params &params,
                                        const bool layer_parallelize) {
  // printf("fc forward\n");

  // // in_data[0]を出力
  // std::cout << "in_data[0]: ";
  // for (size_t i = 0; i < in_data[0].size(); i++) {
  //   printf("in_data[0][%d] = %f,\n", i, in_data[0][i]);
  // }

  // // Wを出力
  // std::cout << std::endl << "W: " << std::endl;
  // for (size_t c = 0; c < params.in_size_; c++) {
  //   printf("W[%d] = %f, \n", c * params.out_size_, W[c * params.out_size_]);
  // }

  // // biasを出力
  // std::cout << "bias: ";
  // printf("bias[%d] = %f,\n", 0, bias[0]);

  // // 実行停止
  // std::exit(0);
#if FC_F_HALF == 0
  for_i(layer_parallelize, in_data.size(), [&](size_t sample) {
    const vec_t &in = in_data[sample];
    vec_t &out      = out_data[sample];

    for (size_t i = 0; i < params.out_size_; i++) {
      out[i] = float_t{0};
      for (size_t c = 0; c < params.in_size_; c++) {
        out[i] += W[c * params.out_size_ + i] * in[c];
      }

      if (params.has_bias_) {
        out[i] += bias[i];
      }
    }
  });

#else

  std::vector<std::vector<half>> in_data_half = two_vector_to_half(in_data);
  std::vector<half> W_half = one_vector_to_half(W);
  std::vector<half> bias_half = one_vector_to_half(bias);
  std::vector<std::vector<half>> out_data_half = two_vector_to_half(out_data);

  for_i(layer_parallelize, in_data_half.size(), [&](size_t sample) {
    const std::vector<half> &in = in_data_half[sample];
    std::vector<half> &out      = out_data_half[sample];

    for (size_t i = 0; i < params.out_size_; i++) {
      out[i] = half{0};
      for (size_t c = 0; c < params.in_size_; c++) {
        out[i] += W_half[c * params.out_size_ + i] * in[c];
      }

      if (params.has_bias_) {
        out[i] += bias_half[i];
      }
    }
  });

  two_half_to_vector(out_data, out_data_half);

#endif
}

inline void fully_connected_op_internal(const tensor_t &prev_out,
                                        const vec_t &W,
                                        tensor_t &dW,
                                        tensor_t &db,
                                        tensor_t &curr_delta,
                                        tensor_t &prev_delta,
                                        const core::fully_params &params,
                                        const bool layer_parallelize) {
  // printf("fc backward\n");
#if FC_B_HALF == 0

#if B_CHECK == 1
  // prev_out_checkを作成
  std::vector<std::vector<float>> prev_out_check(prev_out.size(), std::vector<float>(params.in_size_));
  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    for (size_t i = 0; i < params.in_size_; i++) {
      prev_out_check[sample][i] = prev_out[sample][i];
    }
  }

  // W_checkを作成
  std::vector<float> W_check(W.size());
  for (size_t i = 0; i < W.size(); i++) {
    W_check[i] = W[i];
  }

  // curr_delta_checkを作成
  std::vector<std::vector<float>> curr_delta_check(curr_delta.size(), std::vector<float>(params.out_size_));
  for (size_t sample = 0; sample < curr_delta.size(); sample++) {
    for (size_t i = 0; i < params.out_size_; i++) {
      curr_delta_check[sample][i] = curr_delta[sample][i];
    }
  }

  // prev_delta_checkを作成
  std::vector<std::vector<float>> prev_delta_check(prev_out.size(), std::vector<float>(params.in_size_));
  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    for (size_t i = 0; i < params.in_size_; i++) {
      prev_delta_check[sample][i] = prev_delta[sample][i];
    }
  }

  // dW_checkを作成
  std::vector<std::vector<float>> dW_check(prev_out.size(), std::vector<float>(W.size()));
  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    for (size_t i = 0; i < W.size(); i++) {
      dW_check[sample][i] = dW[sample][i];
    }
  }

  // db_checkを作成
  std::vector<std::vector<float>> db_check(prev_out.size(), std::vector<float>(params.out_size_));
  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    for (size_t i = 0; i < params.out_size_; i++) {
      db_check[sample][i] = db[sample][i];
    }
  }
#endif

  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    for (size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      prev_delta[sample][c] += vectorize::dot(
        &curr_delta[sample][0], &W[c * params.out_size_], params.out_size_);
    }

    for_(layer_parallelize, 0, params.out_size_, [&](const blocked_range &r) {
      // accumulate weight-step using delta
      // dW[c * out_size + i] += current_delta[i] * prev_out[c]
      for (size_t c = 0; c < params.in_size_; c++) {
        vectorize::muladd(&curr_delta[sample][r.begin()], prev_out[sample][c],
                          r.end() - r.begin(),
                          &dW[sample][c * params.out_size_ + r.begin()]);
      }

      if (params.has_bias_) {
        // vec_t& db = *in_grad[2];
        for (size_t i = r.begin(); i < r.end(); i++) {
          db[sample][i] += curr_delta[sample][i];
        }
      }
    });
  }

# if B_CHECK == 1
  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    for (size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      // prev_delta_check[sample][c] += vectorize::dot(
      //   &curr_delta_check[sample][0], &W_check[c * params.out_size_], params.out_size_);

      for (size_t i = 0; i < params.out_size_; i++) {
        prev_delta_check[sample][c] += curr_delta_check[sample][i] * W_check[c * params.out_size_ + i];
      }
    }

    // dWをhalf型で計算
    for_(layer_parallelize, 0, params.out_size_, [&](const blocked_range &r) {
      // accumulate weight-step using delta
      // dW[c * out_size + i] += current_delta[i] * prev_out[c]
      for (size_t c = 0; c < params.in_size_; c++) {
        for (size_t i = r.begin(); i < r.end(); i++) {
          dW_check[sample][c * params.out_size_ + i] += curr_delta_check[sample][i] * prev_out_check[sample][c];
        }
      }
    });

    // dbをhalf型で計算
    // for (size_t sample = 0; sample < prev_out.size(); sample++) {
    //   for (size_t i = 0; i < params.out_size_; i++) {
    //     db_check[sample][i] = half(0.0);
    //     for (size_t c = 0; c < params.in_size_; c++) {
    //       db_check[sample][i] += curr_delta_check[sample][i];
    //     }
    //   }
    // }

    for_(layer_parallelize, 0, params.out_size_, [&](const blocked_range &r) {
      if (params.has_bias_) {
        // vec_t& db = *in_grad[2];
        for (size_t i = r.begin(); i < r.end(); i++) {
          db_check[sample][i] += curr_delta_check[sample][i];
        }
      }
    });
  }


  int flag = 0;
  for (size_t sample = 0; sample < prev_delta.size(); sample++) {
    for (size_t i = 0; i < prev_delta[0].size(); i++) {
      float_t diff = prev_delta_check[sample][i] - prev_delta[sample][i];
      if (diff > 0.0001 || diff < -0.0001) {
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
    for (size_t i = 0; i < dW[0].size(); i++) {
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

  for (size_t sample = 0; sample < prev_out_half.size(); sample++) {
    for (size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      for (size_t i = 0; i < params.out_size_; i++) {
        prev_delta_half[sample][c] += curr_delta_half[sample][i] * W_half[c * params.out_size_ + i];
      }
    }

    // dWをhalf型で計算
    for_(layer_parallelize, 0, params.out_size_, [&](const blocked_range &r) {
      // accumulate weight-step using delta
      // dW[c * out_size + i] += current_delta[i] * prev_out[c]
      for (size_t c = 0; c < params.in_size_; c++) {
        for (size_t i = r.begin(); i < r.end(); i++) {
          dW_half[sample][c * params.out_size_ + i] += curr_delta_half[sample][i] * prev_out_half[sample][c];
        }
      }
      
      if (params.has_bias_) {
        // vec_t& db = *in_grad[2];
        for (size_t i = r.begin(); i < r.end(); i++) {
          db_half[sample][i] += curr_delta_half[sample][i];
        }
      }
    });
  }

  two_half_to_vector(prev_delta, prev_delta_half);
  two_half_to_vector(dW, dW_half);
  two_half_to_vector(db, db_half);

#endif
}

}  // namespace kernels
}  // namespace tiny_dnn
