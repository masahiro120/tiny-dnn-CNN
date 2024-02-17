/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "../../core/framework/op_kernel.h"

#include "../../core/kernels/conv2d_grad_op_avx.h"
#include "../../core/kernels/conv2d_op_internal.h"

namespace tiny_dnn {

class Conv2dGradOp : public core::OpKernel {
 public:
  explicit Conv2dGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    // std::cout << "Conv2dGradOp::compute" << std::endl;
    auto params = OpKernel::params_->conv();

    // incoming/outcoming data
    const tensor_t &prev_out = context.input(0);
    const tensor_t &W        = context.input(1);
    tensor_t &dW             = context.input_grad(1);
    tensor_t &db             = context.input_grad(2);
    tensor_t &prev_delta     = context.input_grad(0);
    tensor_t &curr_delta     = context.output_grad(0);

    // initalize outputs
    fill_tensor(prev_delta, float_t{0});

    // call convolution algorithm depending
    // on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::conv2d_op_internal(prev_out, W[0], dW, db, curr_delta,
                                  prev_delta, params, context.parallelize());
    } else if (engine == core::backend_t::avx) {
      kernels::conv2d_grad_op_avx(prev_out, W[0], dW, db, curr_delta,
                                  prev_delta, params, context.parallelize());
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }

  void compute16(core::OpKernelContext &context) override {
    // std::cout << "Conv2dGradOp::compute" << std::endl;
    auto params = OpKernel::params_->conv();

    // incoming/outcoming data
    const tensor16_t &prev_out = context.input16(0);
    const tensor16_t &W        = context.input16(1);
    tensor16_t &dW             = context.input_grad16(1);
    tensor16_t &db             = context.input_grad16(2);
    tensor16_t &prev_delta     = context.input_grad16(0);
    tensor16_t &curr_delta     = context.output_grad16(0);

    // initalize outputs
    fill_tensor(prev_delta, half{0});

    // std::cout << "prev_delta.size() = " << prev_delta.size() << std::endl;

    // call convolution algorithm depending
    // on the selected engine type

    const core::backend_t engine = context.engine();

    #if CONV_B_HALF == 0
    // 入力データをfloatに変換
    tensor_t prev_out_float;
    for (size_t i = 0; i < prev_out.size(); i++) {
      prev_out_float.emplace_back(vec_t(prev_out[i].size()));
      for (size_t j = 0; j < prev_out[i].size(); j++) {
        prev_out_float[i][j] = float(prev_out[i][j]);
      }
    }

    // 重みデータをfloatに変換
    tensor_t W_float;
    for (size_t i = 0; i < W.size(); i++) {
      W_float.emplace_back(vec_t(W[i].size()));
      for (size_t j = 0; j < W[i].size(); j++) {
        W_float[i][j] = float(W[i][j]);
      }
    }

    // バイアスデータをfloatに変換
    tensor_t db_float;
    for (size_t i = 0; i < db.size(); i++) {
      db_float.emplace_back(vec_t(db[i].size()));
      for (size_t j = 0; j < db[i].size(); j++) {
        db_float[i][j] = float(db[i][j]);
      }
    }

    // デルタデータをfloatに変換
    tensor_t curr_delta_float;
    for (size_t i = 0; i < curr_delta.size(); i++) {
      curr_delta_float.emplace_back(vec_t(curr_delta[i].size()));
      for (size_t j = 0; j < curr_delta[i].size(); j++) {
        curr_delta_float[i][j] = float(curr_delta[i][j]);
      }
    }

    // 前のデルタデータをfloatに変換
    tensor_t prev_delta_float;
    for (size_t i = 0; i < prev_delta.size(); i++) {
      prev_delta_float.emplace_back(vec_t(prev_delta[i].size()));
      for (size_t j = 0; j < prev_delta[i].size(); j++) {
        prev_delta_float[i][j] = float(prev_delta[i][j]);
      }
    }

    // dWをfloatに変換
    tensor_t dW_float;
    for (size_t i = 0; i < dW.size(); i++) {
      dW_float.emplace_back(vec_t(dW[i].size()));
      for (size_t j = 0; j < dW[i].size(); j++) {
        dW_float[i][j] = float(dW[i][j]);
      }
    }

    // dbをfloatに変換
    for (size_t i = 0; i < db.size(); i++) {
      for (size_t j = 0; j < db[i].size(); j++) {
        db[i][j] = half(db[i][j]);
      }
    }

    kernels::conv2d_op_internal(prev_out_float, W_float[0], dW_float, db_float, curr_delta_float, prev_delta_float, params, context.parallelize());

    // 前のデルタデータをhalfに変換
    for (size_t i = 0; i < prev_delta.size(); i++) {
      for (size_t j = 0; j < prev_delta[i].size(); j++) {
        prev_delta[i][j] = half(prev_delta_float[i][j]);
      }
    }

    // dWをhalfに変換
    for (size_t i = 0; i < dW.size(); i++) {
      for (size_t j = 0; j < dW[i].size(); j++) {
        dW[i][j] = half(dW[i][j]);
      }
    }

    // dbをhalfに変換
    for (size_t i = 0; i < db.size(); i++) {
      for (size_t j = 0; j < db[i].size(); j++) {
        db[i][j] = half(db[i][j]);
      }
    }
    
    #else
    kernels::conv2d_op_internal16(prev_out, W[0], dW, db, curr_delta, prev_delta, params, context.parallelize());
    #endif
  }
};

}  // namespace tiny_dnn
