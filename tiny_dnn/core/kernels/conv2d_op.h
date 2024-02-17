/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "../../core/framework/op_kernel.h"

#include "../../core/kernels/conv2d_op_avx.h"
#include "../../core/kernels/conv2d_op_internal.h"
#include "../../core/kernels/conv2d_op_nnpack.h"

namespace tiny_dnn {

class Conv2dOp : public core::OpKernel {
 public:
  explicit Conv2dOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    // std::cout << "Conv2dOp::compute" << std::endl;
    auto params = OpKernel::params_->conv();

    // incomimg/outcoming data
    const tensor_t &in_data = context.input(0);
    const tensor_t &W       = context.input(1);
    const tensor_t &bias    = context.input(2);
    tensor_t &out_data      = context.output(0);

    // initialize outputs
    fill_tensor(out_data, float_t{0});

    // call convolution algorithm depending
    // on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::conv2d_op_internal(in_data, W[0], bias[0], out_data, params,
                                  context.parallelize());
    } else if (engine == core::backend_t::nnpack) {
      kernels::conv2d_op_nnpack(in_data, W[0], bias[0], out_data, params);
    } else if (engine == core::backend_t::avx) {
      kernels::conv2d_op_avx(in_data, W[0], bias[0], out_data, params,
                             context.parallelize());
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }

  void compute16(core::OpKernelContext &context) override {
    // std::cout << "Conv2dOp::compute" << std::endl;
    auto params = OpKernel::params_->conv();

    // incomimg/outcoming data
    const tensor16_t &in_data = context.input16(0);
    const tensor16_t &W       = context.input16(1);
    const tensor16_t &bias    = context.input16(2);
    tensor16_t &out_data      = context.output16(0);

    // initialize outputs
    fill_tensor(out_data, half{0});

    // call convolution algorithm depending
    // on the selected engine type

    const core::backend_t engine = context.engine();

    #if CONV_F_HALF == 0
    // 入力データをfloatに変換
    tensor_t in_data_float;
    for (size_t i = 0; i < in_data.size(); i++) {
      in_data_float.push_back(vec_t());
      for (size_t j = 0; j < in_data[i].size(); j++) {
        in_data_float[i].push_back(float(in_data[i][j]));
      }
    }

    // 重みデータをfloatに変換
    tensor_t W_float;
    for (size_t i = 0; i < W.size(); i++) {
      W_float.push_back(vec_t());
      for (size_t j = 0; j < W[i].size(); j++) {
        W_float[i].push_back(float(W[i][j]));
      }
    }

    // バイアスデータをfloatに変換
    tensor_t bias_float;
    for (size_t i = 0; i < bias.size(); i++) {
      bias_float.push_back(vec_t());
      for (size_t j = 0; j < bias[i].size(); j++) {
        bias_float[i].push_back(float(bias[i][j]));
      }
    }

    // 出力データをfloatに変換
    tensor_t out_data_float;
    for (size_t i = 0; i < out_data.size(); i++) {
      out_data_float.push_back(vec_t());
      for (size_t j = 0; j < out_data[i].size(); j++) {
        out_data_float[i].push_back(float(out_data[i][j]));
      }
    }

    kernels::conv2d_op_internal(in_data_float, W_float[0], bias_float[0], out_data_float, params, context.parallelize());

    // 出力データをhalfに変換
    for (size_t i = 0; i < out_data.size(); i++) {
      for (size_t j = 0; j < out_data[i].size(); j++) {
        out_data[i][j] = half(out_data_float[i][j]);
      }
    }
    #else
    kernels::conv2d_op_internal(in_data, W[0], bias[0], out_data, params, context.parallelize());
    #endif
  }
};

}  // namespace tiny_dnn
