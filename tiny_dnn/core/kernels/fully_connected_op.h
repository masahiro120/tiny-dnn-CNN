/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "../../core/framework/op_kernel.h"

#include "../../core/kernels/fully_connected_op_avx.h"
#include "../../core/kernels/fully_connected_op_cblas.h"
#include "../../core/kernels/fully_connected_op_intel_mkl.h"
#include "../../core/kernels/fully_connected_op_internal.h"
#include "../../core/kernels/fully_connected_op_nnpack.h"

namespace tiny_dnn {

class FullyConnectedOp : public core::OpKernel {
 public:
  explicit FullyConnectedOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->fully();

    // incomimg/outcoming data
    const tensor_t &in_data = context.input(0);
    const tensor_t &W       = context.input(1);
    const tensor_t *bias    = params.has_bias_ ? &context.input(2) : nullptr;
    tensor_t &out_data      = context.output(0);

    // initialize outputs
    fill_tensor(out_data, float_t{0});

    // call the algorithm depending  on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::fully_connected_op_internal(
        in_data, W[0], params.has_bias_ ? (*bias)[0] : vec_t(), out_data,
        params, context.parallelize());
    } else if (engine == core::backend_t::nnpack) {
      kernels::fully_connected_op_nnpack(
        in_data, W[0], params.has_bias_ ? (*bias)[0] : vec_t(), out_data,
        params, context.parallelize());
    } else if (engine == core::backend_t::avx) {
      kernels::fully_connected_op_avx(in_data, W[0],
                                      params.has_bias_ ? (*bias)[0] : vec_t(),
                                      out_data, params, context.parallelize());
    } else if (engine == core::backend_t::cblas) {
      kernels::fully_connected_op_cblas(
        in_data, W[0], params.has_bias_ ? (*bias)[0] : vec_t(), out_data,
        params, context.parallelize());
    } else if (engine == core::backend_t::intel_mkl) {
      kernels::fully_connected_op_intel_mkl(
        in_data, W[0], params.has_bias_ ? (*bias)[0] : vec_t(), out_data,
        params, context.parallelize());
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }

  void compute16(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->fully();

    // incomimg/outcoming data
    const tensor16_t &in_data = context.input16(0);
    const tensor16_t &W       = context.input16(1);
    const tensor16_t *bias    = params.has_bias_ ? &context.input16(2) : nullptr;
    tensor16_t &out_data      = context.output16(0);

    // initialize outputs
    fill_tensor(out_data, half{0});

    // call the algorithm depending  on the selected engine type

    const core::backend_t engine = context.engine();

    #if FC_F_HALF == 0
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
    for (size_t i = 0; i < bias->size(); i++) {
      vec_t temp;
      for (size_t j = 0; j < (*bias)[i].size(); j++) {
        temp.push_back(static_cast<float>((*bias)[i][j]));
      }
      bias_float.push_back(temp);
    }

    // 出力データをfloatに変換
    tensor_t out_data_float;
    for (size_t i = 0; i < out_data.size(); i++) {
      out_data_float.push_back(vec_t());
      for (size_t j = 0; j < out_data[i].size(); j++) {
        out_data_float[i].push_back(float(out_data[i][j]));
      }
    }

    kernels::fully_connected_op_internal(
      in_data_float, W_float[0], params.has_bias_ ? bias_float[0] : vec_t(),
      out_data_float, params, context.parallelize());

    // 出力データをhalfに変換
    for (size_t i = 0; i < out_data.size(); i++) {
      for (size_t j = 0; j < out_data[i].size(); j++) {
        out_data[i][j] = half(out_data_float[i][j]);
      }
    }
    #else

    kernels::fully_connected_op_internal(
      in_data, W[0], params.has_bias_ ? (*bias)[0] : vec16_t(), out_data,
      params, context.parallelize());
    
    #endif
  }
};

}  // namespace tiny_dnn
