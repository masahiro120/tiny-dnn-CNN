/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "../../core/framework/op_kernel.h"

#include "../../core/kernels/maxpool_op_avx.h"
#include "../../core/kernels/maxpool_op_internal.h"

namespace tiny_dnn {

class MaxPoolGradOp : public core::OpKernel {
 public:
  explicit MaxPoolGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto &params = OpKernel::params_->maxpool();

    // incoming/outcoming data
    tensor_t &prev_delta = context.input_grad(0);
    tensor_t &curr_delta = context.output_grad(0);

    // initialize outputs
    fill_tensor(prev_delta, float_t{0});

    // call the algorithm depending on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::maxpool_grad_op_internal(prev_delta, curr_delta,
                                        params.out2inmax, params.in2out,
                                        context.parallelize());
    } else if (engine == core::backend_t::avx) {
      kernels::maxpool_grad_op_avx(prev_delta, curr_delta, params.out2inmax,
                                   params.in2out, context.parallelize());
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }

  void compute16(core::OpKernelContext &context) override {
    auto &params = OpKernel::params_->maxpool();

    // incoming/outcoming data
    tensor16_t &prev_delta = context.input_grad16(0);
    tensor16_t &curr_delta = context.output_grad16(0);

    // initialize outputs
    fill_tensor(prev_delta, half{0});

    // call the algorithm depending on the selected engine type

    const core::backend_t engine = context.engine();

    #if MAXPOOL_F_HALF == 0
    // 入力データをfloatに変換
    tensor_t prev_delta_float;
    for (size_t i = 0; i < prev_delta.size(); i++) {
      prev_delta_float.push_back(vec_t());
      for (size_t j = 0; j < prev_delta[i].size(); j++) {
        prev_delta_float[i].push_back(float(prev_delta[i][j]));
      }
    }

    tensor_t curr_delta_float;
    for (size_t i = 0; i < curr_delta.size(); i++) {
      curr_delta_float.push_back(vec_t());
      for (size_t j = 0; j < curr_delta[i].size(); j++) {
        curr_delta_float[i].push_back(float(curr_delta[i][j]));
      }
    }

    kernels::maxpool_grad_op_internal(prev_delta_float, curr_delta_float,
                                      params.out2inmax, params.in2out,
                                      context.parallelize());

    // floatに変換したデータをhalfに変換
    for (size_t i = 0; i < prev_delta.size(); i++) {
      for (size_t j = 0; j < prev_delta[i].size(); j++) {
        prev_delta[i][j] = half(prev_delta_float[i][j]);
      }
    }

    for (size_t i = 0; i < curr_delta.size(); i++) {
      for (size_t j = 0; j < curr_delta[i].size(); j++) {
        curr_delta[i][j] = half(curr_delta_float[i][j]);
      }
    }

    #else

    kernels::maxpool_grad_op_internal(prev_delta, curr_delta,
                                      params.out2inmax, params.in2out,
                                      context.parallelize());

    #endif
  }
};

}  // namespace tiny_dnn
