/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <unordered_map>

#include "../util/util.h"

#include "../half.hpp"
#include "../half_define.h"

using half_float::half;

std::vector<half> one_vector_to_half(const tiny_dnn::vec_t& array);
tiny_dnn::vec16_t one_vector_to_half16(const tiny_dnn::vec_t& array);
void one_half_to_vector(tiny_dnn::vec_t& array, std::vector<half> array_half);
void one_half_to_vector(tiny_dnn::vec_t& array, tiny_dnn::vec16_t array_half);

namespace tiny_dnn {

/**
 * base class of optimizer
 * usesHessian : true if an optimizer uses hessian (2nd order derivative of loss
 *function)
 **/
struct optimizer {
  optimizer()                  = default;
  optimizer(const optimizer &) = default;
  optimizer(optimizer &&)      = default;
  optimizer &operator=(const optimizer &) = default;
  optimizer &operator=(optimizer &&) = default;
  virtual ~optimizer()               = default;
  virtual void update(const vec_t &dW, vec_t &W, bool parallelize) = 0;
  virtual void update16(const vec16_t &dW, vec16_t &W, bool parallelize) = 0;
  virtual void reset() {}  // override to implement pre-learning action
};

// helper class to hold N values for each weight
template <int N>
struct stateful_optimizer : public optimizer {
  void reset() override {
    for (auto &e : E_) e.clear();
    for (auto &e : E_16_) e.clear();
  }

 protected:
  template <int Index>
  vec_t &get(const vec_t &key) {
    static_assert(Index < N, "index out of range");
    if (E_[Index][&key].empty()) E_[Index][&key].resize(key.size(), float_t());
    return E_[Index][&key];
  }

  template <int Index>
  vec16_t &get(const vec16_t &key) {
    static_assert(Index < N, "index out of range");
    if (E_16_[Index][&key].empty()) E_16_[Index][&key].resize(key.size(), half());
    return E_16_[Index][&key];
  }
  std::unordered_map<const vec_t *, vec_t> E_[N];
  std::unordered_map<const vec16_t *, vec16_t> E_16_[N];
};

/**
 * adaptive gradient method
 *
 * J Duchi, E Hazan and Y Singer,
 * Adaptive subgradient methods for online learning and stochastic optimization
 * The Journal of Machine Learning Research, pages 2121-2159, 2011.
 **/
struct adagrad : public stateful_optimizer<1> {
  adagrad() : alpha(float_t(0.01)), eps(float_t(1e-8)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    vec_t &g = get<0>(W);
    for_i(parallelize, W.size(), [&](size_t i) {
      g[i] += dW[i] * dW[i];
      W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
    });
  }

  float_t alpha;  // learning rate
 private:
  float_t eps;
};

/**
 * RMSprop
 *
 * T Tieleman, and G E Hinton,
 * Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning (2012)
 **/
struct RMSprop : public stateful_optimizer<1> {
  RMSprop() : alpha(float_t(0.0001)), mu(float_t(0.99)), eps(float_t(1e-8)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    vec_t &g = get<0>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
      g[i] = mu * g[i] + (1 - mu) * dW[i] * dW[i];
      W[i] -= alpha * dW[i] / std::sqrt(g[i] + eps);
    });
  }

  float_t alpha;  // learning rate
  float_t mu;     // decay term
 private:
  float_t eps;  // constant value to avoid zero-division
};

/**
 * @brief [a new optimizer (2015)]
 * @details [see Adam: A Method for Stochastic Optimization (Algorithm 1)
 *               http://arxiv.org/abs/1412.6980]
 *
 */
struct adam : public stateful_optimizer<2> {
  adam()
    : alpha(float_t(0.001)),
      b1(float_t(0.9)),
      b2(float_t(0.999)),
      b1_t(float_t(0.9)),
      b2_t(float_t(0.999)),
      eps(float_t(1e-8)),
      alpha_16(half(0.001)),
      b1_16(half(0.9)),
      b2_16(half(0.999)),
      b1_t_16(half(0.9)),
      b2_t_16(half(0.999)),
      eps_16(half(1e-8)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
#if UPDATE == 0
#if 0
    vec_t &mt = get<0>(W);
    vec_t &vt = get<1>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
      mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
      vt[i] = b2 * vt[i] + (float_t(1) - b2) * dW[i] * dW[i];

      // L2 norm based update rule
      W[i] -= alpha * (mt[i] / (float_t(1) - b1_t)) /
              std::sqrt((vt[i] / (float_t(1) - b2_t)) + eps);
    });

    b1_t *= b1;
    b2_t *= b2;
#else

    vec_t &mt = get<0>(W);
    vec_t &vt = get<1>(W);

    vec_t mt_val = mt;
    vec_t vt_val = vt;

    vec_t dW_val = dW;
    vec_t W_val = W;

    for_i(parallelize, W.size(), [&](size_t i) {
      mt_val[i] = b1 * mt_val[i] + (float_t(1) - b1) * dW_val[i];
      vt_val[i] = b2 * vt_val[i] + (float_t(1) - b2) * dW_val[i] * dW_val[i];

      // L2 norm based update rule
      W_val[i] -= alpha * (mt_val[i] / (float_t(1) - b1_t)) /
              std::sqrt((vt_val[i] / (float_t(1) - b2_t)) + eps);
    });

    mt = mt_val;
    vt = vt_val;
    // dW = dW_val;
    W = W_val;

    b1_t *= b1;
    b2_t *= b2;

#endif
#else
    vec_t &mt = get<0>(W);
    vec_t &vt = get<1>(W);

    std::vector<half> mt_half = one_vector_to_half(mt);
    std::vector<half> vt_half = one_vector_to_half(vt);

    std::vector<half> dW_half = one_vector_to_half(dW);
    std::vector<half> W_half = one_vector_to_half(W);

    vec_t mt_val(mt.size());
    one_half_to_vector(mt_val, mt_half);
    vec_t vt_val(vt.size());
    one_half_to_vector(vt_val, vt_half);
    vec_t dW_val(dW.size());
    one_half_to_vector(dW_val, dW_half);
    vec_t W_val(W.size());
    one_half_to_vector(W_val, W_half);

    // for_i(parallelize, W.size(), [&](size_t i) {
    //   mt_half[i] = (half)(b1) * mt_half[i] + (half(1) - (half)(b1)) * dW_half[i];
    //   vt_half[i] = (half)(b2) * vt_half[i] + (half(1) - (half)(b2)) * dW_half[i] * dW_half[i];

    //   // L2 norm based update rule
    //   W_half[i] -= (half)(alpha) * (mt_half[i] / (half(1) - (half)(b1_t))) /
    //           std::sqrt((vt_half[i] / (half(1) - (half)(b2_t))) + (half)(eps));
    // });

    // const half half_min = std::numeric_limits<half>::lowest(); // halfの最小値を取得

    // for_i(parallelize, W.size(), [&](size_t i) {
    //   mt_half[i] = (half)(b1) * mt_half[i] + (half(1) - (half)(b1)) * dW_half[i];
    //   vt_half[i] = (half)(b2) * vt_half[i] + (half(1) - (half)(b2)) * dW_half[i] * dW_half[i];

    //   half denom = (half)std::sqrt((float)((vt_half[i] / (half(1) - (half)(b2_t))) + (half)(eps)));
    //   if (denom == (half)0) {
    //     W_half[i] = (W_half[i] > 0) ? half_min : -half_min;
    //   } else {
    //     half update_value = (half)(alpha) * (mt_half[i] / (half(1) - (half)(b1_t))) / denom;
    //     if (std::isnan(update_value)) {
    //       W_half[i] = (W_half[i] > 0) ? half_min : -half_min;
    //     } else {
    //       W_half[i] -= update_value;
    //     }
    //   }

    //   W_half[i] = std::max(half_min, (half)std::abs((float)W_half[i])) * (W_half[i] > 0 ? 1 : -1);
    // });

    for_i(parallelize, W.size(), [&](size_t i) {
      mt_val[i] = b1 * mt_val[i] + (float_t(1) - b1) * dW_val[i];
      vt_val[i] = b2 * vt_val[i] + (float_t(1) - b2) * dW_val[i] * dW_val[i];

      // L2 norm based update rule
      W_val[i] -= alpha * (mt_val[i] / (float_t(1) - b1_t)) /
              std::sqrt((vt_val[i] / (float_t(1) - b2_t)) + eps);
    });

    mt_half = one_vector_to_half(mt_val);
    vt_half = one_vector_to_half(vt_val);
    dW_half = one_vector_to_half(dW_val);
    W_half = one_vector_to_half(W_val);

    // nan check
    for (size_t i = 0; i < W_half.size(); i++) {
      if (std::isnan(W_half[i])) {
        std::cout << "W nan is detected at " << i << std::endl;
        exit(1);
      }

      if (std::isnan(mt_half[i])) {
        std::cout << "mt nan is detected at " << i << std::endl;
        exit(1);
      }

      if (std::isnan(vt_half[i])) {
        std::cout << "vt nan is detected at " << i << std::endl;
        exit(1);
      }
    }

    one_half_to_vector(mt, mt_half);
    one_half_to_vector(vt, vt_half);
    // one_half_to_vector(dW, dW_half);
    one_half_to_vector(W, W_half);

    b1_t *= b1;
    b2_t *= b2;

#endif
  }

  void update16(const vec16_t &dW, vec16_t &W, bool parallelize) {
  #if UPDATE == 1
    vec16_t &mt = get<0>(W);
    vec16_t &vt = get<1>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
      mt[i] = b1_16 * mt[i] + (half(1) - b1_16) * dW[i];
      vt[i] = b2_16 * vt[i] + (half(1) - b2_16) * dW[i] * dW[i];

      // L2 norm based update rule
      W[i] -= alpha * (mt[i] / (half(1) - b1_t_16)) /
              std::sqrt((vt[i] / (half(1) - b2_t_16)) + eps);
    });

    b1_t_16 *= b1_16;
    b2_t_16 *= b2_16;
  #else
  
    vec16_t &mt = get<0>(W);
    vec16_t &vt = get<1>(W);

    vec_t mt_val(mt.size());
    one_half_to_vector(mt_val, mt);
    vec_t vt_val(vt.size());
    one_half_to_vector(vt_val, vt);

    vec_t dW_val(dW.size());
    one_half_to_vector(dW_val, dW);
    vec_t W_val(W.size());
    one_half_to_vector(W_val, W);

    for_i(parallelize, W.size(), [&](size_t i) {
      mt_val[i] = b1 * mt_val[i] + (float_t(1) - b1) * dW_val[i];
      vt_val[i] = b2 * vt_val[i] + (float_t(1) - b2) * dW_val[i] * dW_val[i];

      // L2 norm based update rule
      W_val[i] -= alpha * (mt_val[i] / (float_t(1) - b1_t)) /
              std::sqrt((vt_val[i] / (float_t(1) - b2_t)) + eps);
    });

    mt = one_vector_to_half16(mt_val);
    vt = one_vector_to_half16(vt_val);
    // dW = one_vector_to_half16(dW_val);
    W = one_vector_to_half16(W_val);

    b1_t *= b1;
    b2_t *= b2;

  #endif
  }

  float_t alpha;  // learning rate
  float_t b1;     // decay term
  float_t b2;     // decay term
  float_t b1_t;   // decay term power t
  float_t b2_t;   // decay term power t

  half alpha_16;  // learning rate
  half b1_16;     // decay term
  half b2_16;     // decay term
  half b1_t_16;   // decay term power t
  half b2_t_16;   // decay term power t

 private:
  float_t eps;  // constant value to avoid zero-division
  half eps_16;  // constant value to avoid zero-division
};

/**
 * @brief [a new optimizer (2015)]
 * @details [see Adam: A Method for Stochastic Optimization (Algorithm 2)
 *               http://arxiv.org/abs/1412.6980]
 *
 */
struct adamax : public stateful_optimizer<2> {
  adamax()
    : alpha(float_t(0.002)),
      b1(float_t(0.9)),
      b2(float_t(0.999)),
      b1_t(b1),
      eps(float_t(1e-8)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    vec_t &mt = get<0>(W);
    vec_t &ut = get<1>(W);

    for_i(parallelize, W.size(), [&](int i) {
      mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
      ut[i] = std::max(b2 * ut[i], std::abs(dW[i]));

      // Lp norm based update rule
      W[i] -= (alpha / (1.0 - b1_t)) * (mt[i] / (ut[i] + eps));
    });

    b1_t *= b1;
  }

  float_t alpha;  // learning rate
  float_t b1;     // decay term
  float_t b2;     // decay term
  float_t b1_t;   // decay term power t

 private:
  float_t eps;  // constant value to avoid zero-division
};

/**
 * SGD without momentum
 *
 * slightly faster than tiny_dnn::momentum
 **/
struct gradient_descent : public optimizer {
  gradient_descent() : alpha(float_t(0.01)), lambda(float_t(0)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    for_i(parallelize, W.size(),
          [&](size_t i) { W[i] = W[i] - alpha * (dW[i] + lambda * W[i]); });
  }

  float_t alpha;   // learning rate
  float_t lambda;  // weight decay
};

/**
 * SGD with momentum
 *
 * B T Polyak,
 * Some methods of speeding up the convergence of iteration methods
 * USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.
 **/
struct momentum : public stateful_optimizer<1> {
 public:
  momentum() : alpha(float_t(0.01)), lambda(float_t(0)), mu(float_t(0.9)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    vec_t &dWprev = get<0>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
      float_t V = mu * dWprev[i] - alpha * (dW[i] + W[i] * lambda);
      W[i] += V;
      dWprev[i] = V;
    });
  }

  float_t alpha;   // learning rate
  float_t lambda;  // weight decay
  float_t mu;      // momentum
};

/**
 * SGD with Nesterov momentum
 *
 * Y Nesterov,
 * A method for unconstrained convex minimization problem with the rate of
 * convergence o(1/k2), Doklady ANSSSR, vol.269, pp.543-547, 1983.
 **/
struct nesterov_momentum : public stateful_optimizer<1> {
 public:
  nesterov_momentum()
    : alpha(float_t(0.01)), lambda(float_t(0)), mu(float_t(0.9)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    vec_t &dWprev = get<0>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
      float_t V = mu * dWprev[i] - alpha * (dW[i] + W[i] * lambda);
      W[i] += (-mu) * dWprev[i] + (1 + mu) * V;
      dWprev[i] = V;
    });
  }

  float_t alpha;   // learning rate
  float_t lambda;  // weight decay
  float_t mu;      // momentum
};

/**
 * RAdam
 *
 * L Liu, H Jiang, P He, W Chen, X Liu, J Gao, and J Han,
 * On the Variance of the Adaptive Learning Rate and Beyond, arXiv preprint arXiv:1908.03265, 2019.
 **/
struct radam : public stateful_optimizer<2> {
  radam()
    : alpha(float_t(0.001)),
      b1(float_t(0.9)),
      b2(float_t(0.999)),
      b1_t(b1),
      eps(float_t(1e-8)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    vec_t &mt = get<0>(W);
    vec_t &vt = get<1>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
      mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
      vt[i] = b2 * vt[i] + (float_t(1) - b2) * dW[i] * dW[i];

      // Compute RAdam update here based on mt, vt, and other RAdam specific parameters

      // W[i] -= computed_update_value;
    });

    b1_t *= b1;
    // Add other RAdam specific updates if necessary
  }

  float_t alpha;  // learning rate
  float_t b1;     // decay term
  float_t b2;     // decay term
  float_t b1_t;   // decay term power t

 private:
  float_t eps;  // constant value to avoid zero-division
};

}  // namespace tiny_dnn
