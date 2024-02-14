/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <iomanip>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "optimizers/optimizer.h"
#include "util/product.h"
#include "util/util.h"
#include "util/weight_init.h"

#ifdef DNN_USE_IMAGE_API
#include "util/image.h"
#endif


#include "half.hpp"
#include "half_define.h"

using namespace half_float;

// #define MARGE_HALF 0

std::vector<half> one_vector_to_half(const tiny_dnn::vec_t& array);
std::vector<std::vector<half>> two_vector_to_half(const tiny_dnn::tensor_t& array);
void one_half_to_vector(tiny_dnn::vec_t& array, std::vector<half> array_half);

namespace tiny_dnn {

class node;
class layer;
class edge;

typedef std::shared_ptr<edge> edgeptr_t;

/**
 * base class of all kind of tinny-cnn data
 **/
class node : public std::enable_shared_from_this<node> {
 public:
  node(size_t in_size, size_t out_size) : prev_(in_size), next_(out_size) {}
  virtual ~node() {}

  const std::vector<edgeptr_t> &prev() const { return prev_; }
  const std::vector<edgeptr_t> &next() const { return next_; }

  size_t prev_port(const edge &e) const {
    auto it = std::find_if(prev_.begin(), prev_.end(),
                           [&](edgeptr_t ep) { return ep.get() == &e; });
    return (size_t)std::distance(prev_.begin(), it);
  }

  size_t next_port(const edge &e) const {
    auto it = std::find_if(next_.begin(), next_.end(),
                           [&](edgeptr_t ep) { return ep.get() == &e; });
    return (size_t)std::distance(next_.begin(), it);
  }

  std::vector<node *> prev_nodes()
    const;  // @todo refactor and remove this method
  std::vector<node *> next_nodes()
    const;  // @todo refactor and remove this method

 protected:
  node() = delete;

  friend void connect(layer *head,
                      layer *tail,
                      size_t head_index,
                      size_t tail_index);

  mutable std::vector<edgeptr_t> prev_;
  mutable std::vector<edgeptr_t> next_;
};

/**
 * class containing input/output data
 **/
class edge {
 public:
  edge(node *prev, const shape3d &shape, vector_type vtype)
    : shape_(shape),
      vtype_(vtype),
      data_({vec_t(shape.size())}),
      grad_({vec_t(shape.size())}),
      data_16_({vec16_t(shape.size())}),
      grad_16_({vec16_t(shape.size())}),
      prev_(prev) {}

  void merge_grads(vec_t *dst) {
#if MARGE_HALF == 0
#if 0
    assert(!grad_.empty());
    const auto &grad_head = grad_[0];
    size_t sz             = grad_head.size();
    dst->resize(sz);
    float_t *pdst = &(*dst)[0];
    // dst = grad_[0]
    std::copy(grad_head.begin(), grad_head.end(), pdst);
    // @todo consider adding parallelism
    for (size_t sample = 1, sample_count = grad_.size(); sample < sample_count;
         ++sample) {
      // dst += grad_[sample]
      vectorize::reduce<float_t>(&grad_[sample][0], sz, pdst);
    }
#else
    assert(!grad_.empty());
    tensor_t grad_val = grad_;
    size_t sz         = grad_val[0].size();
    dst->resize(sz);
    // float_t *pdst = &(*dst)[0];
    vec_t dst_val(sz);
    dst_val = *dst;

    // dst = grad_[0]
    for (size_t i = 0; i < sz; i++) {
      dst_val[i] = grad_val[0][i];
    }

    for (size_t sample = 1, sample_count = grad_val.size(); sample < sample_count;
         ++sample) {
      // dst += grad_[sample]
      for (size_t i = 0; i < sz; i++) {
        dst_val[i] += grad_val[sample][i];
      }
    }

    *dst = dst_val;

#endif
#else
    assert(!grad_.empty());
    // tensor_t grad_val = grad_;
    std::vector<std::vector<half>> grad_half = two_vector_to_half(grad_);
    size_t sz         = grad_half[0].size();
    dst->resize(sz);
    // float_t *pdst = &(*dst)[0];
    vec_t dst_val(sz);
    dst_val = *dst;
    std::vector<half> dst_half = one_vector_to_half(dst_val);

    // dst = grad_[0]
    for (size_t i = 0; i < sz; i++) {
      dst_half[i] = grad_half[0][i];
    }

    for (size_t sample = 1, sample_count = grad_half.size(); sample < sample_count;
         ++sample) {
      // dst += grad_[sample]
      for (size_t i = 0; i < sz; i++) {
        dst_half[i] += grad_half[sample][i];
      }
    }

    one_half_to_vector(dst_val, dst_half);


    *dst = dst_val;
#endif
  }

  void merge_grads(vec16_t *dst) {
    assert(!grad_16_.empty());
    const auto &grad_head = grad_16_[0];
    size_t sz             = grad_head.size();
    dst->resize(sz);
    half *pdst = &(*dst)[0];
    // dst = grad_16_[0]
    std::copy(grad_head.begin(), grad_head.end(), pdst);
    // @todo consider adding parallelism
    for (size_t sample = 1, sample_count = grad_16_.size(); sample < sample_count;
         ++sample) {
      // dst += grad_16_[sample]
      // vectorize::reduce<half>(&grad_16_[sample][0], sz, pdst);
      for (size_t i = 0; i < sz; i++) {
        pdst[i] += grad_16_[sample][i];
      }
    }
  }

  void clear_grads() {
    for (size_t sample = 0, sample_count = grad_.size(); sample < sample_count;
         ++sample) {
      auto &g = grad_[sample];
      vectorize::fill(&g[0], g.size(), float_t{0});
    }
  }

  void clear_grads16() {
    for (size_t sample = 0, sample_count = grad_16_.size(); sample < sample_count;
         ++sample) {
      auto &g = grad_16_[sample];
      vectorize::fill(&g[0], g.size(), half{0});
    }
  }

  tensor_t *get_data() { return &data_; }
  
  tensor16_t *get_data16() { return &data_16_; }

  //自分で作成
  // void *set_data(tensor_t w) {data_ = w;}
  void weight_bias_update(vec_t v) {
    tensor_t v_2d;
    v_2d.push_back(v);
    data_ = v_2d;
  }

  const tensor_t *get_data() const { return &data_; }
  
  const tensor16_t *get_data16() const { return &data_16_; }

  tensor_t *get_gradient() { return &grad_; }

  tensor16_t *get_gradient16() { return &grad_16_; }

  const tensor_t *get_gradient() const { return &grad_; }

  const tensor16_t *get_gradient16() const { return &grad_16_; }

  const std::vector<node *> &next() const { return next_; }
  node *prev() { return prev_; }
  const node *prev() const { return prev_; }

  const shape3d &shape() const { return shape_; }
  vector_type vtype() const { return vtype_; }
  void add_next_node(node *next) { next_.push_back(next); }

 private:
  shape3d shape_;
  vector_type vtype_;
  tensor_t data_;
  tensor_t grad_;
  tensor16_t data_16_;
  tensor16_t grad_16_;
  node *prev_;                // previous node, "producer" of this tensor
  std::vector<node *> next_;  // next nodes, "consumers" of this tensor
};

inline std::vector<node *> node::prev_nodes() const {
  std::vector<node *> vecs;
  for (auto &e : prev_) {
    if (e && e->prev()) {
      vecs.insert(vecs.end(), e->prev());
    }
  }
  return vecs;
}

inline std::vector<node *> node::next_nodes() const {
  std::vector<node *> vecs;
  for (auto &e : next_) {
    if (e) {
      auto n = e->next();
      vecs.insert(vecs.end(), n.begin(), n.end());
    }
  }
  return vecs;
}

template <typename T>
struct layer_tuple {
  layer_tuple(T l1, T l2) {
    layers_.push_back(l1);
    layers_.push_back(l2);
  }
  std::vector<T> layers_;
};

template <
  typename T,
  typename U,
  typename std::enable_if<std::is_base_of<layer, T>::value &&
                          std::is_base_of<layer, U>::value>::type * = nullptr>
layer_tuple<layer *> operator,(T &l1, U &l2) {
  return layer_tuple<layer *>(&l1, &l2);
}

template <
  typename T,
  typename U,
  typename std::enable_if<std::is_base_of<layer, T>::value &&
                          std::is_base_of<layer, U>::value>::type * = nullptr>
layer_tuple<std::shared_ptr<layer>> operator,(std::shared_ptr<T> l1,
                                              std::shared_ptr<U> l2) {
  return layer_tuple<std::shared_ptr<layer>>(l1, l2);
}

template <
  typename T,
  typename std::enable_if<std::is_base_of<layer, T>::value>::type * = nullptr>
layer_tuple<layer *> operator,(layer_tuple<layer *> lhs, T &rhs) {
  lhs.layers_.push_back(&rhs);
  return lhs;
}

template <
  typename T,
  typename std::enable_if<std::is_base_of<layer, T>::value>::type * = nullptr>
layer_tuple<std::shared_ptr<layer>> operator,(
  layer_tuple<std::shared_ptr<layer>> lhs, std::shared_ptr<T> &rhs) {
  lhs.layers_.push_back(rhs);
  return lhs;
}

template <
  typename T,
  typename std::enable_if<std::is_base_of<layer, T>::value>::type * = nullptr>
layer_tuple<layer *> operator,(T &lhs, layer_tuple<layer *> rhs) {
  rhs.layers_.insert(rhs.layers_.begin(), &lhs);
  return rhs;
}

template <
  typename T,
  typename std::enable_if<std::is_base_of<layer, T>::value>::type * = nullptr>
layer_tuple<std::shared_ptr<layer>> operator,(
  std::shared_ptr<T> &lhs, layer_tuple<std::shared_ptr<layer>> rhs) {
  rhs.layers_.insert(rhs.layers_.begin(), lhs);
  return rhs;
}

template <typename T, typename U>
inline std::shared_ptr<U> &operator<<(std::shared_ptr<T> &lhs,
                                      std::shared_ptr<U> &rhs) {
  connect(lhs.get(), rhs.get());
  return rhs;
}

template <typename T>
inline T &operator<<(const layer_tuple<std::shared_ptr<layer>> &lhs, T &rhs) {
  for (size_t i = 0; i < lhs.layers_.size(); i++) {
    connect(&*lhs.layers_[i], &*rhs, 0, i);
  }
  return rhs;
}

template <typename T>
inline const layer_tuple<std::shared_ptr<layer>> &operator<<(
  T &lhs, const layer_tuple<std::shared_ptr<layer>> &rhs) {
  for (size_t i = 0; i < rhs.layers_.size(); i++) {
    connect(&*lhs, &*rhs.layers_[i], i, 0);
  }
  return rhs;
}

template <typename T>
inline T &operator<<(const layer_tuple<layer *> &lhs, T &rhs) {
  for (size_t i = 0; i < lhs.layers_.size(); i++) {
    connect(lhs.layers_[i], &rhs, 0, i);
  }
  return rhs;
}

template <typename T>
inline const layer_tuple<layer *> &operator<<(T &lhs,
                                              const layer_tuple<layer *> &rhs) {
  for (size_t i = 0; i < rhs.layers_.size(); i++) {
    connect(&lhs, rhs.layers_[i], i, 0);
  }
  return rhs;
}
}  // namespace tiny_dnn
