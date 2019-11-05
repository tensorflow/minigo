// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CC_MODEL_TYPES_H_
#define CC_MODEL_TYPES_H_

#include <iostream>
#include <vector>

#include "cc/inline_vector.h"
#include "cc/logging.h"
#include "cc/position.h"
#include "cc/symmetries.h"

namespace minigo {

// Holds the shape of a tensor and provides a place to put shape-related logic
// that isn't coupled to a specific tensor implementation.
class TensorShape {
 public:
  // Maximum number of dimensions supported.
  static constexpr int kMaxDims = 4;

  // Creates an empty tensor shape.
  // Equivalent to calling `TensorShape({})`.
  TensorShape() {}

  // Creates a tensor shape of the given dimensions.
  // CHECK fails if `shape.size() > TensorShape::kMaxDims`.
  TensorShape(std::initializer_list<int> shape) {
    for (auto x : shape) {
      impl_.push_back(x);
    }
  }

  // Returns true if the shape matches.
  // Certain dimensions in the shape can be ignored by passing -1:
  //   TensorShape shape(1, 2, 3, 4);
  //   MG_CHECK(shape.is({1, 2, -1, 4});
  bool is(std::initializer_list<int> shape) const {
    if (static_cast<size_t>(impl_.size()) != shape.size()) {
      return false;
    }
    int i = 0;
    for (auto x : shape) {
      if (x >= 0 && x != impl_[i]) {
        return false;
      }
      i += 1;
    }
    return true;
  }

  // (in)equality comparison operators.
  // Unlike `is`, these operators do not treat negative dimensions specially.
  bool operator==(const TensorShape& other) const {
    if (impl_.size() != other.size()) {
      return false;
    }
    for (int i = 0; i < impl_.size(); ++i) {
      if (impl_[i] != other[i]) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const TensorShape& other) const { return !(*this == other); }

  bool empty() const { return impl_.empty(); }
  int size() const { return impl_.size(); }
  int operator[](int i) const { return impl_[i]; }

  // Returns the number of elements in the tensor.
  int num_elements() const {
    if (empty()) {
      return 0;
    }
    int result = impl_[0];
    for (int i = 1; i < impl_.size(); ++i) {
      result *= impl_[i];
    }
    return result;
  }

 private:
  inline_vector<int, kMaxDims> impl_;
};

std::ostream& operator<<(std::ostream& os, const TensorShape& shape);

// A simple tensor representation that abstracts a real engine-specific
// tensor. Tensor does not own the memory pointed to by `data`.
// Tensors are assumed to be tightly packed for now.
template <typename T>
struct Tensor {
  Tensor() = default;
  Tensor(const TensorShape& shape, T* data) : shape(shape), data(data) {}
  TensorShape shape;
  T* data = nullptr;
};

template <typename T>
class BackedTensor {
 public:
  BackedTensor() = default;
  BackedTensor(const TensorShape& shape) { resize(shape); }

  void resize(const TensorShape& shape) {
    int size;
    if (shape.empty()) {
      size = 0;
    } else {
      size = shape[0];
      for (int i = 1; i < shape.size(); ++i) {
        size *= shape[i];
      }
    }
    if (static_cast<size_t>(size) > buffer_.size()) {
      buffer_.resize(size);
    }
    tensor_ = {shape, buffer_.data()};
  }

  const Tensor<T>& tensor() const { return tensor_; }
  Tensor<T>& tensor() { return tensor_; }

 private:
  Tensor<T> tensor_;
  std::vector<T> buffer_;
};

struct ModelInput {
  // Symmetry to apply to the input features when performing inference.
  symmetry::Symmetry sym = symmetry::kNumSymmetries;

  // position_history[0] holds the current position and position_history[i]
  // holds the position from i moves ago.
  inline_vector<const Position*, kMaxPositionHistory> position_history;
};

struct ModelOutput {
  std::array<float, kNumMoves> policy;
  float value;
};

}  // namespace minigo

#endif  // CC_MODEL_TYPES_H_
