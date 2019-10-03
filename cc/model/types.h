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

#include "cc/position.h"
#include "cc/symmetries.h"

namespace minigo {

// A simple tensor representation that abstracts a real engine-specific
// tensor. Tensor does not own the memory pointed to by `data`.
// Tensors are assumed to be tightly packed for now.
// TODO(tommadams): Make this templated on the data type so we can support
// byte input features, and quantized outputs.
template <typename T>
struct Tensor {
  Tensor() = default;
  Tensor(int n, int h, int w, int c, T* data)
      : n(n), h(h), w(w), c(c), data(data) {}

  // TODO(tommadams): replace (n, h, w, c) with `inline_vector<int, 4> dims`.
  int n = 0;
  int h = 0;
  int w = 0;
  int c = 0;
  T* data = nullptr;
};

template <typename T>
class BackedTensor {
 public:
  BackedTensor() = default;
  BackedTensor(int n, int h, int w, int c) { resize(n, h, w, c); }

  void resize(int n, int h, int w, int c) {
    auto size = n * h * w * c;
    if (static_cast<size_t>(size) > buffer_.size()) {
      buffer_.resize(size);
    }
    tensor_ = {n, h, w, c, buffer_.data()};
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
