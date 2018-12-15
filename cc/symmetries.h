// Copyright 2018 Google LLC
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

#ifndef CC_SYMMETRIES_H_
#define CC_SYMMETRIES_H_

#include <algorithm>
#include <cstring>

#include "cc/logging.h"

namespace minigo {
namespace symmetry {

enum Symmetry {
  // No transform.
  kIdentity,

  // 90 degree anticlockwise rotation.
  kRot90,

  // 180 degree rotation.
  kRot180,

  // 270 degree anticlockwise rotation.
  kRot270,

  // Transpose.
  kFlip,

  // Transpose then 90 degree anticlockwise rotation (vertical reflection).
  kFlipRot90,

  // Transpose then 180 degree rotation.
  kFlipRot180,

  // Transpose then 270 degree anticlockwise rotation (horizontal reflection).
  kFlipRot270,

  kNumSymmetries,
};

inline Symmetry Inverse(Symmetry sym) {
  switch (sym) {
    case kIdentity:
      return kIdentity;
    case kRot90:
      return kRot270;
    case kRot180:
      return kRot180;
    case kRot270:
      return kRot90;
    case kFlip:
      return kFlip;
    case kFlipRot90:
      return kFlipRot90;
    case kFlipRot180:
      return kFlipRot180;
    case kFlipRot270:
      return kFlipRot270;
    default:
      MG_LOG(FATAL) << static_cast<int>(sym);
      return kNumSymmetries;
  }
}

template <int N, int num_channels, typename SrcIt, typename DstIt>
inline void Identity(SrcIt src, DstIt dst) {
  MG_CHECK(dst != src);
  std::copy_n(src, N * N * num_channels, dst);
}

template <int N, int num_channels, typename SrcIt, typename DstIt>
inline void Rot90(SrcIt src, DstIt dst) {
  MG_CHECK(dst != src);
  const int row_stride = num_channels * N;
  for (int j = 0; j < N; ++j) {
    auto s = src + (N - 1 - j) * num_channels;
    for (int i = 0; i < N; ++i) {
      dst = std::copy_n(s, num_channels, dst);
      s += row_stride;
    }
  }
}

template <int N, int num_channels, typename SrcIt, typename DstIt>
inline void Rot180(SrcIt src, DstIt dst) {
  MG_CHECK(dst != src);
  const int row_stride = num_channels * N;
  for (int j = 0; j < N; ++j) {
    auto s = src + (N - 1 - j) * row_stride + (N - 1) * num_channels;
    for (int i = 0; i < N; ++i) {
      dst = std::copy_n(s, num_channels, dst);
      s -= num_channels;
    }
  }
}

template <int N, int num_channels, typename SrcIt, typename DstIt>
inline void Rot270(SrcIt src, DstIt dst) {
  MG_CHECK(dst != src);
  const int row_stride = num_channels * N;
  for (int j = 0; j < N; ++j) {
    auto s = src + (N - 1) * row_stride + j * num_channels;
    for (int i = 0; i < N; ++i) {
      dst = std::copy_n(s, num_channels, dst);
      s -= row_stride;
    }
  }
}

template <int N, int num_channels, typename SrcIt, typename DstIt>
inline void Flip(SrcIt src, DstIt dst) {
  MG_CHECK(dst != src);
  const int row_stride = num_channels * N;
  for (int j = 0; j < N; ++j) {
    auto s = src + j * num_channels;
    for (int i = 0; i < N; ++i) {
      dst = std::copy_n(s, num_channels, dst);
      s += row_stride;
    }
  }
}

template <int N, int num_channels, typename SrcIt, typename DstIt>
inline void FlipRot90(SrcIt src, DstIt dst) {
  MG_CHECK(dst != src);
  const int row_stride = num_channels * N;
  for (int j = 0; j < N; ++j) {
    auto s = src + (N - 1 - j) * row_stride;
    for (int i = 0; i < N; ++i) {
      dst = std::copy_n(s, num_channels, dst);
      s += num_channels;
    }
  }
}

template <int N, int num_channels, typename SrcIt, typename DstIt>
inline void FlipRot180(SrcIt src, DstIt dst) {
  MG_CHECK(dst != src);
  const int row_stride = num_channels * N;
  for (int j = 0; j < N; ++j) {
    auto s = src + (N - 1) * row_stride + (N - 1 - j) * num_channels;
    for (int i = 0; i < N; ++i) {
      dst = std::copy_n(s, num_channels, dst);
      s -= row_stride;
    }
  }
}

template <int N, int num_channels, typename SrcIt, typename DstIt>
inline void FlipRot270(SrcIt src, DstIt dst) {
  MG_CHECK(dst != src);
  const int row_stride = num_channels * N;
  for (int j = 0; j < N; ++j) {
    auto s = src + j * row_stride + (N - 1) * num_channels;
    for (int i = 0; i < N; ++i) {
      dst = std::copy_n(s, num_channels, dst);
      s -= num_channels;
    }
  }
}

template <int N, int num_channels, typename SrcIt, typename DstIt>
inline void ApplySymmetry(Symmetry sym, SrcIt src, DstIt dst) {
  switch (sym) {
    case kIdentity:
      Identity<N, num_channels>(src, dst);
      break;
    case Symmetry::kRot90:
      Rot90<N, num_channels>(src, dst);
      break;
    case Symmetry::kRot180:
      Rot180<N, num_channels>(src, dst);
      break;
    case Symmetry::kRot270:
      Rot270<N, num_channels>(src, dst);
      break;
    case Symmetry::kFlip:
      Flip<N, num_channels>(src, dst);
      break;
    case Symmetry::kFlipRot90:
      FlipRot90<N, num_channels>(src, dst);
      break;
    case Symmetry::kFlipRot180:
      FlipRot180<N, num_channels>(src, dst);
      break;
    case Symmetry::kFlipRot270:
      FlipRot270<N, num_channels>(src, dst);
      break;
    default:
      MG_LOG(FATAL) << static_cast<int>(sym);
      break;
  }
}

template <int N, int num_channels, typename T>
class NchwOutputIterator {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using reference = T&;
  using pointer = T*;
  using iterator_category = std::output_iterator_tag;

  NchwOutputIterator(T* features) : ptr_(features), offset_(0) {}

  NchwOutputIterator& operator++() {
    offset_ += N * N;
    if (offset_ >= N * N * num_channels) {
      offset_ -= N * N * num_channels - 1;
    }
    return *this;
  }

  reference operator*() const { return ptr_[offset_]; }

  bool operator!=(const T* ptr) { return ptr != ptr_ + offset_; }

 private:
  T* ptr_;
  size_t offset_;
};

}  // namespace symmetry
}  // namespace minigo

#endif  // CC_SYMMETRIES_H_
