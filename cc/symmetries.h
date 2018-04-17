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

#include <cstring>

#include "cc/check.h"

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
      MG_CHECK(false);
      return kNumSymmetries;
  }
}

template <typename T, int N, int num_channels>
inline void Identity(const T* src, T* dst) {
  MG_CHECK(src != dst);
  memcpy(dst, src, N * N * num_channels * sizeof(T));
}

template <typename T, int N, int num_channels>
inline void Rot90(const T* src, T* dst) {
  MG_CHECK(src != dst);
  const int col_stride = num_channels;
  const int row_stride = col_stride * N;
  for (int j = 0; j < N; ++j) {
    const auto* s = src + (N - 1 - j) * col_stride;
    auto* d = dst + j * row_stride;
    for (int i = 0; i < N; ++i) {
      memcpy(d, s, col_stride * sizeof(T));
      d += col_stride;
      s += row_stride;
    }
  }
}

template <typename T, int N, int num_channels>
inline void Rot180(const T* src, T* dst) {
  MG_CHECK(src != dst);
  const int col_stride = num_channels;
  const int row_stride = col_stride * N;
  for (int j = 0; j < N; ++j) {
    const auto* s = src + (N - 1 - j) * row_stride + (N - 1) * col_stride;
    auto* d = dst + j * row_stride;
    for (int i = 0; i < N; ++i) {
      memcpy(d, s, col_stride * sizeof(T));
      d += col_stride;
      s -= col_stride;
    }
  }
}

template <typename T, int N, int num_channels>
inline void Rot270(const T* src, T* dst) {
  MG_CHECK(src != dst);
  const int col_stride = num_channels;
  const int row_stride = col_stride * N;
  for (int j = 0; j < N; ++j) {
    const auto* s = src + (N - 1) * row_stride + j * col_stride;
    auto* d = dst + j * row_stride;
    for (int i = 0; i < N; ++i) {
      memcpy(d, s, col_stride * sizeof(T));
      d += col_stride;
      s -= row_stride;
    }
  }
}

template <typename T, int N, int num_channels>
inline void Flip(const T* src, T* dst) {
  MG_CHECK(src != dst);
  const int col_stride = num_channels;
  const int row_stride = col_stride * N;
  for (int j = 0; j < N; ++j) {
    const auto* s = src + j * col_stride;
    auto* d = dst + j * row_stride;
    for (int i = 0; i < N; ++i) {
      memcpy(d, s, col_stride * sizeof(T));
      d += col_stride;
      s += row_stride;
    }
  }
}

template <typename T, int N, int num_channels>
inline void FlipRot90(const T* src, T* dst) {
  MG_CHECK(src != dst);
  const int col_stride = num_channels;
  const int row_stride = col_stride * N;
  for (int j = 0; j < N; ++j) {
    const auto* s = src + (N - 1 - j) * row_stride;
    auto* d = dst + j * row_stride;
    for (int i = 0; i < N; ++i) {
      memcpy(d, s, col_stride * sizeof(T));
      d += col_stride;
      s += col_stride;
    }
  }
}

template <typename T, int N, int num_channels>
inline void FlipRot180(const T* src, T* dst) {
  MG_CHECK(src != dst);
  const int col_stride = num_channels;
  const int row_stride = col_stride * N;
  for (int j = 0; j < N; ++j) {
    const auto* s = src + (N - 1) * row_stride + (N - 1 - j) * col_stride;
    auto* d = dst + j * row_stride;
    for (int i = 0; i < N; ++i) {
      memcpy(d, s, col_stride * sizeof(T));
      d += col_stride;
      s -= row_stride;
    }
  }
}

template <typename T, int N, int num_channels>
inline void FlipRot270(const T* src, T* dst) {
  MG_CHECK(src != dst);
  const int col_stride = num_channels;
  const int row_stride = col_stride * N;
  for (int j = 0; j < N; ++j) {
    const auto* s = src + j * row_stride + (N - 1) * col_stride;
    auto* d = dst + j * row_stride;
    for (int i = 0; i < N; ++i) {
      memcpy(d, s, col_stride * sizeof(T));
      d += col_stride;
      s -= col_stride;
    }
  }
}

template <typename T, int N, int num_channels>
inline void ApplySymmetry(Symmetry sym, const T* src, T* dst) {
  switch (sym) {
    case kIdentity:
      Identity<T, N, num_channels>(src, dst);
      break;
    case Symmetry::kRot90:
      Rot90<T, N, num_channels>(src, dst);
      break;
    case Symmetry::kRot180:
      Rot180<T, N, num_channels>(src, dst);
      break;
    case Symmetry::kRot270:
      Rot270<T, N, num_channels>(src, dst);
      break;
    case Symmetry::kFlip:
      Flip<T, N, num_channels>(src, dst);
      break;
    case Symmetry::kFlipRot90:
      FlipRot90<T, N, num_channels>(src, dst);
      break;
    case Symmetry::kFlipRot180:
      FlipRot180<T, N, num_channels>(src, dst);
      break;
    case Symmetry::kFlipRot270:
      FlipRot270<T, N, num_channels>(src, dst);
      break;
    default:
      MG_CHECK(false);
      break;
  }
}

}  // namespace symmetry
}  // namespace minigo

#endif  // CC_SYMMETRIES_H_
