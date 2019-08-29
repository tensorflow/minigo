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
#include <array>
#include <cstring>
#include <ostream>

#include "cc/coord.h"
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

std::ostream& operator<<(std::ostream& os, Symmetry sym);

// Helpful array of all symmetries that allows iterating over all symmetries
// without casting between int and Symmetry all the time.
extern const std::array<Symmetry, kNumSymmetries> kAllSymmetries;

// Array of inverses.
extern const std::array<Symmetry, kNumSymmetries> kInverseSymmetries;

extern const std::array<std::array<Coord, kNumMoves>, kNumSymmetries> kCoords;

inline Symmetry Inverse(Symmetry sym) { return kInverseSymmetries[sym]; }

template <int N, int num_channels, typename T>
inline void Identity(const T* src, T* dst) {
  MG_CHECK(dst != src);
  std::copy_n(src, N * N * num_channels, dst);
}

template <int N, int num_channels, typename T>
inline void Rot90(const T* src, T* dst) {
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

template <int N, int num_channels, typename T>
inline void Rot180(const T* src, T* dst) {
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

template <int N, int num_channels, typename T>
inline void Rot270(const T* src, T* dst) {
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

template <int N, int num_channels, typename T>
inline void Flip(const T* src, T* dst) {
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

template <int N, int num_channels, typename T>
inline void FlipRot90(const T* src, T* dst) {
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

template <int N, int num_channels, typename T>
inline void FlipRot180(const T* src, T* dst) {
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

template <int N, int num_channels, typename T>
inline void FlipRot270(const T* src, T* dst) {
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

template <int N, int num_channels, typename T>
inline void ApplySymmetry(Symmetry sym, const T* src, T* dst) {
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

Coord ApplySymmetry(Symmetry sym, Coord c);

// Returns the Symmetry obtained by first applying a then b.
Symmetry Concat(Symmetry a, Symmetry b);

}  // namespace symmetry
}  // namespace minigo

#endif  // CC_SYMMETRIES_H_
