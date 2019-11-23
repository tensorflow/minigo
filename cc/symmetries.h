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

#include <array>
#include <cstring>
#include <ostream>

#include "cc/coord.h"
#include "cc/logging.h"

namespace minigo {
namespace symmetry {

enum Symmetry : uint8_t {
  // No transform.
  kIdentity,

  // 90 degree anticlockwise rotation.
  kRot90,

  // 180 degree rotation.
  kRot180,

  // 270 degree clockwise rotation.
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

// Identity symmetry is the same for both interleaved (NHWC) and planar (NCHW)
// data.
template <int N, int num, typename T>
inline void Identity(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  std::memcpy(dst, src, N * N * num * sizeof(T));
}

// Symmetries for interleaved tensors (NHWC).
template <int N, int num_channels, typename T>
inline void Rot90Interleaved(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  const int row_stride = num_channels * N;
  for (int j = 0; j < N; ++j) {
    auto s = src + (N - 1 - j) * num_channels;
    for (int i = 0; i < N; ++i) {
      std::memcpy(dst, s, num_channels * sizeof(T));
      dst += num_channels;
      s += row_stride;
    }
  }
}

template <int N, int num_channels, typename T>
inline void Rot180Interleaved(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  const int row_stride = num_channels * N;
  for (int j = 0; j < N; ++j) {
    auto s = src + (N - 1 - j) * row_stride + (N - 1) * num_channels;
    for (int i = 0; i < N; ++i) {
      std::memcpy(dst, s, num_channels * sizeof(T));
      dst += num_channels;
      s -= num_channels;
    }
  }
}

template <int N, int num_channels, typename T>
inline void Rot270Interleaved(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  const int row_stride = num_channels * N;
  for (int j = 0; j < N; ++j) {
    auto s = src + (N - 1) * row_stride + j * num_channels;
    for (int i = 0; i < N; ++i) {
      std::memcpy(dst, s, num_channels * sizeof(T));
      dst += num_channels;
      s -= row_stride;
    }
  }
}

template <int N, int num_channels, typename T>
inline void FlipInterleaved(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  const int row_stride = num_channels * N;
  for (int j = 0; j < N; ++j) {
    auto s = src + j * num_channels;
    for (int i = 0; i < N; ++i) {
      std::memcpy(dst, s, num_channels * sizeof(T));
      dst += num_channels;
      s += row_stride;
    }
  }
}

template <int N, int num_channels, typename T>
inline void FlipRot90Interleaved(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  const int row_stride = num_channels * N;
  for (int j = 0; j < N; ++j) {
    auto s = src + (N - 1 - j) * row_stride;
    for (int i = 0; i < N; ++i) {
      std::memcpy(dst, s, num_channels * sizeof(T));
      dst += num_channels;
      s += num_channels;
    }
  }
}

template <int N, int num_channels, typename T>
inline void FlipRot180Interleaved(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  const int row_stride = num_channels * N;
  for (int j = 0; j < N; ++j) {
    auto s = src + (N - 1) * row_stride + (N - 1 - j) * num_channels;
    for (int i = 0; i < N; ++i) {
      std::memcpy(dst, s, num_channels * sizeof(T));
      dst += num_channels;
      s -= row_stride;
    }
  }
}

template <int N, int num_channels, typename T>
inline void FlipRot270Interleaved(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  const int row_stride = num_channels * N;
  for (int j = 0; j < N; ++j) {
    auto s = src + j * row_stride + (N - 1) * num_channels;
    for (int i = 0; i < N; ++i) {
      std::memcpy(dst, s, num_channels * sizeof(T));
      dst += num_channels;
      s -= num_channels;
    }
  }
}

// TODO(tommadams): rename ApplySymmetry to ApplySymmetryInterleaved
template <int N, int num_channels, typename T>
inline void ApplySymmetry(Symmetry sym, const T* src, T* dst) {
  switch (sym) {
    case kIdentity:
      Identity<N, num_channels>(src, dst);
      break;
    case Symmetry::kRot90:
      Rot90Interleaved<N, num_channels>(src, dst);
      break;
    case Symmetry::kRot180:
      Rot180Interleaved<N, num_channels>(src, dst);
      break;
    case Symmetry::kRot270:
      Rot270Interleaved<N, num_channels>(src, dst);
      break;
    case Symmetry::kFlip:
      FlipInterleaved<N, num_channels>(src, dst);
      break;
    case Symmetry::kFlipRot90:
      FlipRot90Interleaved<N, num_channels>(src, dst);
      break;
    case Symmetry::kFlipRot180:
      FlipRot180Interleaved<N, num_channels>(src, dst);
      break;
    case Symmetry::kFlipRot270:
      FlipRot270Interleaved<N, num_channels>(src, dst);
      break;
    default:
      MG_LOG(FATAL) << static_cast<int>(sym);
      break;
  }
}

// Symmetries for planar tensors (NCHW).
template <int N, int num_planes, typename T>
inline void Rot90Planar(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  for (int p = 0; p < num_planes; ++p) {
    for (int j = 0; j < N; ++j) {
      auto s = src + (N - 1 - j);
      for (int i = 0; i < N; ++i) {
        *dst++ = *s;
        s += N;
      }
    }
    src += N * N;
  }
}

template <int N, int num_planes, typename T>
inline void Rot180Planar(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  for (int p = 0; p < num_planes; ++p) {
    for (int j = 0; j < N; ++j) {
      auto s = src + (N - 1 - j) * N + (N - 1);
      for (int i = 0; i < N; ++i) {
        *dst++ = *s;
        s -= 1;
      }
    }
    src += N * N;
  }
}

template <int N, int num_planes, typename T>
inline void Rot270Planar(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  for (int p = 0; p < num_planes; ++p) {
    for (int j = 0; j < N; ++j) {
      auto s = src + (N - 1) * N + j;
      for (int i = 0; i < N; ++i) {
        *dst++ = *s;
        s -= N;
      }
    }
    src += N * N;
  }
}

template <int N, int num_planes, typename T>
inline void FlipPlanar(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  for (int p = 0; p < num_planes; ++p) {
    for (int j = 0; j < N; ++j) {
      auto s = src + j;
      for (int i = 0; i < N; ++i) {
        *dst++ = *s;
        s += N;
      }
    }
    src += N * N;
  }
}

template <int N, int num_planes, typename T>
inline void FlipRot90Planar(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  for (int p = 0; p < num_planes; ++p) {
    for (int j = 0; j < N; ++j) {
      auto s = src + (N - 1 - j) * N;
      for (int i = 0; i < N; ++i) {
        *dst++ = *s;
        s += 1;
      }
    }
    src += N * N;
  }
}

template <int N, int num_planes, typename T>
inline void FlipRot180Planar(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  for (int p = 0; p < num_planes; ++p) {
    for (int j = 0; j < N; ++j) {
      auto s = src + (N - 1) * N + (N - 1 - j);
      for (int i = 0; i < N; ++i) {
        *dst++ = *s;
        s -= N;
      }
    }
    src += N * N;
  }
}

template <int N, int num_planes, typename T>
inline void FlipRot270Planar(const T* src, T* dst) {
  MG_DCHECK(dst != src);
  for (int p = 0; p < num_planes; ++p) {
    for (int j = 0; j < N; ++j) {
      auto s = src + j * N + (N - 1);
      for (int i = 0; i < N; ++i) {
        *dst++ = *s;
        s -= 1;
      }
    }
    src += N * N;
  }
}

template <int N, int num_planes, typename T>
inline void ApplySymmetryPlanar(Symmetry sym, const T* src, T* dst) {
  switch (sym) {
    case kIdentity:
      Identity<N, num_planes>(src, dst);
      break;
    case Symmetry::kRot90:
      Rot90Planar<N, num_planes>(src, dst);
      break;
    case Symmetry::kRot180:
      Rot180Planar<N, num_planes>(src, dst);
      break;
    case Symmetry::kRot270:
      Rot270Planar<N, num_planes>(src, dst);
      break;
    case Symmetry::kFlip:
      FlipPlanar<N, num_planes>(src, dst);
      break;
    case Symmetry::kFlipRot90:
      FlipRot90Planar<N, num_planes>(src, dst);
      break;
    case Symmetry::kFlipRot180:
      FlipRot180Planar<N, num_planes>(src, dst);
      break;
    case Symmetry::kFlipRot270:
      FlipRot270Planar<N, num_planes>(src, dst);
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
