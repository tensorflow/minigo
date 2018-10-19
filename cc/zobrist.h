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

#ifndef CC_ZOBRIST_H_
#define CC_ZOBRIST_H_

#include <algorithm>
#include <array>
#include <cstdint>

#include "cc/color.h"
#include "cc/constants.h"
#include "cc/coord.h"

namespace minigo {
namespace zobrist {

using Hash = uint64_t;

extern Hash kBlackToPlayHash;
extern std::array<std::array<Hash, 3>, kNumMoves> kMoveHashes;
extern std::array<Hash, kN * kN> kKoHashes;

inline Hash ToPlayHash(Color color) {
  return color == Color::kBlack ? kBlackToPlayHash : 0;
}

inline Hash MoveHash(Coord c, Color color) {
  return kMoveHashes[c][static_cast<int>(color)];
}

inline Hash KoHash(Coord c) { return c == Coord::kInvalid ? 0 : kKoHashes[c]; }

void Init(uint64_t seed);

}  // namespace zobrist
}  // namespace minigo

#endif  // CC_ZOBRIST_H_
