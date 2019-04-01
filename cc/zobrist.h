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

namespace internal {
extern Hash kBlackToPlayHash;
extern Hash kOpponentPassedHash;
extern std::array<std::array<Hash, 3>, kNumMoves> kMoveHashes;
extern std::array<Hash, kN * kN> kIllegalEmptyPointHashes;
}  // namespace internal

// Non-zero when it's black's turn.
inline Hash ToPlayHash(Color color) {
  return color == Color::kBlack ? internal::kBlackToPlayHash : 0;
}

// Hash set when the previous move was a pass.
inline Hash OpponentPassedHash() { return internal::kOpponentPassedHash; }

// Hashes for moves by black and white.
inline Hash MoveHash(Coord c, Color color) {
  return internal::kMoveHashes[c][static_cast<int>(color)];
}

// Hashes used for empty points that can't be played because of things like
// self-capture, ko or positional superko.
inline Hash IllegalEmptyPointHash(Coord c) {
  return internal::kIllegalEmptyPointHashes[c];
}

void Init(uint64_t seed);

}  // namespace zobrist
}  // namespace minigo

#endif  // CC_ZOBRIST_H_
