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

#include "cc/zobrist.h"

#include "cc/random.h"

namespace minigo {
namespace zobrist {

Hash kBlackToPlayHash;
std::array<std::array<Hash, 3>, kNumMoves> kMoveHashes;
std::array<Hash, kN * kN> kKoHashes;

void Init(uint64_t seed) {
  minigo::Random rnd(seed);

  kBlackToPlayHash = rnd.UniformUint64();
  for (auto& x : kMoveHashes) {
    for (size_t i = 0; i < x.size(); ++i) {
      x[i] = i == 0 ? 0 : rnd.UniformUint64();
    }
  }
  for (auto& x : kKoHashes) {
    x = rnd.UniformUint64();
  }
}

}  // namespace zobrist
}  // namespace minigo
