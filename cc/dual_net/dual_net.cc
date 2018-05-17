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

#include "cc/dual_net/dual_net.h"

#include "cc/color.h"
#include "cc/constants.h"

namespace minigo {

constexpr int DualNet::kNumStoneFeatures;
constexpr int DualNet::kNumBoardFeatures;

void DualNet::InitializeFeatures(const Position& position,
                                 BoardFeatures* features) {
  const auto my_color = position.to_play();
  const auto their_color = OtherColor(my_color);
  const float to_play = my_color == Color::kBlack ? 1 : 0;

  for (int i = 0; i < kN * kN; ++i) {
    int j = i * kNumStoneFeatures;
    auto stone_color = position.stones()[i].color();
    auto my_stone = stone_color == my_color ? 1 : 0;
    auto their_stone = stone_color == their_color ? 1 : 0;
    for (int plane = 0; plane < kPlayerFeature; plane += 2) {
      (*features)[j++] = my_stone;
      (*features)[j++] = their_stone;
    }
    (*features)[j++] = to_play;
  }
}

// The update loop here is a little tricky.
//
// The chart below shows, for each move, how the stones from the last 8 moves
// should be distributed through the input planes.
//
//                                     planes
//   move | to play |   0    1    2    3    4    5   ...  16
//  ------+---------+-----------------------------------------
//     1  |    B    |  B_1  W_1   -    -    -    -   ...   1
//     2  |    W    |  W_2  B_2  W_1  B_1   -    -   ...   0
//     3  |    B    |  B_3  W_3  B_2  W_2  B_1  W_1  ...   1
//     4  |    W    |  W_4  B_4  W_3  B_3  W_2  B_2  ...   0
//    ... |   ...   |  ...  ...  ...  ...  ...  ...  ...  ...
//
// For example: on move 3, planes 0 & 1 hold the black & white stones that are
// on the board before move 3 is played, planes 2 & 3 hold the stones that were
// on the board before move 2 was played, planes 4 & 5 hold the stones that
// were on the board before move 1 was played, etc.
//
// So... to update the features, we need to do four things:
//  1) Shuffle the planes for moves t .. t-6 over to the planes for moves
//     t-1 .. t-7.
//  2) Swap the black and white planes for moves t-1 .. t-7.
//  3) Write the new black and white stones into planes 0 & 1 (or planes 1 & 0
//     depending on who is to play first).
//  4) Write the "to play" feature into plane 16.
//
// Steps 3 and 4 are trivial.
//
// Steps 1 and 2 can be accomplished in one by the following:
//  1) Copy even planes from plane N to plane N + 3.
//  2) Copy odd planes from plane N to plane N + 1.
//
// The code below does this slightly differently, updated the planes in the
// reverse order because that allows old_features and new_features to point to
// the same array, but the end result is the same.
void DualNet::UpdateFeatures(const BoardFeatures& old_features,
                             const Position& position,
                             BoardFeatures* new_features) {
  const auto my_color = position.to_play();
  const auto their_color = OtherColor(my_color);
  const float to_play = my_color == Color::kBlack ? 1 : 0;

  for (int i = 0; i < kN * kN; ++i) {
    auto stone_color = position.stones()[i].color();
    const auto* src = old_features.data() + i * kNumStoneFeatures;
    auto* dst = new_features->data() + i * kNumStoneFeatures;

    dst[kPlayerFeature] = to_play;
    for (int j = kPlayerFeature - 2; j > 0; j -= 2) {
      dst[j + 1] = src[j - 2];
      dst[j] = src[j - 1];
    }
    dst[1] = stone_color == their_color ? 1 : 0;
    dst[0] = stone_color == my_color ? 1 : 0;
  }
}

DualNet::~DualNet() = default;

}  // namespace minigo
