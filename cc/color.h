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

#ifndef CC_COLOR_H_
#define CC_COLOR_H_

#include <ostream>

#include "absl/strings/string_view.h"
#include "cc/logging.h"

namespace minigo {

// Color represents the stone color of each point on the board.
// The position code relies on the values of kEmpty, kBlack and kWhite being
// 0, 1, 2 respectively for some of its bit twiddling.
enum class Color {
  kEmpty,
  kBlack,
  kWhite,
};

inline Color OtherColor(Color color) {
  MG_CHECK(color == Color::kWhite || color == Color::kBlack);
  return color == Color::kWhite ? Color::kBlack : Color::kWhite;
}

// Returns ".", "B" or "W".
absl::string_view ColorToCode(Color color);

std::ostream& operator<<(std::ostream& os, Color color);

}  // namespace minigo

#endif  // CC_COLOR_H_
