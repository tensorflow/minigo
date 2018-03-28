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

#ifndef CC_COORD_H_
#define CC_COORD_H_

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>

#include "absl/strings/string_view.h"
#include "cc/constants.h"

namespace minigo {

// Coord represents the coordinates of a point on the board or, in the case of
// Coord::kPass, a pass move.
class Coord {
 public:
  static constexpr uint16_t kPass = kN * kN;
  static constexpr uint16_t kInvalid = 0xffff;
  static constexpr char kKgsColumns[] = "ABCDEFGHJKLMNOPQRST";

  Coord(uint16_t value) : value_(value) {}  // NOLINT(runtime/explicit)

  Coord(int row, int col) {
    assert(row >= 0 && row < kN);
    assert(col >= 0 && col < kN);
    value_ = row * kN + col;
  }

  // Parse a Coord from a KGS string.
  static Coord FromKgs(absl::string_view str);

  // Parse a Coord from a SGF string.
  static Coord FromSgf(absl::string_view str);

  // Parse a Coord from one of the above string representations.
  static Coord FromString(absl::string_view str);

  // Format the Coord as a KGS string.
  std::string ToKgs() const;

  operator uint16_t() const { return value_; }

 private:
  uint16_t value_;
};

// Formats the coord as a KGS string.
std::ostream& operator<<(std::ostream& os, Coord c);

}  // namespace minigo

#endif  // CC_COORD_H_
