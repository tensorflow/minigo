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

#include <cstdint>
#include <ostream>
#include <string>

#include "absl/strings/string_view.h"
#include "cc/constants.h"
#include "cc/logging.h"

namespace minigo {

// Coord represents the coordinates of a point on the board, a pass, or a
// resign.
class Coord {
 public:
  static constexpr uint16_t kPass = kN * kN;
  static constexpr uint16_t kResign = kPass + 1;
  static constexpr uint16_t kInvalid = 0xffff;
  static const char kGtpColumns[20];

  Coord(uint16_t value) : value_(value) {}  // NOLINT(runtime/explicit)

  Coord(int row, int col) {
    MG_DCHECK(row >= 0 && row < kN);
    MG_DCHECK(col >= 0 && col < kN);
    value_ = row * kN + col;
  }

  // Parse a Coord from a GTP string.
  static Coord FromGtp(absl::string_view str, bool allow_invalid = false);

  // Parse a Coord from a SGF string.
  static Coord FromSgf(absl::string_view str, bool allow_invalid = false);

  // Parse a Coord from one of the above string representations.
  static Coord FromString(absl::string_view str, bool allow_invalid = false);

  // Format the Coord as a GTP string.
  std::string ToGtp() const;

  // Format the Coord as an SGF string.
  std::string ToSgf() const;

  operator uint16_t() const { return value_; }

  // Enable Coords to be used as keys in absl hashed containers.
  template <typename H>
  friend H AbslHashValue(H h, Coord c) {
    return H::combine(std::move(h), c.value_);
  }

 private:
  uint16_t value_;
};

// Formats the coord as a GTP string.
std::ostream& operator<<(std::ostream& os, Coord c);

}  // namespace minigo

#endif  // CC_COORD_H_
