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

#ifndef CC_STONE_H_
#define CC_STONE_H_

#include <cstdint>

#include "cc/color.h"
#include "cc/group.h"
#include "cc/logging.h"

namespace minigo {

// Stone represents either a stone on the board or, when empty() == true, an
// empty point on the board.
// Stone tracks both the color (empty, black or white) and group the ID of the
// stone's.
class Stone {
 public:
  Stone() = default;
  Stone(const Stone& other) = default;
  Stone(Color color, GroupId group_id)
      : value_(static_cast<uint16_t>(color) | (group_id << 2)) {
    MG_DCHECK(color != Color::kEmpty);
  }

  Stone& operator=(const Stone& other) = default;

  bool empty() const { return value_ == 0; }
  Color color() const { return static_cast<Color>(value_ & 3); }
  GroupId group_id() const { return value_ >> 2; }

 private:
  uint16_t value_ = 0;
};

}  // namespace minigo

#endif  // CC_STONE_H_
