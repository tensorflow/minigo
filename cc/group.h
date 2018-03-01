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

#ifndef CC_GROUP_H_
#define CC_GROUP_H_

#include <cstdint>

#include "cc/constants.h"
#include "cc/inline_vector.h"

namespace minigo {

// GroupId is a unique identifier for a group (string) of stones.
using GroupId = uint16_t;

// Group represents a group (string) of stones.
// A group only keeps track of the count of its current liberties, not their
// location.
struct Group {
  Group() = default;
  Group(uint16_t size, uint16_t num_liberties)
      : size(size), num_liberties(num_liberties) {}

  // Maximum number of potential groups on the board.
  // Used in various places to pre-allocate buffers.
  // TODO(tommadams): We can probably reduce the space reserved for potential
  // groups a bit: https://senseis.xmp.net/?MaximumNumberOfLiveGroups
  static constexpr int kMaxNumGroups = kN * kN;

  uint16_t size = 0;
  uint16_t num_liberties = 0;
};

// GroupPool is a simple memory pool for Group objects.
class GroupPool {
 public:
  // Allocates a new Group with the given size and number of liberties, and
  // returns the group's ID.
  GroupId alloc(uint16_t size, uint16_t num_liberties) {
    GroupId id;
    if (!free_ids_.empty()) {
      // We have at least one previously acclocated then freed group, return it.
      id = free_ids_.back();
      free_ids_.pop_back();
      groups_[id] = {size, num_liberties};
    } else {
      // Allocate a new group from the pool.
      id = static_cast<GroupId>(groups_.size());
      groups_.emplace_back(size, num_liberties);
    }
    return id;
  }

  // Free the group, returning it the pool.
  void free(GroupId id) { free_ids_.push_back(id); }

  // Access the Group object by ID.
  Group& operator[](GroupId id) { return groups_[id]; }
  const Group& operator[](GroupId id) const { return groups_[id]; }

 private:
  inline_vector<Group, Group::kMaxNumGroups> groups_;
  inline_vector<GroupId, Group::kMaxNumGroups> free_ids_;
};

}  // namespace minigo

#endif  // CC_GROUP_H_
