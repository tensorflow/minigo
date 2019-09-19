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

#ifndef CC_TINY_SET_H_
#define CC_TINY_SET_H_

#include "cc/inline_vector.h"

namespace minigo {

// tiny_set is a very simple set-like container that uses inline storage.
// Since insertions into the set are O(N), tiny_set should only be used for a
// very small number of elements.
// The Position code uses tiny_sets to keep track of neighboring groups of a
// point on the board.
template <typename T, int Capacity>
class tiny_set : private inline_vector<T, Capacity> {
  using impl = inline_vector<T, Capacity>;

 public:
  using impl::begin;
  using impl::empty;
  using impl::end;
  using impl::size;
  using impl::operator[];

  // Insert an element into the set.
  // Returns true if the insertion took place, or false if the element was
  // already preset in the set.
  bool insert(const T& x) {
    for (const auto& y : *this) {
      if (x == y) {
        return false;
      }
    }
    impl::push_back(x);
    return true;
  }

  // Returns true if the set contains x.
  bool contains(const T& x) const {
    for (const auto& y : *this) {
      if (x == y) {
        return true;
      }
    }
    return false;
  }
};

}  // namespace minigo

#endif  // CC_TINY_SET_H_
