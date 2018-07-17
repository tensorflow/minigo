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

#ifndef CC_INLINE_VECTOR_H_
#define CC_INLINE_VECTOR_H_

#include <cstdint>
#include <new>
#include <utility>

#include "cc/check.h"

namespace minigo {

// inline_vector is an std::vector-like container that uses inline storage, thus
// avoiding heap allocations.
// Since we currently only need to store POD types in inline_vector, this is
// a fairly bare bones implementation.
template <typename T, int Capacity>
class inline_vector {
 public:
  inline_vector() = default;
  ~inline_vector() { clear(); }
  inline_vector(const inline_vector& other) {
    for (const auto& x : other) {
      push_back(x);
    }
  }
  inline_vector& operator=(const inline_vector& other) {
    if (&other != this) {
      clear();
      for (const auto& x : other) {
        push_back(x);
      }
    }
    return *this;
  }

  void clear() {
    for (T& x : *this) {
      x.~T();
    }
    size_ = 0;
  }

  int size() const { return size_; }
  int capacity() const { return Capacity; }
  bool empty() const { return size_ == 0; }

  T* data() { return reinterpret_cast<T*>(storage_); }
  const T* data() const { return reinterpret_cast<const T*>(storage_); }

  T* begin() { return data(); }
  const T* begin() const { return data(); }
  T* end() { return data() + size_; }
  const T* end() const { return data() + size_; }

  T& operator[](int idx) {
    MG_DCHECK(idx >= 0);
    MG_DCHECK(idx < size_);
    return data()[idx];
  }
  const T& operator[](int idx) const {
    MG_DCHECK(idx >= 0);
    MG_DCHECK(idx < size_);
    return data()[idx];
  }

  void push_back(const T& t) {
    MG_CHECK(size_ < Capacity);
    new (data() + size_) T(t);
    ++size_;
  }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    MG_CHECK(size_ < Capacity);
    new (data() + size_) T(std::forward<Args>(args)...);
    ++size_;
  }

  T& back() { return data()[size_ - 1]; }
  const T& back() const { return data()[size_ - 1]; }

  void pop_back() {
    MG_CHECK(size_ > 0);
    --size_;
  }

 private:
  int size_ = 0;
  uint8_t __attribute__((aligned(alignof(T)))) storage_[Capacity * sizeof(T)];
};

}  // namespace minigo

#endif  // CC_INLINE_VECTOR_H_
