// Copyright 2019 Google LLC
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

#ifndef CC_PADDED_ARRAY_H_
#define CC_PADDED_ARRAY_H_

#include <array>

namespace minigo {

constexpr size_t kAlignment = 16;

template <typename T>
class PaddedSpan;

// An array implementation whose internal storage is padded to be a multiple of
// 16 bytes. This means vectorized SSE code can read and write to the array
// without having to worry about the last few elements in the array.
//
// NOTE: this class does NOT guarantee that the base address of the array is
// also aligned to 16 bytes, so vectorized code should always use unaligned
// loads and stores. In practice these aren't significantly slows that aligned
// loads and stores on modern x86 architectures anyway.
template <typename T, size_t Size>
class PaddedArray {
  static constexpr size_t kSizeBytes = sizeof(T) * Size;
  static constexpr size_t kPaddedBytes =
      ((kSizeBytes + kAlignment - 1) / kAlignment) * kAlignment;
  static_assert(kPaddedBytes % sizeof(T) == 0, "padded size isn't aligned");
  static_assert(Size > 0, "size must be > 0");
  static constexpr size_t kPaddedSize = kPaddedBytes / sizeof(T);

 public:
  // Enable implict conversion to PaddedSpan<T>.
  operator PaddedSpan<T>() { return {data(), size()}; }
  operator PaddedSpan<const T>() const { return {data(), size()}; }

  constexpr size_t empty() const { return false; }
  constexpr size_t size() const { return Size; }
  constexpr size_t padded_size() const { return kPaddedSize; }

  const T* data() const { return impl_.data(); }
  T* data() { return impl_.data(); }

  const T& operator[](int i) const { return impl_[i]; }
  T& operator[](int i) { return impl_[i]; }

  const T* begin() const { return impl_.begin(); }
  T* begin() { return impl_.begin(); }

  const T* end() const { return impl_.end(); }
  T* end() { return impl_.end(); }

 private:
  std::array<T, kPaddedSize> impl_;
};

// A span type that can only be constructed from a `PaddedArray`.
// This enables functions to accept a `PaddedArray` without having to know its
// size at compile time, while preserving the guarantee that the storage is
// sufficiently padded.
template <typename T>
class PaddedSpan {
 public:
  constexpr size_t empty() const { return size_ == 0; }
  constexpr size_t size() const { return size_; }

  const T* data() const { return data_; }
  T* data() { return data_; }

  const T& operator[](int i) const { return data_[i]; }
  T& operator[](int i) { return data_[i]; }

  const T* begin() const { return data_; }
  T* begin() { return data_; }

  const T* end() const { return data_ + size_; }
  T* end() { return data_ + size_; }

 private:
  template <typename U, size_t Size>
  friend class PaddedArray;

  PaddedSpan(T* data, size_t size) : data_(data), size_(size) {}

  T* data_;
  size_t size_;
};

}  // namespace minigo

#endif  // CC_PADDED_ARRAY_H_
