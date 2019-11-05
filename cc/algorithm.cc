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

#include "cc/algorithm.h"

#include <emmintrin.h>

#include <cstdint>

namespace minigo {

// ArgMaxSse works by treating the input array as four separate interleaved
// arrays: calculating the maximum of each array independently and then finding
// the maximum of the maximums. If multiple elements have the same value, the
// index of the first one is returned, which replicates the behavior of
// `std::max_element`.
int ArgMaxSse(absl::Span<const float> span) {
  // Handle small arrays.
  if (span.size() <= 4) {
    MG_CHECK(!span.empty());
    size_t idx_max = 0;
    for (size_t i = 1; i < span.size(); ++i) {
      if (span[i] > span[idx_max]) {
        idx_max = i;
      }
    }
    return idx_max;
  }

  // Holds the indices of the maximum elements found so far.
  // On iteration `j` of the loop, `idx_max[i]` holds the maximum of elements
  // `span[4 * k + i]` for all `k` in the range `[0, j)`.
  __m128i idx_max = _mm_set_epi32(3, 2, 1, 0);

  // Holds the values of the maximum elements found so far.
  __m128 val_max = _mm_loadu_ps(span.data());

  // The indices of the elements we'll testing on each iteration of the loop.
  __m128i idx = idx_max;

  // Step size: each iteration compares four elements at time.
  __m128i step = _mm_set1_epi32(4);

  // Round the size of the array down to a multiple of four; we'll handle the
  // last few elements (if any) at the end.
  size_t safe_size = span.size() & ~3;
  for (size_t i = 4; i < safe_size; i += 4) {
    // Load the next four elements.
    idx = _mm_add_epi32(idx, step);
    __m128 val = _mm_loadu_ps(span.data() + i);

    // We need to calculate the following:
    //   `idx_max[i] = val[i] > val_max[i] ? idx[i] : idx_max[i]`
    // This can be accomplished in a few instructions using bitwise operations.

    // `mask[i] = val[i] > val_max[i] ? 0xffffffff : 0`
    __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(val, val_max));

    // `val_max[i] = (mask[i] & idx[i]) | (~mask[i] & idx_max[i])`
    idx_max =
        _mm_or_si128(_mm_and_si128(mask, idx), _mm_andnot_si128(mask, idx_max));

    // Fortunately, we can compute `val_max` in a single operation:
    //   `val_max[i] = val[i] > val_max[i] ? val[i] : val_max[i]`
    val_max = _mm_max_ps(val, val_max);
  }

  // Extract the values of `val_max` and `idx_max`.
  float vals[4];
  int32_t idxs[4];
  _mm_storeu_ps(vals, val_max);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(idxs), idx_max);

  // Find the maximum of maximums found by the SSE code, breaking ties using the
  // smaller index.
  int result = idxs[0];
  for (int i = 1; i < 4; ++i) {
    if ((vals[i] > span[result]) ||
        (vals[i] == span[result] && idxs[i] < result)) {
      result = idxs[i];
    }
  }

  // Handle any remaining elements.
  for (size_t i = safe_size; i < span.size(); ++i) {
    if (span[i] > span[result]) {
      result = i;
    }
  }

  return result;
}

}  // namespace minigo
