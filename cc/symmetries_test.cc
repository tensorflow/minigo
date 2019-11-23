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

#include "cc/symmetries.h"

#include <array>
#include <sstream>

#include "absl/strings/str_join.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::ElementsAreArray;

namespace minigo {
namespace symmetry {
namespace {

TEST(SymmetryTest, TestRot90_1) {
  // clang-format off
  const std::array<float, 16> original = {{
     1,  2,  3,  4,
     5,  6,  7,  8,
     9, 10, 11, 12,
    13, 14, 15, 16,
  }};
  std::array<float, 16> expected = {{
     4,  8, 12, 16,
     3,  7, 11, 15,
     2,  6, 10, 14,
     1,  5,  9, 13,
  }};
  // clang-format on

  std::array<float, 16> actual;
  Rot90Interleaved<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  Rot90Planar<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestRot180_1) {
  // clang-format off
  std::array<float, 16> original = {{
     1,  2,  3,  4,
     5,  6,  7,  8,
     9, 10, 11, 12,
    13, 14, 15, 16,
  }};
  std::array<float, 16> expected = {{
    16, 15, 14, 13,
    12, 11, 10,  9,
     8,  7,  6,  5,
     4,  3,  2,  1,
  }};
  // clang-format on

  std::array<float, 16> actual;
  Rot180Interleaved<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  Rot180Planar<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 16> tmp;
  Rot90Interleaved<4, 1>(original.data(), tmp.data());
  Rot90Interleaved<4, 1>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestRot270_1) {
  // clang-format off
  std::array<float, 16> original = {{
     1,  2,  3,  4,
     5,  6,  7,  8,
     9, 10, 11, 12,
    13, 14, 15, 16,
  }};
  std::array<float, 16> expected = {{
    13,  9,  5,  1,
    14, 10,  6,  2,
    15, 11,  7,  3,
    16, 12,  8,  4,
  }};
  // clang-format on

  std::array<float, 16> actual;
  Rot270Interleaved<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  Rot270Planar<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 16> tmp1, tmp2;
  Rot90Interleaved<4, 1>(original.data(), tmp1.data());
  Rot90Interleaved<4, 1>(tmp1.data(), tmp2.data());
  Rot90Interleaved<4, 1>(tmp2.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlip_1) {
  // clang-format off
  std::array<float, 16> original = {{
     1,  2,  3,  4,
     5,  6,  7,  8,
     9, 10, 11, 12,
    13, 14, 15, 16,
  }};
  std::array<float, 16> expected = {{
     1,  5,  9, 13,
     2,  6, 10, 14,
     3,  7, 11, 15,
     4,  8, 12, 16,
  }};
  // clang-format on

  std::array<float, 16> actual;
  FlipInterleaved<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  FlipPlanar<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlipRot90_1) {
  // clang-format off
  std::array<float, 16> original = {{
     1,  2,  3,  4,
     5,  6,  7,  8,
     9, 10, 11, 12,
    13, 14, 15, 16,
  }};
  std::array<float, 16> expected = {{
    13, 14, 15, 16,
     9, 10, 11, 12,
     5,  6,  7,  8,
     1,  2,  3,  4,
  }};
  // clang-format on

  std::array<float, 16> actual;
  FlipRot90Interleaved<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  FlipRot90Planar<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 16> tmp;
  FlipInterleaved<4, 1>(original.data(), tmp.data());
  Rot90Interleaved<4, 1>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlipRot180_1) {
  // clang-format off
  std::array<float, 16> original = {{
     1,  2,  3,  4,
     5,  6,  7,  8,
     9, 10, 11, 12,
    13, 14, 15, 16,
  }};
  std::array<float, 16> expected = {{
    16, 12,  8,  4,
    15, 11,  7,  3,
    14, 10,  6,  2,
    13,  9,  5,  1
  }};
  // clang-format on

  std::array<float, 16> actual;
  FlipRot180Interleaved<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  FlipRot180Planar<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 16> tmp;
  FlipInterleaved<4, 1>(original.data(), tmp.data());
  Rot180Interleaved<4, 1>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlipRot270_1) {
  // clang-format off
  std::array<float, 16> original = {{
     1,  2,  3,  4,
     5,  6,  7,  8,
     9, 10, 11, 12,
    13, 14, 15, 16,
  }};
  std::array<float, 16> expected = {{
     4,  3,  2,  1,
     8,  7,  6,  5,
    12, 11, 10,  9,
    16, 15, 14, 13,
  }};
  // clang-format on

  std::array<float, 16> actual;
  FlipRot270Interleaved<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  FlipRot270Planar<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 16> tmp;
  FlipInterleaved<4, 1>(original.data(), tmp.data());
  Rot270Interleaved<4, 1>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestRot90Interleaved_3) {
  // clang-format off
  const std::array<float, 48> original = {{
     11,  12,  13,   21,  22,  23,   31,  32,  33,   41,  42,  43,
     51,  52,  53,   61,  62,  63,   71,  72,  73,   81,  82,  83,
     91,  92,  93,  101, 102, 103,  111, 112, 113,  121, 122, 123,
    131, 132, 133,  141, 142, 143,  151, 152, 153,  161, 162, 163,
  }};
  std::array<float, 48> expected = {{
     41,  42,  43,   81,  82,  83,  121, 122, 123,  161, 162, 163,
     31,  32,  33,   71,  72,  73,  111, 112, 113,  151, 152, 153,
     21,  22,  23,   61,  62,  63,  101, 102, 103,  141, 142, 143,
     11,  12,  13,   51,  52,  53,   91,  92,  93,  131, 132, 133,
  }};
  // clang-format on

  std::array<float, 48> actual;
  Rot90Interleaved<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestRot180Interleaved_3) {
  // clang-format off
  std::array<float, 48> original = {{
     11,  12,  13,   21,  22,  23,   31,  32,  33,   41,  42,  43,
     51,  52,  53,   61,  62,  63,   71,  72,  73,   81,  82,  83,
     91,  92,  93,  101, 102, 103,  111, 112, 113,  121, 122, 123,
    131, 132, 133,  141, 142, 143,  151, 152, 153,  161, 162, 163,
  }};
  std::array<float, 48> expected = {{
    161, 162, 163,  151, 152, 153,  141, 142, 143,  131, 132, 133,
    121, 122, 123,  111, 112, 113,  101, 102, 103,   91,  92,  93,
     81,  82,  83,   71,  72,  73,   61,  62,  63,   51,  52,  53,
     41,  42,  43,   31,  32,  33,   21,  22,  23,   11,  12,  13,
  }};
  // clang-format on

  std::array<float, 48> actual;
  Rot180Interleaved<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp;
  Rot90Interleaved<4, 3>(original.data(), tmp.data());
  Rot90Interleaved<4, 3>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestRot270Interleaved_3) {
  // clang-format off
  std::array<float, 48> original = {{
     11,  12,  13,   21,  22,  23,   31,  32,  33,   41,  42,  43,
     51,  52,  53,   61,  62,  63,   71,  72,  73,   81,  82,  83,
     91,  92,  93,  101, 102, 103,  111, 112, 113,  121, 122, 123,
    131, 132, 133,  141, 142, 143,  151, 152, 153,  161, 162, 163,
  }};
  std::array<float, 48> expected = {{
    131, 132, 133,   91,  92,  93,   51,  52,  53,   11,  12,  13,
    141, 142, 143,  101, 102, 103,   61,  62,  63,   21,  22,  23,
    151, 152, 153,  111, 112, 113,   71,  72,  73,   31,  32,  33,
    161, 162, 163,  121, 122, 123,   81,  82,  83,   41,  42,  43,
  }};
  // clang-format on

  std::array<float, 48> actual;
  Rot270Interleaved<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp1, tmp2;
  Rot90Interleaved<4, 3>(original.data(), tmp1.data());
  Rot90Interleaved<4, 3>(tmp1.data(), tmp2.data());
  Rot90Interleaved<4, 3>(tmp2.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlipInterleaved_3) {
  // clang-format off
  std::array<float, 48> original = {{
     11,  12,  13,   21,  22,  23,   31,  32,  33,   41,  42,  43,
     51,  52,  53,   61,  62,  63,   71,  72,  73,   81,  82,  83,
     91,  92,  93,  101, 102, 103,  111, 112, 113,  121, 122, 123,
    131, 132, 133,  141, 142, 143,  151, 152, 153,  161, 162, 163,
  }};
  std::array<float, 48> expected = {{
     11,  12,  13,   51,  52,  53,   91,  92,  93,  131, 132, 133,
     21,  22,  23,   61,  62,  63,  101, 102, 103,  141, 142, 143,
     31,  32,  33,   71,  72,  73,  111, 112, 113,  151, 152, 153,
     41,  42,  43,   81,  82,  83,  121, 122, 123,  161, 162, 163,
  }};
  // clang-format on

  std::array<float, 48> actual;
  FlipInterleaved<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlipRot90Interleaved_3) {
  // clang-format off
  std::array<float, 48> original = {{
     11,  12,  13,   21,  22,  23,   31,  32,  33,   41,  42,  43,
     51,  52,  53,   61,  62,  63,   71,  72,  73,   81,  82,  83,
     91,  92,  93,  101, 102, 103,  111, 112, 113,  121, 122, 123,
    131, 132, 133,  141, 142, 143,  151, 152, 153,  161, 162, 163,
  }};
  std::array<float, 48> expected = {{
    131, 132, 133,  141, 142, 143,  151, 152, 153,  161, 162, 163,
     91,  92,  93,  101, 102, 103,  111, 112, 113,  121, 122, 123,
     51,  52,  53,   61,  62,  63,   71,  72,  73,   81,  82,  83,
     11,  12,  13,   21,  22,  23,   31,  32,  33,   41,  42,  43,
  }};
  // clang-format on

  std::array<float, 48> actual;
  FlipRot90Interleaved<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp;
  FlipInterleaved<4, 3>(original.data(), tmp.data());
  Rot90Interleaved<4, 3>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlipRot180Interleaved_3) {
  // clang-format off
  std::array<float, 48> original = {{
     11,  12,  13,   21,  22,  23,   31,  32,  33,   41,  42,  43,
     51,  52,  53,   61,  62,  63,   71,  72,  73,   81,  82,  83,
     91,  92,  93,  101, 102, 103,  111, 112, 113,  121, 122, 123,
    131, 132, 133,  141, 142, 143,  151, 152, 153,  161, 162, 163,
  }};
  std::array<float, 48> expected = {{
    161, 162, 163,  121, 122, 123,   81,  82,  83,   41,  42,  43,
    151, 152, 153,  111, 112, 113,   71,  72,  73,   31,  32,  33,
    141, 142, 143,  101, 102, 103,   61,  62,  63,   21,  22,  23,
    131, 132, 133,   91,  92,  93,   51,  52,  53,   11,  12,  13,
  }};
  // clang-format on

  std::array<float, 48> actual;
  FlipRot180Interleaved<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp;
  FlipInterleaved<4, 3>(original.data(), tmp.data());
  Rot180Interleaved<4, 3>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlipRot270Interleaved_3) {
  // clang-format off
  std::array<float, 48> original = {{
     11,  12,  13,   21,  22,  23,   31,  32,  33,   41,  42,  43,
     51,  52,  53,   61,  62,  63,   71,  72,  73,   81,  82,  83,
     91,  92,  93,  101, 102, 103,  111, 112, 113,  121, 122, 123,
    131, 132, 133,  141, 142, 143,  151, 152, 153,  161, 162, 163,
  }};
  std::array<float, 48> expected = {{
    41,  42,  43,   31,  32,  33,   21,  22,  23,   11,  12,  13,
    81,  82,  83,   71,  72,  73,   61,  62,  63,   51,  52,  53,
   121, 122, 123,  111, 112, 113,  101, 102, 103,   91,  92,  93,
   161, 162, 163,  151, 152, 153,  141, 142, 143,  131, 132, 133,
  }};
  // clang-format on

  std::array<float, 48> actual;
  FlipRot270Interleaved<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp;
  FlipInterleaved<4, 3>(original.data(), tmp.data());
  Rot270Interleaved<4, 3>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestRot90Planar_3) {
  // clang-format off
  const std::array<float, 48> original = {{
     11,  21,  31,  41,
     51,  61,  71,  81,
     91, 101, 111, 121,
    131, 141, 151, 161,

     12,  22,  32,  42,
     52,  62,  72,  82,
     92, 102, 112, 122,
    132, 142, 152, 162,

     13,  23,  33,  43,
     53,  63,  73,  83,
     93, 103, 113, 123,
    133, 143, 153, 163,
  }};
  std::array<float, 48> expected = {{
     41,  81, 121, 161,
     31,  71, 111, 151,
     21,  61, 101, 141,
     11,  51,  91, 131,

     42,  82, 122, 162,
     32,  72, 112, 152,
     22,  62, 102, 142,
     12,  52,  92, 132,

     43,  83, 123, 163,
     33,  73, 113, 153,
     23,  63, 103, 143,
     13,  53,  93, 133,
  }};
  // clang-format on

  std::array<float, 48> actual;
  Rot90Planar<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestRot180Planar_3) {
  // clang-format off
  std::array<float, 48> original = {{
     11,  21,  31,  41,
     51,  61,  71,  81,
     91, 101, 111, 121,
    131, 141, 151, 161,

     12,  22,  32,  42,
     52,  62,  72,  82,
     92, 102, 112, 122,
    132, 142, 152, 162,

     13,  23,  33,  43,
     53,  63,  73,  83,
     93, 103, 113, 123,
    133, 143, 153, 163,
  }};
  std::array<float, 48> expected = {{
    161, 151, 141, 131,
    121, 111, 101,  91,
     81,  71,  61,  51,
     41,  31,  21,  11,

    162, 152, 142, 132,
    122, 112, 102,  92,
     82,  72,  62,  52,
     42,  32,  22,  12,

    163, 153, 143, 133,
    123, 113, 103,  93,
     83,  73,  63,  53,
     43,  33,  23,  13,
  }};
  // clang-format on

  std::array<float, 48> actual;
  Rot180Planar<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp;
  Rot90Planar<4, 3>(original.data(), tmp.data());
  Rot90Planar<4, 3>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestRot270Planar_3) {
  // clang-format off
  std::array<float, 48> original = {{
     11,  21,  31,  41,
     51,  61,  71,  81,
     91, 101, 111, 121,
    131, 141, 151, 161,

     12,  22,  32,  42,
     52,  62,  72,  82,
     92, 102, 112, 122,
    132, 142, 152, 162,

     13,  23,  33,  43,
     53,  63,  73,  83,
     93, 103, 113, 123,
    133, 143, 153, 163,
  }};
  std::array<float, 48> expected = {{
    131,  91,  51,  11,
    141, 101,  61,  21,
    151, 111,  71,  31,
    161, 121,  81,  41,

    132,  92,  52,  12,
    142, 102,  62,  22,
    152, 112,  72,  32,
    162, 122,  82,  42,

    133,  93,  53,  13,
    143, 103,  63,  23,
    153, 113,  73,  33,
    163, 123,  83,  43,
  }};
  // clang-format on

  std::array<float, 48> actual;
  Rot270Planar<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp1, tmp2;
  Rot90Planar<4, 3>(original.data(), tmp1.data());
  Rot90Planar<4, 3>(tmp1.data(), tmp2.data());
  Rot90Planar<4, 3>(tmp2.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlipPlanar_3) {
  // clang-format off
  std::array<float, 48> original = {{
     11,  21,  31,  41,
     51,  61,  71,  81,
     91, 101, 111, 121,
    131, 141, 151, 161,

     12,  22,  32,  42,
     52,  62,  72,  82,
     92, 102, 112, 122,
    132, 142, 152, 162,

     13,  23,  33,  43,
     53,  63,  73,  83,
     93, 103, 113, 123,
    133, 143, 153, 163,
  }};
  std::array<float, 48> expected = {{
     11,  51,  91, 131,
     21,  61, 101, 141,
     31,  71, 111, 151,
     41,  81, 121, 161,

     12,  52,  92, 132,
     22,  62, 102, 142,
     32,  72, 112, 152,
     42,  82, 122, 162,

     13,  53,  93, 133,
     23,  63, 103, 143,
     33,  73, 113, 153,
     43,  83, 123, 163,
  }};
  // clang-format on

  std::array<float, 48> actual;
  FlipPlanar<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlipRot90Planar_3) {
  // clang-format off
  std::array<float, 48> original = {{
     11,  21,  31,  41,
     51,  61,  71,  81,
     91, 101, 111, 121,
    131, 141, 151, 161,

     12,  22,  32,  42,
     52,  62,  72,  82,
     92, 102, 112, 122,
    132, 142, 152, 162,

     13,  23,  33,  43,
     53,  63,  73,  83,
     93, 103, 113, 123,
    133, 143, 153, 163,
  }};
  std::array<float, 48> expected = {{
    131, 141, 151, 161,
     91, 101, 111, 121,
     51,  61,  71,  81,
     11,  21,  31,  41,

    132, 142, 152, 162,
     92, 102, 112, 122,
     52,  62,  72,  82,
     12,  22,  32,  42,

    133, 143, 153, 163,
     93, 103, 113, 123,
     53,  63,  73,  83,
     13,  23,  33,  43,
  }};
  // clang-format on

  std::array<float, 48> actual;
  FlipRot90Planar<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp;
  FlipPlanar<4, 3>(original.data(), tmp.data());
  Rot90Planar<4, 3>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlipRot180Planar_3) {
  // clang-format off
  std::array<float, 48> original = {{
     11,  21,  31,  41,
     51,  61,  71,  81,
     91, 101, 111, 121,
    131, 141, 151, 161,

     12,  22,  32,  42,
     52,  62,  72,  82,
     92, 102, 112, 122,
    132, 142, 152, 162,

     13,  23,  33,  43,
     53,  63,  73,  83,
     93, 103, 113, 123,
    133, 143, 153, 163,
  }};
  std::array<float, 48> expected = {{
    161, 121,  81,  41,
    151, 111,  71,  31,
    141, 101,  61,  21,
    131,  91,  51,  11,

    162, 122,  82,  42,
    152, 112,  72,  32,
    142, 102,  62,  22,
    132,  92,  52,  12,

    163, 123,  83,  43,
    153, 113,  73,  33,
    143, 103,  63,  23,
    133,  93,  53,  13,
  }};
  // clang-format on

  std::array<float, 48> actual;
  FlipRot180Planar<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp;
  FlipPlanar<4, 3>(original.data(), tmp.data());
  Rot180Planar<4, 3>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlipRot270Planar_3) {
  // clang-format off
  std::array<float, 48> original = {{
     11,  21,  31,  41,
     51,  61,  71,  81,
     91, 101, 111, 121,
    131, 141, 151, 161,

     12,  22,  32,  42,
     52,  62,  72,  82,
     92, 102, 112, 122,
    132, 142, 152, 162,

     13,  23,  33,  43,
     53,  63,  73,  83,
     93, 103, 113, 123,
    133, 143, 153, 163,
  }};
  std::array<float, 48> expected = {{
     41,  31,  21,   11,
     81,  71,  61,   51,
    121, 111, 101,   91,
    161, 151, 141,  131,

     42,  32,  22,   12,
     82,  72,  62,   52,
    122, 112, 102,   92,
    162, 152, 142,  132,

     43,  33,  23,   13,
     83,  73,  63,   53,
    123, 113, 103,   93,
    163, 153, 143,  133,
  }};
  // clang-format on

  std::array<float, 48> actual;
  FlipRot270Planar<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp;
  FlipPlanar<4, 3>(original.data(), tmp.data());
  Rot270Planar<4, 3>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, Inverses) {
  // clang-format off
  std::array<float, 48> original = {{
     11,  12,  13,   21,  22,  23,   31,  32,  33,   41,  42,  43,
     51,  52,  53,   61,  62,  63,   71,  72,  73,   81,  82,  83,
     91,  92,  93,  101, 102, 103,  111, 112, 113,  121, 122, 123,
    131, 132, 133,  141, 142, 143,  151, 152, 153,  161, 162, 163,
  }};
  // clang-format on

  for (auto sym : kAllSymmetries) {
    std::array<float, 48> transformed, inverse;
    ApplySymmetry<4, 3>(sym, original.data(), transformed.data());
    ApplySymmetry<4, 3>(Inverse(sym), transformed.data(), inverse.data());

    EXPECT_THAT(inverse, ElementsAreArray(original)) << sym;

    ApplySymmetryPlanar<4, 3>(sym, original.data(), transformed.data());
    ApplySymmetryPlanar<4, 3>(Inverse(sym), transformed.data(), inverse.data());

    EXPECT_THAT(inverse, ElementsAreArray(original)) << sym;
  }
}

// Verify the ApplySymmetry overload for Coord matches that for arrays.
TEST(SymmetryTest, Coord) {
  std::array<int, kN * kN> original;
  for (int i = 0; i < kN * kN; ++i) {
    original[i] = i;
  }

  for (auto sym : kAllSymmetries) {
    EXPECT_EQ(Coord::kPass, ApplySymmetry(sym, Coord::kPass));
    EXPECT_EQ(Coord::kResign, ApplySymmetry(sym, Coord::kResign));
    EXPECT_EQ(Coord::kInvalid, ApplySymmetry(sym, Coord::kInvalid));

    std::array<int, kN * kN> transformed;
    ApplySymmetry<kN, 1>(sym, original.data(), transformed.data());

    for (int i = 0; i < kN * kN; ++i) {
      EXPECT_EQ(original[i], transformed[ApplySymmetry(sym, i)]) << sym;
    }
  }
}

// Build the symmetry concat table and verify it matches the one in the .cc
// file.
TEST(SymmetryTest, ConcatTable) {
  Symmetry table[kNumSymmetries][kNumSymmetries];

  // Table with which we will figure out how to concatenate two symmetries.
  std::array<int, 4> original = {0, 1, 2, 3};
  for (auto a : kAllSymmetries) {
    std::array<int, 4> after_a;
    ApplySymmetry<2, 1>(a, original.data(), after_a.data());

    for (auto b : kAllSymmetries) {
      std::array<int, 4> after_a_b;
      ApplySymmetry<2, 1>(b, after_a.data(), after_a_b.data());

      bool found = false;
      for (auto c : kAllSymmetries) {
        std::array<int, 4> concat;
        ApplySymmetry<2, 1>(c, original.data(), concat.data());
        if (concat == after_a_b) {
          MG_CHECK(!found);
          found = true;
          table[a][b] = c;
        }
      }

      MG_CHECK(found);
    }
  }

  MG_LOG(INFO)
      << "constexpr Symmetry kConcatTable[kNumSymmetries][kNumSymmetries] = {";
  for (int i = 0; i < kNumSymmetries; ++i) {
    std::vector<std::string> row;
    for (int j = 0; j < kNumSymmetries; ++j) {
      std::ostringstream oss;
      oss << table[i][j];
      row.push_back(oss.str());
    }
    MG_LOG(INFO) << "  {" << absl::StrJoin(row, ", ") << "},";
  }
  MG_LOG(INFO) << "};";

  for (auto a : kAllSymmetries) {
    for (auto b : kAllSymmetries) {
      EXPECT_EQ(table[a][b], Concat(a, b));
    }
  }
}

TEST(SymmetryTest, ConcatSymmetryCoord) {
  for (int i = 0; i < kN * kN; ++i) {
    for (auto a : kAllSymmetries) {
      Coord after_a = ApplySymmetry(a, i);

      for (auto b : kAllSymmetries) {
        Coord after_a_b = ApplySymmetry(b, after_a);

        auto c = Concat(a, b);
        Coord after_c = ApplySymmetry(c, i);
        EXPECT_EQ(after_a_b, after_c)
            << "coord:" << Coord(i) << "  a:" << a << "  after_a:" << after_a
            << "  b:" << b << "  after_a_b:" << after_a_b << "  c:" << c
            << "  after_c:" << after_c;
      }
    }
  }
}

TEST(SymmetryTest, ConcatSymmetryArray) {
  std::array<int, kN * kN> original;
  for (int i = 0; i < kN * kN; ++i) {
    original[i] = i;
  }

  for (auto a : kAllSymmetries) {
    std::array<int, kN * kN> after_a;
    ApplySymmetry<kN, 1>(a, original.data(), after_a.data());

    for (auto b : kAllSymmetries) {
      std::array<int, kN * kN> after_a_b;
      ApplySymmetry<kN, 1>(b, after_a.data(), after_a_b.data());

      auto c = Concat(a, b);
      std::array<int, kN * kN> after_c;
      ApplySymmetry<kN, 1>(c, original.data(), after_c.data());
      EXPECT_EQ(after_a_b, after_c);
    }
  }
}

}  // namespace
}  // namespace symmetry
}  // namespace minigo
