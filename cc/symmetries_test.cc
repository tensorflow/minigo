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
  Rot90<4, 1>(original.data(), actual.data());
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
  Rot180<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 16> tmp;
  Rot90<4, 1>(original.data(), tmp.data());
  Rot90<4, 1>(tmp.data(), actual.data());
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
  Rot270<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 16> tmp1, tmp2;
  Rot90<4, 1>(original.data(), tmp1.data());
  Rot90<4, 1>(tmp1.data(), tmp2.data());
  Rot90<4, 1>(tmp2.data(), actual.data());
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
  Flip<4, 1>(original.data(), actual.data());
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
  FlipRot90<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 16> tmp;
  Flip<4, 1>(original.data(), tmp.data());
  Rot90<4, 1>(tmp.data(), actual.data());
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
  FlipRot180<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 16> tmp;
  Flip<4, 1>(original.data(), tmp.data());
  Rot180<4, 1>(tmp.data(), actual.data());
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
  FlipRot270<4, 1>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 16> tmp;
  Flip<4, 1>(original.data(), tmp.data());
  Rot270<4, 1>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestRot90_3) {
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
  Rot90<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestRot180_3) {
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
  Rot180<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp;
  Rot90<4, 3>(original.data(), tmp.data());
  Rot90<4, 3>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestRot270_3) {
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
  Rot270<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp1, tmp2;
  Rot90<4, 3>(original.data(), tmp1.data());
  Rot90<4, 3>(tmp1.data(), tmp2.data());
  Rot90<4, 3>(tmp2.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlip_3) {
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
  Flip<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlipRot90_3) {
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
  FlipRot90<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp;
  Flip<4, 3>(original.data(), tmp.data());
  Rot90<4, 3>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlipRot180_3) {
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
  FlipRot180<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp;
  Flip<4, 3>(original.data(), tmp.data());
  Rot180<4, 3>(tmp.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));
}

TEST(SymmetryTest, TestFlipRot270_3) {
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
  FlipRot270<4, 3>(original.data(), actual.data());
  EXPECT_THAT(actual, ElementsAreArray(expected));

  std::array<float, 48> tmp;
  Flip<4, 3>(original.data(), tmp.data());
  Rot270<4, 3>(tmp.data(), actual.data());
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

    EXPECT_THAT(inverse, ElementsAreArray(original));
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
