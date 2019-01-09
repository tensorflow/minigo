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

#include "cc/coord.h"

#include "gtest/gtest.h"

namespace minigo {
namespace {

TEST(CoordTest, TestFromGtp) {
  EXPECT_EQ(Coord::kPass, Coord::FromGtp("pass"));

  if (kN == 9) {
    EXPECT_EQ(Coord(0, 0), Coord::FromGtp("A9"));
    EXPECT_EQ(Coord(0, 7), Coord::FromGtp("H9"));
    EXPECT_EQ(Coord(0, 8), Coord::FromGtp("J9"));
    EXPECT_EQ(Coord(8, 0), Coord::FromGtp("A1"));
    EXPECT_EQ(Coord(8, 7), Coord::FromGtp("H1"));
    EXPECT_EQ(Coord(8, 8), Coord::FromGtp("J1"));
  } else {
    EXPECT_EQ(Coord(0, 0), Coord::FromGtp("A19"));
    EXPECT_EQ(Coord(0, 7), Coord::FromGtp("H19"));
    EXPECT_EQ(Coord(0, 8), Coord::FromGtp("J19"));
    EXPECT_EQ(Coord(18, 0), Coord::FromGtp("A1"));
    EXPECT_EQ(Coord(18, 7), Coord::FromGtp("H1"));
    EXPECT_EQ(Coord(18, 8), Coord::FromGtp("J1"));
    EXPECT_EQ(Coord(18, 18), Coord::FromGtp("T1"));
  }
}

TEST(CoordTest, GtpRoundTrip) {
  EXPECT_EQ(Coord::kPass, Coord::FromGtp(Coord(Coord::kPass).ToGtp()));
  for (int row = 0; row < kN; ++row) {
    for (int col = 0; col < kN; ++col) {
      Coord c(row, col);
      EXPECT_EQ(c, Coord::FromGtp(c.ToGtp()));
    }
  }
}

TEST(CoordTest, SgfRoundTrip) {
  EXPECT_EQ(Coord::kPass, Coord::FromSgf(Coord(Coord::kPass).ToSgf()));
  for (int row = 0; row < kN; ++row) {
    for (int col = 0; col < kN; ++col) {
      Coord c(row, col);
      EXPECT_EQ(c, Coord::FromSgf(c.ToSgf()));
    }
  }
}

}  // namespace
}  // namespace minigo
