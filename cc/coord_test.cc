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

TEST(CoordTest, TestFromKgs) {
  EXPECT_EQ(Coord::kPass, Coord::FromKgs("pass"));

  if (kN == 9) {
    EXPECT_EQ(Coord(0, 0), Coord::FromKgs("A9"));
    EXPECT_EQ(Coord(0, 7), Coord::FromKgs("H9"));
    EXPECT_EQ(Coord(0, 8), Coord::FromKgs("J9"));
    EXPECT_EQ(Coord(8, 0), Coord::FromKgs("A1"));
    EXPECT_EQ(Coord(8, 7), Coord::FromKgs("H1"));
    EXPECT_EQ(Coord(8, 8), Coord::FromKgs("J1"));
  } else {
    EXPECT_EQ(Coord(0, 0), Coord::FromKgs("A19"));
    EXPECT_EQ(Coord(0, 7), Coord::FromKgs("H19"));
    EXPECT_EQ(Coord(0, 8), Coord::FromKgs("J19"));
    EXPECT_EQ(Coord(18, 0), Coord::FromKgs("A1"));
    EXPECT_EQ(Coord(18, 7), Coord::FromKgs("H1"));
    EXPECT_EQ(Coord(18, 8), Coord::FromKgs("J1"));
    EXPECT_EQ(Coord(18, 18), Coord::FromKgs("T1"));
  }
}

}  // namespace
}  // namespace minigo
