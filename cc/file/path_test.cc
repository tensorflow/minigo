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

#include "cc/file/path.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace minigo {
namespace file {
namespace {

TEST(PathTest, NormalizeSlashes) {
  auto actual = NormalizeSlashes("/foo/bar\\baz\\");

  std::string expected;
  if (kSepChar == '/') {
    expected = "/foo/bar/baz/";
  } else {
    expected = "\\foo\\bar\\baz\\";
  }

  EXPECT_EQ(expected, actual);
}

TEST(PathTest, JoinPath) {
  std::string expected;
  if (kSepChar == '/') {
    expected = "foo/bar/baz";
  } else {
    expected = "foo\\bar\\baz";
  }

  auto actual = JoinPath("foo", "bar", "baz");
  EXPECT_EQ(expected, actual);
}

TEST(PathTest, SplitPath) {
  using Pair = std::pair<absl::string_view, absl::string_view>;
  auto path = JoinPath("a", "b", "c.d");
  EXPECT_EQ(Pair(JoinPath("a", "b"), "c.d"), SplitPath(path));
}

TEST(PathTest, Dirname) {
  EXPECT_EQ(JoinPath("a", "b"), Dirname(JoinPath("a", "b", "c.d")));
}

TEST(PathTest, Basename) {
  EXPECT_EQ("c.d", Basename(JoinPath("a", "b", "c.d")));
}

TEST(PathTest, Stem) { EXPECT_EQ("c", Stem(JoinPath("a", "b", "c.d"))); }

}  // namespace
}  // namespace file
}  // namespace minigo
