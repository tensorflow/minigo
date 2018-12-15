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

#include "cc/file/utils.h"

#include <cstdlib>
#include <string>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/file/path.h"
#include "cc/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace minigo {
namespace file {
namespace {

std::string FullPath(const char* basename) {
  const char* tmpdir = std::getenv("TEST_TMPDIR");
  MG_CHECK(tmpdir != nullptr) << "TEST_TMPDIR environment variable not found";
  return JoinPath(tmpdir, basename);
}

TEST(UtilsTest, ReadAndWriteFile) {
  // Recursively create a directory using both forward and back slashes.
  auto dir = FullPath("foo/bar\\read_write");
  ASSERT_TRUE(RecursivelyCreateDir(dir));

  // Attempting to create an already existing directory shouldn't fail.
  ASSERT_TRUE(RecursivelyCreateDir(dir));

  // Write to a file under the new directory.
  auto path = JoinPath(dir, "test.txt");
  std::string expected_contents = "this is a test";
  ASSERT_TRUE(WriteFile(path, expected_contents));

  // Read the file back.
  std::string actual_contents;
  ASSERT_TRUE(ReadFile(path, &actual_contents));
  ASSERT_EQ(expected_contents, actual_contents);
}

TEST(UtilsTest, GetModTime) {
  // Recursively create a directory using both forward and back slashes.
  auto dir = FullPath("foo/bar\\mod_date");
  ASSERT_TRUE(RecursivelyCreateDir(dir));

  // Write a file.
  auto path = JoinPath(dir, "a");
  ASSERT_TRUE(WriteFile(path, ""));

  // Get the modification time.
  uint64_t actual_time;
  ASSERT_TRUE(GetModTime(path, &actual_time));

  // The modification time should be almost the same as the current time.
  // We allow for a fairly large deviation (up to 1 minute) to minimize the
  // chances of test flakiness.
  auto delta = absl::Now() - absl::FromUnixMicros(actual_time);
  ASSERT_LT(delta, absl::Minutes(1));
}

TEST(UtilsTest, ListDir) {
  // Recursively create a directory using both forward and back slashes.
  auto dir = FullPath("foo/bar\\list_dir");
  ASSERT_TRUE(RecursivelyCreateDir(dir));

  // Write a few files.
  ASSERT_TRUE(WriteFile(JoinPath(dir, "a"), ""));
  ASSERT_TRUE(WriteFile(JoinPath(dir, "b"), ""));
  ASSERT_TRUE(WriteFile(JoinPath(dir, "c"), ""));

  // List the directory.
  std::vector<std::string> files;
  ASSERT_TRUE(ListDir(dir, &files));

  // The order of the returned list is undefined.
  EXPECT_THAT(files, ::testing::UnorderedElementsAreArray({"a", "b", "c"}));
}

}  // namespace
}  // namespace file
}  // namespace minigo
