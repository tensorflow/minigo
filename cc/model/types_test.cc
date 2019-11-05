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

#include "cc/model/types.h"

#include "gtest/gtest.h"

namespace minigo {
namespace {

TEST(TypesTest, Basic) {
  TensorShape shape({2, 3, 4});
  EXPECT_FALSE(shape.empty());
  EXPECT_EQ(3, shape.size());
  EXPECT_EQ(24, shape.num_elements());
  EXPECT_EQ(2, shape[0]);
  EXPECT_EQ(3, shape[1]);
  EXPECT_EQ(4, shape[2]);

  EXPECT_TRUE(TensorShape().empty());
  EXPECT_TRUE(TensorShape({}).empty());

  EXPECT_EQ(TensorShape({}), TensorShape());
  EXPECT_EQ(TensorShape({1, 2}), TensorShape({1, 2}));
  EXPECT_EQ(TensorShape({1, 2, 3, 4}), TensorShape({1, 2, 3, 4}));
  EXPECT_NE(TensorShape({1, 2, 3, 4}), TensorShape({}));
  EXPECT_NE(TensorShape({1, 2, 3, 4}), TensorShape({4, 5, 6, 7}));
  EXPECT_NE(TensorShape({1, 2, 3, 4}), TensorShape({1, 2, 3, -1}));
  EXPECT_NE(TensorShape({1, 2, 3, 4}), TensorShape({-1, -1, -1, -1}));
}

TEST(TypesTest, ShapeIs) {
  TensorShape shape({1, 2, 3, 4});

  EXPECT_FALSE(shape.is({}));
  EXPECT_FALSE(shape.is({1}));
  EXPECT_FALSE(shape.is({1, 2}));
  EXPECT_FALSE(shape.is({1, 2, 3}));
  EXPECT_FALSE(shape.is({-1}));
  EXPECT_FALSE(shape.is({-1, -1}));
  EXPECT_FALSE(shape.is({-1, -1, -1}));

  EXPECT_TRUE(shape.is({1, 2, 3, 4}));
  EXPECT_TRUE(shape.is({-1, 2, 3, 4}));
  EXPECT_TRUE(shape.is({-1, -1, 3, 4}));
  EXPECT_TRUE(shape.is({-1, -1, 3, -1}));
  EXPECT_TRUE(shape.is({-1, -1, -1, -1}));
  EXPECT_TRUE(shape.is({1, -1, 3, 4}));
  EXPECT_TRUE(shape.is({1, 2, -1, 4}));
  EXPECT_TRUE(shape.is({1, 2, 3, -1}));
}

}  // namespace
}  // namespace minigo
