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

#include <utility>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"

namespace minigo {

constexpr uint16_t Coord::kPass;
constexpr uint16_t Coord::kInvalid;
constexpr char Coord::kKgsColumns[];

namespace {

std::pair<bool, Coord> TryParseKgs(absl::string_view str) {
  if (str == "pass") {
    return {true, Coord::kPass};
  }

  auto col_char = str[0];
  if (col_char < 'A' || col_char > 'T' || col_char == 'I') {
    return {false, Coord::kPass};
  }
  int col = col_char < 'I' ? col_char - 'A' : 8 + col_char - 'J';

  auto row_str = str.substr(1);
  int row;
  if (!absl::SimpleAtoi(row_str, &row) || row <= 0 || row > kN) {
    return {false, Coord::kPass};
  }
  return {true, {kN - row, col}};
}

std::pair<bool, Coord> TryParseSgf(absl::string_view str) {
  if (str.empty()) {
    return {true, Coord::kPass};
  }
  if (str.size() != 2) {
    return {false, Coord::kPass};
  }

  int col = str[0] - 'a';
  int row = str[1] - 'a';
  if (row < 0 || row >= kN || col < 0 || col >= kN) {
    return {false, Coord::kPass};
  }
  return {true, {row, col}};
}

std::pair<bool, Coord> TryParseString(absl::string_view str) {
  auto result = TryParseKgs(str);
  if (result.first) {
    return result;
  }
  return TryParseSgf(str);
}

}  // namespace

Coord Coord::FromKgs(absl::string_view str) {
  auto result = TryParseKgs(str);
  assert(result.first);
  return result.second;
}

Coord Coord::FromSgf(absl::string_view str) {
  auto result = TryParseSgf(str);
  assert(result.first);
  return result.second;
}

Coord Coord::FromString(absl::string_view str) {
  auto result = TryParseString(str);
  assert(result.first);
  return result.second;
}

std::string Coord::ToKgs() const {
  if (*this == kPass) {
    return "pass";
  } else if (*this == kInvalid) {
    return "invalid";
  }
  int row = value_ / kN;
  int col = value_ % kN;
  absl::string_view col_str = kKgsColumns;
  return absl::StrCat(col_str.substr(col, 1), kN - row);
}

std::ostream& operator<<(std::ostream& os, Coord c) {
  if (c == Coord::kPass) {
    return os << "pass";
  } else if (c == Coord::kInvalid) {
    return os << "invalid";
  } else {
    uint16_t value = c;
    int row = value / kN;
    int col = value % kN;
    return os << Coord::kKgsColumns[col] << (kN - row);
  }
}

}  // namespace minigo
