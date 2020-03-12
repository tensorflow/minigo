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

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"

namespace minigo {

constexpr uint16_t Coord::kPass;
constexpr uint16_t Coord::kResign;
constexpr uint16_t Coord::kInvalid;
const char Coord::kGtpColumns[20] = "ABCDEFGHJKLMNOPQRST";

namespace {

Coord TryParseGtp(absl::string_view str) {
  auto upper = absl::AsciiStrToUpper(str);
  if (upper == "PASS") {
    return Coord::kPass;
  }
  if (upper == "RESIGN") {
    return Coord::kResign;
  }

  auto col_char = upper[0];
  if (col_char < 'A' || col_char > 'T' || col_char == 'I') {
    return Coord::kInvalid;
  }
  int col = col_char < 'I' ? col_char - 'A' : 8 + col_char - 'J';

  auto row_str = upper.substr(1);
  int row;
  if (!absl::SimpleAtoi(row_str, &row) || row <= 0 || row > kN) {
    return Coord::kInvalid;
  }
  return {kN - row, col};
}

Coord TryParseSgf(absl::string_view str) {
  if (str.empty() || (kN <= 19 && str == "tt")) {
    return Coord::kPass;
  }
  if (str.size() != 2) {
    return Coord::kInvalid;
  }

  int col = str[0] - 'a';
  int row = str[1] - 'a';
  if (row < 0 || row >= kN || col < 0 || col >= kN) {
    return Coord::kInvalid;
  }
  return {row, col};
}

Coord TryParseString(absl::string_view str) {
  auto result = TryParseGtp(str);
  if (result != Coord::kInvalid) {
    return result;
  }
  return TryParseSgf(str);
}

}  // namespace

Coord Coord::FromGtp(absl::string_view str, bool allow_invalid) {
  auto c = TryParseGtp(str);
  MG_CHECK(allow_invalid || c != Coord::kInvalid) << "\"" << str << "\"";
  return c;
}

Coord Coord::FromSgf(absl::string_view str, bool allow_invalid) {
  auto c = TryParseSgf(str);
  MG_CHECK(allow_invalid || c != Coord::kInvalid) << "\"" << str << "\"";
  return c;
}

Coord Coord::FromString(absl::string_view str, bool allow_invalid) {
  auto c = TryParseString(str);
  MG_CHECK(allow_invalid || c != Coord::kInvalid) << "\"" << str << "\"";
  return c;
}

std::string Coord::ToGtp() const {
  if (*this == kPass) {
    return "pass";
  } else if (*this == kResign) {
    return "resign";
  } else if (*this == kInvalid) {
    return "invalid";
  }
  int row = value_ / kN;
  int col = value_ % kN;
  absl::string_view col_str = kGtpColumns;
  return absl::StrCat(col_str.substr(col, 1), kN - row);
}

std::string Coord::ToSgf() const {
  if (*this == kPass) {
    return "";
  } else if (*this == kInvalid) {
    return "invalid";
  }

  // We should not be writing resign moves to SGF files.
  MG_CHECK(*this != kResign);

  int row = value_ / kN;
  int col = value_ % kN;
  char buffer[2];
  buffer[0] = 'a' + col;
  buffer[1] = 'a' + row;
  return {buffer, 2};
}

std::ostream& operator<<(std::ostream& os, Coord c) {
  if (c == Coord::kPass) {
    return os << "pass";
  } else if (c == Coord::kResign) {
    return os << "resign";
  } else if (c == Coord::kInvalid) {
    return os << "invalid";
  } else {
    uint16_t value = c;
    int row = value / kN;
    int col = value % kN;
    return os << Coord::kGtpColumns[col] << (kN - row);
  }
}

}  // namespace minigo
