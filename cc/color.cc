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

#include "cc/color.h"

namespace minigo {

namespace {
const auto kBlackCode = "B";
const auto kWhiteCode = "W";
const auto kEmptyCode = ".";
}  // namespace

std::ostream& operator<<(std::ostream& os, Color color) {
  return os << ColorToCode(color);
}

absl::string_view ColorToCode(Color color) {
  switch (color) {
    case Color::kEmpty:
      return kEmptyCode;
    case Color::kBlack:
      return kBlackCode;
    case Color::kWhite:
      return kWhiteCode;
    default:
      MG_LOG(FATAL) << "<" << static_cast<int>(color) << ">";
      return "BAD";
  }
}

}  // namespace minigo
