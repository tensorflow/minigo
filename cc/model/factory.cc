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

#include "cc/model/factory.h"

namespace minigo {

namespace {

// Visitor class used to log a ModelProperty's value using absl::visit.
class LogModelProperty {
 public:
  explicit LogModelProperty(std::ostream* os) : os_(os) {}

  template <typename T>
  void operator()(const T& value) const {
    (*os_) << value;
  }

 private:
  std::ostream* os_;
};

}  // namespace

std::ostream& operator<<(std::ostream& os, const ModelProperty& p) {
  absl::visit(LogModelProperty(&os), p);
  return os;
}

ModelFactory::~ModelFactory() = default;

}  // namespace minigo
