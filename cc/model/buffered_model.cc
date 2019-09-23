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

#include "cc/model/buffered_model.h"

namespace minigo {

BufferedModel::BufferedModel(std::string name,
                             std::vector<std::unique_ptr<Model>> impls)
    : Model(std::move(name), impls[0]->feature_type(),
            static_cast<int>(impls.size())) {
  for (auto& x : impls) {
    // Make sure all impls use the same input features.
    MG_CHECK(x->feature_type() == feature_type());
    impls_.Push(std::move(x));
  }
}

void BufferedModel::RunMany(const std::vector<const Input*>& inputs,
                            std::vector<Output*>* outputs,
                            std::string* model_name) {
  auto impl = impls_.Pop();
  impl->RunMany(inputs, outputs, model_name);
  impls_.Push(std::move(impl));
}

}  // namespace minigo
