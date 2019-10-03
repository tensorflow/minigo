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

#ifndef CC_DUAL_NET_FAKE_DUAL_NET_H_
#define CC_DUAL_NET_FAKE_DUAL_NET_H_

#include <array>

#include "absl/types/span.h"
#include "cc/model/model.h"

namespace minigo {

class FakeDualNet : public Model {
 public:
  FakeDualNet() : FakeDualNet(absl::Span<const float>(), 0) {}
  FakeDualNet(absl::Span<const float> priors, float value);

  void RunMany(const std::vector<const ModelInput*>& inputs,
               std::vector<ModelOutput*>* outputs,
               std::string* model_name) override;

 private:
  std::array<float, kNumMoves> priors_;
  float value_;
};

class FakeDualNetFactory : public ModelFactory {
 public:
  std::unique_ptr<Model> NewModel(const std::string& descriptor) override;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_FAKE_DUAL_NET_H_
