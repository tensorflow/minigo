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

#ifndef CC_DUAL_NET_FAKE_NET_H_
#define CC_DUAL_NET_FAKE_NET_H_

#include <array>

#include "cc/dual_net/dual_net.h"

namespace minigo {

class FakeNet : public DualNet {
 public:
  FakeNet() : FakeNet(absl::Span<const float>(), 0) {}
  FakeNet(absl::Span<const float> priors, float value);

  void RunMany(absl::Span<const BoardFeatures> features,
               absl::Span<Output> outputs, std::string* model) override;

 private:
  std::array<float, kNumMoves> priors_;
  float value_;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_FAKE_NET_H_
