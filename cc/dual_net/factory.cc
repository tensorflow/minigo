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

#include "cc/dual_net/factory.h"

#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "cc/dual_net/fake_dual_net.h"
#include "cc/dual_net/random_dual_net.h"
#include "gflags/gflags.h"

#ifdef MG_ENABLE_TF_DUAL_NET
#include "cc/dual_net/tf_dual_net.h"
#endif  // MG_ENABLE_TF_DUAL_NET

#ifdef MG_ENABLE_LITE_DUAL_NET
#include "cc/dual_net/lite_dual_net.h"
#endif  // MG_ENABLE_LITE_DUAL_NET

#ifdef MG_ENABLE_TPU_DUAL_NET
#include "cc/dual_net/tpu_dual_net.h"
#endif  // MG_ENABLE_TPU_DUAL_NET

namespace minigo {

std::unique_ptr<ModelFactory> NewModelFactory(absl::string_view engine,
                                              absl::string_view device) {
  if (engine == "fake") {
    return absl::make_unique<FakeDualNetFactory>();
  }
  if (engine == "random") {
    uint64_t seed = 0;
    MG_CHECK(absl::SimpleAtoi(device, &seed)) << "\"" << device << "\"";
    return absl::make_unique<RandomDualNetFactory>(seed);
  }

#ifdef MG_ENABLE_TF_DUAL_NET
  if (engine == "tf") {
    int id = -1;
    if (!device.empty()) {
      MG_CHECK(absl::SimpleAtoi(device, &id)) << "\"" << device << "\"";
    }
    return absl::make_unique<TfDualNetFactory>(id);
  }
#endif  // MG_ENABLE_TF_DUAL_NET

#ifdef MG_ENABLE_LITE_DUAL_NET
  if (engine == "lite") {
    return absl::make_unique<LiteDualNetFactory>();
  }
#endif  // MG_ENABLE_LITE_DUAL_NET

#ifdef MG_ENABLE_TPU_DUAL_NET
  if (engine == "tpu") {
    return absl::make_unique<TpuDualNetFactory>(device);
  }
#endif  // MG_ENABLE_TPU_DUAL_NET

  MG_LOG(FATAL) << "Unrecognized inference engine \"" << engine << "\"";
  return nullptr;
}

}  // namespace minigo
