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
#include "absl/strings/str_split.h"
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

std::ostream& operator<<(std::ostream& os, const ModelDescriptor& desc) {
  return os << desc.engine << "," << desc.model;
}

ModelDescriptor ParseModelDescriptor(absl::string_view descriptor) {
  std::vector<std::string> parts =
      absl::StrSplit(descriptor, absl::MaxSplits(',', 1));
  MG_CHECK(parts.size() == 1 || parts.size() == 2);
  ModelDescriptor result;
  result.engine = std::move(parts[0]);
  if (parts.size() == 2) {
    result.model = std::move(parts[1]);
  }
  return result;
}

std::unique_ptr<ModelFactory> NewModelFactory(absl::string_view engine_desc) {
  MG_CHECK(!engine_desc.empty());

  std::vector<std::string> parts =
      absl::StrSplit(engine_desc, absl::MaxSplits(':', 1));
  auto engine = parts[0];
  std::string arg_str;
  if (parts.size() == 2) {
    arg_str = std::move(parts[1]);
  }

  if (engine == "fake") {
    return absl::make_unique<FakeDualNetFactory>();
  }
  if (engine == "random") {
    MG_CHECK(!arg_str.empty());
    uint64_t seed = 0;
    MG_CHECK(absl::SimpleAtoi(arg_str, &seed));
    return absl::make_unique<RandomDualNetFactory>(13 * seed);
  }

#ifdef MG_ENABLE_TF_DUAL_NET
  if (engine == "tf") {
    MG_CHECK(arg_str.empty());
    return absl::make_unique<TfDualNetFactory>();
  }
#endif  // MG_ENABLE_TF_DUAL_NET

#ifdef MG_ENABLE_LITE_DUAL_NET
  if (engine == "lite") {
    MG_CHECK(arg_str.empty());
    return absl::make_unique<LiteDualNetFactory>();
  }
#endif  // MG_ENABLE_LITE_DUAL_NET

#ifdef MG_ENABLE_TPU_DUAL_NET
  if (engine == "tpu") {
    std::vector<std::string> args =
        absl::StrSplit(arg_str, absl::MaxSplits(':', 1));
    MG_CHECK(args.size() == 2) << "\"" << arg_str << "\"";
    int buffer_count = 0;
    MG_CHECK(absl::SimpleAtoi(args[0], &buffer_count)) << args[0];
    return absl::make_unique<TpuDualNetFactory>(buffer_count, args[1]);
  }
#endif  // MG_ENABLE_TPU_DUAL_NET

  MG_LOG(FATAL) << "Unrecognized inference engine \"" << engine << "\"";
  return nullptr;
}

}  // namespace minigo
