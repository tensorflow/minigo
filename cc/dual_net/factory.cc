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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "cc/dual_net/fake_dual_net.h"
#include "gflags/gflags.h"

#ifdef MG_ENABLE_TF_DUAL_NET
#include "cc/dual_net/tf_dual_net.h"
#ifndef MG_DEFAULT_ENGINE
#define MG_DEFAULT_ENGINE "tf"
#endif  // MG_DEFAULT_ENGINE
#endif  // MG_ENABLE_TF_DUAL_NET

#ifdef MG_ENABLE_LITE_DUAL_NET
#include "cc/dual_net/lite_dual_net.h"
#ifndef MG_DEFAULT_ENGINE
#define MG_DEFAULT_ENGINE "lite"
#endif  // MG_DEFAULT_ENGINE
#endif  // MG_ENABLE_LITE_DUAL_NET

#ifdef MG_ENABLE_TPU_DUAL_NET
#include "cc/dual_net/tpu_dual_net.h"
#ifndef MG_DEFAULT_ENGINE
#define MG_DEFAULT_ENGINE "tpu"
#endif  // MG_DEFAULT_ENGINE
#endif  // MG_ENABLE_TPU_DUAL_NET

#ifdef MG_ENABLE_TRT_DUAL_NET
#include "cc/dual_net/trt_dual_net.h"
#ifndef MG_DEFAULT_ENGINE
#define MG_DEFAULT_ENGINE "trt"
#endif  // MG_DEFAULT_ENGINE
#endif  // MG_ENABLE_TRT_DUAL_NET

#ifndef MG_DEFAULT_ENGINE
#define MG_DEFAULT_ENGINE "fake"
#endif  // MG_DEFAULT_ENGINE

// Inference engine flags.
DEFINE_string(engine, MG_DEFAULT_ENGINE,
              "The inference engine to use. Accepted values: \"fake\""
#ifdef MG_ENABLE_TF_DUAL_NET
              " \"tf\""
#endif
#ifdef MG_ENABLE_LITE_DUAL_NET
              " \"lite\""
#endif
#ifdef MG_ENABLE_TPU_DUAL_NET
              " \"tpu\""
#endif
#ifdef MG_ENABLE_TRT_DUAL_NET
              " \"trt\""
#endif
);

// TPU flags.
DEFINE_string(
    tpu_name, "",
    "Cloud TPU name to run inference on, e.g. \"grpc://10.240.2.10:8470\"");

namespace minigo {

std::unique_ptr<DualNet> NewDualNet(const std::string& model_path) {
  if (FLAGS_engine == "fake") {
    return absl::make_unique<FakeDualNet>();
  }

  if (FLAGS_engine == "tf") {
#ifdef MG_ENABLE_TF_DUAL_NET
    return NewTfDualNet(model_path);
#else
    MG_FATAL() << "Binary wasn't compiled with tf inference support";
#endif  // MG_ENABLE_TF_DUAL_NET
  }

  if (FLAGS_engine == "lite") {
#ifdef MG_ENABLE_LITE_DUAL_NET
    return NewLiteDualNet(model_path);
#else
    MG_FATAL() << "Binary wasn't compiled with lite inference support";
#endif  // MG_ENABLE_LITE_DUAL_NET
  }

  if (FLAGS_engine == "tpu") {
#ifdef MG_ENABLE_TPU_DUAL_NET
    return absl::make_unique<TpuDualNet>(model_path, FLAGS_tpu_name);
#else
    MG_FATAL() << "Binary wasn't compiled with tpu inference support";
#endif  // MG_ENABLE_TPU_DUAL_NET
  }

  if (FLAGS_engine == "trt") {
#ifdef MG_ENABLE_TRT_DUAL_NET
    return NewTrtDualNet(model_path);
#else
    MG_FATAL() << "Binary wasn't compiled with trt inference support";
#endif  // MG_ENABLE_TRT_DUAL_NET
  }

  MG_FATAL() << "Unrecognized inference engine \"" << FLAGS_engine << "\"";
  return nullptr;
}

}  // namespace minigo
