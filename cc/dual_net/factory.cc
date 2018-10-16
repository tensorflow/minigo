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
#include "cc/dual_net/batching_dual_net.h"
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

// Flags defined in main.cc.
DECLARE_int32(virtual_losses);
DECLARE_int32(parallel_games);

// Inference engine flags.
DEFINE_string(engine, MG_DEFAULT_ENGINE,
              "The inference engine to use. Accepted values:"
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

// Batching flags.
// TODO(tommadams): Calculate batch_size from other arguments (virtual_lossess,
// parallel_games, etc) instead of having an explicit flag.
DEFINE_int32(batch_size, 256, "Inference batch size.");

namespace minigo {

DualNetFactory::~DualNetFactory() = default;

std::unique_ptr<DualNetFactory> NewDualNetFactory(
    const std::string& model_path) {
  std::unique_ptr<DualNet> dual_net;
  int batch_size = FLAGS_batch_size;

  if (FLAGS_engine == "tf") {
#ifdef MG_ENABLE_TF_DUAL_NET
    dual_net = NewTfDualNet(model_path);
#else
    MG_FATAL() << "Binary wasn't compiled with tf inference support";
#endif  // MG_ENABLE_TF_DUAL_NET
  }

  if (FLAGS_engine == "lite") {
#ifdef MG_ENABLE_LITE_DUAL_NET
    dual_net = NewLiteDualNet(model_path);
#else
    MG_FATAL() << "Binary wasn't compiled with lite inference support";
#endif  // MG_ENABLE_LITE_DUAL_NET
  }

  if (FLAGS_engine == "tpu") {
#ifdef MG_ENABLE_TPU_DUAL_NET
    int inferences = FLAGS_virtual_losses * FLAGS_parallel_games;
    int buffering = FLAGS_parallel_games == 1 ? 1 : 2;
    MG_CHECK(inferences % buffering == 0);
    batch_size = inferences / buffering;
    dual_net = absl::make_unique<TpuDualNet>(model_path, FLAGS_tpu_name,
                                             buffering, batch_size);
#else
    MG_FATAL() << "Binary wasn't compiled with tpu inference support";
#endif  // MG_ENABLE_TPU_DUAL_NET
  }

  if (FLAGS_engine == "trt") {
#ifdef MG_ENABLE_TRT_DUAL_NET
    dual_net = NewTrtDualNet(model_path, FLAGS_batch_size);
#else
    MG_FATAL() << "Binary wasn't compiled with trt inference support";
#endif  // MG_ENABLE_TRT_DUAL_NET
  }

  if (dual_net == nullptr) {
    MG_FATAL() << "Unrecognized inference engine \"" << FLAGS_engine << "\"";
  }

  return NewBatchingFactory(std::move(dual_net),
                            static_cast<size_t>(batch_size));
}

}  // namespace minigo
