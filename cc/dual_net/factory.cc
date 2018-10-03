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
#include "gflags/gflags.h"

#ifdef MG_ENABLE_REMOTE_DUAL_NET
#include <thread>
#include "cc/dual_net/inference_server.h"
#ifndef MG_DEFAULT_ENGINE
#define MG_DEFAULT_ENGINE "remote"
#endif  // MG_DEFAULT_ENGINE
#endif  // MG_ENABLE_REMOTE_DUAL_NET

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

DEFINE_string(engine, MG_DEFAULT_ENGINE,
              "The inference engine to use. Accepted values:"
#ifdef MG_ENABLE_REMOTE_DUAL_NET
              " \"remote\""
#endif
#ifdef MG_ENABLE_TF_DUAL_NET
              " \"tf\""
#endif
#ifdef MG_ENABLE_LITE_DUAL_NET
              " \"lite\""
#endif
);

DEFINE_string(checkpoint_dir, "",
              "Path to a directory containing TensorFlow model checkpoints. "
              "The inference worker will monitor this when a new checkpoint "
              "is found, load the model and use it for futher inferences. "
              "Only valid when remote inference is enabled.");
DEFINE_bool(use_tpu, true,
            "If true, the remote inference will be run on a TPU. Ignored when "
            "remote_inference=false.");
DEFINE_string(tpu_name, "", "Cloud TPU name, e.g. grpc://10.240.2.2:8470.");
DEFINE_int32(conv_width, 256, "Width of the model's convolution filters.");
DEFINE_int32(fc_width, 256,
             "Width of the model's fully connected layer in value head.");
DEFINE_int32(parallel_tpus, 8,
             "If model=remote, the number of TPU cores to run on in parallel.");
DEFINE_int32(port, 50051, "The port opened by the InferenceService server.");
DECLARE_int32(virtual_losses);

namespace minigo {

namespace {

#ifdef MG_ENABLE_REMOTE_DUAL_NET
class RemoteDualNetFactory : public DualNetFactory {
 public:
  explicit RemoteDualNetFactory(std::string model_path, int parallel_games)
      : DualNetFactory(std::move(model_path)) {
    inference_worker_thread_ = std::thread([this]() {
      std::vector<std::string> cmd_parts = {
          absl::StrCat("BOARD_SIZE=", kN),
          "python",
          "inference_worker.py",
          absl::StrCat("--model=", model()),
          absl::StrCat("--checkpoint_dir=", FLAGS_checkpoint_dir),
          absl::StrCat("--use_tpu=", FLAGS_use_tpu),
          absl::StrCat("--tpu_name=", FLAGS_tpu_name),
          absl::StrCat("--conv_width=", FLAGS_conv_width),
          absl::StrCat("--fc_width=", FLAGS_fc_width),
          absl::StrCat("--parallel_tpus=", FLAGS_parallel_tpus),
      };
      auto cmd = absl::StrJoin(cmd_parts, " ");
      FILE* f = popen(cmd.c_str(), "r");
      for (;;) {
        int c = fgetc(f);
        if (c == EOF) {
          break;
        }
        fputc(c, stderr);
      }
      fputc('\n', stderr);
    });

    int games_per_inference = std::max(1, parallel_games / 2);
    server_ = absl::make_unique<InferenceServer>(
        FLAGS_virtual_losses, games_per_inference, FLAGS_port);
  }

  ~RemoteDualNetFactory() override {
    server_.reset(nullptr);
    inference_worker_thread_.join();
  }

  std::unique_ptr<DualNet> New() override { return server_->NewDualNet(); }

 private:
  std::thread inference_worker_thread_;
  std::unique_ptr<InferenceServer> server_;
};
#endif  // MG_ENABLE_REMOTE_DUAL_NET

#ifdef MG_ENABLE_TF_DUAL_NET
class TfDualNetFactory : public DualNetFactory {
 public:
  TfDualNetFactory(std::string model_path)
      : DualNetFactory(std::move(model_path)) {}

  std::unique_ptr<DualNet> New() override {
    return absl::make_unique<TfDualNet>(model());
  }
};
#endif  // MG_ENABLE_TF_DUAL_NET

#ifdef MG_ENABLE_LITE_DUAL_NET
class LiteDualNetFactory : public DualNetFactory {
 public:
  LiteDualNetFactory(std::string model_path)
      : DualNetFactory(std::move(model_path)) {}

  std::unique_ptr<DualNet> New() override {
    return absl::make_unique<LiteDualNet>(model());
  }
};
#endif  // MG_ENABLE_LITE_DUAL_NET

}  // namespace

DualNetFactory::~DualNetFactory() = default;

std::unique_ptr<DualNetFactory> NewDualNetFactory(std::string model_path,
                                                  int parallel_games) {
  if (FLAGS_engine == "remote") {
#ifdef MG_ENABLE_REMOTE_DUAL_NET
    return absl::make_unique<RemoteDualNetFactory>(std::move(model_path),
                                                   parallel_games);
#else
    MG_FATAL() << "Binary wasn't compiled with remote inference support";
#endif  // MG_ENABLE_REMOTE_DUAL_NET
  }

  if (FLAGS_engine == "tf") {
#ifdef MG_ENABLE_TF_DUAL_NET
    return absl::make_unique<TfDualNetFactory>(std::move(model_path));
#else
    MG_FATAL() << "Binary wasn't compiled with tf inference support";
#endif  // MG_ENABLE_TF_DUAL_NET
  }

  if (FLAGS_engine == "lite") {
#ifdef MG_ENABLE_LITE_DUAL_NET
    return absl::make_unique<LiteDualNetFactory>(std::move(model_path));
#else
    MG_FATAL() << "Binary wasn't compiled with lite inference support";
#endif  // MG_ENABLE_LITE_DUAL_NET
  }

  MG_FATAL() << "Unrecognized inference engine \"" << FLAGS_engine << "\"";
  return nullptr;
}

}  // namespace minigo
