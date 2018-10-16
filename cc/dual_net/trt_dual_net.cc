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

#include "cc/dual_net/trt_dual_net.h"

#include <bitset>
#include <fstream>
#include <thread>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "cc/check.h"
#include "cc/constants.h"
#include "cc/thread_safe_queue.h"
#include "cuda/include/cuda_runtime_api.h"
#include "tensorrt/include/NvInfer.h"
#include "tensorrt/include/NvUffParser.h"

namespace minigo {

namespace {

bool DeviceHasNativeReducedPrecision(int device) {
  cudaDeviceProp props;
  MG_CHECK(cudaGetDeviceProperties(&props, device) == cudaSuccess);
  if (props.major > 6) {
    return true;
  }
  if (props.major == 6) {
    return props.minor != 1;
  }
  if (props.major == 5) {
    return props.minor >= 3;
  }
  return false;
}

class TrtDualNet : public DualNet {
  class TrtLogger : public nvinfer1::ILogger {
   public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
      switch (severity) {
        case Severity::kINTERNAL_ERROR:
          std::cerr << "TensorRT internal error: " << msg << std::endl;
          break;
        case Severity::kERROR:
          std::cerr << "TensorRT error: " << msg << std::endl;
          break;
        case Severity::kWARNING:
          std::cerr << "TensorRT warning: " << msg << std::endl;
          break;
        default:
          break;
      }
    }
  };

  class TrtWorker {
   public:
    TrtWorker(nvinfer1::ICudaEngine* engine, size_t max_batch_size)
        : max_batch_size_(max_batch_size) {
      context_ = engine->createExecutionContext();
      MG_CHECK(context_);

      void* host_ptr;
      size_t input_size = max_batch_size * kN * kN * DualNet::kNumStoneFeatures;
      MG_CHECK(cudaHostAlloc(&host_ptr, input_size * sizeof(float),
                             cudaHostAllocWriteCombined) == cudaSuccess);
      pos_tensor_ = static_cast<float*>(host_ptr);
      size_t output_size = max_batch_size * (kNumMoves + 1);
      MG_CHECK(cudaHostAlloc(&host_ptr, output_size * sizeof(float),
                             cudaHostAllocDefault) == cudaSuccess);
      value_output_ = static_cast<float*>(host_ptr);
      policy_output_ = value_output_ + max_batch_size;
    }

    ~TrtWorker() {
      cudaFreeHost(value_output_);
      cudaFreeHost(pos_tensor_);

      context_->destroy();
    }

    void RunMany(std::vector<const BoardFeatures*> features,
                 std::vector<Output*> outputs) {
      auto batch_size = features.size();
      MG_CHECK(batch_size <= max_batch_size_);

      auto* feature_data = pos_tensor_;
      // Copy the features into the input tensor.
      for (const auto* feature : features) {
        feature_data =
            std::copy_n(feature->data(), kNumBoardFeatures, feature_data);
      }

      // Run the model.
      void* buffers[] = {pos_tensor_, policy_output_, value_output_};
      context_->execute(batch_size_, buffers);

      // Copy the policy and value out of the output tensors.
      for (size_t i = 0; i < batch_size; ++i) {
        memcpy(outputs[i]->policy.data(), policy_output_ + i * kNumMoves,
               sizeof(outputs[i]->policy));
        outputs[i]->value = value_output_[i];
      }
    }

   private:
    nvinfer1::IExecutionContext* context_;

    float* pos_tensor_;
    float* policy_output_;
    float* value_output_;
    const size_t max_batch_size_;
  };

  struct InferenceData {
    std::vector<const DualNet::BoardFeatures*> features;
    std::vector<DualNet::Output*> outputs;
    absl::Notification* notification;
  };

 public:
  TrtDualNet(std::string graph_path, size_t max_batch_size)
      : graph_path_(graph_path),
        running_(true),
        max_batch_size_(max_batch_size) {
    runtime_ = nvinfer1::createInferRuntime(logger_);
    MG_CHECK(runtime_);

    auto* parser = nvuffparser::createUffParser();

    // Note: TensorRT ignores the input order argument and always assumes NCHW.
    parser->registerInput("pos_tensor",
                          nvinfer1::DimsCHW(DualNet::kNumStoneFeatures, kN, kN),
                          nvuffparser::UffInputOrder::kNCHW);

    parser->registerOutput("policy_output");
    parser->registerOutput("value_output");

    cudaSetDevice(0);
    auto* builder = nvinfer1::createInferBuilder(logger_);
    auto* network = builder->createNetwork();

    if (!std::ifstream(graph_path).good()) {
      graph_path = absl::StrCat(graph_path, ".uff");
    }

    MG_CHECK(
        parser->parse(graph_path.c_str(), *network, nvinfer1::DataType::kFLOAT))
        << ". File path: '" << graph_path << "'";

    builder->setMaxBatchSize(max_batch_size_);
    builder->setMaxWorkspaceSize(1ull << 30);  // One gigabyte.

    int device_count = 0;
    MG_CHECK(cudaGetDeviceCount(&device_count) == cudaSuccess);

    bool enable_fp16_mode = true;
    for (int device_id = 0; device_id < device_count; ++device_id) {
      enable_fp16_mode &= DeviceHasNativeReducedPrecision(device_id);
    }
    // If all GPUs support fast fp16 math, enable it.
    builder->setFp16Mode(enable_fp16_mode);

    auto* engine = [&] {
      // Building TensorRT engines is not thread-safe.
      static absl::Mutex mutex;
      absl::MutexLock lock(&mutex);
      return builder->buildCudaEngine(*network);
    }();
    MG_CHECK(engine);

    network->destroy();
    builder->destroy();
    parser->destroy();

    using Pair = std::pair<int, nvinfer1::ICudaEngine*>;
    std::vector<Pair> pairs = {Pair(0, engine)};

    auto* blob = engine->serialize();
    MG_CHECK(blob);

    for (int device_id = 1; device_id < device_count; ++device_id) {
      cudaSetDevice(device_id);
      // TODO(csigg): it would be faster to deserialize engines in parallel.
      pairs.push_back(
          Pair(device_id, runtime_->deserializeCudaEngine(
                              blob->data(), blob->size(), nullptr)));
    }

    auto functor = [this](const Pair& pair) {
      pthread_setname_np(pthread_self(), "TrtWorker");
      cudaSetDevice(pair.first);
      TrtWorker worker(pair.second, max_batch_size_);
      while (running_) {
        InferenceData inference;
        if (inference_queue_.PopWithTimeout(&inference, absl::Seconds(1))) {
          worker.RunMany(std::move(inference.features),
                         std::move(inference.outputs));
          inference.notification->Notify();
        }
      }
    };

    for (auto& pair : pairs) {
      MG_CHECK(pair.second) << "Failed to deserialize TensorRT engine.";
      engines_.push_back(pair.second);
      worker_threads_.emplace_back(functor, pair);
      worker_threads_.emplace_back(functor, pair);
    }
    blob->destroy();
  }

  ~TrtDualNet() override {
    running_ = false;
    for (auto& thread : worker_threads_) {
      thread.join();
    }
    for (auto* engine : engines_) {
      engine->destroy();
    }
    runtime_->destroy();
  }

  void RunMany(std::vector<const BoardFeatures*> features,
               std::vector<Output*> outputs, std::string* model) {
    MG_DCHECK(features.size() == outputs.size());

    absl::Notification notification;
    inference_queue_.Push(
        {std::move(features), std::move(outputs), &notification});
    notification.WaitForNotification();

    if (model != nullptr) {
      *model = graph_path_;
    }
  }

  InputLayout GetInputLayout() const override {
    // TensorRT requires the input to be in NCHW layout.
    return InputLayout::kNCHW;
  }

 private:
  std::string graph_path_;

  TrtLogger logger_;
  nvinfer1::IRuntime* runtime_;
  std::vector<nvinfer1::ICudaEngine*> engines_;

  ThreadSafeQueue<InferenceData> inference_queue_;
  std::vector<std::thread> worker_threads_;
  std::atomic<bool> running_;
  const size_t max_batch_size_;
};

}  // namespace

std::unique_ptr<DualNet> NewTrtDualNet(const std::string& model_path,
                                       size_t max_batch_size) {
  return absl::make_unique<TrtDualNet>(model_path, max_batch_size);
}

}  // namespace minigo
