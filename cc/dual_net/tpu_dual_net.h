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

#ifndef CC_DUAL_NET_TPU_DUAL_NET_H_
#define CC_DUAL_NET_TPU_DUAL_NET_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "cc/constants.h"
#include "cc/model/factory.h"
#include "cc/model/model.h"
#include "cc/model/factory.h"
#include "cc/random.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

namespace minigo {

class TpuDualNetFactory;

class TpuDualNet : public Model {
 public:
  TpuDualNet(const std::string& graph_path,
             const FeatureDescriptor& feature_desc,
             tensorflow::DataType input_type,
             std::shared_ptr<tensorflow::Session> session, int num_replicas,
             TpuDualNetFactory* factory);
  ~TpuDualNet() override;

  void RunMany(const std::vector<const ModelInput*>& inputs,
               std::vector<ModelOutput*>* outputs,
               std::string* model_name) override;

 private:
  void Reserve(size_t capacity);

  std::shared_ptr<tensorflow::Session> session_;
  tensorflow::Session::CallableHandle handle_;
  std::vector<tensorflow::Tensor> inputs_;
  std::vector<tensorflow::Tensor> outputs_;
  size_t batch_capacity_ = 0;
  const int num_replicas_;
  const std::string graph_path_;
  tensorflow::DataType input_type_;
  TpuDualNetFactory* factory_;
};

class TpuDualNetFactory : public ModelFactory {
 public:
  TpuDualNetFactory(std::string tpu_name);
  ~TpuDualNetFactory() override;

  // Close any tensorflow sessions that are no longer used by any TpuDualNet
  // instances. Called by the TpuDualNet destructor.
  void CloseOrphanedSessions();

  std::unique_ptr<Model> NewModel(const ModelDefinition& def) override;

 private:
  struct LoadedModel {
    tensorflow::DataType input_type = tensorflow::DT_INVALID;
    int num_replicas = 0;
    std::shared_ptr<tensorflow::Session> session;
    FeatureDescriptor feature_desc;
  };

  LoadedModel GetModel(const ModelDefinition& def);

  std::unique_ptr<tensorflow::Session> main_session_;
  std::string tpu_name_;

  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, LoadedModel> models_ GUARDED_BY(&mutex_);
};

}  // namespace minigo

#endif  // CC_DUAL_NET_TPU_DUAL_NET_H_
