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

#ifndef CC_MODEL_BUFFERED_MODEL_H_
#define CC_MODEL_BUFFERED_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "cc/model/model.h"
#include "cc/thread_safe_queue.h"

namespace minigo {

class BufferedModel : public Model {
 public:
  BufferedModel(std::string name, std::vector<std::unique_ptr<Model>> impls);

  const std::string& name() const { return name_; }

  void RunMany(const std::vector<const Input*>& inputs,
               std::vector<Output*>* outputs, std::string* model_name) override;

 private:
  ThreadSafeQueue<std::unique_ptr<Model>> impls_;
  const std::string name_;
};

}  // namespace minigo

#endif  //  CC_MODEL_BUFFERED_MODEL_H_
