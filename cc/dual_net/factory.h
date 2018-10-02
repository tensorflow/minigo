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

#ifndef MINIGO_CC_DUAL_NET_FACTORY_H_
#define MINIGO_CC_DUAL_NET_FACTORY_H_

#include <memory>
#include <string>
#include <utility>

#include "cc/dual_net/dual_net.h"

namespace minigo {

class DualNetFactory {
 public:
  explicit DualNetFactory(std::string model_path)
      : model_path_(std::move(model_path)) {}
  virtual ~DualNetFactory();
  virtual std::unique_ptr<DualNet> New() = 0;

  const std::string& model() const { return model_path_; }

 private:
  const std::string model_path_;
};

std::unique_ptr<DualNetFactory> NewDualNetFactory(std::string model_path,
                                                  int parallel_games);

}  // namespace minigo

#endif  // MINIGO_CC_DUAL_NET_FACTORY_H_
