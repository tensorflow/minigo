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

#include "cc/dual_net/dual_net.h"

namespace minigo {

// Creates one of the DualNetFactory implementations based on the engine flags.
std::unique_ptr<DualNetFactory> NewDualNetFactory();

}  // namespace minigo

#endif  // MINIGO_CC_DUAL_NET_FACTORY_H_
