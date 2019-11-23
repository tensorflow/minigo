// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at //
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CC_MODEL_MODEL_LOADER_H_
#define CC_MODEL_MODEL_LOADER_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "cc/model/factory.h"
#include "cc/model/model.h"

namespace minigo {

// Load a ModelDefinition from the given path.
ModelDefinition LoadModelDefinition(const std::string& path);

// Look or create up the factory instance for given engine and device.
ModelFactory* GetModelFactory(const std::string& engine,
                              const std::string& device);

// Shutdown all model factories created via calls to GetModelFactory.
void ShutdownModelFactories();

// Helper to get a factory from a definition and device by using the
// definition's engine metadata.
ModelFactory* GetModelFactory(const ModelDefinition& def,
                              const std::string& device);

// Helper to load and instantiate a model in one shot.
std::unique_ptr<Model> NewModel(const std::string& path,
                                const std::string& device);

}  // namespace minigo

#endif  //  CC_MODEL_MODEL_LOADER_H_
