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

#ifndef CC_MODEL_MODEL_FACTORY_H_
#define CC_MODEL_MODEL_FACTORY_H_

#include <memory>
#include <ostream>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "cc/model/model.h"

namespace minigo {

using ModelProperty =
    absl::variant<std::string, bool, int64_t, uint64_t, float>;

std::ostream& operator<<(std::ostream& os, const ModelProperty& p);

// Although the metadata is stored in the Minigo file as JSON, it is
// converted on load to a simpler representation to avoid pulling an entire
// JSON library into this header (at the time of writing the nlohmann::json
// header is more than 22,000 lines).
class ModelMetadata {
 public:
  // Explicit setters to avoid hard to interpret template compiler errors.
  void Set(absl::string_view key, const char* value) {
    impl_[key] = std::string(value);
  }
  void Set(absl::string_view key, absl::string_view value) {
    impl_[key] = std::string(value);
  }
  void Set(absl::string_view key, std::string value) {
    impl_[key] = std::move(value);
  }
  void Set(absl::string_view key, bool value) { impl_[key] = value; }
  void Set(absl::string_view key, int64_t value) { impl_[key] = value; }
  void Set(absl::string_view key, uint64_t value) { impl_[key] = value; }
  void Set(absl::string_view key, float value) { impl_[key] = value; }

  bool Has(absl::string_view key) const { return impl_.contains(key); }

  template <typename T>
  const T& Get(absl::string_view key) const {
    const auto& prop = impl_.at(key);
    MG_DCHECK(absl::holds_alternative<T>(prop)) << prop;
    return absl::get<T>(prop);
  }

  template <typename T>
  bool TryGet(absl::string_view key, T* value) const {
    auto it = impl_.find(key);
    if (it == impl_.end()) {
      return false;
    }
    if (!absl::holds_alternative<T>(it->second)) {
      return false;
    }
    *value = absl::get<T>(it->second);
    return true;
  }

 private:
  absl::flat_hash_map<std::string, ModelProperty> impl_;
};

struct ModelDefinition {
  std::string path;
  ModelMetadata metadata;
  std::string model_bytes;
};

// Factory that creates Model instances.
// All implementations are required to be thread safe.
class ModelFactory {
 public:
  virtual ~ModelFactory();

  // Create a single model.
  virtual std::unique_ptr<Model> NewModel(const ModelDefinition& def) = 0;
};

}  // namespace minigo

#endif  //  CC_MODEL_MODEL_FACTORY_H_
