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

#include "cc/model/loader.h"

#include <cstdint>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "cc/dual_net/random_dual_net.h"
#include "cc/file/utils.h"
#include "cc/json.h"
#include "cc/logging.h"

#ifdef MG_ENABLE_TF_DUAL_NET
#include "cc/dual_net/tf_dual_net.h"
#endif  // MG_ENABLE_TF_DUAL_NET

#ifdef MG_ENABLE_LITE_DUAL_NET
#include "cc/dual_net/lite_dual_net.h"
#endif  // MG_ENABLE_LITE_DUAL_NET

#ifdef MG_ENABLE_TPU_DUAL_NET
#include "cc/dual_net/tpu_dual_net.h"
#endif  // MG_ENABLE_TPU_DUAL_NET

namespace minigo {

namespace {

// Header for a Minigo model file.
struct ModelHeader {
  char magic[8];
  uint64_t version;
  uint64_t file_size;
  uint64_t metadata_size;
};

// A registry of ModelFactories.
// Some factories (e.g. TpuDualNetFactory) must be kept around for the
// duration of the process because they keep alive a connection to an
// accelerator.
class FactoryRegistry {
 public:
  static FactoryRegistry* Get() {
    static FactoryRegistry impl;
    return &impl;
  }

  // Gets or registers a new model factory for the given `engine` and `device`.
  ModelFactory* GetFactory(const std::string& engine,
                           const std::string& device) {
    absl::MutexLock lock(&mutex_);

    // Look for an already registered factory.
    for (const auto& f : factories_) {
      if (f.engine == engine && f.device == device) {
        return f.factory.get();
      }
    }

    // Register a new factory.
    auto factory = NewModelFactory(engine, device);
    auto* result = factory.get();
    factories_.emplace_back(engine, device, std::move(factory));
    return result;
  }

  // Clears the registry, destroying all registered factories.
  void Clear() {
    absl::MutexLock lock(&mutex_);
    factories_.clear();
  }

 private:
  struct RegisteredFactory {
    RegisteredFactory(const std::string& engine, const std::string& device,
                      std::unique_ptr<ModelFactory> factory)
        : engine(engine), device(device), factory(std::move(factory)) {}
    const std::string engine;
    const std::string device;
    std::unique_ptr<ModelFactory> factory;
  };

  std::unique_ptr<ModelFactory> NewModelFactory(const std::string& engine,
                                                const std::string& device)
      EXCLUSIVE_LOCKS_REQUIRED(&mutex_) {
    if (engine == "random") {
      return absl::make_unique<RandomDualNetFactory>();
    }

#ifdef MG_ENABLE_TF_DUAL_NET
    if (engine == "tf") {
      int id = -1;
      if (!device.empty()) {
        MG_CHECK(absl::SimpleAtoi(device, &id)) << "\"" << device << "\"";
      }
      return absl::make_unique<TfDualNetFactory>(id);
    }
#endif  // MG_ENABLE_TF_DUAL_NET

#ifdef MG_ENABLE_LITE_DUAL_NET
    if (engine == "lite") {
      return absl::make_unique<LiteDualNetFactory>();
    }
#endif  // MG_ENABLE_LITE_DUAL_NET

#ifdef MG_ENABLE_TPU_DUAL_NET
    if (engine == "tpu") {
      return absl::make_unique<TpuDualNetFactory>(device);
    }
#endif  // MG_ENABLE_TPU_DUAL_NET

    MG_LOG(FATAL) << "Unrecognized inference engine \"" << engine << "\"";
    return nullptr;
  }

  absl::Mutex mutex_;
  std::vector<RegisteredFactory> factories_ GUARDED_BY(&mutex_);
};

ModelDefinition CreateRandomModelDefinition(absl::string_view descriptor) {
  ModelDefinition def;

  std::vector<absl::string_view> parts = absl::StrSplit(descriptor, ':');
  MG_CHECK(parts.size() == 2);

  uint64_t seed;
  MG_CHECK(absl::SimpleAtoi(parts[1], &seed));

  def.path = std::string(descriptor);
  def.metadata.Set("engine", "random");
  def.metadata.Set("input_features", parts[0]);
  def.metadata.Set("seed", seed);
  def.metadata.Set("policy_stddev", 0.4f);
  def.metadata.Set("value_stddev", 0.4f);
  return def;
}

ModelDefinition ReadModelDefinition(const std::string& path) {
  ModelDefinition def;

  def.path = path;

  std::string contents;
  MG_CHECK(file::ReadFile(path, &contents));

  ModelHeader header;
  MG_CHECK(contents.size() >= sizeof(header));
  memcpy(&header, contents.data(), sizeof(header));

  absl::string_view magic(header.magic, sizeof(header.magic));
  MG_CHECK(magic == "<minigo>") << "\"" << magic << "\"";
  MG_CHECK(header.version == 1) << header.version;
  MG_CHECK(header.file_size == contents.size()) << header.file_size;
  MG_CHECK(header.metadata_size + sizeof(header) <= header.file_size);

  const auto* json_begin = contents.data() + sizeof(header);
  const auto* json_end = json_begin + header.metadata_size;
  auto j = nlohmann::json::parse(json_begin, json_end);
  for (const auto& kv : j.items()) {
    const auto& key = kv.key();
    const auto& value = kv.value();
    switch (value.type()) {
      case nlohmann::json::value_t::boolean:
        def.metadata.Set(key, value.get<bool>());
        break;
      case nlohmann::json::value_t::string:
        def.metadata.Set(key, value.get<std::string>());
        break;
      case nlohmann::json::value_t::number_float:
        def.metadata.Set(key, value.get<float>());
        break;
      case nlohmann::json::value_t::number_integer:
        def.metadata.Set(key, value.get<int64_t>());
        break;
      case nlohmann::json::value_t::number_unsigned:
        def.metadata.Set(key, value.get<uint64_t>());
        break;

      default:
        MG_LOG(FATAL) << "unsupported metadata type "
                      << static_cast<int>(value.type()) << " for key \"" << key
                      << "\"";
    }
  }

  // TODO(tommadams): this copies almost the entire file. Add a proper file
  // abstraction that can load partial files to avoid this (currently the only
  // abstraction we have is file::ReadFile, which can only read an entire file's
  // contents in one shot).
  def.model_bytes = contents.substr(sizeof(header) + header.metadata_size);

  MG_CHECK(def.metadata.Has("engine"));
  MG_CHECK(def.metadata.Has("input_features"));
  MG_CHECK(def.metadata.Has("input_layout"));
  MG_CHECK(def.metadata.Has("board_size"));
  MG_CHECK(def.metadata.Get<uint64_t>("board_size") == kN);

  return def;
}

}  // namespace

ModelDefinition LoadModelDefinition(const std::string& path) {
  absl::string_view path_view(path);
  if (absl::ConsumePrefix(&path_view, "random:")) {
    return CreateRandomModelDefinition(path_view);
  } else {
    return ReadModelDefinition(path);
  }
}

ModelFactory* GetModelFactory(const std::string& engine,
                              const std::string& device) {
  return FactoryRegistry::Get()->GetFactory(engine, device);
}

void ShutdownModelFactories() { FactoryRegistry::Get()->Clear(); }

ModelFactory* GetModelFactory(const ModelDefinition& def,
                              const std::string& device) {
  const auto& engine = def.metadata.Get<std::string>("engine");
  return GetModelFactory(engine, device);
}

std::unique_ptr<Model> NewModel(const std::string& path,
                                const std::string& device) {
  auto def = LoadModelDefinition(path);
  auto* factory = GetModelFactory(def, device);
  return factory->NewModel(def);
}

}  // namespace minigo
