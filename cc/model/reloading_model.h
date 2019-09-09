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

#ifndef CC_MODEL_RELOADING_MODEL_H_
#define CC_MODEL_RELOADING_MODEL_H_

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "cc/model/model.h"

namespace minigo {

class ReloadingModelUpdater;

// Lightweight wrapper around a real Model instance.
// This class exists to enable ReloadingModelUpdater to update the wrapped
// model when a newer one is found.
class ReloadingModel : public Model {
 public:
  ReloadingModel(std::string name, ReloadingModelUpdater* updater,
                 std::unique_ptr<Model> impl);
  ~ReloadingModel() override;

  void RunMany(const std::vector<const Input*>& inputs,
               std::vector<Output*>* outputs, std::string* model_name) override;

  // Replaces the wrapped implementation with a new one constructed from the
  // given factory & model.
  // Called by ReloadingModelUpdater::Poll when it finds a new model.
  void UpdateImpl(std::unique_ptr<Model> model_impl);

 private:
  ReloadingModelUpdater* updater_;
  absl::Mutex mutex_;
  std::unique_ptr<Model> model_impl_ GUARDED_BY(&mutex_);
};

// Constructs ReloadingModel instances.
// Wraps another ModelFactory that constructs the real Model instances that
// ReloadingModel wraps.
class ReloadingModelFactory : public ModelFactory {
 public:
  ReloadingModelFactory(std::unique_ptr<ModelFactory> impl,
                        absl::Duration poll_interval);

  ~ReloadingModelFactory() override;

  // Constructs a new Model instance from the latest model that matches
  // model_pattern.
  // The model_pattern is a file path that contains exactly one "%d" scanf
  // matcher in the basename part (not the dirname part), e.g.:
  //  "foo/bar/%d-shipname.tflite"
  //  "some/dir/model.ckpt-%d.pb"
  std::unique_ptr<Model> NewModel(const std::string& model_pattern) override;

 private:
  void ThreadRun();

  absl::Mutex mutex_;

  // Map from model pattern to updater than scans for models that match that
  // pattern.
  absl::flat_hash_map<std::string, std::unique_ptr<ReloadingModelUpdater>>
      updaters_ GUARDED_BY(&mutex_);

  std::atomic<bool> running_{true};
  std::unique_ptr<ModelFactory> factory_impl_;
  const absl::Duration poll_interval_;
  std::thread thread_;
};

class ReloadingModelUpdater {
 public:
  // Blocks until at least one matching model path is found.
  ReloadingModelUpdater(const std::string& pattern, ModelFactory* factory_impl);

  // Scans directory_ for a new model that matches basename_pattern_.
  // If a new model is found, all registered ReloadingModels are updated
  // using Model instances created from the updater's factory_.
  // Returns true if a new model was found.
  bool Poll();

  // Unregisters a model with the updater.
  // There isn't a matching RegisterModel method because updater registers
  // models when it creates them.
  // Called by the model's destructor.
  void UnregisterModel(ReloadingModel* model);

  // Returns a new ReloadingModel instance that wraps a new Model instance
  // created by the factory_.
  std::unique_ptr<ReloadingModel> NewReloadingModel();

  // Exposed for testing.
  static bool ParseModelPathPattern(const std::string& pattern,
                                    std::string* directory,
                                    std::string* basename_pattern);
  static bool MatchBasename(const std::string& basename,
                            const std::string& pattern, int* generation);

 private:
  // The directory we're watching for new files.
  std::string directory_;

  // Pattern used to match files in directory_.
  std::string basename_pattern_;

  // basename_pattern_ with "%n" appended, which is used to ensure that the full
  // basename matches the pattern (and not just a prefix).
  std::string basename_and_length_pattern_;

  mutable absl::Mutex mutex_;
  ModelFactory* factory_impl_ GUARDED_BY(&mutex_);
  std::string latest_model_path_ GUARDED_BY(&mutex_);
  absl::flat_hash_set<ReloadingModel*> models_ GUARDED_BY(&mutex_);
};

}  // namespace minigo

#endif  // CC_MODEL_RELOADING_MODEL_H_
