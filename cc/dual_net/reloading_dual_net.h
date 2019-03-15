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

#ifndef CC_DUAL_NET_RELOADING_DUAL_NET_H_
#define CC_DUAL_NET_RELOADING_DUAL_NET_H_

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "cc/dual_net/dual_net.h"

namespace minigo {

class ReloadingDualNetUpdater;

// Lightweight wrapper around a real DualNet instance.
// This class exists to enable ReloadingDualNetUpdater to update the wrapped
// model when a newer one is found.
class ReloadingDualNet : public DualNet {
 public:
  ReloadingDualNet(std::string name, ReloadingDualNetUpdater* updater,
                   std::unique_ptr<DualNet> impl);
  ~ReloadingDualNet() override;

  void RunMany(std::vector<const BoardFeatures*> features,
               std::vector<Output*> outputs, std::string* model) override;

  void Reserve(size_t capacity) override;

  // Replaces the wrapped implementation with a new one constructed from the
  // given factory & model.
  // Called by ReloadingDualNetUpdater::Poll when it finds a new model.
  void UpdateImpl(std::unique_ptr<DualNet> model_impl);

 private:
  ReloadingDualNetUpdater* updater_;
  absl::Mutex mutex_;
  std::unique_ptr<DualNet> model_impl_ GUARDED_BY(&mutex_);
};

// Constructs ReloadingDualNet instances.
// Wraps another DualNetFactory that constructs the real DualNet instances that
// ReloadingDualNet wraps.
class ReloadingDualNetFactory : public DualNetFactory {
 public:
  ReloadingDualNetFactory(std::unique_ptr<DualNetFactory> impl,
                          absl::Duration poll_interval);

  ~ReloadingDualNetFactory() override;

  int GetBufferCount() const override;

  // Constructs a new DualNet instance from the latest model that matches
  // model_pattern.
  // The model_pattern is a file path that contains exactly one "%d" scanf
  // matcher in the basename part (not the dirname part), e.g.:
  //  "foo/bar/%d-shipname.tflite"
  //  "some/dir/model.ckpt-%d.pb"
  std::unique_ptr<DualNet> NewDualNet(
      const std::string& model_pattern) override;

 private:
  void ThreadRun();

  absl::Mutex mutex_;

  // Map from model pattern to updater than scans for models that match that
  // pattern.
  absl::flat_hash_map<std::string, std::unique_ptr<ReloadingDualNetUpdater>>
      updaters_ GUARDED_BY(&mutex_);

  std::atomic<bool> running_;
  std::unique_ptr<DualNetFactory> factory_impl_;
  const absl::Duration poll_interval_;
  std::thread thread_;
};

class ReloadingDualNetUpdater {
 public:
  // Blocks until at least one matching model path is found.
  ReloadingDualNetUpdater(const std::string& pattern,
                          DualNetFactory* factory_impl);

  // Scans directory_ for a new model that matches basename_pattern_.
  // If a new model is found, all registered ReloadingDualNets are updated
  // using DualNet instances created from the updater's factory_.
  // Returns true if a new model was found.
  bool Poll();

  // Unregisters a model with the updater.
  // There isn't a matching RegisterModel method because updater registers
  // models when it creates them.
  // Called by the model's destructor.
  void UnregisterModel(ReloadingDualNet* model);

  // Returns a new ReloadingDualNet instance that wraps a new DualNet instance
  // created by the factory_.
  std::unique_ptr<ReloadingDualNet> NewReloadingDualNet();

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
  DualNetFactory* factory_impl_ GUARDED_BY(&mutex_);
  std::string latest_model_path_ GUARDED_BY(&mutex_);
  absl::flat_hash_set<ReloadingDualNet*> models_ GUARDED_BY(&mutex_);
};

}  // namespace minigo

#endif  // CC_DUAL_NET_RELOADING_DUAL_NET_H_
