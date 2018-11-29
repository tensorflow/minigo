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

#include "cc/dual_net/reloading_dual_net.h"

#include <cstdio>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"

namespace minigo {

// ReloadingDualNetUpdater methods follow.

ReloadingDualNetUpdater::ReloadingDualNetUpdater(const std::string& pattern,
                                                 DualNetFactory* factory_impl)
    : factory_impl_(factory_impl) {
  auto pair = file::SplitPath(pattern);

  directory_ = std::string(pair.first);
  MG_CHECK(directory_.find('%') == std::string::npos &&
           directory_.find('*') == std::string::npos)
      << "invalid pattern \"" << pattern
      << "\": directory part must not contain '*' or '%'";

  basename_pattern_ = std::string(pair.second);
  auto it = basename_pattern_.find('%');
  MG_CHECK(it != std::string::npos && basename_pattern_.find("%d") == it &&
           basename_pattern_.rfind("%d") == it)
      << "invalid pattern \"" << pattern
      << "\": basename must contain exactly one \"%d\" and no other matchers";

  // Append "%n" to the end of the basename pattern. This is used when calling
  // sscanf to ensure we match the model's full basename and not just a prefix.
  basename_pattern_ = absl::StrCat(pair.second, "%n");

  // Wait for at least one matching model to be found.
  if (!Poll()) {
    std::cerr << "Waiting for model that matches pattern \"" << pattern << "\""
              << std::endl;
    do {
      absl::SleepFor(absl::Seconds(1));
    } while (!Poll());
  }
}

bool ReloadingDualNetUpdater::Poll() {
  // List all the files in the given directory.
  std::vector<std::string> basenames;
  if (!file::ListDir(directory_, &basenames)) {
    return false;
  }

  // Find the file basename that contains the largest integer.
  const std::string* latest_basename = nullptr;
  int latest_gen = -1;
  for (const auto& basename : basenames) {
    int gen = 0;
    int n = 0;
    if (sscanf(basename.c_str(), basename_pattern_.c_str(), &gen, &n) != 1 ||
        n != static_cast<int>(basename.size())) {
      continue;
    }
    if (latest_basename == nullptr || gen > latest_gen) {
      latest_basename = &basename;
      latest_gen = gen;
    }
  }

  if (latest_basename == nullptr) {
    // Didn't find any matching files.
    return false;
  }

  // Build the full path to the latest model.
  auto path = file::JoinPath(directory_, *latest_basename);

  {
    absl::MutexLock lock(&mutex_);
    if (path == latest_model_path_) {
      // The latest model hasn't changed.
      return false;
    }

    // Create new model instances for all registered ReloadingDualNets.
    latest_model_path_ = std::move(path);
    std::cerr << "Loading new model \"" << latest_model_path_ << "\""
              << std::endl;
    for (auto* model : models_) {
      model->UpdateImpl(factory_impl_->NewDualNet(latest_model_path_));
    }
  }
  return true;
}

void ReloadingDualNetUpdater::UnregisterModel(ReloadingDualNet* model) {
  absl::MutexLock lock(&mutex_);
  MG_CHECK(models_.erase(model) != 0);
}

std::unique_ptr<ReloadingDualNet>
ReloadingDualNetUpdater::NewReloadingDualNet() {
  absl::MutexLock lock(&mutex_);
  // Create the real model.
  auto model_impl = factory_impl_->NewDualNet(latest_model_path_);

  // Wrap the model.
  auto model = absl::make_unique<ReloadingDualNet>(this, std::move(model_impl));

  // Register the wrapped model.
  MG_CHECK(models_.emplace(model.get()).second);
  return model;
}

// ReloadingDualNet methods follow.

ReloadingDualNet::ReloadingDualNet(ReloadingDualNetUpdater* updater,
                                   std::unique_ptr<DualNet> impl)
    : updater_(updater), model_impl_(std::move(impl)) {}

ReloadingDualNet::~ReloadingDualNet() { updater_->UnregisterModel(this); }

void ReloadingDualNet::RunMany(std::vector<const BoardFeatures*> features,
                               std::vector<Output*> outputs,
                               std::string* model) {
  absl::MutexLock lock(&mutex_);
  model_impl_->RunMany(std::move(features), std::move(outputs), model);
}

void ReloadingDualNet::Reserve(size_t capacity) {
  absl::MutexLock lock(&mutex_);
  model_impl_->Reserve(capacity);
}

void ReloadingDualNet::UpdateImpl(std::unique_ptr<DualNet> impl) {
  absl::MutexLock lock(&mutex_);
  model_impl_ = std::move(impl);
}

// ReloadingDualNetFactory methods follow.

ReloadingDualNetFactory::ReloadingDualNetFactory(
    std::unique_ptr<DualNetFactory> impl, absl::Duration poll_interval)
    : factory_impl_(std::move(impl)), poll_interval_(poll_interval) {
  running_ = true;
  thread_ = std::thread(&ReloadingDualNetFactory::ThreadRun, this);
}

ReloadingDualNetFactory::~ReloadingDualNetFactory() {
  running_ = false;
  thread_.join();
}

int ReloadingDualNetFactory::GetBufferCount() const {
  return factory_impl_->GetBufferCount();
}

std::unique_ptr<DualNet> ReloadingDualNetFactory::NewDualNet(
    const std::string& model_pattern) {
  absl::MutexLock lock(&mutex_);
  auto it = updaters_.find(model_pattern);
  if (it == updaters_.end()) {
    auto updater = absl::make_unique<ReloadingDualNetUpdater>(
        model_pattern, factory_impl_.get());
    it = updaters_.emplace(model_pattern, std::move(updater)).first;
  }
  return it->second->NewReloadingDualNet();
}

void ReloadingDualNetFactory::ThreadRun() {
  while (running_) {
    absl::SleepFor(poll_interval_);
    absl::MutexLock lock(&mutex_);
    for (auto& kv : updaters_) {
      kv.second->Poll();
    }
  }
}

}  // namespace minigo
