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

#include "cc/model/reloading_model.h"

#include <cstdio>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"

namespace minigo {

// ReloadingModelUpdater methods follow.

bool ReloadingModelUpdater::ParseModelPathPattern(
    const std::string& pattern, std::string* directory,
    std::string* basename_pattern) {
  auto pair = file::SplitPath(pattern);

  *directory = std::string(pair.first);
  if (directory->find('%') != std::string::npos ||
      directory->find('*') != std::string::npos) {
    MG_LOG(ERROR) << "invalid pattern \"" << pattern
                  << "\": directory part must not contain '*' or '%'";
    return false;
  }
  if (directory->empty()) {
    MG_LOG(ERROR) << "directory not be empty";
    return false;
  }

  *basename_pattern = std::string(pair.second);
  auto it = basename_pattern->find('%');
  if (it == std::string::npos || basename_pattern->find("%d") != it ||
      basename_pattern->rfind("%d") != it) {
    MG_LOG(ERROR) << "invalid pattern \"" << pattern
                  << "\": basename must contain "
                  << " exactly one \"%d\" and no other matchers";
    return false;
  }
  return true;
}

bool ReloadingModelUpdater::MatchBasename(const std::string& basename,
                                          const std::string& pattern,
                                          int* generation) {
  int gen = 0;
  int n = 0;
  if (sscanf(basename.c_str(), pattern.c_str(), &gen, &n) != 1 ||
      n != static_cast<int>(basename.size())) {
    return false;
  }
  *generation = gen;
  return true;
}

ReloadingModelUpdater::ReloadingModelUpdater(const std::string& pattern,
                                             ModelFactory* factory_impl)
    : factory_impl_(factory_impl) {
  MG_CHECK(ParseModelPathPattern(pattern, &directory_, &basename_pattern_));

  // Append "%n" to the end of the basename pattern. This is used when calling
  // sscanf to ensure we match the model's full basename and not just a prefix.
  basename_and_length_pattern_ = absl::StrCat(basename_pattern_, "%n");

  // Wait for at least one matching model to be found.
  if (!Poll()) {
    MG_LOG(INFO) << "Waiting for model that matches pattern \"" << pattern
                 << "\"";
    do {
      absl::SleepFor(absl::Seconds(1));
    } while (!Poll());
  }
}

bool ReloadingModelUpdater::Poll() {
  // List all the files in the given directory.
  std::vector<std::string> basenames;
  if (!file::ListDir(directory_, &basenames)) {
    return false;
  }

  // Find the file basename that contains the largest integer.
  const std::string* latest_basename = nullptr;
  int latest_generation = -1;
  for (const auto& basename : basenames) {
    int generation = 0;
    if (!MatchBasename(basename, basename_and_length_pattern_, &generation)) {
      continue;
    }
    if (latest_basename == nullptr || generation > latest_generation) {
      latest_basename = &basename;
      latest_generation = generation;
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

    // Create new model instances for all registered ReloadingModels.
    latest_model_path_ = std::move(path);
    MG_LOG(INFO) << "Loading new model \"" << latest_model_path_ << "\"";
    for (auto* model : models_) {
      model->UpdateImpl(factory_impl_->NewModel(latest_model_path_));
    }
  }
  return true;
}

void ReloadingModelUpdater::UnregisterModel(ReloadingModel* model) {
  absl::MutexLock lock(&mutex_);
  MG_CHECK(models_.erase(model) != 0);
}

std::unique_ptr<ReloadingModel> ReloadingModelUpdater::NewReloadingModel() {
  absl::MutexLock lock(&mutex_);
  // Create the real model.
  auto model_impl = factory_impl_->NewModel(latest_model_path_);

  // Wrap the model.
  auto model = absl::make_unique<ReloadingModel>(basename_pattern_, this,
                                                 std::move(model_impl));

  // Register the wrapped model.
  MG_CHECK(models_.emplace(model.get()).second);
  return model;
}

// ReloadingModel methods follow.

ReloadingModel::ReloadingModel(std::string name, ReloadingModelUpdater* updater,
                               std::unique_ptr<Model> impl)
    : Model(std::move(name), impl->feature_type(), impl->buffer_count()),
      updater_(updater),
      model_impl_(std::move(impl)) {}

ReloadingModel::~ReloadingModel() { updater_->UnregisterModel(this); }

void ReloadingModel::RunMany(const std::vector<const Input*>& inputs,
                             std::vector<Output*>* outputs,
                             std::string* model_name) {
  absl::MutexLock lock(&mutex_);
  model_impl_->RunMany(inputs, outputs, model_name);
}

void ReloadingModel::UpdateImpl(std::unique_ptr<Model> model_impl) {
  absl::MutexLock lock(&mutex_);
  model_impl_ = std::move(model_impl);
}

// ReloadingModelFactory methods follow.

ReloadingModelFactory::ReloadingModelFactory(std::unique_ptr<ModelFactory> impl,
                                             absl::Duration poll_interval)
    : factory_impl_(std::move(impl)), poll_interval_(poll_interval) {
  thread_ = std::thread(&ReloadingModelFactory::ThreadRun, this);
}

ReloadingModelFactory::~ReloadingModelFactory() {
  running_ = false;
  thread_.join();
}

std::unique_ptr<Model> ReloadingModelFactory::NewModel(
    const std::string& model_pattern) {
  absl::MutexLock lock(&mutex_);
  auto it = updaters_.find(model_pattern);
  if (it == updaters_.end()) {
    auto updater = absl::make_unique<ReloadingModelUpdater>(
        model_pattern, factory_impl_.get());
    it = updaters_.emplace(model_pattern, std::move(updater)).first;
  }
  return it->second->NewReloadingModel();
}

void ReloadingModelFactory::ThreadRun() {
  while (running_) {
    absl::SleepFor(poll_interval_);
    absl::MutexLock lock(&mutex_);
    for (auto& kv : updaters_) {
      kv.second->Poll();
    }
  }
}

}  // namespace minigo
