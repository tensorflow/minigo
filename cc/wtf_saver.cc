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

#include "cc/wtf_saver.h"

#include <iostream>
#include <functional>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/time/time.h"
#include "cc/async/poll_thread.h"
#include "cc/logging.h"

namespace minigo {

WtfSaver::WtfSaver(std::string path, absl::Duration poll_interval)
    : path_(std::move(path)) {
  poll_thread_ = absl::make_unique<PollThread>(
      "WtfSaver", poll_interval, std::bind(&WtfSaver::Poll, this));
  poll_thread_->Start();

  options_ = wtf::Runtime::SaveOptions::ForStreamingFile(&checkpoint_);

  // Overwrite on the first write.
  options_.open_mode = std::ios_base::trunc | std::ios_base::binary;
}

WtfSaver::~WtfSaver() {
  poll_thread_->Join();
}

void WtfSaver::Poll() {
  MG_CHECK(wtf::Runtime::GetInstance()->SaveToFile(path_, options_));
  MG_LOG(INFO) << "Wrote \"" << path_ << "\"";

  // Append for subsequent writes.
  options_.open_mode = std::ios_base::app;
}

}  // namespace minigo

