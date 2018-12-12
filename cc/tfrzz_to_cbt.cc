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

#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/constants.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/tf_utils.h"
#include "gflags/gflags.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

DEFINE_int32(conversion_batch, 50, "How many games to process in each batch");
DEFINE_uint32(concurrency, 16,
              "How many processes to permit execution concurrently");
DEFINE_string(output_bigtable, "",
              "Output Bigtable specification, of the form: "
              "project,instance,table. "
              "If empty, no examples are written to Bigtable.");
DEFINE_string(glob_pattern, "", "Input filename glob pattern");
DEFINE_bool(async, false, "Run in background after incrementing game counter.");

void wait_for_children(std::set<int>* pids, size_t maximum_children) {
  while (pids->size() > maximum_children) {
    int status;
    int child_pid = wait(&status);
    if (status != 0) {
      MG_LOG(FATAL) << "Child pid " << child_pid << " did not succeed";
    }
    auto where = pids->find(child_pid);
    if (where == pids->end()) {
      MG_LOG(FATAL) << "Child pid " << child_pid << " not found";
    } else {
      pids->erase(where);
    }
  }
}

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);

  std::vector<std::string> bigtable_spec =
      absl::StrSplit(FLAGS_output_bigtable, ',');
  bool use_bigtable = bigtable_spec.size() == 3;
  if (!FLAGS_output_bigtable.empty() && !use_bigtable) {
    MG_LOG(FATAL)
        << "Bigtable output must be of the form: project,instance,table";
    return 1;
  }

  std::deque<std::string> paths(argv + 1, argv + argc);
  if (!paths.empty()) {
    std::cout << paths.size() << " files detected on command line."
              << std::endl;
  }
  auto glob_pattern = FLAGS_glob_pattern;
  if (!glob_pattern.empty()) {
    std::vector<std::string> glob_expansion;
    TF_CHECK_OK(tensorflow::Env::Default()->GetMatchingPaths(glob_pattern,
                                                             &glob_expansion));
    std::copy(glob_expansion.begin(), glob_expansion.end(),
              std::back_inserter(paths));
    std::cout << "Added " << glob_expansion.size() << " files for a total of "
              << paths.size() << " files to process." << std::endl;
  }

  std::set<int> pending_children;
  auto total_games = paths.size();
  auto const& project = bigtable_spec[0];
  auto const& instance = bigtable_spec[1];
  auto const& table = bigtable_spec[2];

  uint64_t final_game_counter = minigo::tf_utils::IncrementGameCounter(
      project, instance, table, "game_counter", total_games);
  uint64_t game_counter = final_game_counter - total_games;
  std::cout << "Initial game counter: " << game_counter << std::endl
            << "Final game counter will be: " << final_game_counter
            << std::endl;

  if (FLAGS_async) {
    // Now let our caller be free to launch the next.
    if (int handover = fork()) {
      std::cerr << "PID " << handover << " will continue orchestration."
                << std::endl;
      return 0;
    }
  }

  int conversion_batch = FLAGS_conversion_batch;
  auto full_start = absl::Now();
  while (!paths.empty()) {
    std::vector<std::string> batch;
    for (int i = 0; !paths.empty() && i < conversion_batch; ++i) {
      batch.push_back(paths.front());
      paths.pop_front();
    }
    // Run each batch in a separate process in order to work
    // around https://github.com/grpc/grpc/issues/15340.
    int pid = fork();
    if (pid == 0) {
      minigo::tf_utils::PortGamesToBigtable(project, instance, table, batch,
                                            game_counter);
      return 0;
    }
    game_counter += batch.size();
    pending_children.insert(pid);
    wait_for_children(&pending_children, FLAGS_concurrency);
  }

  wait_for_children(&pending_children, 0);
  auto full_stop = absl::Now();
  double elapsed = absl::ToDoubleSeconds(full_stop - full_start);
  if (!FLAGS_async) {
    std::cerr << "Total games/second: " << total_games / elapsed << std::endl;
  }

  return 0;
}
