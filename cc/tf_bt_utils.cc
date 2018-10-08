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

#include "cc/tf_utils.h"

#include <cstdint>

#include "absl/strings/str_format.h"
#include "cc/constants.h"
#include "cc/mcts_player.h"
#include "google/cloud/bigtable/read_modify_write_rule.h"
#include "google/cloud/bigtable/table.h"
#include "tensorflow/core/example/example.pb.h"

using google::cloud::bigtable::bigendian64_t;
using google::cloud::bigtable::CreateDefaultDataClient;
using google::cloud::bigtable::ReadModifyWriteRule;
using google::cloud::bigtable::SetCell;
using google::cloud::bigtable::Table;

namespace minigo {
namespace tf_utils {

// Fetches the tensorflow Examples from a MctsPlayer.
// Linked from tf_utils.cc
std::vector<tensorflow::Example> MakeExamples(const MctsPlayer& player);

// Writes a list of tensorflow Example protos to a series of Bigtable rows.
void WriteTfExamples(Table& table, const std::string& row_prefix,
                     const std::vector<tensorflow::Example>& examples) {
  int move = 0;
  for (const auto& example : examples) {
    std::string data;
    example.SerializeToString(&data);
    auto row_name = absl::StrFormat("%s_m_%06d", row_prefix, move);
    google::cloud::bigtable::SingleRowMutation mutation(row_name);
    mutation.emplace_back(SetCell("tfexample", "example", data));
    mutation.emplace_back(SetCell("metadata", "move", std::to_string(move)));
    table.Apply(std::move(mutation));
    move++;
  }
}

void WriteGameExamples(const std::string& gcp_project_name,
                       const std::string& instance_name,
                       const std::string& table_name,
                       const MctsPlayer& player) {
  auto examples = MakeExamples(player);
  Table table(CreateDefaultDataClient(gcp_project_name, instance_name,
                                      google::cloud::bigtable::ClientOptions()),
              table_name);
  // This will be everything from a single game, so retrieve the game
  // counter from the Bigtable and increment it atomically.
  using namespace google::cloud::bigtable;
  auto rule =
      ReadModifyWriteRule::IncrementAmount("metadata", "game_counter", 1);
  auto row = table.ReadModifyWriteRow("table_state", rule);
  uint64_t game_counter = 0;
  std::chrono::microseconds age{};
  for (auto const& cell : row.cells()) {
    if (cell.family_name() == "metadata" &&
        cell.column_qualifier() == "game_counter" && cell.timestamp() > age) {
      age = cell.timestamp();
      game_counter = cell.value_as<bigendian64_t>().get();
    }
  }

  auto row_prefix = absl::StrFormat("g_%010d", game_counter);
  WriteTfExamples(table, row_prefix, examples);
  std::cerr << "Bigtable rows written to prefix " << row_prefix << " : "
            << examples.size() << std::endl;
}

}  // namespace tf_utils
}  // namespace minigo
