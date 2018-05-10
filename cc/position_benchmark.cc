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

#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "cc/coord.h"
#include "cc/position.h"

using minigo::BoardVisitor;
using minigo::Color;
using minigo::Coord;
using minigo::GroupVisitor;
using minigo::kDefaultKomi;
using minigo::Position;

namespace {

void BM_PlayGame(benchmark::State& state) {  // NOLINT(runtime/references)
  std::vector<std::string> str_moves = {
      "pd", "dd", "qp", "dp", "fq", "hq", "oq", "cn", "qj", "nc", "pf", "pb",
      "cf", "fc", "qc", "ld", "bd", "ch", "cc", "ce", "be", "df", "dg", "cg",
      "bf", "ef", "jq", "eq", "dm", "fp", "jc", "kc", "eg", "fg", "di", "dj",
      "ei", "ci", "ej", "ek", "dk", "cj", "fk", "el", "dl", "fl", "gj", "bl",
      "gl", "fm", "fo", "gp", "gm", "fn", "go", "gn", "hn", "eo", "ho", "en",
      "im", "pk", "pj", "ok", "oj", "nk", "qk", "pm", "ql", "oo", "nm", "mn",
      "mm", "lm", "ll", "nq", "nr", "pq", "pp", "op", "or", "ln", "mk", "mq",
      "mr", "lq", "jd", "gf", "kf", "om", "nn", "no", "nj", "qb", "dc", "hc",
      "qn", "lr", "iq", "ko", "hp", "gr", "hr", "gq", "jo", "lp", "he", "ge",
      "hd", "gd", "hb", "hf", "gb", "gi", "hi", "fi", "fj", "hh", "ii", "fb",
      "ih", "ca", "ba", "ea", "db", "da", "bb", "je", "ke", "id", "kb", "kd",
      "ic", "ie", "lb", "md", "mb", "nb", "ga", "rb", "rc", "mf", "lg", "mg",
      "mh", "og", "pg", "sc", "oc", "of", "od", "ne", "sb", "sa", "sd", "sb",
      "rd", "nh", "lh", "ph", "qh", "kl", "jl", "lk", "ml", "km", "kk", "po",
      "qo", "pr", "qr", "ms", "qq", "bg", "ag", "ah", "af", "bk", "ob", "oa",
      "jf", "pe", "oe", "qg", "qf", "qi", "rg", "pi", "ri", "ni", "hg", "ib",
      "jb", "mi", "li", "gc", "fh", "gg", "gh", "jm", "jk", "mj", "lj", "jn",
      "ma", "na", "cd", "de", "oi", "oh", "is", "ig", "jg", "dh", "eh", "if",
      "kr", "qm", "rm", "ks", "js", "ls", "ec", "ed", "le", "me", "kq", "io",
      "ip", "jp", "gs", "fs", "hs", "ia", "ja", "ns", "ps", "kp", "in", "pc",
      "pl", "ol", "ha", "nd", "qe", "on", "lf", "fa", "lk", "cb", "nl", "pn",
      "os", "eb", "mc", "lc", "hh", "jo",
  };

  std::vector<Coord> moves;
  for (const auto& str_move : str_moves) {
    moves.push_back(Coord::FromSgf(str_move));
  }

  BoardVisitor bv;
  GroupVisitor gv;
  std::vector<Position> boards;
  boards.reserve(str_moves.size());
  for (auto _ : state) {
    for (int i = 0; i < 1000; ++i) {
      // For a fair comparison with the Python performance, create a new board
      // for each move.
      boards.clear();
      boards.emplace_back(&bv, &gv, Color::kBlack);
      for (const auto& move : moves) {
        boards.push_back(boards.back());
        boards.back().PlayMove(move);
      }
    }
  }
}

BENCHMARK(BM_PlayGame);

}  // namespace

BENCHMARK_MAIN();
