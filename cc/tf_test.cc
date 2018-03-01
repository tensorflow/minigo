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

#include <memory>
#include <vector>

#include "cc/color.h"
#include "cc/constants.h"
#include "cc/dual_net.h"
#include "cc/position.h"

constexpr char kPrintWhite[] = "\x1b[0;31;47m";
constexpr char kPrintBlack[] = "\x1b[0;31;40m";
constexpr char kPrintEmpty[] = "\x1b[0;31;43m";
constexpr char kPrintNormal[] = "\x1b[0m";

namespace minigo {

class SimplePlayer {
 public:
  SimplePlayer() : position_(&bv_, &gv_) {
    dual_net_.Initialize("/tmp/graph/test.pb");
  }

  bool PlayMove() {
    // Run the network.
    auto output = dual_net_.Run(position_);
    if (!output.status.ok()) {
      return false;
    }

    // Clear the screen.
    printf("\e[1;1H\e[2J");

    // Print the output of the network.
    for (int row = 0; row < kN; ++row) {
      for (int col = 0; col < kN; ++col) {
        auto color = position_.stones()[row * kN + col].color();
        if (color == Color::kEmpty) {
          printf(kPrintEmpty);
        } else if (color == Color::kBlack) {
          printf(kPrintBlack);
        } else if (color == Color::kWhite) {
          printf(kPrintWhite);
        }
        float policy = output.policy.tensor<float, 2>()(0, row * kN + col);
        if (policy > 0.001) {
          printf(" %04.1f", 100 * policy);
        } else {
          printf(" .   ");
        }
      }
      printf("\n");
      for (int col = 0; col < kN; ++col) {
        auto color = position_.stones()[row * kN + col].color();
        if (color == Color::kEmpty) {
          printf(kPrintEmpty);
        } else if (color == Color::kBlack) {
          printf(kPrintBlack);
        } else if (color == Color::kWhite) {
          printf(kPrintWhite);
        }
        printf("     ");
      }
      printf("\n");
    }
    printf(kPrintNormal);
    printf("pass: %04.1f\n",
           100 * output.policy.tensor<float, 2>()(0, kN * kN));
    printf("value: %+f\n", output.value.scalar<float>()());

    // Find the best move.
    Coord best_move = Coord::kPass;
    float best_policy = output.policy.tensor<float, 2>()(0, kN * kN);
    for (int row = 0; row < kN; ++row) {
      for (int col = 0; col < kN; ++col) {
        auto color = position_.stones()[row * kN + col].color();
        float policy = output.policy.tensor<float, 2>()(0, row * kN + col);
        if (color == Color::kEmpty && policy > best_policy &&
            position_.IsMoveLegal({row, col}, position_.to_play())) {
          best_policy = policy;
          best_move = Coord(row, col);
        }
      }
    }

    // Play the best move.
    auto best_move_str = best_move.ToKgs();
    printf("\n=== To play: %s\n",
           position_.to_play() == Color::kBlack ? "B" : "W");
    printf("=== Playing move: %s\n\n", best_move_str.c_str());
    position_.PlayMove(best_move);
    fflush(stdout);

    return true;
  }

 private:
  DualNet dual_net_;
  BoardVisitor bv_;
  GroupVisitor gv_;
  Position position_;
};

}  // namespace minigo

int main() {
  minigo::SimplePlayer player;
  for (int i = 0; i < 500; ++i) {
    player.PlayMove();
  }
  return 0;
}
