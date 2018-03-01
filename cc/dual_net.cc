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

#include "cc/dual_net.h"

#include "cc/constants.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

using tensorflow::DT_FLOAT;
using tensorflow::Env;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::ReadBinaryProto;
using tensorflow::Session;
using tensorflow::SessionOptions;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::error::CANCELLED;

namespace minigo {

DualNet::DualNet() = default;

DualNet::~DualNet() {
  if (session_ != nullptr) {
    session_->Close();
  }
}

Status DualNet::Initialize(const std::string& graph_path) {
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(ReadBinaryProto(Env::Default(), graph_path, &graph_def));
  session_.reset(NewSession(SessionOptions()));
  TF_RETURN_IF_ERROR(session_->Create(graph_def));

  inputs_.clear();
  inputs_.emplace_back("pos_tensor",
                       Tensor(DT_FLOAT, TensorShape({1, kN, kN, 17})));

  output_names_.clear();
  output_names_.push_back("policy_output");
  output_names_.push_back("value_output");

  return Status::OK();
}

DualNet::Output DualNet::Run(const Position& position) {
  UpdateFeatures(position);

  DualNet::Output output;
  output.status = session_->Run(inputs_, output_names_, {}, &outputs_);
  if (output.status.ok()) {
    output.policy = std::move(outputs_[0]);
    output.value = std::move(outputs_[1]);
  }
  outputs_.clear();

  return output;
}

void DualNet::UpdateFeatures(const Position& position) {
  auto features = inputs_[0].second.tensor<float, 4>();
  float to_play = position.to_play() == Color::kBlack ? 1 : 0;
  auto my_color = position.to_play();
  auto their_color = my_color == Color::kBlack ? Color::kWhite : Color::kBlack;
  for (int row = 0; row < kN; ++row) {
    for (int col = 0; col < kN; ++col) {
      features(0, row, col, 16) = to_play;
      for (int plane = 14; plane > 0; plane -= 2) {
        features(0, row, col, plane) = features(0, row, col, plane - 1);
        features(0, row, col, plane + 1) = features(0, row, col, plane - 2);
      }
      Stone s = position.stones()[row * kN + col];
      features(0, row, col, 0) = s.color() == my_color ? 1 : 0;
      features(0, row, col, 1) = s.color() == their_color ? 1 : 0;
    }
  }

  // for (int row = 0; row < kN; ++row) {
  //   for (int col = 0; col < kN; ++col) {
  //     int v = 0;
  //     int m = 1;
  //     for (int plane = 0; plane < 17; ++plane) {
  //       v += m * features(0, row, col, plane);
  //       m *= 2;
  //     }
  //     printf(" %5x", v);
  //   }
  //   printf("\n");
  // }
}

}  // namespace minigo
