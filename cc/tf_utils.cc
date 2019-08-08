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

#include <algorithm>
#include <array>
#include <memory>

#include "cc/constants.h"
#include "cc/dual_net/dual_net.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

using tensorflow::io::RecordWriter;
using tensorflow::io::RecordWriterOptions;

namespace minigo {
namespace tf_utils {

namespace {

template <typename T, size_t N>
std::array<uint8_t, N> ConvertToBytes(const std::array<T, N>& src) {
  std::array<uint8_t, N> dst;
  std::copy(src.begin(), src.end(), dst.begin());
  return dst;
}

template <typename T>
tensorflow::Feature MakeBytesFeature(const T& data) {
  tensorflow::Feature feature;
  feature.mutable_bytes_list()->add_value(
      reinterpret_cast<const void*>(data.data()),
      sizeof(typename T::value_type) * data.size());
  return feature;
}

// Converts board features, and the pi & value outputs of MTCS to a tensorflow
// example proto.
tensorflow::Example MakeTfExample(const DualNet::BoardFeatures& features,
                                  const std::array<float, kNumMoves>& pi,
                                  float outcome) {
  tensorflow::Example example;
  auto& dst_features = *example.mutable_features()->mutable_feature();

  // The input features are expected to be uint8 bytes.
  dst_features["x"] = MakeBytesFeature(ConvertToBytes(features));

  // pi is expected to be a float array serialized as bytes.
  dst_features["pi"] = MakeBytesFeature(pi);

  // outcome is a single float.
  dst_features["outcome"].mutable_float_list()->add_value(outcome);

  return example;
}

// Writes a list of tensorflow Example protos to a zlib compressed TFRecord
// file.
void WriteTfExamples(const std::string& path,
                     const std::vector<tensorflow::Example>& examples) {
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(path, &file));

  RecordWriterOptions options;
  options.compression_type = RecordWriterOptions::ZLIB_COMPRESSION;
  RecordWriter writer(file.get(), options);

  std::string data;
  for (const auto& example : examples) {
    example.SerializeToString(&data);
    TF_CHECK_OK(writer.WriteRecord(data));
  }

  TF_CHECK_OK(writer.Close());
  TF_CHECK_OK(file->Close());
}

}  // namespace

std::vector<tensorflow::Example> MakeExamples(const Game& game) {
  // Write the TensorFlow examples.
  std::vector<tensorflow::Example> examples;
  examples.reserve(game.num_moves());
  DualNet::BoardFeatures features;
  std::vector<const Position::Stones*> recent_positions;
  for (size_t i = 0; i < game.moves().size(); ++i) {
    const auto* move = game.moves()[i].get();
    game.GetStoneHistory(i, DualNet::kMoveHistory, &recent_positions);
    DualNet::SetFeatures(recent_positions, move->color, &features);
    if (move->trainable) {
      examples.push_back(MakeTfExample(features, move->search_pi, game.result()));
    }
  }
  return examples;
}

void WriteGameExamples(const std::string& output_dir,
                       const std::string& output_name, const Game& game) {
  MG_CHECK(file::RecursivelyCreateDir(output_dir));
  auto output_path = file::JoinPath(output_dir, output_name + ".tfrecord.zz");

  auto examples = MakeExamples(game);
  WriteTfExamples(output_path, examples);
}

}  // namespace tf_utils
}  // namespace minigo
