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

#include <memory>
#include "absl/types/span.h"
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
  for (size_t i = 0; i < N; ++i) {
    dst[i] = static_cast<uint8_t>(src[i]);
  }
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

}  // namespace

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

void WriteTfExamples(const std::string& path,
                     absl::Span<const tensorflow::Example> examples) {
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

tensorflow::Status WriteFile(const std::string& path,
                             absl::string_view contents) {
  tensorflow::Status status;

  std::unique_ptr<tensorflow::WritableFile> file;
  status = tensorflow::Env::Default()->NewWritableFile(path, &file);
  if (!status.ok()) {
    return status;
  }

  // *sigh* absl::string_view doesn't automatically convert to
  // tensorflow::StringPiece even though I'm pretty sure they're exactly the
  // same.
  status = file->Append({contents.data(), contents.size()});
  if (!status.ok()) {
    return status;
  }

  return file->Close();
}

tensorflow::Status ReadFile(const std::string& path, std::string* contents) {
  auto* env = tensorflow::Env::Default();
  tensorflow::uint64 size;
  TF_RETURN_IF_ERROR(env->GetFileSize(path, &size));

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(path, &file));

  contents->resize(size);
  tensorflow::StringPiece s;
  TF_RETURN_IF_ERROR(file->Read(0u, size, &s, &(*contents)[0]));
  contents->resize(s.size());

  return tensorflow::Status::OK();
}

tensorflow::Status GetModTime(const std::string& path, uint64_t* mtime_usec) {
  tensorflow::FileStatistics stat;
  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->Stat(path, &stat));
  *mtime_usec = static_cast<uint64_t>(stat.mtime_nsec);
  return tensorflow::Status::OK();
}

}  // namespace tf_utils
}  // namespace minigo
