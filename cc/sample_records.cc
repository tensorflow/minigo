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

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/random.h"
#include "cc/thread.h"
#include "gflags/gflags.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

DEFINE_double(sample_frac, 0, "Fraction of records to read from each file.");
DEFINE_int32(num_threads, 1, "Number of threads to run on.");
DEFINE_int32(compression, 1,
             "Compression level between 0 (disabled) and 9. Default is 1.");
DEFINE_uint64(seed, 0, "Random seed.");
DEFINE_bool(shuffle, false, "Whether to shuffle the sampled records.");
DEFINE_string(dst, "",
              "Destination path. If path has a .zz suffix, the file will be "
              "automatically compressed.");

namespace minigo {

class ReadThread : public Thread {
 public:
  ReadThread(std::vector<std::string> paths, float sample_frac)
      : rnd_(FLAGS_seed, Random::kUniqueStream),
        paths_(std::move(paths)),
        sample_frac_(sample_frac) {}

  std::vector<std::string>& sampled_records() { return sampled_records_; }
  const std::vector<std::string>& sampled_records() const {
    return sampled_records_;
  }

 private:
  void Run() override {
    tensorflow::io::RecordReaderOptions options;
    for (const auto& path : paths_) {
      std::unique_ptr<tensorflow::RandomAccessFile> file;
      TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(path, &file));
      if (absl::EndsWith(path, ".zz")) {
        options.compression_type =
            tensorflow::io::RecordReaderOptions::ZLIB_COMPRESSION;
      } else {
        options.compression_type = tensorflow::io::RecordReaderOptions::NONE;
      }

      tensorflow::io::SequentialRecordReader reader(file.get(), options);
      std::string record;
      for (;;) {
        auto status = reader.ReadRecord(&record);
        if (status.code() == tensorflow::error::OUT_OF_RANGE) {
          // Reached the end of the file.
          break;
        } else if (!status.ok()) {
          // Some other error.
          MG_LOG(WARNING) << "Error reading record from \"" << path
                          << "\": " << status;
          continue;
        }

        if (rnd_() < sample_frac_) {
          sampled_records_.push_back(std::move(record));
        }
      }
    }
  }

  Random rnd_;
  const std::vector<std::string> paths_;
  const float sample_frac_;
  std::vector<std::string> sampled_records_;
};

template <typename T>
void MoveAppend(std::vector<T>* src, std::vector<T>* dst) {
  if (dst->empty()) {
    *dst = std::move(*src);
  } else {
    std::move(std::begin(*src), std::end(*src), std::back_inserter(*dst));
    src->clear();
  }
}

void WriteRecords(const std::vector<std::string>& records,
                  const std::string& path, int compression) {
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(path, &file));

  tensorflow::io::RecordWriterOptions options;
  if (compression > 0) {
    MG_CHECK(compression <= 9);
    options.compression_type =
        tensorflow::io::RecordWriterOptions::ZLIB_COMPRESSION;
    options.zlib_options.compression_level = compression;
  } else {
    options.compression_type = tensorflow::io::RecordWriterOptions::NONE;
  }

  tensorflow::io::RecordWriter writer(file.get(), options);
  for (const auto& record : records) {
    TF_CHECK_OK(writer.WriteRecord(record));
  }

  TF_CHECK_OK(writer.Close());
  TF_CHECK_OK(file->Close());
}

void Run(std::vector<std::string> src_paths, const std::string& dst_path) {
  MG_CHECK(!src_paths.empty());
  MG_CHECK(!dst_path.empty());

  int num_paths = static_cast<int>(src_paths.size());
  int num_threads = std::min<int>(FLAGS_num_threads, num_paths);

  MG_LOG(INFO) << absl::Now() << " : reading " << num_paths << " records on "
               << num_threads << " threads";

  std::vector<std::unique_ptr<ReadThread>> threads;
  for (int i = 0; i < num_threads; ++i) {
    // Get the record paths that this thread should run on.
    int begin = i * num_paths / num_threads;
    int end = (i + 1) * num_paths / num_threads;
    std::vector<std::string> thread_paths;
    for (int j = begin; j < end; ++j) {
      thread_paths.push_back(std::move(src_paths[j]));
    }
    threads.push_back(absl::make_unique<ReadThread>(std::move(thread_paths),
                                                    FLAGS_sample_frac));
  }
  for (auto& t : threads) {
    t->Start();
  }
  for (auto& t : threads) {
    t->Join();
  }

  // Concatenate sampled records.
  size_t n = 0;
  for (const auto& t : threads) {
    n += t->sampled_records().size();
  }
  MG_LOG(INFO) << absl::Now() << " : sampled " << n << " records";
  MG_LOG(INFO) << absl::Now() << " : concatenating";
  std::vector<std::string> records;
  records.reserve(n);
  for (const auto& t : threads) {
    MoveAppend(&t->sampled_records(), &records);
  }

  // Shuffle the records if requested.
  if (FLAGS_shuffle) {
    Random rnd(FLAGS_seed, Random::kUniqueStream);
    MG_LOG(INFO) << absl::Now() << " : shuffling";
    auto gen = [&rnd](int i) { return rnd.UniformInt(0, i); };
    std::random_shuffle(records.begin(), records.end(), gen);
  }

  // Write result.
  MG_LOG(INFO) << absl::Now() << " : writing to " << dst_path;
  WriteRecords(records, dst_path, FLAGS_compression);
  MG_LOG(INFO) << absl::Now() << " : done";
}

}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  std::vector<std::string> src_paths;
  for (int i = 1; i < argc; ++i) {
    src_paths.emplace_back(argv[i]);
  }
  minigo::Run(std::move(src_paths), FLAGS_dst);
  return 0;
}
