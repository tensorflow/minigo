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

#ifndef CC_MODEL_MODEL_H_
#define CC_MODEL_MODEL_H_

#include <string>
#include <vector>

#include "cc/color.h"
#include "cc/constants.h"
#include "cc/inline_vector.h"
#include "cc/position.h"
#include "cc/symmetries.h"

namespace minigo {

class Model {
 public:
  enum class FeatureType {
    kAgz,
    kExtra,

    kNumFeatureTypes,
  };

  static constexpr int kNumAgzFeaturePlanes = 17;
  static constexpr int kNumExtraFeaturePlanes = 20;

  // A simple tensor representation that abstracts a real engine-specific
  // tensor. Tensor does not own the memory pointed to by `data`.
  // Tensors are assumed to be tightly packed for now.
  // TODO(tommadams): Make this templated on the data type so we can support
  // byte input features, and quantized outputs.
  template <typename T>
  struct Tensor {
    Tensor() = default;
    Tensor(int n, int h, int w, int c, T* data)
        : n(n), h(h), w(w), c(c), data(data) {}

    int n = 0;
    int h = 0;
    int w = 0;
    int c = 0;
    T* data = nullptr;
  };

  struct Input {
    // Symmetry to apply to the input features when performing inference.
    symmetry::Symmetry sym = symmetry::kNumSymmetries;

    // position_history[0] holds the current position and position_history[i]
    // holds the position from i moves ago.
    inline_vector<const Position*, kMaxPositionHistory> position_history;
  };

  struct Output {
    std::array<float, kNumMoves> policy;
    float value;
  };

  static int GetNumFeaturePlanes(FeatureType feature_type);

  static void ApplySymmetry(symmetry::Symmetry sym, const Output& src,
                            Output* dst);

  // TODO(tommadams): is there some way to avoid having buffer_count in the base
  // class? All subclasses except BufferedModel set this to 1.
  Model(std::string name, FeatureType feature_type, int buffer_count);
  virtual ~Model();

  const std::string& name() const { return name_; }
  FeatureType feature_type() const { return feature_type_; }

  // Returns the ideal number of inference requests in flight for this model.
  int buffer_count() const { return buffer_count_; }

  virtual void RunMany(const std::vector<const Input*>& inputs,
                       std::vector<Output*>* outputs,
                       std::string* model_name) = 0;

 private:
  const std::string name_;
  const FeatureType feature_type_;
  const int buffer_count_;
};

// Factory that creates Model instances.
// All implementations are required to be thread safe.
class ModelFactory {
 public:
  virtual ~ModelFactory();

  // Create a single model.
  virtual std::unique_ptr<Model> NewModel(const std::string& descriptor) = 0;
};

}  // namespace minigo

#endif  //  CC_MODEL_MODEL_H_
