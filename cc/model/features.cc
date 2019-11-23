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

#include "cc/model/features.h"

#include "cc/logging.h"

namespace minigo {

FeatureDescriptor FeatureDescriptor::Create(absl::string_view input_features,
                                            absl::string_view input_layout) {
  FeatureDescriptor::Layout layout;
  if (input_layout == "nhwc") {
    layout = FeatureDescriptor::Layout::kNhwc;
  } else if (input_layout == "nchw") {
    layout = FeatureDescriptor::Layout::kNchw;
  } else {
    MG_LOG(FATAL) << "Unrecognized input layout \"" << input_layout << "\"";
    return {};
  }

  if (input_features == "agz") {
    return FeatureDescriptor::Create<AgzFeatures>(layout);
  } else if (input_features == "mlperf07") {
    return FeatureDescriptor::Create<ExtraFeatures>(layout);
  } else {
    MG_LOG(FATAL) << "Unrecognized input features \"" << input_features << "\"";
    return {};
  }
}

}  // namespace minigo
