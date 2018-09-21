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

function heatMapDq(qs: Float32Array) {
  let result: Float32Array[] = [];
  qs.forEach((q) => {
    let rgb = q > 0 ? 0 : 255;
    let a = Math.min(Math.abs(q / 100), 0.6);
    result.push(new Float32Array([rgb, rgb, rgb, a]));
  });
  return result;
}

function heatMapN(ns: Float32Array) {
  let result: Float32Array[] = [];
  let nSum = 0;
  ns.forEach((n) => {
    nSum += n;
  });
  nSum = Math.max(nSum, 1);
  ns.forEach((n) => {
    let a = Math.min(Math.sqrt(n / nSum), 0.6);
    if (a > 0) {
      a = 0.1 + 0.9 * a;
    }
    result.push(new Float32Array([0, 0, 0, a]));
  });
  return result;
}

export {
  heatMapDq,
  heatMapN,
}

