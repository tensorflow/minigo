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

import {Color, otherColor, Point, BoardSize} from './board'

function getElement(id: string) {
  return document.getElementById(id) as HTMLElement;
}

function querySelector(selector: string) {
  return document.querySelector(selector) as HTMLElement;
}

function parseGtpPoint(gtpCoord: string, N: BoardSize): Point | 'pass' | 'resign' {
  if (gtpCoord == 'pass' || gtpCoord == 'resign') {
    return gtpCoord;
  }
  let col = gtpCoord.charCodeAt(0) - 65;
  if (col >= 8) {
    --col;
  }
  let row = N - parseInt(gtpCoord.slice(1), 10);
  return {row: row, col: col};
}

function parseVariation(str: string, N: BoardSize, toPlay: Color) {
  if (str.trim() == '') {
    return [];
  }
  let moves = str.split(' ');
  let variation = [];
  let color = toPlay;
  for (let move of moves) {
    let p = parseGtpPoint(move, N)
    if (p != 'pass' && p != 'resign') {
      variation.push({p: p, color: color});
    }
    color = otherColor(color);
  }
  return variation;
}

export {
  getElement,
  parseGtpPoint,
  parseVariation,
  querySelector,
}
