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

import {BoardSize, Color, otherColor, Move} from './base'

function getElement(id: string) {
  return document.getElementById(id) as HTMLElement;
}

function querySelector(selector: string) {
  return document.querySelector(selector) as HTMLElement;
}

function parseGtpColor(color: string) {
  let c = color[0].toLowerCase();
  return c == 'b' ? Color.Black : Color.White;
}

function parseGtpMove(gtpCoord: string, size: BoardSize): Move {
  if (gtpCoord == 'pass' || gtpCoord == 'resign') {
    return gtpCoord;
  }
  let col = gtpCoord.charCodeAt(0) - 65;
  if (col >= 8) {
    --col;
  }
  let row = size - parseInt(gtpCoord.slice(1), 10);
  return {row: row, col: col};
}

function parseMoves(moves: string[], size: BoardSize): Move[] {
  let variation: Move[] = [];
  for (let move of moves) {
    variation.push(parseGtpMove(move, size));
  }
  return variation;
}

function pixelRatio() {
  return window.devicePixelRatio || 1;
}

function emptyBoard(size: number): Color[] {
  let result: Color[] = [];
  result.fill(Color.Empty, size * size);
  return result;
}

export {
  emptyBoard,
  getElement,
  parseGtpColor,
  parseGtpMove,
  parseMoves,
  pixelRatio,
  querySelector,
}
