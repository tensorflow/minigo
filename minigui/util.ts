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

import {BoardSize, Color, otherColor, Move, N} from './base'

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

function emptyBoard(): Color[] {
  let result = new Array<Color>(N * N);
  result.fill(Color.Empty);
  return result;
}

function partialUpdate(src: any, dst: any, propNames: string[]) {
  for (let name of propNames) {
    if (src[name] != null) {
      dst[name] = src[name];
    }
  }
  return dst;
}

// Converts a result like "B+R" or "W+3.5" to a human-readable string like
// "Black wins by resignation" or "White wins by 3.5 points".
function toPrettyResult(result: string) {
  let prettyResult: string;
  if (result[0] == 'W') {
    prettyResult = 'White wins by ';
  } else {
    prettyResult = 'Black wins by ';
  }
  if (result[2] == 'R') {
    prettyResult += 'resignation';
  } else {
    prettyResult += result.substr(2) + ' points';
  }
  return prettyResult;
}

export {
  emptyBoard,
  getElement,
  parseGtpColor,
  parseGtpMove,
  parseMoves,
  partialUpdate,
  pixelRatio,
  querySelector,
  toPrettyResult,
}
