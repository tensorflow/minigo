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

const COL_LABELS = 'ABCDEFGHJKLMNOPQRST';

enum Color {
  Empty,
  Black,
  White,
}

function otherColor(color: Color) {
  if (color != Color.White && color != Color.Black) {
    throw new Error(`invalid color ${color}`);
  }
  return color == Color.White ? Color.Black : Color.White;
}

function gtpColor(color: Color) {
  if (color != Color.White && color != Color.Black) {
    throw new Error(`invalid color ${color}`);
  }
  return color == Color.Black ? 'b' : 'w'
}

function stonesEqual(a: Color[], b: Color[]) {
  if (a.length != b.length) {
    throw new Error(
      `Expected arrays of equal length, got lengths ${a.length} & ${b.length}`);
  }
  for (let i = 0; i < a.length; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

enum BoardSize {
  Nine = 9,
  Nineteen = 19,
}

let N = BoardSize.Nineteen;

function setBoardSize(size: number) {
  if (size == BoardSize.Nine || size == BoardSize.Nineteen) {
    N = size;
  } else {
    throw new Error(`Unsupported board size ${size}`);
  }
}

// Canvas coordinate.
interface Coord {
  x: number;
  y: number;
}

// Point on the board.
interface Point {
  row: number;
  col: number;
}

type Move = Point | 'pass' | 'resign';

function moveIsPoint(move: Nullable<Move>): move is Point {
  return move != null && move != 'pass' && move != 'resign';
}

function toGtp(move: Move) {
  if (move == 'pass' || move == 'resign') {
    return move;
  }
  let row = N - move.row;
  let col = COL_LABELS[move.col];
  return `${col}${row}`;
}

function movesEqual(a: Nullable<Move>, b: Nullable<Move>) {
  if (moveIsPoint(a) && moveIsPoint(b)) {
    return a.row == b.row && a.col == b.col;
  } else {
    return a == b;
  }
}

type Nullable<T> = T | null;

export {
  BoardSize,
  COL_LABELS,
  Color,
  Coord,
  Move,
  N,
  Nullable,
  Point,
  gtpColor,
  moveIsPoint,
  movesEqual,
  otherColor,
  setBoardSize,
  stonesEqual,
  toGtp,
}
