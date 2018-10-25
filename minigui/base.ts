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

enum BoardSize {
  Nine = 9,
  Nineteen = 19,
}

let N = BoardSize.Nineteen;

function setBoardSize(size: number) {
  switch (size) {
  }
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
  otherColor,
  setBoardSize,
}
