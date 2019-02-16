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

import {Color, Move, N, Nullable, Point, moveIsPoint, movesEqual, otherColor, stonesEqual, toGtp} from './base'
import * as util from './util'

namespace Annotation {
  export enum Shape {
    Dot,
  }
}

interface Annotation {
  p: Point;
  shape: Annotation.Shape;
  colors: string[];
}

class Position {
  id: string;
  parent: Nullable<Position> = null;
  stones: Color[];
  lastMove: Nullable<Move> = null;
  toPlay: Color;
  gameOver: boolean;
  isMainLine = true;

  n = 0;
  q: Nullable<number> = null;
  moveNum: number;

  // A map of variations.
  // The principal variation is keyed by "pv".
  // The current live tree search is keyed by "live".
  // All other variations are keyed by their GTP string.
  variations = new Map<string, Position.Variation>();

  annotations: Annotation[] = [];
  childN: Nullable<number[]> = null;
  childQ: Nullable<number[]> = null;

  // children[0] is the main line. Subsequent children are variations.
  children: Position[] = [];

  // captures[0] is the number of stones that black has captured.
  // captures[1] is the number of stones that white has captured.
  captures: number[] = [0, 0];

  comment = "";

  treeStats: Position.TreeStats = {
    numNodes: 0,
    numLeafNodes: 0,
    maxDepth: 0,
  }

  constructor(j: Position.Definition) {
    this.id = j.id;
    this.moveNum = j.moveNum;
    this.toPlay = util.parseColor(j.toPlay);
    this.stones = [];
    if (j.stones !== undefined) {
      const stoneMap: {[index: string]: Color} = {
        '.': Color.Empty,
        'X': Color.Black,
        'O': Color.White,
      };
      for (let i = 0; i < N * N; ++i) {
        this.stones.push(stoneMap[j.stones[i]]);
      }
    } else {
      for (let i = 0; i < N * N; ++i) {
        this.stones.push(Color.Empty);
      }
    }
    if (j.move) {
      this.lastMove = util.parseMove(j.move);
    }
    this.gameOver = j.gameOver || false;
    this.moveNum = j.moveNum;
    if (j.comment) {
      this.comment = j.comment;
    }
    if (j.caps !== undefined) {
      this.captures[0] = j.caps[0];
      this.captures[1] = j.caps[1];
    }

    if (moveIsPoint(this.lastMove)) {
      this.annotations.push({
        p: this.lastMove,
        shape: Annotation.Shape.Dot,
        colors: ['#ef6c02'],
      });
    }
  }

  addChild(p: Position) {
    if (p.lastMove == null) {
      throw new Error('Child nodes shouldn\'t have a null lastMove');
    }
    if (p.parent != null) {
      throw new Error('Node already has a parent');
    }

    // If the position already has a child with the given move, verify that the
    // stones are equal and return the existing child.
    for (let child of this.children) {
      if (movesEqual(child.lastMove, p.lastMove)) {
        throw new Error(`Position already has child ${toGtp(p.lastMove)}`);
      }
    }

    // Create a new child.
    p.isMainLine = this.isMainLine && this.children.length == 0;
    p.parent = this;
    this.children.push(p);
  }

  getChild(move: Move) {
    for (let child of this.children) {
      if (movesEqual(child.lastMove, move)) {
        return child;
      }
    }
    return null;
  }

  update(update: Position.Update) {
    if (update.n !== undefined) { this.n = update.n; }
    if (update.q !== undefined) { this.q = update.q; }
    if (update.childN !== undefined) { this.childN = update.childN; }
    if (update.childQ !== undefined) {
      this.childQ = [];
      for (let q of update.childQ) { this.childQ.push(q / 1000); }
    }
    if (update.treeStats !== undefined) { this.treeStats = update.treeStats; }
    if (update.variations !== undefined) {
      this.variations.clear();
      let pv: Nullable<Position.Variation> = null;
      for (let key in update.variations) {
        let variation = {
          n: update.variations[key].n,
          q: update.variations[key].q,
          moves: util.parseMoves(update.variations[key].moves),
        };
        this.variations.set(key, variation);
        if (pv == null || variation.n > pv.n) {
          pv = variation;
        }
      }
      if (pv != null) {
        this.variations.set("pv", pv);
      }
    }
  }

  // Returns a copy of the complete variation that this position is a part of,
  // starting from the root, down to the last move in the line.
  // The varation of the root node is its main line.
  getFullLine() {
    // Get all ancestors.
    let result: Position[] = [];
    let node: Nullable<Position>
    for (node = this.parent; node != null; node = node.parent) {
      result.push(node);
    }
    result.reverse();

    // Append descendants.
    for (node = this; node != null; node = node.children[0]) {
      result.push(node);
    }

    return result;
  }
}

namespace Position {
  // Definition of a Position
  export interface Definition {
    id: string;
    parentId?: string;
    moveNum: number;
    toPlay: string;
    stones?: string;
    gameOver?: boolean;
    move?: string;
    comment?: string;
    caps?: number[];
  }

  export interface Variation {
    n: number;
    q: number;
    moves: Move[];
  };

  export interface TreeStats {
    numNodes: number;
    numLeafNodes: number;
    maxDepth: number;
  }

  export interface Update {
    id: string;
    n?: number;
    q?: number;
    childN?: number[];
    childQ?: number[];
    treeStats?: TreeStats;
    variations?: {
      [index:string]: {
        n: number;
        q: number;
        moves: string[];
      }
    };
  }
}

export {
  Annotation,
  Position,
};
