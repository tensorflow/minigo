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

import {Color, Move, Nullable, Point, movesEqual, otherColor, stonesEqual, toKgs} from './base'
import {partialUpdate} from './util'

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
  n = 0;
  q: Nullable<number> = null;
  moveNum: number;
  search: Move[] = [];

  // A map of variations.
  // The pricinpal variation is keyed by "pv".
  // The current tree search is keyed by "search".
  // All other variations are keyed by their KGS string.
  variations = new Map<string, Move[]>();

  annotations: Annotation[] = [];
  childN: Nullable<number[]> = null;
  childQ: Nullable<number[]> = null;

  // children[0] is the main line. Subsequent children are variations.
  children: Position[] = [];

  constructor(public id: string,
              public parent: Nullable<Position>,
              public stones: Color[],
              public lastMove: Nullable<Move>,
              public toPlay: Color,
              public gameOver: boolean,
              public isMainLine: boolean) {
    this.moveNum = parent != null ? parent.moveNum + 1 : 0;
    if (lastMove != null && lastMove != 'pass' && lastMove != 'resign') {
      this.annotations.push({
        p: lastMove,
        shape: Annotation.Shape.Dot,
        colors: ['#ef6c02'],
      });
    }
  }

  addChild(id: string, move: Move, stones: Color[], gameOver: boolean) {
    // If the position already has a child with the given move, verify that the
    // stones are equal and return the existing child.
    for (let child of this.children) {
      if (child.lastMove == null) {
        throw new Error('Child node shouldn\'t have a null lastMove');
      }
      if (movesEqual(child.lastMove, move)) {
        if (!stonesEqual(stones, child.stones)) {
          throw new Error(`Position has child ${toKgs(move)} with different stones`);
        }
        return child;
      }
    }

    // Create a new child.
    let isMainLine = this.isMainLine && this.children.length == 0;
    let child = new Position(id, this, stones, move, otherColor(this.toPlay),
                             gameOver, isMainLine);
    this.children.push(child);
    return child;
  }

  update(update: Position.Update) {
    // Update simple properties.
    const props = ['n', 'q', 'childN', 'childQ'];
    partialUpdate(update, this, props);

    // Variations need special handling.
    if (update.variations != null) {
      for (let key in update.variations) {
        this.variations.set(key, update.variations[key]);
      }
      // If the update has a principal variation also update the variation
      // at it's first move.
      if ("pv" in update.variations) {
        let pv = update.variations["pv"];
        if (pv.length > 0) {
          this.variations.set(toKgs(pv[0]), pv);
        }
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
  export interface Update {
    // Number of reads under this position.
    n?: number;

    // This position's Q score.
    q?: number;

    // A map of variations.
    // The pricinpal variation is keyed by "pv".
    // The current tree search is keyed by "search".
    // All other variations are keyed by their KGS string.
    variations?: {[index: string]: Move[]};

    // Child visit counts.
    childN?: number[];

    // Child Q scores.
    childQ?: number[];
  }
}

export {
  Annotation,
  Position,
};
