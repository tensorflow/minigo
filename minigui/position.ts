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

import {Color, Move, Nullable, Point, movesEqual, otherColor, stonesEqual} from './base'
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
  q = 0;
  moveNum: number;
  search: Move[] = [];
  pv: Move[] = [];
  // TODO(tommadams): remove dq
  dq: Nullable<number[]> = null;
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
              public isMainline: boolean) {
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
          throw new Error(`Position has child ${move} with different stones`);
        }
        return child;
      }
    }

    // Create a new child.
    let isMainline = this.isMainline && this.children.length == 0;
    let child = new Position(id, this, stones, move, otherColor(this.toPlay),
                             gameOver, isMainline);
    this.children.push(child);
    return child;
  }

  update(update: Position.Update) {
    // TODO(tommadams): It would be more flexible to allow the position
    // to store any/all properties return in the SearchMsg, without having to
    // specify this property list.
    const props = ['n', 'q', 'dq', 'pv', 'search', 'childN', 'childQ'];
    partialUpdate(update, this, props);
  }
}


namespace Position {
  export interface Update {
    n?: number;
    q?: number;
    pv?: Move[];
    search?: Move[];
    childN?: number[];
    childQ?: number[];
    dq?: number[];  // TODO(tommadams): remove this
  }
}

export {
  Annotation,
  Position,
};
