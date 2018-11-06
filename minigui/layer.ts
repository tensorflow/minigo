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

import {Annotation} from './position'
import {BoardSize, COL_LABELS, Color, Coord, Point, Move, N, Nullable, movesEqual, otherColor} from './base'
import {Board} from './board'
import {pixelRatio} from './util'

const STAR_POINTS = {
  [BoardSize.Nine]: [[2, 2], [2, 6], [6, 2], [6, 6], [4, 4]],
  [BoardSize.Nineteen]: [[3, 3], [3, 9], [3, 15],
                         [9, 3], [9, 9], [9, 15],
                         [15, 3], [15, 9], [15, 15]],
};

interface DataObj {
  [key: string]: any;
}

type DataProp<T> = T | undefined;

abstract class Layer {
  private _show = true;

  get show() {
    return this._show;
  }
  set show(x: boolean) {
    if (x != this._show) {
      this._show = x;
      this.board.draw();
    }
  }

  board: Board;
  protected boardToCanvas: (row: number, col: number) => Coord;

  addToBoard(board: Board) {
    this.board = board;
    this.boardToCanvas = board.boardToCanvas.bind(board);
  }

  abstract clear(): void;

  // Returns true if dataObj contained updated data for the layer and it should
  // be redrawn.
  abstract update(dataObj: DataObj): boolean;

  abstract draw(): void;
}

abstract class StaticLayer extends Layer {
  clear() {}

  update(dataObj: DataObj) {
    return false;
  }
}

abstract class DataLayer extends Layer {
  protected getData<T>(obj: any, propName: string): DataProp<T> {
    let prop: any = obj[propName];
    if (prop === undefined) {
      return undefined;
    }
    return prop as T;
  }
}

class Grid extends StaticLayer {
  private style = '#864';

  draw() {
    let starPointRadius = Math.min(4, Math.max(this.board.stoneRadius / 5, 2.5));
    let ctx = this.board.ctx;
    let pr = pixelRatio();

    ctx.strokeStyle = this.style;
    ctx.lineWidth = pr;
    ctx.lineCap = 'round';

    ctx.beginPath();
    for (let i = 0; i < N; ++i) {
      let left = this.boardToCanvas(i, 0);
      let right = this.boardToCanvas(i, N - 1);
      let top = this.boardToCanvas(0, i);
      let bottom = this.boardToCanvas(N - 1, i);
      ctx.moveTo(0.5 + Math.round(left.x), 0.5 + Math.round(left.y));
      ctx.lineTo(0.5 + Math.round(right.x), 0.5 + Math.round(right.y));
      ctx.moveTo(0.5 + Math.round(top.x), 0.5 + Math.round(top.y));
      ctx.lineTo(0.5 + Math.round(bottom.x), 0.5 + Math.round(bottom.y));
    }
    ctx.stroke();

    // Draw star points.
    ctx.fillStyle = this.style;
    ctx.strokeStyle = '';
    for (let p of STAR_POINTS[N]) {
      let c = this.boardToCanvas(p[0], p[1]);
      ctx.beginPath();
      ctx.arc(c.x + 0.5, c.y + 0.5, starPointRadius * pr, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

class Label extends StaticLayer {
  draw() {
    let ctx = this.board.ctx;

    let textHeight = Math.floor(0.6 * this.board.stoneRadius);
    ctx.font = `${textHeight}px sans-serif`;
    ctx.fillStyle = '#9d7c4d';

    // Draw column labels.
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    for (let i = 0; i < N; ++i) {
      let c = this.boardToCanvas(-0.66, i);
      ctx.fillText(COL_LABELS[i], c.x, c.y);
    }
    ctx.textBaseline = 'top';
    for (let i = 0; i < N; ++i) {
      let c = this.boardToCanvas(N - 0.33, i);
      ctx.fillText(COL_LABELS[i], c.x, c.y);
    }

    // Draw row labels.
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < N; ++i) {
      let c = this.boardToCanvas(i, -0.66);
      ctx.fillText((N - i).toString(), c.x, c.y);
    }
    ctx.textAlign = 'left';
    for (let i = 0; i < N; ++i) {
      let c = this.boardToCanvas(i, N - 0.33);
      ctx.fillText((N - i).toString(), c.x, c.y);
    }
  }
}

class Caption extends StaticLayer {
  constructor(public caption: string) {
    super();
  }

  draw() {
    let ctx = this.board.ctx;

    let textHeight = Math.floor(0.8 * this.board.stoneRadius);
    ctx.font = `${textHeight}px sans-serif`;
    ctx.fillStyle = '#9d7c4d';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    let c = this.boardToCanvas(N - 0.45, (N - 1) / 2);
    ctx.fillText(this.caption, c.x, c.y);
  }
}

class HeatMap extends DataLayer {
  private colors: Nullable<Float32Array[]> = null;

  constructor(private dataPropName: string,
      private colorizeFn: (src: number[] | Float32Array) => Float32Array[]) {
    super();
  }

  clear() {
    if (this.colors) {
      this.colors = null;
      this.board.draw();
    }
  }

  update(dataObj: DataObj) {
    let data = this.getData<Nullable<number[]>>(dataObj, this.dataPropName);
    if (data === undefined) {
      return false;
    }
    this.colors = data != null ? this.colorizeFn(data) : null;
    return true;
  }

  draw() {
    if (!this.colors) {
      return;
    }

    let ctx = this.board.ctx;
    let w = this.board.pointW;
    let h = this.board.pointH;
    let stones = this.board.stones;
    let p = {row: 0, col: 0};
    let i = 0;
    for (p.row = 0; p.row < N; ++p.row) {
      for (p.col = 0; p.col < N; ++p.col) {
        let rgba = this.colors[i];
        if (stones[i++] != Color.Empty) {
          continue;
        }
        ctx.fillStyle = `rgba(${rgba[0]}, ${rgba[1]}, ${rgba[2]}, ${rgba[3]}`;
        let c = this.boardToCanvas(p.row, p.col);
        ctx.fillRect(c.x - 0.5 * w, c.y - 0.5 * h, w, h);
      }
    }
  }
}

abstract class StoneBaseLayer extends DataLayer {
  protected blackStones: Point[] = [];
  protected whiteStones: Point[] = [];

  constructor(protected alpha: number) {
    super();
  }

  clear() {
    if (this.blackStones.length > 0 || this.whiteStones.length > 0) {
      this.blackStones = [];
      this.whiteStones = [];
      this.board.draw();
    }
  }

  draw() {
    this.board.drawStones(this.blackStones, Color.Black, this.alpha);
    this.board.drawStones(this.whiteStones, Color.White, this.alpha);
  }
}

class BoardStones extends StoneBaseLayer {
  constructor() {
    super(1);
  }

  update(dataObj: DataObj) {
    let stones = this.getData<Color[]>(dataObj, 'stones');
    if (stones === undefined) {
      return false;
    }

    this.blackStones = [];
    this.whiteStones = [];
    if (stones != null) {
      let i = 0;
      for (let row = 0; row < N; ++row) {
        for (let col = 0; col < N; ++col) {
          let color = stones[i++];
          if (color == Color.Black) {
            this.blackStones.push({row: row, col: col});
          } else if (color == Color.White) {
            this.whiteStones.push({row: row, col: col});
          }
        }
      }
    }
    return true;
  }
}

interface VariationLabel {
  p: Point;
  s: string;
}

class Variation extends StoneBaseLayer {
  private _childVariation: Nullable<Move> = null;
  get childVariation() {
    return this._childVariation;
  }
  set childVariation(p: Nullable<Move>) {
    if (!movesEqual(p, this._childVariation)) {
      this._childVariation = p;
      this.board.draw();
    }
  }

  private blackLabels: VariationLabel[] = [];
  private whiteLabels: VariationLabel[] = [];

  constructor(private dataPropName: string, alpha = 0.4) {
    super(alpha);
  }

  clear() {
    super.clear();
    this.childVariation = null;
  }

  update(dataObj: DataObj) {
    let variation = this.getData<Move[]>(dataObj, this.dataPropName);
    if (variation === undefined) {
      return false;
    }

    this.parseVariation(variation);

    // We assume here that every update contains a new variation.
    // The search variation will by definition be different every time and
    // the engine only sends a principle variation when it changes, so this is
    // currently a safe assumption.
    // TODO(tommadams): return false if the varation was previously empty.
    return true;
  }

  protected parseVariation(variation: Nullable<Move[]>) {
    let toPlay = this.board.toPlay;
    this.blackStones = [];
    this.whiteStones = [];
    this.blackLabels = [];
    this.whiteLabels = [];

    if (variation == null || variation.length == 0) {
      return;
    }

    if (!movesEqual(variation[0], this.childVariation)) {
      return;
    }

    // The playedCount array keeps track of the number of times each point on
    // the board is played within the variation. For points that are played more
    // we only show the earliest move and mark it with an asterisk.
    let playedCount = new Uint16Array(N * N);
    let firstPlayed: VariationLabel[] = [];

    toPlay = otherColor(toPlay);
    for (let i = 0; i < variation.length; ++i) {
      let move = variation[i];

      toPlay = otherColor(toPlay);
      if (move == 'pass' || move == 'resign') {
        continue;
      }

      let idx = move.row * N + move.col;
      let label = {p: move, s: (i + 1).toString()};
      let count = ++playedCount[idx];
      if (toPlay == Color.Black) {
        this.blackStones.push(move);
        if (count == 1) {
          this.blackLabels.push(label);
        }
      } else {
        this.whiteStones.push(move);
        if (count == 1) {
          this.whiteLabels.push(label);
        }
      }
      if (count == 1) {
        firstPlayed[idx] = label;
      } else if (count == 2) {
        firstPlayed[idx].s += '*';
      }
    }
  }

  draw() {
    super.draw()

    let ctx = this.board.ctx;

    let textHeight = Math.floor(this.board.stoneRadius);
    ctx.font = `${textHeight}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    this.drawLabels(this.blackLabels, '#fff');
    this.drawLabels(this.whiteLabels, '#000');
  }

  protected drawLabels(labels: VariationLabel[], style: string) {
    let ctx = this.board.ctx;
    ctx.fillStyle = style;
    for (let label of labels) {
      let c = this.boardToCanvas(label.p.row, label.p.col);
      ctx.fillText(label.s, c.x, c.y);
    }
  }
}

class Annotations extends DataLayer {
  private annotations: Annotation[] = [];

  constructor(private dataPropName = 'annotations') {
    super();
  }

  clear() {
    if (this.annotations.length > 0) {
      this.annotations = [];
      this.board.draw();
    }
  }

  update(dataObj: DataObj) {
    let annotations = this.getData<Annotation[]>(dataObj, this.dataPropName);
    if (annotations === undefined) {
      return false;
    }
    this.annotations = annotations != null ? annotations : [];
    return true;
  }

  draw() {
    if (this.annotations == null || this.annotations.length == 0) {
      return;
    }

    let ctx = this.board.ctx;
    for (let annotation of this.annotations) {
      let c = this.boardToCanvas(annotation.p.row, annotation.p.col);
      let sr = this.board.stoneRadius;
      switch (annotation.shape) {
        case Annotation.Shape.Dot:
          ctx.fillStyle = annotation.color;
          ctx.beginPath();
          ctx.arc(c.x, c.y, 0.16 * sr, 0, 2 * Math.PI);
          ctx.fill();
          break;

        case Annotation.Shape.Triangle:
          ctx.lineWidth = 3 * pixelRatio();
          ctx.lineCap = 'round';
          ctx.strokeStyle = annotation.color;
          ctx.beginPath();
          ctx.moveTo(c.x, c.y - 0.7 * sr);
          ctx.lineTo(c.x - 0.6 * sr, c.y + 0.42 * sr);
          ctx.lineTo(c.x + 0.6 * sr, c.y + 0.42 * sr);
          ctx.lineTo(c.x, c.y - 0.7 * sr);
          ctx.stroke();
          break;
      }
    }
  }
}

class NextMove {
  p: Point;
  constructor(idx: number, public n: number, public q: number,
              public alpha: number) {
    this.p = {
      row: Math.floor(idx / N),
      col: idx % N,
    };
  }
}

class Q extends DataLayer {
  private nextMoves: NextMove[] = [];

  hasPoint(p: Point) {
    for (let move of this.nextMoves) {
      if (movesEqual(p, move.p)) {
        return true;
      }
    }
    return false;
  }

  clear() {
    if (this.nextMoves.length > 0) {
      this.nextMoves = [];
      this.board.draw();
    }
  }

  update(dataObj: DataObj) {
    let childN = this.getData<number[]>(dataObj, 'n');
    let childQ = this.getData<number[]>(dataObj, 'childQ');
    if (childN == null || childQ == null) {
      return false;
    }

    this.nextMoves = [];

    // Build a list of indices into childN & childQ sorted in descending N
    // then descending Q.
    let indices = [];
    for (let i = 0; i < N * N; ++i) {
      indices.push(i);
    }
    indices.sort((a: number, b: number) => {
      let n = childN as number[];
      let q = childQ as number[];
      if (n[b] != n[a]) {
        return n[b] - n[a];
      }
      return q[b] - q[a];
    });

    // We haven't done any reads yet.
    let maxN = childN[indices[0]];
    if (maxN == 0) {
      return true;
    }

    let sumN = 0;
    for (let n of childN) {
      sumN += n;
    }

    let logMaxN = Math.log(maxN);

    // Build the list of suggested next moves.
    // Limit the maximum number of suggestions to 9.
    let idx = indices[0];
    for (let i = 0; i < indices.length; ++i) {
      let idx = indices[i];
      let n = childN[idx];
      if (n == 0) {
        break;
      }
      let q = childQ[idx] / 10;
      let alpha = Math.log(n) / logMaxN;
      alpha *= alpha;
      if (n < sumN / 100) {
        break;
      }
      this.nextMoves.push(new NextMove(idx, n, q, alpha));
    }

    return true;
  }

  draw() {
    if (this.nextMoves.length == 0) {
      return;
    }

    let ctx = this.board.ctx;
    let pr = pixelRatio();

    let stoneRgb = this.board.toPlay == Color.Black ? 0 : 255;
    let textRgb = 255 - stoneRgb;

    for (let nextMove of this.nextMoves) {
      ctx.fillStyle =
          `rgba(${stoneRgb}, ${stoneRgb}, ${stoneRgb}, ${nextMove.alpha})`;
      let c = this.boardToCanvas(nextMove.p.row, nextMove.p.col);
      ctx.beginPath();
      ctx.arc(c.x + 0.5, c.y + 0.5, this.board.stoneRadius, 0, 2 * Math.PI);
      ctx.fill();
    }

    let textHeight = Math.floor(0.8 * this.board.stoneRadius);
    ctx.font = `${textHeight}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = `rgba(${textRgb}, ${textRgb}, ${textRgb}, 0.8)`;
    let scoreScale = this.board.toPlay == Color.Black ? 1 : -1;
    for (let nextMove of this.nextMoves) {
      let c = this.boardToCanvas(nextMove.p.row, nextMove.p.col);
      let winRate = (scoreScale * nextMove.q + 100) / 2;
      ctx.fillText(winRate.toFixed(1), c.x, c.y);
    }
  }
}

export {
  Annotation,
  Annotations,
  BoardStones,
  Caption,
  DataObj,
  Grid,
  HeatMap,
  Label,
  Layer,
  Q,
  Variation,
}
