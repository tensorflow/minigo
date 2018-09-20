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

import {BoardSize, Color, otherColor, Coord, Point, Move} from './base'
import {Annotation, Board, COL_LABELS} from './board'
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

type DataProp<T> = T | undefined | null;

abstract class Layer {
  protected boardToCanvas: (row: number, col: number) => Coord;

  constructor(public board: Board) {
    this.boardToCanvas = board.boardToCanvas.bind(board);
  }

  // Returns true if dataObj contained updated data for the layer and it should
  // be redrawn.
  abstract update(dataObj: DataObj): boolean;

  abstract draw(): void;
}

abstract class StaticLayer extends Layer {
  update(dataObj: DataObj) {
    return false;
  }
}

abstract class DataLayer extends Layer {
  constructor(board: Board, private dataPropName: string) {
    super(board);
  }

  protected getData<T>(dataObj: any): DataProp<T> {
    let prop: any = dataObj[this.dataPropName];
    if (prop === undefined) {
      return undefined;
    }
    return prop as T | null;
  }
}

class Grid extends StaticLayer {
  private style = '#864';

  draw() {
    let starPointRadius = Math.min(4, Math.max(this.board.stoneRadius / 10, 2.5));
    let ctx = this.board.ctx;
    let size = this.board.size;
    let pr = pixelRatio();

    ctx.strokeStyle = this.style;
    ctx.lineWidth = pr;
    ctx.lineCap = 'round';

    ctx.beginPath();
    for (let i = 0; i < size; ++i) {
      let left = this.boardToCanvas(i, 0);
      let right = this.boardToCanvas(i, size - 1);
      let top = this.boardToCanvas(0, i);
      let bottom = this.boardToCanvas(size - 1, i);
      ctx.moveTo(0.5 + Math.round(left.x), 0.5 + Math.round(left.y));
      ctx.lineTo(0.5 + Math.round(right.x), 0.5 + Math.round(right.y));
      ctx.moveTo(0.5 + Math.round(top.x), 0.5 + Math.round(top.y));
      ctx.lineTo(0.5 + Math.round(bottom.x), 0.5 + Math.round(bottom.y));
    }
    ctx.stroke();

    // Draw star points.
    ctx.fillStyle = this.style;
    ctx.strokeStyle = '';
    for (let p of STAR_POINTS[size]) {
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
    let size = this.board.size;

    let textHeight = Math.floor(0.3 * this.board.stoneRadius);
    ctx.font = `${textHeight}px sans-serif`;
    ctx.fillStyle = '#9d7c4d';

    // Draw column labels.
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    for (let i = 0; i < size; ++i) {
      let c = this.boardToCanvas(-0.66, i);
      ctx.fillText(COL_LABELS[i], c.x, c.y);
    }
    ctx.textBaseline = 'top';
    for (let i = 0; i < size; ++i) {
      let c = this.boardToCanvas(size - 0.33, i);
      ctx.fillText(COL_LABELS[i], c.x, c.y);
    }

    // Draw row labels.
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < size; ++i) {
      let c = this.boardToCanvas(i, -0.66);
      ctx.fillText((size - i).toString(), c.x, c.y);
    }
    ctx.textAlign = 'left';
    for (let i = 0; i < size; ++i) {
      let c = this.boardToCanvas(i, size - 0.33);
      ctx.fillText((size - i).toString(), c.x, c.y);
    }
  }
}

class Caption extends StaticLayer {
  constructor(board: Board, private caption: string) {
    super(board);
  }

  draw() {
    let ctx = this.board.ctx;
    let size = this.board.size;

    let textHeight = Math.floor(0.4 * this.board.stoneRadius);
    ctx.font = `${textHeight}px sans-serif`;
    ctx.fillStyle = '#9d7c4d';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    let c = this.boardToCanvas(size - 0.45, (size - 1) / 2);
    ctx.fillText(this.caption, c.x, c.y);
  }
}

class HeatMap extends DataLayer {
  private colors: Float32Array[] | null = null;

  constructor(board: Board, dataPropName: string,
              private colorizeFn: (src: number[]) => Float32Array[]) {
    super(board, dataPropName);
  }

  update(dataObj: DataObj) {
    let data = this.getData<number[] | null>(dataObj);
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
    let size = this.board.size;
    let w = this.board.pointW;
    let h = this.board.pointH;
    let stones = this.board.stones;
    let p = {row: 0, col: 0};
    let i = 0;
    for (p.row = 0; p.row < size; ++p.row) {
      for (p.col = 0; p.col < size; ++p.col) {
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

  constructor(board: Board, dataPropName: string, protected alpha: number) {
    super(board, dataPropName);
  }

  draw() {
    this.board.drawStones(this.blackStones, Color.Black, this.alpha);
    this.board.drawStones(this.whiteStones, Color.White, this.alpha);
  }
}

class BoardStones extends StoneBaseLayer {
  constructor(board: Board, dataPropName = 'stones', alpha = 1) {
    super(board, dataPropName, alpha);
  }

  update(dataObj: DataObj) {
    let stones = this.getData<Color[]>(dataObj);
    if (stones === undefined) {
      return false;
    }

    this.blackStones = [];
    this.whiteStones = [];
    if (stones != null) {
      let size = this.board.size;
      let i = 0;
      for (let row = 0; row < size; ++row) {
        for (let col = 0; col < size; ++col) {
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
  private blackLabels: VariationLabel[] = [];
  private whiteLabels: VariationLabel[] = [];

  constructor(board: Board, dataPropName: string, alpha = 0.4) {
    super(board, dataPropName, alpha);
  }

  update(dataObj: DataObj) {
    let variation = this.getData<Move[]>(dataObj);
    if (variation === undefined) {
      return false;
    }

    let toPlay = this.board.toPlay;
    let size = this.board.size;
    this.blackStones = [];
    this.whiteStones = [];
    this.blackLabels = [];
    this.whiteLabels = [];

    if (variation == null) {
      // We assume here that every update contains a new variation.
      // The search variation will by definition be different every time and
      // the engine only sends a principle variation when it changes, so this is
      // currently a safe assumption.
      // TODO(tommadams): return false if the varation was previously empty.
      return true;
    }

    // The playedCount array keeps track of the number of times each point on
    // the board is played within the variation. For points that are played more
    // we only show the earliest move and mark it with an asterisk.
    let playedCount = new Uint16Array(size * size);
    let firstPlayed: VariationLabel[] = [];

    toPlay = otherColor(toPlay);
    for (let i = 0; i < variation.length; ++i) {
      let move = variation[i];

      toPlay = otherColor(toPlay);
      if (move == 'pass' || move == 'resign') {
        continue;
      }

      let idx = move.row * size + move.col;
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

    return true;
  }

  draw() {
    super.draw()

    let ctx = this.board.ctx;
    let size = this.board.size;

    let textHeight = Math.floor(0.5 * this.board.stoneRadius);
    ctx.font = `${textHeight}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    this.drawLabels(this.blackLabels, '#fff');
    this.drawLabels(this.whiteLabels, '#000');
  }

  private drawLabels(labels: VariationLabel[], style: string) {
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

  constructor(board: Board, dataPropName = 'annotations') {
    super(board, dataPropName);
  }

  update(dataObj: DataObj) {
    let annotations = this.getData<Annotation[]>(dataObj);
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
          ctx.arc(c.x, c.y, 0.08 * sr, 0, 2 * Math.PI);
          ctx.fill();
          break;

        case Annotation.Shape.Triangle:
          ctx.lineWidth = 3 * pixelRatio();
          ctx.lineCap = 'round';
          ctx.strokeStyle = annotation.color;
          ctx.beginPath();
          ctx.moveTo(c.x, c.y - 0.35 * sr);
          ctx.lineTo(c.x - 0.3 * sr, c.y + 0.21 * sr);
          ctx.lineTo(c.x + 0.3 * sr, c.y + 0.21 * sr);
          ctx.lineTo(c.x, c.y - 0.35 * sr);
          ctx.stroke();
          break;
      }
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
  Variation,
}
