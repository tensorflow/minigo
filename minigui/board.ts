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

import {getElement} from './util'

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

interface Move {
  p: Point | null;
  color: Color;
}

const COL_LABELS = 'ABCDEFGHJKLMNOPQRST';

const STAR_POINTS = {
  [BoardSize.Nine]: [[2, 2], [2, 6], [6, 2], [6, 6], [4, 4]],
  [BoardSize.Nineteen]: [[3, 3], [3, 9], [3, 15],
                         [9, 3], [9, 9], [9, 15],
                         [15, 3], [15, 9], [15, 15]],
};

interface BoardOptions {
  // Caption to display at the bottom of the board (if any).
  caption?: string;

  // Margin in pixels to add to the board.
  // Default is 0;
  margin?: number;

  // Whether or not to draw labels for each row and column on the board.
  // Default is false.
  labelRowCol?: boolean;

  // Radius of the start points in pixels.
  // Default is 2.5.
  starPointRadius?: number;
}

class Board {
  toPlay = Color.Black;
  stones: Array<Color>;

  protected ctx: CanvasRenderingContext2D;
  protected pixelRatio: number;
  protected backgroundColor: string;
  protected lineColor: string;
  protected stoneRadius: number;
  protected pointW: number;
  protected pointH: number;
  protected caption: string | null;
  protected marks: Array<string | null>;
  protected variation = new Array<Move>();
  protected heatMap: Float32Array;
  protected labelRowCol: boolean;
  protected margin: number;
  protected starPointRadius: number;

  constructor(parent: HTMLElement | string, public size: BoardSize, options: BoardOptions = {}) {
    if (typeof(parent) == 'string') {
      parent = getElement(parent);
    }

    this.backgroundColor = '#db6';
    this.lineColor = '#864';
    this.caption = options.caption || '';
    this.labelRowCol = options.labelRowCol || false;
    this.margin = options.margin || 0;
    this.starPointRadius = options.starPointRadius || 2.5;
    this.heatMap = new Float32Array(this.size * this.size);

    this.stones = new Array<Color>(this.size * this.size);
    this.marks = new Array<string | null>(this.size * this.size);
    for (let i = 0; i < this.stones.length; ++i) {
      this.stones[i] = Color.Empty;
      this.marks[i] = null;
    }

    this.pixelRatio = window.devicePixelRatio || 1;

    let canvas = document.createElement('canvas');
    canvas.style.margin = `${this.margin}px`;
    canvas.width = this.pixelRatio * (parent.offsetWidth - 2 * this.margin);
    canvas.height = this.pixelRatio * (parent.offsetHeight - 2 * this.margin);
    canvas.style.width = `${parent.offsetWidth - 2 * this.margin}px`;
    canvas.style.height = `${parent.offsetHeight - 2 * this.margin}px`;
    this.ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

    this.pointW = canvas.width / (this.size + 1);
    this.pointH = canvas.height / (this.size + 1);
    this.stoneRadius = Math.min(this.pointW, this.pointH);

    parent.appendChild(canvas);
  }

  setStone(p: Point, color: Color) {
    this.stones[p.row * this.size + p.col] = color;
  }

  setStones(stones: Array<Color>) {
    for (let i = 0; i < stones.length; ++i) {
      this.stones[i] = stones[i];
    }
  }

  getStone(p: Point) {
    return this.stones[p.row * this.size + p.col];
  }

  // style is any valid canvas fill style.
  setMark(p: Point, style: string) {
    this.marks[p.row * this.size + p.col] = style;
  }

  clearMark(p: Point) {
    this.marks[p.row * this.size + p.col] = null;
  }

  clearMarks() {
    for (let i = 0; i < this.marks.length; ++i) {
      this.marks[i] = null;
    }
  }

  clearHeatMap() {
    this.heatMap.fill(0);
    this.draw();
  }

  setHeatMap(map: Array<number>) {
    for (let i = 0; i < map.length; ++i) {
      this.heatMap[i] = map[i];
    }
    this.draw();
  }

  setVariation(variation: Array<Move>) {
    this.variation = variation;
    this.draw();
  }

  clearVariation() {
    this.variation = [];
    this.draw();
  }

  draw() {
    this.drawBackground();
    this.drawLabels();
    this.drawHeatMap();
    this.drawBoardStones();
    this.drawMarks();
    this.drawVariation();
  }

  protected drawBackground() {
    let ctx = this.ctx;
    ctx.fillStyle = this.backgroundColor;
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  protected drawLabels() {
    let ctx = this.ctx;
    let n = this.size;

    // Draw lines.
    ctx.strokeStyle = this.lineColor;
    ctx.lineWidth = this.pixelRatio;
    ctx.lineCap = 'round';

    ctx.beginPath();
    for (let i = 0; i < n; ++i) {
      let left = this.boardToCanvas(i, 0);
      let right = this.boardToCanvas(i, n - 1);
      let top = this.boardToCanvas(0, i);
      let bottom = this.boardToCanvas(n - 1, i);
      ctx.moveTo(0.5 + left.x, 0.5 + left.y);
      ctx.lineTo(0.5 + right.x, 0.5 + right.y);
      ctx.moveTo(0.5 + top.x, 0.5 + top.y);
      ctx.lineTo(0.5 + bottom.x, 0.5 + bottom.y);
    }
    ctx.stroke();

    // Draw star points.
    ctx.fillStyle = this.lineColor;
    ctx.strokeStyle = '';
    for (let p of STAR_POINTS[n]) {
      let c = this.boardToCanvas(p[0], p[1]);
      ctx.beginPath();
      ctx.arc(c.x + 0.5, c.y + 0.5, this.starPointRadius * this.pixelRatio, 0, 2 * Math.PI);
      ctx.fill();
    }

    let textHeight = Math.floor(0.3 * this.stoneRadius);
    ctx.font = `${textHeight}px sans-serif`;

    if (this.labelRowCol) {
      ctx.fillStyle = '#9d7c4d';
      // Draw column labels.
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      for (let i = 0; i < n; ++i) {
        let c = this.boardToCanvas(-0.66, i);
        ctx.fillText(COL_LABELS[i], c.x, c.y);
      }
      ctx.textBaseline = 'top';
      for (let i = 0; i < n; ++i) {
        let c = this.boardToCanvas(this.size - 0.33, i);
        ctx.fillText(COL_LABELS[i], c.x, c.y);
      }

      // Draw row labels.
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let i = 0; i < n; ++i) {
        let c = this.boardToCanvas(i, -0.66);
        ctx.fillText((n - i).toString(), c.x, c.y);
      }
      ctx.textAlign = 'left';
      for (let i = 0; i < n; ++i) {
        let c = this.boardToCanvas(i, this.size - 0.33);
        ctx.fillText((n - i).toString(), c.x, c.y);
      }
    }

    // Draw board caption.
    if (this.caption) {
      textHeight = Math.floor(0.4 * this.stoneRadius);
      ctx.font = `${textHeight}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      let c = this.boardToCanvas(n - 0.45, (n - 1) / 2);
      ctx.fillText(this.caption, c.x, c.y);
    }
  }

  protected drawBoardStones() {
    let blackStones = new Array<Point>();
    let whiteStones = new Array<Point>();
    let i = 0;
    for (let row = 0; row < this.size; ++row) {
      for (let col = 0; col < this.size; ++col) {
        let color = this.stones[i++];
        if (color == Color.Black) {
          blackStones.push({row: row, col: col});
        } else if (color == Color.White) {
          whiteStones.push({row: row, col: col});
        }
      }
    }

    this.drawStones(blackStones, Color.Black, 1);
    this.drawStones(whiteStones, Color.White, 1);
  }

  protected drawHeatMap() {
    let ctx = this.ctx;
    let w = this.pointW;
    let h = this.pointH;
    let p = {row: 0, col: 0};
    let i = 0;
    for (p.row = 0; p.row < this.size; ++p.row) {
      for (p.col = 0; p.col < this.size; ++p.col) {
        let x = this.heatMap[i];
        if (this.stones[i++] != Color.Empty) {
          continue;
        }
        if (x < 0) {
          ctx.fillStyle = `rgba(255, 255, 255, ${-x}`;
        } else {
          ctx.fillStyle = `rgba(0, 0, 0, ${x}`;
        }
        let c = this.boardToCanvas(p.row, p.col);
        ctx.fillRect(c.x - 0.5 * w, c.y - 0.5 * h, w, h);
      }
    }
  }

  protected drawVariation() {
    // The playedCount array keeps track of the number of times each point on
    // the board is played within the variation. For points that are played more
    // we only show the earliest move and mark it with an asterisk.
    let playedCount = new Uint16Array(this.size * this.size);
    let firstPlayed = new Uint16Array(this.size * this.size);

    let ctx = this.ctx;
    let blackStones = new Array<Point>();
    let whiteStones = new Array<Point>();
    for (let i = 0; i < this.variation.length; ++i) {
      let move = this.variation[i];
      if (!move.p) {
        continue;
      }
      let idx = move.p.row * this.size + move.p.col;
      if (++playedCount[idx] > 1) {
        continue;
      }
      firstPlayed[idx] = i;
      if (move.color == Color.Black) {
        blackStones.push(move.p);
      } else {
        whiteStones.push(move.p);
      }
    }
    this.drawStones(blackStones, Color.Black, 0.4);
    this.drawStones(whiteStones, Color.White, 0.4);

    let textHeight = Math.floor(0.5 * this.stoneRadius);
    ctx.font = `${textHeight}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < this.variation.length; ++i) {
      let move = this.variation[i];
      if (!move.p) {
        continue;
      }
      let idx = move.p.row * this.size + move.p.col;
      if (firstPlayed[idx] != i) {
        continue;
      }

      // Draw move number in the center of each stone.
      ctx.fillStyle = move.color == Color.Black ? '#fff' : '#000';
      let c = this.boardToCanvas(move.p.row, move.p.col);
      let str = (i + 1).toString();
      if (playedCount[idx] > 1) {
        str += '*';
      }
      ctx.fillText(str, c.x, c.y);
    }
  }

  protected drawMarks() {
    let ctx = this.ctx;
    let w = 0.2 * this.pointW;
    let h = 0.2 * this.pointH;
    let i = 0;
    for (let row = 0; row < this.size; ++row) {
      for (let col = 0; col < this.size; ++col) {
        let style = this.marks[i++];
        if (style == null) {
          continue;
        }
        ctx.fillStyle = style;
        let c = this.boardToCanvas(row, col);
        ctx.beginPath();
        ctx.arc(c.x, c.y, 0.08 * this.stoneRadius, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }

  protected setShadow(blur: number, offset: number, alpha: number) {
    let ctx = this.ctx;
    ctx.shadowBlur = blur * this.pixelRatio;
    ctx.shadowOffsetX = offset * this.pixelRatio;
    ctx.shadowOffsetY = offset * this.pixelRatio;
    ctx.shadowColor = `rgba(0, 0, 0, ${alpha})`;
  }

  protected drawStones(ps: Array<Point>, color: Color, alpha: number) {
    if (ps.length == 0) {
      return;
    }
    let ctx = this.ctx;
    if (color != Color.Black && color != Color.White) {
      throw new Error(`Can't draw stones with color ${color}`);
    }

    if (alpha == 1) {
      this.setShadow(4, 1.5, color == Color.Black ? 0.4 : 0.3);
    }

    if (color == Color.Black) {
      ctx.fillStyle = this.blackFill(alpha);
    } else {
      ctx.fillStyle = this.whiteFill(alpha);
    }

    let r = 0.48 * this.stoneRadius;
    for (let p of ps) {
      let c = this.boardToCanvas(p.row, p.col);
      ctx.beginPath();
      ctx.translate(c.x, c.y);
      ctx.arc(0, 0, r, 0, 2 * Math.PI);
      ctx.fill();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
    }

    if (alpha == 1) {
      this.setShadow(0, 0, 0);
    }
  }

  protected canvasToBoard(x: number, y: number, threshold?: number): Point | null {
    y = y * (this.size + 1) / this.ctx.canvas.height - 0.5;
    x = x * (this.size + 1) / this.ctx.canvas.width - 0.5;
    let row = Math.floor(y);
    let col = Math.floor(x);
    if (row < 0 || row >= this.size || col < 0 || col >= this.size) {
      return null;
    }

    if (threshold) {
      let fx = 0.5 - (x - col);
      let fy = 0.5 - (y - row);
      let disSqr = fx * fx + fy * fy;
      if (disSqr > threshold * threshold) {
        return null;
      }
    }

    return {row: row, col: col};
  }

  protected boardToCanvas(row: number, col: number) {
    return {
      x: this.ctx.canvas.width * (col + 1.0) / (this.size + 1),
      y: this.ctx.canvas.height * (row + 1.0) / (this.size + 1)
    };
  }

  protected blackFill(alpha: number) {
    let ofs = -0.25 * this.stoneRadius;
    let grad = this.ctx.createRadialGradient(
        ofs, ofs, 0, ofs, ofs, this.stoneRadius);
    grad.addColorStop(0, `rgba(68, 68, 68, ${alpha})`);
    grad.addColorStop(1, `rgba(16, 16, 16, ${alpha})`);
    return grad;
  }

  protected whiteFill(alpha: number) {
    let ofs = -0.1 * this.stoneRadius;
    let grad = this.ctx.createRadialGradient(
        ofs, ofs, 0, ofs, ofs, 0.6 * this.stoneRadius);
    grad.addColorStop(0.4, `rgba(255, 255, 255, ${alpha})`);
    grad.addColorStop(1, `rgba(204, 204, 204, ${alpha})`);
    return grad;
  }
}

class ClickableBoard extends Board {
  protected p: Point | null;
  protected listeners = new Array<(p: Point) => void>();
  public enabled = false;

  constructor(parent: HTMLElement | string, public size: BoardSize, options: BoardOptions = {}) {
    super(parent, size, options);

    this.ctx.canvas.addEventListener('mousemove', (e) => {
      let p = this.canvasToBoard(
          e.offsetX * this.pixelRatio, e.offsetY * this.pixelRatio, 0.4);
      if (p != null && this.getStone(p) != Color.Empty) {
        p = null;
      }

      let changed: boolean;
      if (p != null) {
        changed = this.p == null || this.p.row != p.row || this.p.col != p.col;
      } else {
        changed = this.p != null;
      }

      if (changed) {
        this.p = p;
        this.draw();
      }
    });

    this.ctx.canvas.addEventListener('mouseleave', (e) => {
      this.p = null;
      this.draw();
    });

    this.ctx.canvas.addEventListener('click', (e) => {
      if (!this.p || !this.enabled) {
        return;
      }
      for (let listener of this.listeners) {
        listener(this.p);
      }
      this.p = null;
    });
  }

  onClick(cb: (p: Point) => void) {
    this.listeners.push(cb);
  }

  draw() {
    super.draw();
    let p = this.enabled ? this.p : null;
    this.ctx.canvas.style.cursor = p ? 'pointer' : null;
    if (p) {
      this.drawStones([p], this.toPlay, 0.6);
    }
  }
}

export {
  Board,
  BoardOptions,
  BoardSize,
  ClickableBoard,
  COL_LABELS,
  Color,
  Move,
  Point,
  otherColor,
};
