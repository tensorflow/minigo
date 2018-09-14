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

import {getElement, pixelRatio} from './util'
import {DataObj, Grid, Layer} from './layer'
import {BoardSize, Color, Coord, Move, Point} from './base'

const COL_LABELS = 'ABCDEFGHJKLMNOPQRST';

namespace Annotation {
  export enum Shape {
    Dot,
    Triangle,
  }
}

interface Annotation {
  p: Point;
  shape: Annotation.Shape;
  color: string;
}

interface StateObj {
  toPlay?: Color;
  stones?: Color[];
  [key: string]: any;
}

interface BoardOptions {
  // Margin in pixels to add to the board.
  // Default is 0;
  margin?: number;
}

class Board {
  toPlay = Color.Black;
  stones: Color[] = [];
  ctx: CanvasRenderingContext2D;

  backgroundColor: string;
  margin: number;
  pointW: number;
  pointH: number;
  stoneRadius: number;

  protected layers: Layer[];

  constructor(parent: HTMLElement | string, public size: BoardSize,
              layerDescs: any[], options: BoardOptions = {}) {
    if (typeof(parent) == 'string') {
      parent = getElement(parent);
    }

    this.backgroundColor = '#db6';
    this.margin = options.margin || 0;

    for (let i = 0; i < this.size * this.size; ++i) {
      this.stones.push(Color.Empty);
    }

    let pr = pixelRatio();
    let canvas = document.createElement('canvas');
    canvas.style.margin = `${this.margin}px`;
    canvas.width = pr * (parent.offsetWidth - 2 * this.margin);
    canvas.height = pr * (parent.offsetHeight - 2 * this.margin);
    canvas.style.width = `${parent.offsetWidth - 2 * this.margin}px`;
    canvas.style.height = `${parent.offsetHeight - 2 * this.margin}px`;
    this.ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

    this.pointW = this.ctx.canvas.width / (this.size + 1);
    this.pointH = this.ctx.canvas.height / (this.size + 1);
    this.stoneRadius = Math.min(this.pointW, this.pointH);

    this.layers = [new Grid(this)];
    this.addLayers(layerDescs);

    parent.appendChild(canvas);
  }

  addLayers(descs: any[]) {
    for (let desc of descs) {
      let layer: Layer;
      if (Array.isArray(desc)) {
        let ctor= desc[0];
        let args = desc.slice(1);
        layer = new ctor(this, ...args);
      } else {
        let ctor= desc;
        layer = new ctor(this);
      }
      this.layers.push(layer);
    }
  }

  update(state: StateObj) {
    if (state.toPlay !== undefined) {
      this.toPlay = state.toPlay;
    }
    if (state.stones !== undefined) {
      this.stones = state.stones;
    }
    for (let layer of this.layers) {
      layer.update(state);
    }
  }

  draw() {
    let ctx = this.ctx;
    ctx.fillStyle = this.backgroundColor;
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    for (let layer of this.layers) {
      layer.draw();
    }
  }

  getStone(p: Point) {
    return this.stones[p.row * this.size + p.col];
  }

  canvasToBoard(x: number, y: number, threshold?: number): Point | null {
    let pr = pixelRatio();
    x *= pr;
    y *= pr;

    let canvas = this.ctx.canvas;
    let size = this.size;

    y = y * (size + 1) / canvas.height - 0.5;
    x = x * (size + 1) / canvas.width - 0.5;
    let row = Math.floor(y);
    let col = Math.floor(x);
    if (row < 0 || row >= size || col < 0 || col >= size) {
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

  boardToCanvas(row: number, col: number): Coord {
    let canvas = this.ctx.canvas;
    let size = this.size;

    return {
      x: canvas.width * (col + 1.0) / (size + 1),
      y: canvas.height * (row + 1.0) / (size + 1)
    };
  }

  drawStones(ps: Point[], color: Color, alpha: number) {
    if (ps.length == 0) {
      return;
    }

    let ctx = this.ctx;
    let pr = pixelRatio();

    if (alpha == 1) {
      ctx.shadowBlur = 4 * pr;
      ctx.shadowOffsetX = 1.5 * pr;
      ctx.shadowOffsetY = 1.5 * pr;
      ctx.shadowColor = `rgba(0, 0, 0, ${color == Color.Black ? 0.4 : 0.3})`;
    }

    ctx.fillStyle = this.stoneFill(color, alpha);

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
      ctx.shadowColor = 'rgba(0, 0, 0, 0)';
    }
  }

  private stoneFill(color: Color, alpha: number) {
    let grad: CanvasGradient;
    if (color == Color.Black) {
      let ofs = -0.25 * this.stoneRadius;
      grad = this.ctx.createRadialGradient(
          ofs, ofs, 0, ofs, ofs, this.stoneRadius);
      grad.addColorStop(0, `rgba(68, 68, 68, ${alpha})`);
      grad.addColorStop(1, `rgba(16, 16, 16, ${alpha})`);
    } else if (color == Color.White) {
      let ofs = -0.1 * this.stoneRadius;
      grad = this.ctx.createRadialGradient(
          ofs, ofs, 0, ofs, ofs, 0.6 * this.stoneRadius);
      grad.addColorStop(0.4, `rgba(255, 255, 255, ${alpha})`);
      grad.addColorStop(1, `rgba(204, 204, 204, ${alpha})`);
    } else {
      throw new Error(`Invalid color ${color}`);
    }
    return grad;
  }
}

class ClickableBoard extends Board {
  protected p: Point | null;
  protected listeners = new Array<(p: Point) => void>();
  public enabled = false;

  constructor(parent: HTMLElement | string, public size: BoardSize,
              layerDescs: any[], options: BoardOptions = {}) {
    super(parent, size, layerDescs, options);

    this.ctx.canvas.addEventListener('mousemove', (e) => {
      // Find the point on the board being hovered over.
      let p = this.canvasToBoard(e.offsetX, e.offsetY, 0.4);

      // Clear the hovered point if there's already a stone on the board there.
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
  Annotation,
  Board,
  BoardOptions,
  ClickableBoard,
  COL_LABELS,
};
