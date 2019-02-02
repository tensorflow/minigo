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

import {BoardSize, COL_LABELS, Color, Coord, Move, N, Nullable, Point} from './base'
import {Grid, Layer} from './layer'
import {Position} from './position'
import {getElement, pixelRatio} from './util'
import {View} from './view'

class Board extends View {
  ctx: CanvasRenderingContext2D;

  backgroundColor: string;
  pointW: number;
  pointH: number;
  stoneRadius: number;
  elem: HTMLElement;

  protected layers: Layer[] = [];

  constructor(parent: HTMLElement | string, public position: Position, layers: Layer[]) {
    super();

    if (typeof(parent) == 'string') {
      parent = getElement(parent);
    }
    this.elem = parent;

    this.backgroundColor = '#db6';

    let canvas = document.createElement('canvas');
    this.ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

    parent.appendChild(canvas);
    window.addEventListener('resize', () => {
      this.resizeCanvas();
      this.draw();
    });
    this.resizeCanvas();

    this.addLayer(new Grid());
    this.addLayers(layers);
  }

  private resizeCanvas() {
    let pr = pixelRatio();
    let canvas = this.ctx.canvas;
    let parent = canvas.parentElement as HTMLElement;
    canvas.width = pr * (parent.offsetWidth);
    canvas.height = pr * (parent.offsetHeight);
    canvas.style.width = `${parent.offsetWidth}px`;
    canvas.style.height = `${parent.offsetHeight}px`;
    this.pointW = this.ctx.canvas.width / (N + 1);
    this.pointH = this.ctx.canvas.height / (N + 1);
    this.stoneRadius = 0.96 * Math.min(this.pointW, this.pointH) / 2;
  }

  newGame(rootPosition: Position) {
    this.position = rootPosition;
    for (let layer of this.layers) {
      layer.clear();
    }
    this.draw();
  }

  addLayer(layer: Layer) {
    this.layers.push(layer);
    layer.addToBoard(this);
  }

  addLayers(layers: Layer[]) {
    for (let layer of layers) {
      this.addLayer(layer);
    }
  }

  // Update the board's layers from the given state object.
  // Board layers that are derived from DataLayer will look in the state object
  // for a named property. If that property exists, the layer will update its
  // internal state from it.
  setPosition(position: Position) {
    if (this.position == position) {
      return;
    }

    this.position = position;
    let allProps = new Set<string>(Object.keys(position));
    for (let layer of this.layers) {
      layer.update(allProps);
    }
    this.draw();
  }

  update(update: Position | Position.Update) {
    let anythingChanged = false;
    let keys = new Set<string>(Object.keys(update));
    for (let layer of this.layers) {
      if (layer.update(keys)) {
        anythingChanged = true;
      }
    }
    if (anythingChanged) {
      this.draw();
    }
  }

  drawImpl() {
    let ctx = this.ctx;
    ctx.fillStyle = this.backgroundColor;
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    for (let layer of this.layers) {
      if (layer.show) {
        layer.draw();
      }
    }
  }

  getStone(p: Point) {
    return this.position.stones[p.row * N + p.col];
  }

  canvasToBoard(x: number, y: number, threshold?: number): Nullable<Point> {
    let pr = pixelRatio();
    x *= pr;
    y *= pr;

    let canvas = this.ctx.canvas;

    y = y * (N + 1) / canvas.height - 0.5;
    x = x * (N + 1) / canvas.width - 0.5;
    let row = Math.floor(y);
    let col = Math.floor(x);
    if (row < 0 || row >= N || col < 0 || col >= N) {
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

    return {
      x: canvas.width * (col + 1.0) / (N + 1),
      y: canvas.height * (row + 1.0) / (N + 1)
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

    let r = this.stoneRadius;
    for (let p of ps) {
      let c = this.boardToCanvas(p.row, p.col);
      ctx.beginPath();
      ctx.translate(c.x + 0.5, c.y + 0.5);
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
      let ofs = -0.5 * this.stoneRadius;
      grad = this.ctx.createRadialGradient(
          ofs, ofs, 0, ofs, ofs, 2 * this.stoneRadius);
      grad.addColorStop(0, `rgba(68, 68, 68, ${alpha})`);
      grad.addColorStop(1, `rgba(16, 16, 16, ${alpha})`);
    } else if (color == Color.White) {
      let ofs = -0.2 * this.stoneRadius;
      grad = this.ctx.createRadialGradient(
          ofs, ofs, 0, ofs, ofs, 1.2 * this.stoneRadius);
      grad.addColorStop(0.4, `rgba(255, 255, 255, ${alpha})`);
      grad.addColorStop(1, `rgba(204, 204, 204, ${alpha})`);
    } else {
      throw new Error(`Invalid color ${color}`);
    }
    return grad;
  }
}

type ClickListener = (p: Point) => void;

class ClickableBoard extends Board {
  enabled = false;
  protected p: Point | null;
  protected listeners: ClickListener[] = [];

  constructor(parent: HTMLElement | string, position: Position, layerDescs: any[]) {
    super(parent, position, layerDescs);

    this.ctx.canvas.addEventListener('mousemove', (e) => {
      // Find the point on the board being hovered over.
      let p = this.canvasToBoard(e.offsetX, e.offsetY, 0.45);

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
      if (this.p != null) {
        this.p = null;
        this.draw();
      }
    });

    this.ctx.canvas.addEventListener('click', (e) => {
      if (!this.p || !this.enabled) {
        return;
      }
      for (let listener of this.listeners) {
        listener(this.p);
      }
      this.p = null;
      this.draw();
    });
  }

  onClick(cb: ClickListener) {
    this.listeners.push(cb);
  }

  setPosition(position: Position) {
    if (position != this.position) {
      this.p = null;
      super.setPosition(position);
    }
  }

  drawImpl() {
    super.drawImpl();
    let p = this.enabled ? this.p : null;
    this.ctx.canvas.style.cursor = p ? 'pointer' : null;
    if (p) {
      this.drawStones([p], this.position.toPlay, 0.6);
    }
  }
}

export {
  Board,
  ClickableBoard,
  COL_LABELS,
};
