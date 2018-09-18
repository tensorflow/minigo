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

type MoveChangedCallback = (move: number | null) => void;

class WinrateGraph {
  protected ctx: CanvasRenderingContext2D;
  protected points = new Array<[number, number]>();
  protected pixelRatio = window.devicePixelRatio || 1;
  protected marginTop = 20 * this.pixelRatio;
  protected marginBottom = 20 * this.pixelRatio;
  protected marginLeft = 30 * this.pixelRatio;
  protected marginRight = 60 * this.pixelRatio;
  protected minPoints = 10;

  protected w: number;
  protected h: number;
  protected selectedMove: number | null = null;
  protected moveChangedCallback: MoveChangedCallback | null = null;

  constructor(parent: HTMLElement | string) {
    if (typeof(parent) == 'string') {
      parent = getElement(parent);
    }

    let canvas = document.createElement('canvas');
    canvas.width = this.pixelRatio * parent.offsetWidth;
    canvas.height = this.pixelRatio * parent.offsetHeight;
    canvas.style.width = `${parent.offsetWidth}px`;
    canvas.style.height = `${parent.offsetHeight}px`;
    this.ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

    this.w = canvas.width - this.marginLeft - this.marginRight;
    this.h = canvas.height - this.marginTop - this.marginBottom;

    parent.appendChild(canvas);

    canvas.addEventListener('mousemove', (e) => {
      this.onMouseMove(e.offsetX, e.offsetY);
    });
    canvas.addEventListener('mouseleave', () => {
      this.onMouseLeave();
    });

    this.draw();
  }

  clear() {
    this.points = [];
    this.draw();
  }

  // Sets the predicted winrate for the given move.
  setWinrate(move: number, winrate: number) {
    this.points[move] = [move, winrate];
    this.draw();
  }

  onMoveChanged(callback: MoveChangedCallback) {
    this.moveChangedCallback = callback;
  }

  private draw() {
    let ctx = this.ctx;
    let w = this.w;
    let h = this.h;
    let xScale = Math.max(this.points.length - 1, this.minPoints);

    // Reset the transform to identity and clear.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Apply a translation such that (0, 0) is the center of the pixel at the
    // top left of the graph.
    ctx.translate(this.marginLeft + 0.5, this.marginTop + 0.5);

    // Round caps & joins look nice.
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Draw the resign threshold lines.
    ctx.lineWidth = 1 * this.pixelRatio;
    ctx.strokeStyle = '#56504b';
    ctx.beginPath();
    ctx.moveTo(0, Math.round(0.95 * h));
    ctx.lineTo(w, Math.round(0.95 * h));
    ctx.moveTo(0, Math.round(0.05 * h));
    ctx.lineTo(w, Math.round(0.05 * h));
    ctx.stroke();

    // Draw the horizontal & vertical axis.
    let lineWidth = 3 * this.pixelRatio;
    ctx.lineWidth = lineWidth;

    ctx.strokeStyle = '#96928f';
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, h);
    ctx.moveTo(0, 0.5 * h);
    ctx.lineTo(w, 0.5 * h);
    ctx.stroke();

    // Draw the Y axis labels.
    let textHeight = 14 * this.pixelRatio;
    ctx.font = `${textHeight}px sans-serif`;
    ctx.fillStyle = '#96928f';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText('B', -0.5 * textHeight, Math.round(0.05 * h));
    ctx.fillText('W', -0.5 * textHeight, Math.round(0.95 * h));

    // Offset the start of the X axis by the widh of the vertical axis.
    // It looks nicer that way.
    let xOfs = Math.floor(lineWidth / 2);
    ctx.translate(xOfs, 0);
    w -= xOfs;

    // Draw the selected move (if any).
    if (this.selectedMove !== null) {
      let x = Math.round(w * this.selectedMove / xScale);
      ctx.lineWidth = 1;
      ctx.strokeStyle = '#96928f';
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    // Draw the graph.
    if (this.points.length >= 2) {
      ctx.lineWidth = lineWidth;
      ctx.strokeStyle = '#eee';
      ctx.beginPath();
      let [x, y] = this.points[0];
      ctx.moveTo(w * x / xScale, h * (0.5 - 0.5 * y));
      for (let i = 1; i < this.points.length; ++i) {
        [x, y] = this.points[i];
        ctx.lineTo(w * x / xScale, h * (0.5 - 0.5 * y));
      }
      ctx.stroke();
    }

    // Draw the value label.
    ctx.textAlign = 'left';
    ctx.fillStyle = '#ffe';
    let y;
    if (this.points.length > 0) {
      y = this.points[this.points.length - 1][1];
    } else {
      y = 0;
    }
    let score = y;
    y = h * (0.5 - 0.5 * y);
    let txt: string;
    if (score > 0) {
      txt = `B:${Math.round(score * 100)}%`;
    } else {
      txt = `W:${Math.round(-score * 100)}%`;
    }
    ctx.fillText(txt, w + 8, y);
  }

  private onMouseMove(x: number, y: number) {
    if (this.points.length < 2) {
      return;
    }
    let n = Math.max(this.minPoints, this.points.length - 1);
    let newMove = Math.round(n * (this.pixelRatio * x - this.marginLeft) / this.w);
    newMove = Math.max(0, Math.min(newMove, this.points.length - 1));
    if (newMove != this.selectedMove) {
      this.selectedMove = newMove;
      this.draw();
      if (this.moveChangedCallback) {
        this.moveChangedCallback(this.selectedMove);
      }
    }
  }

  private onMouseLeave() {
    this.selectedMove = null;
    this.draw();
    if (this.moveChangedCallback) {
      this.moveChangedCallback(this.selectedMove);
    }
  }
}

export {WinrateGraph}
