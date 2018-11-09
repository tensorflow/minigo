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

import {Nullable} from './base'
import {Position} from './position'
import {getElement, pixelRatio} from './util'
import {View} from './view'

const MIN_POINTS = 10;

function arraysApproxEqual(a: number[], b: number[], threshold: number) {
  if (a.length != b.length) {
    return false;
  }
  for (let i = 0; i < a.length; ++i) {
    if (Math.abs(a[i] - b[i]) > threshold) {
      return false;
    }
  }
  return true;
}

class WinrateGraph extends View {
  protected ctx: CanvasRenderingContext2D;
  protected marginTop: number;
  protected marginBottom: number;
  protected marginLeft: number;
  protected marginRight: number;
  protected textHeight: number;

  protected w: number;
  protected h: number;

  protected mainLine: number[] = [];
  protected variation: number[] = [];

  protected moveNum = 0;

  protected xScale = MIN_POINTS;

  constructor(parent: HTMLElement | string) {
    super();
    if (typeof(parent) == 'string') {
      parent = getElement(parent);
    }

    let canvas = document.createElement('canvas');
    this.ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
    parent.appendChild(canvas);
    this.resizeCanvas();

    window.addEventListener('resize', () => {
      this.resizeCanvas();
      this.draw();
    });

    this.draw();
  }

  private resizeCanvas() {
    let pr = pixelRatio();
    let canvas = this.ctx.canvas;
    let parent = canvas.parentElement as HTMLElement;
    canvas.width = pr * parent.offsetWidth;
    canvas.height = pr * parent.offsetHeight;
    canvas.style.width = `${parent.offsetWidth}px`;
    canvas.style.height = `${parent.offsetHeight}px`;

    this.marginTop = Math.floor(0.05 * canvas.width);
    this.marginBottom = Math.floor(0.05 * canvas.width);
    this.marginLeft = Math.floor(0.075 * canvas.width);
    this.marginRight = Math.floor(0.125 * canvas.width);
    this.w = canvas.width - this.marginLeft - this.marginRight;
    this.h = canvas.height - this.marginTop - this.marginBottom;
    this.textHeight = 0.06 * this.h;
  }

  clear() {
    this.mainLine = [];
    this.variation = [];
    this.moveNum = 0;
    this.draw();
  }

  // Sets the predicted winrate for the given move.
  update(position: Position) {
    let anythingChanged = position.moveNum != this.moveNum;
    this.moveNum = position.moveNum;

    // Find the end of this variation.
    while (position.children.length > 0) {
      position = position.children[0];
    }

    // Get the score history for this line.
    let values: number[] = [];
    let p: Nullable<Position> = position;
    while (p != null) {
      values.push(p.q);
      p = p.parent;
    }
    values.reverse();

    if (position.isMainLine) {
      // When the given position is on the main line, update it and remove
      // the variation.
      anythingChanged =
          anythingChanged || !arraysApproxEqual(values, this.mainLine, 0.001);
      if (anythingChanged) {
        this.mainLine = values;
      }
      this.variation = [];
    } else {
      // When the given position is a variation, update it but leave the main
      // line alone.
      anythingChanged =
          anythingChanged || !arraysApproxEqual(values, this.variation, 0.001);
      if (anythingChanged) {
        this.variation = values;
      }
    }

    if (anythingChanged) {
      this.xScale = Math.max(
          this.mainLine.length - 1, this.variation.length - 1, MIN_POINTS);
      this.draw();
    }
  }

  drawImpl() {
    let pr = pixelRatio();
    let ctx = this.ctx;
    let w = this.w;
    let h = this.h;

    // Reset the transform to identity and clear.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Apply a translation such that (0, 0) is the center of the pixel at the
    // top left of the graph.
    ctx.translate(this.marginLeft + 0.5, this.marginTop + 0.5);

    // Draw the horizontal & vertical axis and the move.
    ctx.lineWidth = pr;

    ctx.strokeStyle = '#96928f';
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, h);
    ctx.moveTo(0, Math.floor(0.5 * h));
    ctx.lineTo(w, Math.floor(0.5 * h));
    ctx.stroke();

    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(Math.round(w * this.moveNum / this.xScale), 0);
    ctx.lineTo(Math.round(w * this.moveNum / this.xScale), h);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw the Y axis labels.
    ctx.font = `${this.textHeight}px sans-serif`;
    ctx.fillStyle = '#96928f';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText('B', -0.5 * this.textHeight, Math.round(0.05 * h));
    ctx.fillText('W', -0.5 * this.textHeight, Math.round(0.95 * h));

    if (this.variation.length == 0) {
      this.drawPlot(this.mainLine, pr, '#ffe');
    } else {
      this.drawPlot(this.mainLine, pr, '#96928f');
      this.drawPlot(this.variation, pr, '#ffe');
    }

    // Draw the value label.
    ctx.textAlign = 'left';
    ctx.fillStyle = '#ffe';
    let y = 0;
    let values = this.variation.length > 0 ? this.variation : this.mainLine;
    if (values.length > 0) {
      y = values[this.moveNum];
    }
    let score = 50 + 50 * y;
    y = h * (0.5 - 0.5 * y);
    let txt: string;
    if (score > 50) {
      txt = `B:${Math.round(score)}%`;
    } else {
      txt = `W:${Math.round(100 - score)}%`;
    }
    ctx.fillText(txt, w + 8, y);
  }

  private drawPlot(values: number[], lineWidth: number, style: string) {
    if (values.length < 2) {
      return;
    }

    let ctx = this.ctx;
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = style;
    ctx.beginPath();
    ctx.moveTo(0, this.h * (0.5 - 0.5 * values[0]));
    for (let x = 0; x < values.length; ++x) {
      let y = values[x];
      ctx.lineTo(this.w * x / this.xScale, this.h * (0.5 - 0.5 * y));
    }
    ctx.stroke();
  }
}

export {WinrateGraph}
