// Copyright 2019 Google LLC
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

namespace Graph {
export interface GraphOptions {
  xStart: number;
  xEnd: number;
  yStart: number;
  yEnd: number;
  xTicks?: boolean;
  yTicks?: boolean;
  marginTop?: number;
  marginBottom?: number;
  marginLeft?: number;
  marginRight?: number;
}

export type Point = number[];

export interface PlotOptions {
  width?: number;
  style?: string;
  snap?: boolean;
  dash?: number[];
}
}

const DEFAULT_PLOT_OPTIONS: Graph.PlotOptions = {
  width: 1,
  snap: false,
};

abstract class Graph extends View {
  protected ctx: CanvasRenderingContext2D;

  private marginTopPct: number;
  private marginBottomPct: number;
  private marginLeftPct: number;
  private marginRightPct: number;
  private xTicks: boolean;
  private yTicks: boolean;
  private lineDash = false;

  protected xScale: number;
  protected yScale: number;
  protected xTickPoints: number[] = [];
  protected yTickPoints: number[] = [];
  protected marginTop: number;
  protected marginBottom: number;
  protected marginLeft: number;
  protected marginRight: number;
  protected xStart = 0;
  protected xEnd = 1;
  protected yStart = 0;
  protected yEnd = 1;
  protected moveNum = 0;

  constructor(parent: HTMLElement | string, private options: Graph.GraphOptions) {
    super();
    this.xStart = options.xStart;
    this.xEnd = options.xEnd;
    this.yStart = options.yStart;
    this.yEnd = options.yEnd;
    this.xTicks = options.xTicks || false;
    this.yTicks = options.yTicks || false;
    this.marginTopPct = options.marginTop || 0.05;
    this.marginBottomPct = options.marginBottom || 0.05;
    this.marginLeftPct = options.marginLeft || 0.05;
    this.marginRightPct = options.marginRight || 0.05;

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
    let ctx = this.ctx;

    let canvas = ctx.canvas;
    let parent = canvas.parentElement as HTMLElement;
    canvas.width = pr * parent.offsetWidth;
    canvas.height = pr * parent.offsetHeight;
    canvas.style.width = `${parent.offsetWidth}px`;
    canvas.style.height = `${parent.offsetHeight}px`;

    this.marginTop = Math.floor(this.marginTopPct * canvas.width);
    this.marginBottom = Math.floor(this.marginBottomPct * canvas.width);
    this.marginLeft = Math.floor(this.marginLeftPct * canvas.width);
    this.marginRight = Math.floor(this.marginRightPct * canvas.width);
    let w = canvas.width - this.marginLeft - this.marginRight;
    let h = canvas.height - this.marginTop - this.marginBottom;
    this.xScale = w / (this.xEnd - this.xStart);
    this.yScale = h / (this.yEnd - this.yStart);

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }

  setMoveNum(moveNum: number) {
    if (moveNum != this.moveNum) {
      this.moveNum = moveNum;
      this.draw();
    }
  }

  drawImpl() {
    this.updateScale();

    let pr = pixelRatio();
    let ctx = this.ctx;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Apply a translation such that (0, 0) is the center of the pixel at the
    // top left of the graph.
    ctx.translate(this.marginLeft + 0.5, this.marginTop + 0.5);

    ctx.lineWidth = pr;
    ctx.strokeStyle = '#96928f';

    this.beginPath();
    this.moveTo(this.xStart, 0, true);
    this.lineTo(this.xEnd, 0, true);
    this.moveTo(0, this.yStart, true);
    this.lineTo(0, this.yEnd, true);

    if (this.xTicks) {
      this.xTickPoints = this.calculateTickPoints(this.xStart, this.xEnd);
      let y = 4 * pr / this.yScale;
      if (this.yEnd > this.yStart) {
        y = -y;
      }
      for (let x of this.xTickPoints) {
        this.moveTo(x, 0, true);
        this.lineTo(x, y, true);
      }
    }

    if (this.yTicks) {
      this.yTickPoints = this.calculateTickPoints(this.yStart, this.yEnd);
      let x = 4 * pr / this.xScale;
      if (this.xEnd > this.xStart) {
        x = -x;
      }
      for (let y of this.yTickPoints) {
        this.moveTo(0, y, true);
        this.lineTo(x, y, true);
      }
    }
    this.stroke();
  }

  protected drawText(text: string, x: number, y: number, snap=false) {
    x = this.xScale * (x - this.xStart);
    y = this.yScale * (y - this.yStart);
    if (snap) {
      x = Math.round(x);
      y = Math.round(y);
    }
    this.ctx.fillText(text, x, y);
  }

  protected drawPlot(points: Graph.Point[], options=DEFAULT_PLOT_OPTIONS) {
    if (points.length == 0) {
      return;
    }
    let pr = pixelRatio();
    let ctx = this.ctx;
    let snap = options.snap || false;
    ctx.lineWidth = (options.width || 1) * pr;
    if (options.style) {
      ctx.strokeStyle = options.style;
    }
    this.beginPath(options.dash || null);
    this.moveTo(points[0][0], points[0][1], snap);
    for (let i = Math.min(1, points.length - 1); i < points.length; ++i) {
      let p = points[i];
      this.lineTo(p[0], p[1], snap);
    }
    this.stroke();
  }

  private beginPath(dash: Nullable<number[]> = null) {
    let ctx = this.ctx;
    if (dash != null) {
      ctx.lineCap = 'square';
      ctx.setLineDash(dash);
      this.lineDash = true;
    } else if (this.lineDash) {
      ctx.lineCap = 'round';
      ctx.setLineDash([]);
    }
    ctx.beginPath();
  }

  private stroke() {
    this.ctx.stroke();
  }

  private moveTo(x: number, y: number, snap=false) {
    x = this.xScale * (x - this.xStart);
    y = this.yScale * (y - this.yStart);
    if (snap) {
      x = Math.round(x);
      y = Math.round(y);
    }
    this.ctx.moveTo(x, y);
  }

  private lineTo(x: number, y: number, snap=false) {
    x = this.xScale * (x - this.xStart);
    y = this.yScale * (y - this.yStart);
    if (snap) {
      x = Math.round(x);
      y = Math.round(y);
    }
    this.ctx.lineTo(x, y);
  }

  private updateScale() {
    let canvas = this.ctx.canvas;
    let w = canvas.width - this.marginLeft - this.marginRight;
    let h = canvas.height - this.marginTop - this.marginBottom;
    this.xScale = w / (this.xEnd - this.xStart);
    this.yScale = h / (this.yEnd - this.yStart);
  }

  private calculateTickPoints(start: number, end: number) {
    // Calculate the spacing of the ticks.
    let x = Math.abs(end - start);
    let spacing = 1;
    if (x >= 1) {
      let scale = Math.pow(10, Math.max(0, Math.floor(Math.log10(x) - 1)));
      let top2 = Math.floor(x / scale);
      if (top2 <= 10) {
        spacing = 1;
      } else if (top2 <= 20) {
        spacing = 2;
      } else if (top2 <= 50) {
        spacing = 5;
      } else {
        spacing = 10;
      }
      spacing *= scale;
    }

    // Build the array of tick points.
    let min = Math.min(start, end);
    let max = Math.max(start, end);
    let positive: number[] = [];
    let negative: number[] = [];
    if (min <= 0 && max >= 0) {
      positive.push(0);
    }
    for (let x = spacing; x <= max; x += spacing) {
      positive.push(x);
    }
    for (let x = -spacing; x >= min; x -= spacing) {
      negative.push(x);
    }
    negative.reverse();

    return negative.concat(positive);
  }
}

export {Graph}

