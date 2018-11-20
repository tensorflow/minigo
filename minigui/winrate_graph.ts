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

  // Horizontal scaling factor used when plotting. It's the maximum move number
  // seen, with a minimum of MIN_POINTS. Note that because win rate evaluation
  // is computed lazily after an SGF file has been loaded, the maximum move
  // number can be larger than the prefix of the current variation that we have
  // data to plot. Scaling by the maximum move number therefore gives an
  // indication to the user how many more moves need to be evaluated before we
  // have a win rate estimation for the complete game.
  protected xScale = MIN_POINTS;

  protected rootPosition: Nullable<Position> = null;
  protected activePosition: Nullable<Position> = null;

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

  newGame(rootPosition: Position) {
    this.rootPosition = rootPosition;
    this.activePosition = rootPosition;
    this.mainLine = [];
    this.variation = [];
    this.xScale = MIN_POINTS;
    this.draw();
  }

  setActive(position: Position) {
    if (position != this.activePosition) {
      this.xScale = Math.max(this.xScale, position.moveNum);
      this.activePosition = position;
      this.update(position);
      this.draw();
    }
  }

  update(position: Position) {
    if (this.rootPosition == null || this.activePosition == null) {
      return;
    }
    if (!position.isMainLine) {
      // The updated position isn't on the main line, we may not need to update
      // anything.
      if (this.activePosition.isMainLine) {
        // Only showing the main line, nothing to do.
        return;
      } else if (this.activePosition.getFullLine().indexOf(position) == -1) {
        // This position isn't from the current variation, nothing to do.
        return;
      }
    }

    // Always check if the main line plot needs updating, since the main line
    // and any currently active variation likely share a prefix of moves.
    let anythingChanged = false;
    let mainLine = this.getWinRate(this.rootPosition.getFullLine());
    if (!arraysApproxEqual(mainLine, this.mainLine, 0.001)) {
      anythingChanged = true;
      this.mainLine = mainLine;
    }

    // If the updated position is a variation, check if the variation plot needs
    // updating.
    if (!position.isMainLine) {
      let variation = this.getWinRate(position.getFullLine());
      if (!arraysApproxEqual(variation, this.variation, 0.001)) {
        anythingChanged = true;
        this.variation = variation;
      }
    }

    if (anythingChanged) {
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

    ctx.lineCap = 'butt';
    ctx.lineJoin = 'butt';

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

    // Draw the Y axis labels.
    ctx.font = `${this.textHeight}px sans-serif`;
    ctx.fillStyle = '#96928f';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText('B', -0.5 * this.textHeight, Math.round(0.05 * h));
    ctx.fillText('W', -0.5 * this.textHeight, Math.round(0.95 * h));

    if (this.activePosition == null) {
      return;
    }

    let moveNum = this.activePosition.moveNum;
    ctx.setLineDash([1, 2]);
    ctx.beginPath();
    ctx.moveTo(Math.round(w * moveNum / this.xScale), 0.5);
    ctx.lineTo(Math.round(w * moveNum / this.xScale), h - 0.5);
    ctx.stroke();
    ctx.setLineDash([]);

    if (this.activePosition.isMainLine) {
      this.drawPlot(this.mainLine, pr, '#ffe');
    } else {
      this.drawPlot(this.mainLine, pr, '#615b56');
      this.drawPlot(this.variation, pr, '#ffe');
    }

    // Draw the value label.
    ctx.textAlign = 'left';
    ctx.fillStyle = '#ffe';
    let q = 0;
    let values =
        this.activePosition.isMainLine ? this.mainLine : this.variation;
    if (values.length > 0) {
      q = values[Math.min(moveNum, values.length - 1)];
    }
    let score = 50 + 50 * q;
    let y = h * (0.5 - 0.5 * q);
    let txt: string;
    if (score > 50) {
      txt = `B:${Math.round(score)}%`;
    } else {
      txt = `W:${Math.round(100 - score)}%`;
    }
    ctx.fillText(txt, w + 8, y);
  }

  // Returns the win rate estimation for the prefix of `variation` that has
  // valid Q values (the backend has either performed tree search or win rate
  // evaluation on every position in the prefix at least once).
  private getWinRate(variation: Position[]) {
    let result: number[] = [];
    for (let p of variation) {
      if (p.q == null) {
        // Stop when we reach the first position that doesn't have a valid Q.
        break;
      }
      result.push(p.q);
    }
    return result;
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
