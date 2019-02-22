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
import {Graph} from './graph'

function arraysApproxEqual(a: Nullable<number>[], b: Nullable<number>[],
                           threshold: number) {
  if (a.length != b.length) {
    return false;
  }
  for (let i = 0; i < a.length; ++i) {
    let ai = a[i];
    let bi = b[i];
    if ((ai == null) != (bi == null)) {
      return false;
    }
    if (ai != null && bi != null && Math.abs(ai - bi) > threshold) {
      return false;
    }
  }
  return true;
}

class WinrateGraph extends Graph {
  protected ctx: CanvasRenderingContext2D;

  protected mainLine: Nullable<number>[] = [];
  protected variation: Nullable<number>[] = [];

  protected rootPosition: Nullable<Position> = null;
  protected activePosition: Nullable<Position> = null;

  constructor(parent: HTMLElement | string) {
    super(parent, {
      xStart: 0,
      xEnd: 10,
      yStart: 1,
      yEnd: -1,
      marginTop: 0.05,
      marginBottom: 0.05,
      marginLeft: 0.075,
      marginRight: 0.125,
    });
    if (typeof(parent) == 'string') {
      parent = getElement(parent);
    }
  }

  newGame() {
    this.rootPosition = null;
    this.activePosition = null;
    this.mainLine = [];
    this.variation = [];
    this.xEnd = 10;
    this.draw();
  }

  setActive(position: Position) {
    if (this.rootPosition == null && position.parent == null) {
      this.rootPosition = position;
    }
    if (position != this.activePosition) {
      this.xEnd = Math.max(this.xEnd, position.moveNum);
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
    super.drawImpl();

    let pr = pixelRatio();
    let ctx = this.ctx;

    // Draw the Y axis labels.
    let textHeight = 0.25 * this.marginRight;
    ctx.font = `${textHeight}px sans-serif`;
    ctx.fillStyle = '#96928f';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    this.drawText('B', -4 / this.xScale, 0.95);
    this.drawText('W', -4 / this.xScale, -0.95);

    if (this.activePosition == null) {
      return;
    }

    let moveNum = this.activePosition.moveNum;
    this.drawPlot(
      [[moveNum, this.yStart], [moveNum, this.yEnd]], {
      dash: [0, 3],
      width: 1,
      style: '#96928f',
      snap: true,
    });

    if (this.activePosition.isMainLine) {
      this.drawVariation('#ffe', this.mainLine);
    } else {
      this.drawVariation('#615b56', this.mainLine);
      this.drawVariation('#ffe', this.variation);
    }

    // Draw the value label.
    ctx.textAlign = 'left';
    ctx.fillStyle = '#ffe';
    let q: Nullable<number> = 0;
    let values =
        this.activePosition.isMainLine ? this.mainLine : this.variation;
    // Find the first valid Q value.
    for (let i = Math.min(moveNum, values.length - 1); i >= 0; --i) {
      if (values[i] != null) {
        q = values[i];
        break;
      }
    }
    if (q != null) {
      let score = 50 + 50 * q;
      let txt: string;
      if (score > 50) {
        txt = `B:${Math.round(score)}%`;
      } else {
        txt = `W:${Math.round(100 - score)}%`;
      }
      this.drawText(txt, this.xEnd + 4 / this.xScale, q);
    }
  }

  // Returns the win rate estimation for the prefix of `variation` that has
  // valid Q values (the backend has either performed tree search or win rate
  // evaluation on every position in the prefix at least once).
  private getWinRate(variation: Position[]) {
    let result: Nullable<number>[] = [];
    for (let p of variation) {
      result.push(p.q);
    }
    return result;
  }

  private drawVariation(style: string, values: Nullable<number>[]) {
    if (values.length < 2) {
      return;
    }

    let points: number[][] = [];
    for (let i = 0; i < values.length; ++i) {
      let v = values[i];
      if (v != null) {
        points.push([i, v]);
      }
    }
    super.drawPlot(points, {
      width: 1,
      style: style,
    });
  }
}

export {WinrateGraph}
