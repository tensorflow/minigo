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
import {Position} from './position'

class VariationTree {
  private ctx: CanvasRenderingContext2D;
  private mainline: Position[] = [];

  constructor(parent: HTMLElement | string) {
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

  update(mainline: Position[]) {
    this.mainline = mainline;
    this.draw();
  }

  private resizeCanvas() {
  }

  private draw() {
  }
}

