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

class ViewPainter {
  private pendingViews: View[] = [];

  draw(view: View) {
    if (this.pendingViews.length == 0) {
      window.requestAnimationFrame(() => {
        for (let view of this.pendingViews) {
          view.drawImpl();
        }
        this.pendingViews = [];
      });
    }

    if (this.pendingViews.indexOf(view) == -1) {
      this.pendingViews.push(view);
    }
  }
}

let painter = new ViewPainter();

abstract class View {
  draw() { painter.draw(this); }
  abstract drawImpl(): void;
}

export {View}
