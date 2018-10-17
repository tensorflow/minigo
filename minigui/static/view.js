define(["require", "exports"], function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class ViewPainter {
        constructor() {
            this.pendingViews = [];
        }
        draw(view) {
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
    class View {
        draw() { painter.draw(this); }
    }
    exports.View = View;
});
//# sourceMappingURL=view.js.map