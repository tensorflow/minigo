define(["require", "exports", "./util"], function (require, exports, util_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class VariationTree {
        constructor(parent) {
            this.mainline = [];
            if (typeof (parent) == 'string') {
                parent = util_1.getElement(parent);
            }
            let canvas = document.createElement('canvas');
            this.ctx = canvas.getContext('2d');
            parent.appendChild(canvas);
            this.resizeCanvas();
            window.addEventListener('resize', () => {
                this.resizeCanvas();
                this.draw();
            });
            this.draw();
        }
        update(mainline) {
            this.mainline = mainline;
            this.draw();
        }
        resizeCanvas() {
        }
        draw() {
        }
    }
});
//# sourceMappingURL=variation_tree.js.map