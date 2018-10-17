define(["require", "exports", "./util", "./view", "./layer", "./base"], function (require, exports, util_1, view_1, layer_1, base_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.COL_LABELS = base_1.COL_LABELS;
    var Annotation;
    (function (Annotation) {
        let Shape;
        (function (Shape) {
            Shape[Shape["Dot"] = 0] = "Dot";
            Shape[Shape["Triangle"] = 1] = "Triangle";
        })(Shape = Annotation.Shape || (Annotation.Shape = {}));
    })(Annotation || (Annotation = {}));
    exports.Annotation = Annotation;
    class Board extends view_1.View {
        constructor(parent, size, layerDescs) {
            super();
            this.size = size;
            this.toPlay = base_1.Color.Black;
            this.stones = [];
            if (typeof (parent) == 'string') {
                parent = util_1.getElement(parent);
            }
            this.elem = parent;
            this.backgroundColor = '#db6';
            for (let i = 0; i < this.size * this.size; ++i) {
                this.stones.push(base_1.Color.Empty);
            }
            let canvas = document.createElement('canvas');
            this.ctx = canvas.getContext('2d');
            parent.appendChild(canvas);
            window.addEventListener('resize', () => {
                this.resizeCanvas();
                this.draw();
            });
            this.resizeCanvas();
            this.layers = [new layer_1.Grid(this)];
            this.addLayers(layerDescs);
        }
        resizeCanvas() {
            let pr = util_1.pixelRatio();
            let canvas = this.ctx.canvas;
            let parent = canvas.parentElement;
            canvas.width = pr * (parent.offsetWidth);
            canvas.height = pr * (parent.offsetHeight);
            canvas.style.width = `${parent.offsetWidth}px`;
            canvas.style.height = `${parent.offsetHeight}px`;
            this.pointW = this.ctx.canvas.width / (this.size + 1);
            this.pointH = this.ctx.canvas.height / (this.size + 1);
            this.stoneRadius = Math.min(this.pointW, this.pointH);
        }
        addLayers(descs) {
            for (let desc of descs) {
                let layer;
                if (Array.isArray(desc)) {
                    let ctor = desc[0];
                    let args = desc.slice(1);
                    layer = new ctor(this, ...args);
                }
                else {
                    let ctor = desc;
                    layer = new ctor(this);
                }
                this.layers.push(layer);
            }
        }
        update(state) {
            if (state.toPlay !== undefined) {
                this.toPlay = state.toPlay;
            }
            if (state.stones !== undefined) {
                this.stones = state.stones;
            }
            let anything_changed = false;
            for (let layer of this.layers) {
                if (layer.update(state)) {
                    anything_changed = true;
                }
            }
            return anything_changed;
        }
        drawImpl() {
            let ctx = this.ctx;
            ctx.fillStyle = this.backgroundColor;
            ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            for (let layer of this.layers) {
                layer.draw();
            }
        }
        getStone(p) {
            return this.stones[p.row * this.size + p.col];
        }
        canvasToBoard(x, y, threshold) {
            let pr = util_1.pixelRatio();
            x *= pr;
            y *= pr;
            let canvas = this.ctx.canvas;
            let size = this.size;
            y = y * (size + 1) / canvas.height - 0.5;
            x = x * (size + 1) / canvas.width - 0.5;
            let row = Math.floor(y);
            let col = Math.floor(x);
            if (row < 0 || row >= size || col < 0 || col >= size) {
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
            return { row: row, col: col };
        }
        boardToCanvas(row, col) {
            let canvas = this.ctx.canvas;
            let size = this.size;
            return {
                x: canvas.width * (col + 1.0) / (size + 1),
                y: canvas.height * (row + 1.0) / (size + 1)
            };
        }
        drawStones(ps, color, alpha) {
            if (ps.length == 0) {
                return;
            }
            let ctx = this.ctx;
            let pr = util_1.pixelRatio();
            if (alpha == 1) {
                ctx.shadowBlur = 4 * pr;
                ctx.shadowOffsetX = 1.5 * pr;
                ctx.shadowOffsetY = 1.5 * pr;
                ctx.shadowColor = `rgba(0, 0, 0, ${color == base_1.Color.Black ? 0.4 : 0.3})`;
            }
            ctx.fillStyle = this.stoneFill(color, alpha);
            let r = 0.48 * this.stoneRadius;
            for (let p of ps) {
                let c = this.boardToCanvas(p.row, p.col);
                ctx.beginPath();
                ctx.translate(c.x, c.y);
                ctx.arc(0, 0, r, 0, 2 * Math.PI);
                ctx.fill();
                ctx.setTransform(1, 0, 0, 1, 0, 0);
            }
            if (alpha == 1) {
                ctx.shadowColor = 'rgba(0, 0, 0, 0)';
            }
        }
        stoneFill(color, alpha) {
            let grad;
            if (color == base_1.Color.Black) {
                let ofs = -0.25 * this.stoneRadius;
                grad = this.ctx.createRadialGradient(ofs, ofs, 0, ofs, ofs, this.stoneRadius);
                grad.addColorStop(0, `rgba(68, 68, 68, ${alpha})`);
                grad.addColorStop(1, `rgba(16, 16, 16, ${alpha})`);
            }
            else if (color == base_1.Color.White) {
                let ofs = -0.1 * this.stoneRadius;
                grad = this.ctx.createRadialGradient(ofs, ofs, 0, ofs, ofs, 0.6 * this.stoneRadius);
                grad.addColorStop(0.4, `rgba(255, 255, 255, ${alpha})`);
                grad.addColorStop(1, `rgba(204, 204, 204, ${alpha})`);
            }
            else {
                throw new Error(`Invalid color ${color}`);
            }
            return grad;
        }
    }
    exports.Board = Board;
    class ClickableBoard extends Board {
        constructor(parent, size, layerDescs) {
            super(parent, size, layerDescs);
            this.size = size;
            this.listeners = [];
            this.enabled = false;
            this.ctx.canvas.addEventListener('mousemove', (e) => {
                let p = this.canvasToBoard(e.offsetX, e.offsetY, 0.4);
                if (p != null && this.getStone(p) != base_1.Color.Empty) {
                    p = null;
                }
                let changed;
                if (p != null) {
                    changed = this.p == null || this.p.row != p.row || this.p.col != p.col;
                }
                else {
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
        onClick(cb) {
            this.listeners.push(cb);
        }
        drawImpl() {
            super.drawImpl();
            let p = this.enabled ? this.p : null;
            this.ctx.canvas.style.cursor = p ? 'pointer' : null;
            if (p) {
                this.drawStones([p], this.toPlay, 0.6);
            }
        }
    }
    exports.ClickableBoard = ClickableBoard;
});
//# sourceMappingURL=board.js.map