define(["require", "exports", "./base", "./layer", "./util", "./view"], function (require, exports, base_1, layer_1, util_1, view_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.COL_LABELS = base_1.COL_LABELS;
    class Board extends view_1.View {
        constructor(parent, position, layers) {
            super();
            this.position = position;
            this.layers = [];
            if (typeof (parent) == 'string') {
                parent = util_1.getElement(parent);
            }
            this.elem = parent;
            this.backgroundColor = '#db6';
            let canvas = document.createElement('canvas');
            this.ctx = canvas.getContext('2d');
            parent.appendChild(canvas);
            window.addEventListener('resize', () => {
                this.resizeCanvas();
                this.draw();
            });
            this.resizeCanvas();
            this.addLayer(new layer_1.Grid());
            this.addLayers(layers);
        }
        resizeCanvas() {
            let pr = util_1.pixelRatio();
            let canvas = this.ctx.canvas;
            let parent = canvas.parentElement;
            canvas.width = pr * (parent.offsetWidth);
            canvas.height = pr * (parent.offsetHeight);
            canvas.style.width = `${parent.offsetWidth}px`;
            canvas.style.height = `${parent.offsetHeight}px`;
            this.pointW = this.ctx.canvas.width / (base_1.N + 1);
            this.pointH = this.ctx.canvas.height / (base_1.N + 1);
            this.stoneRadius = 0.96 * Math.min(this.pointW, this.pointH) / 2;
        }
        newGame(rootPosition) {
            this.position = rootPosition;
            for (let layer of this.layers) {
                layer.clear();
            }
            this.draw();
        }
        addLayer(layer) {
            this.layers.push(layer);
            layer.addToBoard(this);
        }
        addLayers(layers) {
            for (let layer of layers) {
                this.addLayer(layer);
            }
        }
        setPosition(position) {
            if (this.position == position) {
                return;
            }
            this.position = position;
            let allProps = new Set(Object.keys(position));
            for (let layer of this.layers) {
                layer.update(allProps);
            }
            this.draw();
        }
        update(update) {
            let anythingChanged = false;
            let keys = new Set(Object.keys(update));
            for (let layer of this.layers) {
                if (layer.update(keys)) {
                    anythingChanged = true;
                }
            }
            if (anythingChanged) {
                this.draw();
            }
        }
        drawImpl() {
            let ctx = this.ctx;
            ctx.fillStyle = this.backgroundColor;
            ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            for (let layer of this.layers) {
                if (layer.show) {
                    layer.draw();
                }
            }
        }
        getStone(p) {
            return this.position.stones[p.row * base_1.N + p.col];
        }
        canvasToBoard(x, y, threshold) {
            let pr = util_1.pixelRatio();
            x *= pr;
            y *= pr;
            let canvas = this.ctx.canvas;
            y = y * (base_1.N + 1) / canvas.height - 0.5;
            x = x * (base_1.N + 1) / canvas.width - 0.5;
            let row = Math.floor(y);
            let col = Math.floor(x);
            if (row < 0 || row >= base_1.N || col < 0 || col >= base_1.N) {
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
            return {
                x: canvas.width * (col + 1.0) / (base_1.N + 1),
                y: canvas.height * (row + 1.0) / (base_1.N + 1)
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
            let r = this.stoneRadius;
            for (let p of ps) {
                let c = this.boardToCanvas(p.row, p.col);
                ctx.beginPath();
                ctx.translate(c.x + 0.5, c.y + 0.5);
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
                let ofs = -0.5 * this.stoneRadius;
                grad = this.ctx.createRadialGradient(ofs, ofs, 0, ofs, ofs, 2 * this.stoneRadius);
                grad.addColorStop(0, `rgba(68, 68, 68, ${alpha})`);
                grad.addColorStop(1, `rgba(16, 16, 16, ${alpha})`);
            }
            else if (color == base_1.Color.White) {
                let ofs = -0.2 * this.stoneRadius;
                grad = this.ctx.createRadialGradient(ofs, ofs, 0, ofs, ofs, 1.2 * this.stoneRadius);
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
        constructor(parent, position, layerDescs) {
            super(parent, position, layerDescs);
            this.enabled = false;
            this.listeners = [];
            this.ctx.canvas.addEventListener('mousemove', (e) => {
                let p = this.canvasToBoard(e.offsetX, e.offsetY, 0.45);
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
        setPosition(position) {
            if (position != this.position) {
                this.p = null;
                super.setPosition(position);
            }
        }
        drawImpl() {
            super.drawImpl();
            let p = this.enabled ? this.p : null;
            this.ctx.canvas.style.cursor = p ? 'pointer' : null;
            if (p) {
                this.drawStones([p], this.position.toPlay, 0.6);
            }
        }
    }
    exports.ClickableBoard = ClickableBoard;
});
//# sourceMappingURL=board.js.map