define(["require", "exports", "./util", "./view"], function (require, exports, util_1, view_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    const MIN_POINTS = 10;
    function arraysApproxEqual(a, b, threshold) {
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
    class WinrateGraph extends view_1.View {
        constructor(parent) {
            super();
            this.mainLine = [];
            this.variation = [];
            this.xScale = MIN_POINTS;
            this.rootPosition = null;
            this.activePosition = null;
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
        }
        resizeCanvas() {
            let pr = util_1.pixelRatio();
            let canvas = this.ctx.canvas;
            let parent = canvas.parentElement;
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
        newGame(rootPosition) {
            this.rootPosition = rootPosition;
            this.activePosition = rootPosition;
            this.mainLine = [];
            this.variation = [];
            this.xScale = MIN_POINTS;
            this.draw();
        }
        setActive(position) {
            if (position != this.activePosition) {
                this.xScale = Math.max(this.xScale, position.moveNum);
                this.activePosition = position;
                this.update(position);
                this.draw();
            }
        }
        update(position) {
            if (this.rootPosition == null || this.activePosition == null) {
                return;
            }
            if (!position.isMainLine) {
                if (this.activePosition.isMainLine) {
                    return;
                }
                else if (this.activePosition.getFullLine().indexOf(position) == -1) {
                    return;
                }
            }
            let anythingChanged = false;
            let mainLine = this.getWinRate(this.rootPosition.getFullLine());
            if (!arraysApproxEqual(mainLine, this.mainLine, 0.001)) {
                anythingChanged = true;
                this.mainLine = mainLine;
            }
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
            let pr = util_1.pixelRatio();
            let ctx = this.ctx;
            let w = this.w;
            let h = this.h;
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            ctx.lineCap = 'square';
            ctx.lineJoin = 'miter';
            ctx.translate(this.marginLeft + 0.5, this.marginTop + 0.5);
            ctx.lineWidth = pr;
            ctx.strokeStyle = '#96928f';
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(0, h);
            ctx.moveTo(0, Math.floor(0.5 * h));
            ctx.lineTo(w, Math.floor(0.5 * h));
            ctx.stroke();
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
            }
            else {
                this.drawPlot(this.mainLine, pr, '#615b56');
                this.drawPlot(this.variation, pr, '#ffe');
            }
            ctx.textAlign = 'left';
            ctx.fillStyle = '#ffe';
            let q = 0;
            let values = this.activePosition.isMainLine ? this.mainLine : this.variation;
            if (values.length > 0) {
                q = values[Math.min(moveNum, values.length - 1)];
            }
            let score = 50 + 50 * q;
            let y = h * (0.5 - 0.5 * q);
            let txt;
            if (score > 50) {
                txt = `B:${Math.round(score)}%`;
            }
            else {
                txt = `W:${Math.round(100 - score)}%`;
            }
            ctx.fillText(txt, w + 8, y);
        }
        getWinRate(variation) {
            let result = [];
            for (let p of variation) {
                if (p.q == null) {
                    break;
                }
                result.push(p.q);
            }
            return result;
        }
        drawPlot(values, lineWidth, style) {
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
    exports.WinrateGraph = WinrateGraph;
});
//# sourceMappingURL=winrate_graph.js.map