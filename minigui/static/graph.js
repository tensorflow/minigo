define(["require", "exports", "./util", "./view"], function (require, exports, util_1, view_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    const DEFAULT_PLOT_OPTIONS = {
        width: 1,
        snap: false,
    };
    class Graph extends view_1.View {
        constructor(parent, options) {
            super();
            this.options = options;
            this.lineDash = false;
            this.xTickPoints = [];
            this.yTickPoints = [];
            this.xStart = 0;
            this.xEnd = 1;
            this.yStart = 0;
            this.yEnd = 1;
            this.moveNum = 0;
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
        resizeCanvas() {
            let pr = util_1.pixelRatio();
            let ctx = this.ctx;
            let canvas = ctx.canvas;
            let parent = canvas.parentElement;
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
        setMoveNum(moveNum) {
            if (moveNum != this.moveNum) {
                this.moveNum = moveNum;
                this.draw();
            }
        }
        drawImpl() {
            this.updateScale();
            let pr = util_1.pixelRatio();
            let ctx = this.ctx;
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
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
        drawText(text, x, y, snap = false) {
            x = this.xScale * (x - this.xStart);
            y = this.yScale * (y - this.yStart);
            if (snap) {
                x = Math.round(x);
                y = Math.round(y);
            }
            this.ctx.fillText(text, x, y);
        }
        drawPlot(points, options = DEFAULT_PLOT_OPTIONS) {
            if (points.length == 0) {
                return;
            }
            let pr = util_1.pixelRatio();
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
        beginPath(dash = null) {
            let ctx = this.ctx;
            if (dash != null) {
                ctx.lineCap = 'square';
                ctx.setLineDash(dash);
                this.lineDash = true;
            }
            else if (this.lineDash) {
                ctx.lineCap = 'round';
                ctx.setLineDash([]);
            }
            ctx.beginPath();
        }
        stroke() {
            this.ctx.stroke();
        }
        moveTo(x, y, snap = false) {
            x = this.xScale * (x - this.xStart);
            y = this.yScale * (y - this.yStart);
            if (snap) {
                x = Math.round(x);
                y = Math.round(y);
            }
            this.ctx.moveTo(x, y);
        }
        lineTo(x, y, snap = false) {
            x = this.xScale * (x - this.xStart);
            y = this.yScale * (y - this.yStart);
            if (snap) {
                x = Math.round(x);
                y = Math.round(y);
            }
            this.ctx.lineTo(x, y);
        }
        updateScale() {
            let canvas = this.ctx.canvas;
            let w = canvas.width - this.marginLeft - this.marginRight;
            let h = canvas.height - this.marginTop - this.marginBottom;
            this.xScale = w / (this.xEnd - this.xStart);
            this.yScale = h / (this.yEnd - this.yStart);
        }
        calculateTickPoints(start, end) {
            let x = Math.abs(end - start);
            let spacing = 1;
            if (x >= 1) {
                let scale = Math.pow(10, Math.max(0, Math.floor(Math.log10(x) - 1)));
                let top2 = Math.floor(x / scale);
                if (top2 <= 10) {
                    spacing = 1;
                }
                else if (top2 <= 20) {
                    spacing = 2;
                }
                else if (top2 <= 50) {
                    spacing = 5;
                }
                else {
                    spacing = 10;
                }
                spacing *= scale;
            }
            let min = Math.min(start, end);
            let max = Math.max(start, end);
            let positive = [];
            let negative = [];
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
    exports.Graph = Graph;
});
//# sourceMappingURL=graph.js.map