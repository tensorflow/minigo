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
            this.moveNum = 0;
            this.xScale = MIN_POINTS;
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
        clear() {
            this.mainLine = [];
            this.variation = [];
            this.moveNum = 0;
            this.draw();
        }
        update(position) {
            let anythingChanged = position.moveNum != this.moveNum;
            this.moveNum = position.moveNum;
            while (position.children.length > 0) {
                position = position.children[0];
            }
            let values = [];
            let p = position;
            while (p != null) {
                values.push(p.q);
                p = p.parent;
            }
            values.reverse();
            if (position.isMainLine) {
                this.mainLine = values;
                this.variation = [];
                anythingChanged =
                    anythingChanged || !arraysApproxEqual(values, this.mainLine, 0.001);
            }
            else {
                this.variation = values;
                anythingChanged =
                    anythingChanged || arraysApproxEqual(values, this.variation, 0.001);
            }
            if (anythingChanged) {
                this.xScale = Math.max(this.mainLine.length - 1, this.variation.length - 1, MIN_POINTS);
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
            ctx.translate(this.marginLeft + 0.5, this.marginTop + 0.5);
            ctx.lineWidth = pr;
            ctx.strokeStyle = '#96928f';
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(0, h);
            ctx.moveTo(0, Math.floor(0.5 * h));
            ctx.lineTo(w, Math.floor(0.5 * h));
            ctx.stroke();
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(Math.round(w * this.moveNum / this.xScale), 0);
            ctx.lineTo(Math.round(w * this.moveNum / this.xScale), h);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.font = `${this.textHeight}px sans-serif`;
            ctx.fillStyle = '#96928f';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText('B', -0.5 * this.textHeight, Math.round(0.05 * h));
            ctx.fillText('W', -0.5 * this.textHeight, Math.round(0.95 * h));
            if (this.variation.length == 0) {
                this.drawPlot(this.mainLine, pr, '#ffe');
            }
            else {
                this.drawPlot(this.mainLine, pr, '#96928f');
                this.drawPlot(this.variation, pr, '#ffe');
            }
            ctx.textAlign = 'left';
            ctx.fillStyle = '#ffe';
            let y = 0;
            let values = this.variation.length > 0 ? this.variation : this.mainLine;
            if (values.length > 0) {
                y = values[this.moveNum];
            }
            let score = 50 + 50 * y;
            y = h * (0.5 - 0.5 * y);
            let txt;
            if (score > 50) {
                txt = `B:${Math.round(score)}%`;
            }
            else {
                txt = `W:${Math.round(100 - score)}%`;
            }
            ctx.fillText(txt, w + 8, y);
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