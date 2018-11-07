define(["require", "exports", "./util"], function (require, exports, util_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class WinrateGraph {
        constructor(parent) {
            this.points = new Array();
            this.minPoints = 10;
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
            this.points = [];
            this.draw();
        }
        setWinrate(move, winrate) {
            this.points[move] = [move, winrate];
            this.draw();
        }
        draw() {
            let pr = util_1.pixelRatio();
            let ctx = this.ctx;
            let w = this.w;
            let h = this.h;
            let xScale = Math.max(this.points.length - 1, this.minPoints);
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            ctx.translate(this.marginLeft + 0.5, this.marginTop + 0.5);
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.lineWidth = pr;
            ctx.strokeStyle = '#56504b';
            ctx.beginPath();
            ctx.moveTo(0, Math.round(0.95 * h));
            ctx.lineTo(w, Math.round(0.95 * h));
            ctx.moveTo(0, Math.round(0.05 * h));
            ctx.lineTo(w, Math.round(0.05 * h));
            ctx.stroke();
            let lineWidth = 3 * pr;
            ctx.lineWidth = lineWidth;
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
            let xOfs = Math.floor(lineWidth / 2);
            ctx.translate(xOfs, 0);
            w -= xOfs;
            if (this.points.length >= 2) {
                ctx.lineWidth = pr;
                ctx.strokeStyle = '#eee';
                ctx.beginPath();
                let [x, y] = this.points[0];
                ctx.moveTo(w * x / xScale, h * (0.5 - 0.5 * y));
                for (let i = 1; i < this.points.length; ++i) {
                    [x, y] = this.points[i];
                    ctx.lineTo(w * x / xScale, h * (0.5 - 0.5 * y));
                }
                ctx.stroke();
            }
            ctx.textAlign = 'left';
            ctx.fillStyle = '#ffe';
            let y;
            if (this.points.length > 0) {
                y = this.points[this.points.length - 1][1];
            }
            else {
                y = 0;
            }
            let score = y;
            y = h * (0.5 - 0.5 * y);
            let txt;
            if (score > 0) {
                txt = `B:${Math.round(score * 100)}%`;
            }
            else {
                txt = `W:${Math.round(-score * 100)}%`;
            }
            ctx.fillText(txt, w + 8, y);
        }
    }
    exports.WinrateGraph = WinrateGraph;
});
//# sourceMappingURL=winrate_graph.js.map