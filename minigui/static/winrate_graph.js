define(["require", "exports", "./util", "./graph"], function (require, exports, util_1, graph_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
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
    class WinrateGraph extends graph_1.Graph {
        constructor(parent) {
            super(parent, {
                xStart: 0,
                xEnd: 10,
                yStart: 1,
                yEnd: -1,
                marginTop: 0.05,
                marginBottom: 0.05,
                marginLeft: 0.075,
                marginRight: 0.125,
            });
            this.mainLine = [];
            this.variation = [];
            this.rootPosition = null;
            this.activePosition = null;
            if (typeof (parent) == 'string') {
                parent = util_1.getElement(parent);
            }
        }
        newGame(rootPosition) {
            this.rootPosition = rootPosition;
            this.activePosition = rootPosition;
            this.mainLine = [];
            this.variation = [];
            this.xEnd = 10;
            this.draw();
        }
        setActive(position) {
            if (position != this.activePosition) {
                this.xEnd = Math.max(this.xEnd, position.moveNum);
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
            super.drawImpl();
            let pr = util_1.pixelRatio();
            let ctx = this.ctx;
            let textHeight = 0.25 * this.marginRight;
            ctx.font = `${textHeight}px sans-serif`;
            ctx.fillStyle = '#96928f';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            this.drawText('B', -4 / this.xScale, 0.95);
            this.drawText('W', -4 / this.xScale, -0.95);
            if (this.activePosition == null) {
                return;
            }
            let moveNum = this.activePosition.moveNum;
            this.drawPlot([[moveNum, this.yStart], [moveNum, this.yEnd]], {
                dash: [0, 3],
                width: 1,
                style: '#96928f',
                snap: true,
            });
            if (this.activePosition.isMainLine) {
                this.drawVariation('#ffe', this.mainLine);
            }
            else {
                this.drawVariation('#615b56', this.mainLine);
                this.drawVariation('#ffe', this.variation);
            }
            ctx.textAlign = 'left';
            ctx.fillStyle = '#ffe';
            let q = 0;
            let values = this.activePosition.isMainLine ? this.mainLine : this.variation;
            if (values.length > 0) {
                q = values[Math.min(moveNum, values.length - 1)];
            }
            let score = 50 + 50 * q;
            let txt;
            if (score > 50) {
                txt = `B:${Math.round(score)}%`;
            }
            else {
                txt = `W:${Math.round(100 - score)}%`;
            }
            this.drawText(txt, this.xEnd + 4 / this.xScale, q);
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
        drawVariation(style, values) {
            if (values.length < 2) {
                return;
            }
            let points = [];
            for (let i = 0; i < values.length; ++i) {
                if (values[i] != null) {
                    points[i] = [i, values[i]];
                }
            }
            super.drawPlot(points, {
                width: 1,
                style: style,
            });
        }
    }
    exports.WinrateGraph = WinrateGraph;
});
//# sourceMappingURL=winrate_graph.js.map