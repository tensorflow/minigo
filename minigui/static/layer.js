define(["require", "exports", "./base", "./board", "./util"], function (require, exports, base_1, board_1, util_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.Annotation = board_1.Annotation;
    const STAR_POINTS = {
        [base_1.BoardSize.Nine]: [[2, 2], [2, 6], [6, 2], [6, 6], [4, 4]],
        [base_1.BoardSize.Nineteen]: [[3, 3], [3, 9], [3, 15],
            [9, 3], [9, 9], [9, 15],
            [15, 3], [15, 9], [15, 15]],
    };
    class Layer {
        constructor(board) {
            this.board = board;
            this.boardToCanvas = board.boardToCanvas.bind(board);
        }
    }
    exports.Layer = Layer;
    class StaticLayer extends Layer {
        update(dataObj) {
            return false;
        }
    }
    class DataLayer extends Layer {
        constructor(board, dataPropName) {
            super(board);
            this.dataPropName = dataPropName;
        }
        getData(dataObj) {
            let prop = dataObj[this.dataPropName];
            if (prop === undefined) {
                return undefined;
            }
            return prop;
        }
    }
    class Grid extends StaticLayer {
        constructor() {
            super(...arguments);
            this.style = '#864';
        }
        draw() {
            let starPointRadius = Math.min(4, Math.max(this.board.stoneRadius / 10, 2.5));
            let ctx = this.board.ctx;
            let size = this.board.size;
            let pr = util_1.pixelRatio();
            ctx.strokeStyle = this.style;
            ctx.lineWidth = pr;
            ctx.lineCap = 'round';
            ctx.beginPath();
            for (let i = 0; i < size; ++i) {
                let left = this.boardToCanvas(i, 0);
                let right = this.boardToCanvas(i, size - 1);
                let top = this.boardToCanvas(0, i);
                let bottom = this.boardToCanvas(size - 1, i);
                ctx.moveTo(0.5 + Math.round(left.x), 0.5 + Math.round(left.y));
                ctx.lineTo(0.5 + Math.round(right.x), 0.5 + Math.round(right.y));
                ctx.moveTo(0.5 + Math.round(top.x), 0.5 + Math.round(top.y));
                ctx.lineTo(0.5 + Math.round(bottom.x), 0.5 + Math.round(bottom.y));
            }
            ctx.stroke();
            ctx.fillStyle = this.style;
            ctx.strokeStyle = '';
            for (let p of STAR_POINTS[size]) {
                let c = this.boardToCanvas(p[0], p[1]);
                ctx.beginPath();
                ctx.arc(c.x + 0.5, c.y + 0.5, starPointRadius * pr, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
    }
    exports.Grid = Grid;
    class Label extends StaticLayer {
        draw() {
            let ctx = this.board.ctx;
            let size = this.board.size;
            let textHeight = Math.floor(0.3 * this.board.stoneRadius);
            ctx.font = `${textHeight}px sans-serif`;
            ctx.fillStyle = '#9d7c4d';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'alphabetic';
            for (let i = 0; i < size; ++i) {
                let c = this.boardToCanvas(-0.66, i);
                ctx.fillText(base_1.COL_LABELS[i], c.x, c.y);
            }
            ctx.textBaseline = 'top';
            for (let i = 0; i < size; ++i) {
                let c = this.boardToCanvas(size - 0.33, i);
                ctx.fillText(base_1.COL_LABELS[i], c.x, c.y);
            }
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (let i = 0; i < size; ++i) {
                let c = this.boardToCanvas(i, -0.66);
                ctx.fillText((size - i).toString(), c.x, c.y);
            }
            ctx.textAlign = 'left';
            for (let i = 0; i < size; ++i) {
                let c = this.boardToCanvas(i, size - 0.33);
                ctx.fillText((size - i).toString(), c.x, c.y);
            }
        }
    }
    exports.Label = Label;
    class Caption extends StaticLayer {
        constructor(board, caption) {
            super(board);
            this.caption = caption;
        }
        draw() {
            let ctx = this.board.ctx;
            let size = this.board.size;
            let textHeight = Math.floor(0.4 * this.board.stoneRadius);
            ctx.font = `${textHeight}px sans-serif`;
            ctx.fillStyle = '#9d7c4d';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            let c = this.boardToCanvas(size - 0.45, (size - 1) / 2);
            ctx.fillText(this.caption, c.x, c.y);
        }
    }
    exports.Caption = Caption;
    class HeatMap extends DataLayer {
        constructor(board, dataPropName, colorizeFn) {
            super(board, dataPropName);
            this.colorizeFn = colorizeFn;
            this.colors = null;
        }
        update(dataObj) {
            let data = this.getData(dataObj);
            if (data === undefined) {
                return false;
            }
            this.colors = data != null ? this.colorizeFn(data) : null;
            return true;
        }
        draw() {
            if (!this.colors) {
                return;
            }
            let ctx = this.board.ctx;
            let size = this.board.size;
            let w = this.board.pointW;
            let h = this.board.pointH;
            let stones = this.board.stones;
            let p = { row: 0, col: 0 };
            let i = 0;
            for (p.row = 0; p.row < size; ++p.row) {
                for (p.col = 0; p.col < size; ++p.col) {
                    let rgba = this.colors[i];
                    if (stones[i++] != base_1.Color.Empty) {
                        continue;
                    }
                    ctx.fillStyle = `rgba(${rgba[0]}, ${rgba[1]}, ${rgba[2]}, ${rgba[3]}`;
                    let c = this.boardToCanvas(p.row, p.col);
                    ctx.fillRect(c.x - 0.5 * w, c.y - 0.5 * h, w, h);
                }
            }
        }
    }
    exports.HeatMap = HeatMap;
    class StoneBaseLayer extends DataLayer {
        constructor(board, dataPropName, alpha) {
            super(board, dataPropName);
            this.alpha = alpha;
            this.blackStones = [];
            this.whiteStones = [];
        }
        draw() {
            this.board.drawStones(this.blackStones, base_1.Color.Black, this.alpha);
            this.board.drawStones(this.whiteStones, base_1.Color.White, this.alpha);
        }
    }
    class BoardStones extends StoneBaseLayer {
        constructor(board, dataPropName = 'stones', alpha = 1) {
            super(board, dataPropName, alpha);
        }
        update(dataObj) {
            let stones = this.getData(dataObj);
            if (stones === undefined) {
                return false;
            }
            this.blackStones = [];
            this.whiteStones = [];
            if (stones != null) {
                let size = this.board.size;
                let i = 0;
                for (let row = 0; row < size; ++row) {
                    for (let col = 0; col < size; ++col) {
                        let color = stones[i++];
                        if (color == base_1.Color.Black) {
                            this.blackStones.push({ row: row, col: col });
                        }
                        else if (color == base_1.Color.White) {
                            this.whiteStones.push({ row: row, col: col });
                        }
                    }
                }
            }
            return true;
        }
    }
    exports.BoardStones = BoardStones;
    class Variation extends StoneBaseLayer {
        constructor(board, dataPropName, alpha = 0.4) {
            super(board, dataPropName, alpha);
            this.blackLabels = [];
            this.whiteLabels = [];
        }
        update(dataObj) {
            let variation = this.getData(dataObj);
            if (variation === undefined) {
                return false;
            }
            let toPlay = this.board.toPlay;
            let size = this.board.size;
            this.blackStones = [];
            this.whiteStones = [];
            this.blackLabels = [];
            this.whiteLabels = [];
            if (variation == null) {
                return true;
            }
            let playedCount = new Uint16Array(size * size);
            let firstPlayed = [];
            toPlay = base_1.otherColor(toPlay);
            for (let i = 0; i < variation.length; ++i) {
                let move = variation[i];
                toPlay = base_1.otherColor(toPlay);
                if (move == 'pass' || move == 'resign') {
                    continue;
                }
                let idx = move.row * size + move.col;
                let label = { p: move, s: (i + 1).toString() };
                let count = ++playedCount[idx];
                if (toPlay == base_1.Color.Black) {
                    this.blackStones.push(move);
                    if (count == 1) {
                        this.blackLabels.push(label);
                    }
                }
                else {
                    this.whiteStones.push(move);
                    if (count == 1) {
                        this.whiteLabels.push(label);
                    }
                }
                if (count == 1) {
                    firstPlayed[idx] = label;
                }
                else if (count == 2) {
                    firstPlayed[idx].s += '*';
                }
            }
            return true;
        }
        draw() {
            super.draw();
            let ctx = this.board.ctx;
            let size = this.board.size;
            let textHeight = Math.floor(0.5 * this.board.stoneRadius);
            ctx.font = `${textHeight}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            this.drawLabels(this.blackLabels, '#fff');
            this.drawLabels(this.whiteLabels, '#000');
        }
        drawLabels(labels, style) {
            let ctx = this.board.ctx;
            ctx.fillStyle = style;
            for (let label of labels) {
                let c = this.boardToCanvas(label.p.row, label.p.col);
                ctx.fillText(label.s, c.x, c.y);
            }
        }
    }
    exports.Variation = Variation;
    class Annotations extends DataLayer {
        constructor(board, dataPropName = 'annotations') {
            super(board, dataPropName);
            this.annotations = [];
        }
        update(dataObj) {
            let annotations = this.getData(dataObj);
            if (annotations === undefined) {
                return false;
            }
            this.annotations = annotations != null ? annotations : [];
            return true;
        }
        draw() {
            if (this.annotations == null || this.annotations.length == 0) {
                return;
            }
            let ctx = this.board.ctx;
            for (let annotation of this.annotations) {
                let c = this.boardToCanvas(annotation.p.row, annotation.p.col);
                let sr = this.board.stoneRadius;
                switch (annotation.shape) {
                    case board_1.Annotation.Shape.Dot:
                        ctx.fillStyle = annotation.color;
                        ctx.beginPath();
                        ctx.arc(c.x, c.y, 0.08 * sr, 0, 2 * Math.PI);
                        ctx.fill();
                        break;
                    case board_1.Annotation.Shape.Triangle:
                        ctx.lineWidth = 3 * util_1.pixelRatio();
                        ctx.lineCap = 'round';
                        ctx.strokeStyle = annotation.color;
                        ctx.beginPath();
                        ctx.moveTo(c.x, c.y - 0.35 * sr);
                        ctx.lineTo(c.x - 0.3 * sr, c.y + 0.21 * sr);
                        ctx.lineTo(c.x + 0.3 * sr, c.y + 0.21 * sr);
                        ctx.lineTo(c.x, c.y - 0.35 * sr);
                        ctx.stroke();
                        break;
                }
            }
        }
    }
    exports.Annotations = Annotations;
});
//# sourceMappingURL=layer.js.map