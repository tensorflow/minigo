define(["require", "exports", "./position", "./base", "./util"], function (require, exports, position_1, base_1, util_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    const STAR_POINTS = {
        [base_1.BoardSize.Nine]: [[2, 2], [2, 6], [6, 2], [6, 6], [4, 4]],
        [base_1.BoardSize.Nineteen]: [[3, 3], [3, 9], [3, 15],
            [9, 3], [9, 9], [9, 15],
            [15, 3], [15, 9], [15, 15]],
    };
    class Layer {
        constructor() {
            this._show = true;
        }
        get show() {
            return this._show;
        }
        set show(x) {
            if (x != this._show) {
                this._show = x;
                this.board.draw();
            }
        }
        addToBoard(board) {
            this.board = board;
            this.boardToCanvas = board.boardToCanvas.bind(board);
        }
    }
    exports.Layer = Layer;
    class StaticLayer extends Layer {
        clear() { }
        update(props) {
            return false;
        }
    }
    class Grid extends StaticLayer {
        constructor() {
            super(...arguments);
            this.style = '#864';
        }
        draw() {
            let starPointRadius = Math.min(4, Math.max(this.board.stoneRadius / 5, 2.5));
            let ctx = this.board.ctx;
            let pr = util_1.pixelRatio();
            ctx.strokeStyle = this.style;
            ctx.lineWidth = pr;
            ctx.lineCap = 'round';
            ctx.beginPath();
            for (let i = 0; i < base_1.N; ++i) {
                let left = this.boardToCanvas(i, 0);
                let right = this.boardToCanvas(i, base_1.N - 1);
                let top = this.boardToCanvas(0, i);
                let bottom = this.boardToCanvas(base_1.N - 1, i);
                ctx.moveTo(0.5 + Math.round(left.x), 0.5 + Math.round(left.y));
                ctx.lineTo(0.5 + Math.round(right.x), 0.5 + Math.round(right.y));
                ctx.moveTo(0.5 + Math.round(top.x), 0.5 + Math.round(top.y));
                ctx.lineTo(0.5 + Math.round(bottom.x), 0.5 + Math.round(bottom.y));
            }
            ctx.stroke();
            ctx.fillStyle = this.style;
            ctx.strokeStyle = '';
            for (let p of STAR_POINTS[base_1.N]) {
                let c = this.boardToCanvas(p[0], p[1]);
                ctx.beginPath();
                ctx.arc(c.x + 0.5, c.y + 0.5, starPointRadius * pr, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
    }
    exports.Grid = Grid;
    class Label extends StaticLayer {
        constructor(size = 0.6) {
            super();
            this.size = size;
        }
        draw() {
            let ctx = this.board.ctx;
            let textHeight = Math.floor(this.size * this.board.stoneRadius);
            ctx.font = `${textHeight}px sans-serif`;
            ctx.fillStyle = '#9d7c4d';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'alphabetic';
            for (let i = 0; i < base_1.N; ++i) {
                let c = this.boardToCanvas(-0.66, i);
                ctx.fillText(base_1.COL_LABELS[i], c.x, c.y);
            }
            ctx.textBaseline = 'top';
            for (let i = 0; i < base_1.N; ++i) {
                let c = this.boardToCanvas(base_1.N - 0.33, i);
                ctx.fillText(base_1.COL_LABELS[i], c.x, c.y);
            }
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (let i = 0; i < base_1.N; ++i) {
                let c = this.boardToCanvas(i, -0.66);
                ctx.fillText((base_1.N - i).toString(), c.x, c.y);
            }
            ctx.textAlign = 'left';
            for (let i = 0; i < base_1.N; ++i) {
                let c = this.boardToCanvas(i, base_1.N - 0.33);
                ctx.fillText((base_1.N - i).toString(), c.x, c.y);
            }
        }
    }
    exports.Label = Label;
    class Caption extends StaticLayer {
        constructor(caption) {
            super();
            this.caption = caption;
        }
        draw() {
            let ctx = this.board.ctx;
            let textHeight = Math.floor(0.8 * this.board.stoneRadius);
            ctx.font = `${textHeight}px sans-serif`;
            ctx.fillStyle = '#9d7c4d';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            let c = this.boardToCanvas(base_1.N - 0.45, (base_1.N - 1) / 2);
            ctx.fillText(this.caption, c.x, c.y);
        }
    }
    exports.Caption = Caption;
    class HeatMapBase extends Layer {
        constructor() {
            super(...arguments);
            this.colors = [];
        }
        clear() {
            if (this.colors.length > 0) {
                this.colors = [];
                this.board.draw();
            }
        }
        draw() {
            if (this.colors.length == 0) {
                return;
            }
            let ctx = this.board.ctx;
            let w = this.board.pointW;
            let h = this.board.pointH;
            let stones = this.board.position.stones;
            let p = { row: 0, col: 0 };
            let i = 0;
            for (p.row = 0; p.row < base_1.N; ++p.row) {
                for (p.col = 0; p.col < base_1.N; ++p.col) {
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
    (function (HeatMapBase) {
        HeatMapBase.TRANSPARENT = new Float32Array([0, 0, 0, 0]);
    })(HeatMapBase || (HeatMapBase = {}));
    class VisitCountHeatMap extends HeatMapBase {
        update(props) {
            if (!props.has('childN')) {
                return false;
            }
            this.colors = [];
            if (this.board.position.childN == null) {
                return true;
            }
            let n = Math.max(this.board.position.n, 1);
            for (let childN of this.board.position.childN) {
                if (childN == 0) {
                    this.colors.push(HeatMapBase.TRANSPARENT);
                }
                else {
                    let a = Math.min(Math.sqrt(childN / n), 0.6);
                    if (a > 0) {
                        a = 0.1 + 0.9 * a;
                    }
                    this.colors.push(new Float32Array([0, 0, 0, a]));
                }
            }
            return true;
        }
    }
    exports.VisitCountHeatMap = VisitCountHeatMap;
    class DeltaQHeatMap extends HeatMapBase {
        update(props) {
            if (!props.has('childQ')) {
                return false;
            }
            let position = this.board.position;
            this.colors = [];
            if (position.childQ == null) {
                return true;
            }
            let q = position.q || 0;
            for (let i = 0; i < base_1.N * base_1.N; ++i) {
                if (position.stones[i] != base_1.Color.Empty) {
                    this.colors.push(HeatMapBase.TRANSPARENT);
                }
                else {
                    let childQ = position.childQ[i];
                    let dq = childQ - q;
                    let rgb = dq > 0 ? 0 : 255;
                    let a = Math.min(Math.abs(dq), 0.6);
                    this.colors.push(new Float32Array([rgb, rgb, rgb, a]));
                }
            }
            return true;
        }
    }
    exports.DeltaQHeatMap = DeltaQHeatMap;
    class StoneBaseLayer extends Layer {
        constructor(alpha) {
            super();
            this.alpha = alpha;
            this.blackStones = [];
            this.whiteStones = [];
        }
        clear() {
            if (this.blackStones.length > 0 || this.whiteStones.length > 0) {
                this.blackStones = [];
                this.whiteStones = [];
                this.board.draw();
            }
        }
        draw() {
            this.board.drawStones(this.blackStones, base_1.Color.Black, this.alpha);
            this.board.drawStones(this.whiteStones, base_1.Color.White, this.alpha);
        }
    }
    class BoardStones extends StoneBaseLayer {
        constructor() {
            super(1);
        }
        update(props) {
            if (!props.has('stones')) {
                return false;
            }
            let position = this.board.position;
            this.blackStones = [];
            this.whiteStones = [];
            let i = 0;
            for (let row = 0; row < base_1.N; ++row) {
                for (let col = 0; col < base_1.N; ++col) {
                    let color = position.stones[i++];
                    if (color == base_1.Color.Black) {
                        this.blackStones.push({ row: row, col: col });
                    }
                    else if (color == base_1.Color.White) {
                        this.whiteStones.push({ row: row, col: col });
                    }
                }
            }
            return true;
        }
    }
    exports.BoardStones = BoardStones;
    class Variation extends StoneBaseLayer {
        constructor(_showVariation, alpha = 0.4) {
            super(alpha);
            this._showVariation = _showVariation;
            this.variation = [];
            this.blackLabels = [];
            this.whiteLabels = [];
        }
        get showVariation() {
            return this._showVariation;
        }
        set showVariation(x) {
            if (x == this._showVariation) {
                return;
            }
            this._showVariation = x;
            this.update(new Set(["variations"]));
            this.board.draw();
        }
        clear() {
            super.clear();
            this.variation = [];
            this.blackLabels = [];
            this.whiteLabels = [];
        }
        update(props) {
            if (!props.has("variations")) {
                return false;
            }
            let position = this.board.position;
            let variation = position.variations.get(this.showVariation);
            if (variation === undefined) {
                this.clear();
                return false;
            }
            if (this.variationsEqual(variation.moves, this.variation)) {
                return false;
            }
            this.variation = variation.moves.slice(0);
            this.parseVariation(this.variation);
            return true;
        }
        variationsEqual(a, b) {
            if (a.length != b.length) {
                return false;
            }
            for (let i = 0; i < a.length; ++i) {
                if (!base_1.movesEqual(a[i], b[i])) {
                    return false;
                }
            }
            return true;
        }
        parseVariation(variation) {
            let toPlay = this.board.position.toPlay;
            this.blackStones = [];
            this.whiteStones = [];
            this.blackLabels = [];
            this.whiteLabels = [];
            if (variation.length == 0) {
                return true;
            }
            let playedCount = new Uint16Array(base_1.N * base_1.N);
            let firstPlayed = [];
            toPlay = base_1.otherColor(toPlay);
            for (let i = 0; i < variation.length; ++i) {
                let move = variation[i];
                toPlay = base_1.otherColor(toPlay);
                if (move == 'pass' || move == 'resign') {
                    continue;
                }
                let idx = move.row * base_1.N + move.col;
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
            let textHeight = Math.floor(this.board.stoneRadius);
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
    (function (Variation) {
        let Type;
        (function (Type) {
            Type[Type["Principal"] = 0] = "Principal";
            Type[Type["CurrentSearch"] = 1] = "CurrentSearch";
            Type[Type["SpecificMove"] = 2] = "SpecificMove";
        })(Type = Variation.Type || (Variation.Type = {}));
    })(Variation || (Variation = {}));
    exports.Variation = Variation;
    class Annotations extends Layer {
        constructor() {
            super(...arguments);
            this.annotations = new Map();
        }
        clear() {
            if (this.annotations.size > 0) {
                this.annotations.clear();
                this.board.draw();
            }
        }
        update(props) {
            if (!props.has('annotations')) {
                return false;
            }
            let position = this.board.position;
            this.annotations.clear();
            for (let annotation of position.annotations) {
                let byShape = this.annotations.get(annotation.shape);
                if (byShape === undefined) {
                    byShape = [];
                    this.annotations.set(annotation.shape, byShape);
                }
                byShape.push(annotation);
            }
            return true;
        }
        draw() {
            if (this.annotations.size == 0) {
                return;
            }
            let sr = this.board.stoneRadius;
            let pr = util_1.pixelRatio();
            let ctx = this.board.ctx;
            ctx.lineCap = 'round';
            this.annotations.forEach((annotations, shape) => {
                switch (shape) {
                    case position_1.Annotation.Shape.Dot:
                        for (let annotation of annotations) {
                            let c = this.boardToCanvas(annotation.p.row, annotation.p.col);
                            ctx.fillStyle = annotation.colors[0];
                            ctx.beginPath();
                            ctx.arc(c.x + 0.5, c.y + 0.5, 0.16 * sr, 0, 2 * Math.PI);
                            ctx.fill();
                        }
                        break;
                }
            });
        }
    }
    exports.Annotations = Annotations;
    class Search extends Variation {
        constructor() {
            super('');
            this.variations = [];
        }
        addToBoard(board) {
            super.addToBoard(board);
            this.board.ctx.canvas.addEventListener('mousemove', (e) => {
                let p = this.board.canvasToBoard(e.offsetX, e.offsetY, 0.45);
                if (p != null && this.hasVariation(p)) {
                    let gtp = base_1.toGtp(p);
                    if (this.showVariation == gtp) {
                        return;
                    }
                    this.showVariation = base_1.toGtp(p);
                }
                else if (this.showVariation != '') {
                    this.showVariation = '';
                }
            });
            this.board.ctx.canvas.addEventListener('mouseleave', () => {
                this.showVariation = '';
            });
        }
        hasVariation(p) {
            for (let v of this.variations) {
                if (base_1.movesEqual(p, v.p)) {
                    return true;
                }
            }
            return false;
        }
        update(props) {
            return super.update(props) || props.has('variations');
        }
        draw() {
            if (this.showVariation != '') {
                super.draw();
                return;
            }
            let ctx = this.board.ctx;
            let pr = util_1.pixelRatio();
            let toPlay = this.board.position.toPlay;
            let stoneRgb = toPlay == base_1.Color.Black ? 0 : 255;
            let maxN = 0;
            this.board.position.variations.forEach((v) => {
                maxN = Math.max(maxN, v.n);
            });
            if (maxN <= 1) {
                return;
            }
            let logMaxN = Math.log(maxN);
            this.variations = [];
            this.board.position.variations.forEach((v, key) => {
                try {
                    if (v.n == 0 || v.moves.length == 0 || !base_1.moveIsPoint(util_1.parseMove(key))) {
                        return;
                    }
                }
                catch {
                    return;
                }
                let alpha = Math.log(v.n) / logMaxN;
                alpha = Math.max(0, Math.min(alpha, 1));
                if (alpha < 0.1) {
                    return;
                }
                this.variations.push({
                    p: v.moves[0],
                    alpha: alpha,
                    q: v.q,
                });
            });
            for (let v of this.variations) {
                let c = this.boardToCanvas(v.p.row, v.p.col);
                let x = c.x + 0.5;
                let y = c.y + 0.5;
                ctx.fillStyle = `rgba(${stoneRgb}, ${stoneRgb}, ${stoneRgb}, ${v.alpha})`;
                ctx.beginPath();
                ctx.arc(x, y, 0.85 * this.board.stoneRadius, 0, 2 * Math.PI);
                ctx.fill();
            }
            let textHeight = Math.floor(0.8 * this.board.stoneRadius);
            ctx.font = `${textHeight}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = toPlay == base_1.Color.Black ? '#fff' : '#000';
            let scoreScale = this.board.position.toPlay == base_1.Color.Black ? 1 : -1;
            for (let v of this.variations) {
                let c = this.boardToCanvas(v.p.row, v.p.col);
                let winRate = 50 + 50 * scoreScale * v.q;
                ctx.fillText(winRate.toFixed(1), c.x, c.y);
            }
            ;
        }
    }
    exports.Search = Search;
});
//# sourceMappingURL=layer.js.map