define(["require", "exports", "./base", "./util"], function (require, exports, base_1, util) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    var Annotation;
    (function (Annotation) {
        let Shape;
        (function (Shape) {
            Shape[Shape["Dot"] = 0] = "Dot";
        })(Shape = Annotation.Shape || (Annotation.Shape = {}));
    })(Annotation || (Annotation = {}));
    exports.Annotation = Annotation;
    class Position {
        constructor(j) {
            this.parent = null;
            this.lastMove = null;
            this.isMainLine = true;
            this.n = 0;
            this.q = null;
            this.variations = new Map();
            this.annotations = [];
            this.childN = null;
            this.childQ = null;
            this.children = [];
            this.captures = [0, 0];
            this.comment = "";
            this.treeStats = {
                numNodes: 0,
                numLeafNodes: 0,
                maxDepth: 0,
            };
            this.id = j.id;
            this.moveNum = j.moveNum;
            this.toPlay = util.parseColor(j.toPlay);
            this.stones = [];
            if (j.stones !== undefined) {
                const stoneMap = {
                    '.': base_1.Color.Empty,
                    'X': base_1.Color.Black,
                    'O': base_1.Color.White,
                };
                for (let i = 0; i < base_1.N * base_1.N; ++i) {
                    this.stones.push(stoneMap[j.stones[i]]);
                }
            }
            else {
                for (let i = 0; i < base_1.N * base_1.N; ++i) {
                    this.stones.push(base_1.Color.Empty);
                }
            }
            if (j.move) {
                this.lastMove = util.parseMove(j.move);
            }
            this.gameOver = j.gameOver || false;
            this.moveNum = j.moveNum;
            if (j.comment) {
                this.comment = j.comment;
            }
            if (j.caps !== undefined) {
                this.captures[0] = j.caps[0];
                this.captures[1] = j.caps[1];
            }
            if (base_1.moveIsPoint(this.lastMove)) {
                this.annotations.push({
                    p: this.lastMove,
                    shape: Annotation.Shape.Dot,
                    colors: ['#ef6c02'],
                });
            }
        }
        addChild(p) {
            if (p.lastMove == null) {
                throw new Error('Child nodes shouldn\'t have a null lastMove');
            }
            if (p.parent != null) {
                throw new Error('Node already has a parent');
            }
            for (let child of this.children) {
                if (base_1.movesEqual(child.lastMove, p.lastMove)) {
                    throw new Error(`Position already has child ${base_1.toGtp(p.lastMove)}`);
                }
            }
            p.isMainLine = this.isMainLine && this.children.length == 0;
            p.parent = this;
            this.children.push(p);
        }
        getChild(move) {
            for (let child of this.children) {
                if (base_1.movesEqual(child.lastMove, move)) {
                    return child;
                }
            }
            return null;
        }
        update(update) {
            if (update.n !== undefined) {
                this.n = update.n;
            }
            if (update.q !== undefined) {
                this.q = update.q;
            }
            if (update.childN !== undefined) {
                this.childN = update.childN;
            }
            if (update.childQ !== undefined) {
                this.childQ = [];
                for (let q of update.childQ) {
                    this.childQ.push(q / 1000);
                }
            }
            if (update.treeStats !== undefined) {
                this.treeStats = update.treeStats;
            }
            if (update.variations !== undefined) {
                this.variations.clear();
                let pv = null;
                for (let key in update.variations) {
                    let variation = {
                        n: update.variations[key].n,
                        q: update.variations[key].q,
                        moves: util.parseMoves(update.variations[key].moves),
                    };
                    this.variations.set(key, variation);
                    if (pv == null || variation.n > pv.n) {
                        pv = variation;
                    }
                }
                if (pv != null) {
                    this.variations.set("pv", pv);
                }
            }
        }
        getFullLine() {
            let result = [];
            let node;
            for (node = this.parent; node != null; node = node.parent) {
                result.push(node);
            }
            result.reverse();
            for (node = this; node != null; node = node.children[0]) {
                result.push(node);
            }
            return result;
        }
    }
    exports.Position = Position;
    (function (Position) {
        ;
    })(Position || (Position = {}));
    exports.Position = Position;
});
//# sourceMappingURL=position.js.map