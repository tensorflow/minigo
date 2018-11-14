define(["require", "exports", "./base", "./util"], function (require, exports, base_1, util_1) {
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
        constructor(id, parent, stones, lastMove, toPlay, gameOver, isMainLine) {
            this.id = id;
            this.parent = parent;
            this.stones = stones;
            this.lastMove = lastMove;
            this.toPlay = toPlay;
            this.gameOver = gameOver;
            this.isMainLine = isMainLine;
            this.n = 0;
            this.q = 0;
            this.search = [];
            this.variations = new Map();
            this.annotations = [];
            this.childN = null;
            this.childQ = null;
            this.children = [];
            this.moveNum = parent != null ? parent.moveNum + 1 : 0;
            if (lastMove != null && lastMove != 'pass' && lastMove != 'resign') {
                this.annotations.push({
                    p: lastMove,
                    shape: Annotation.Shape.Dot,
                    colors: ['#ef6c02'],
                });
            }
        }
        addChild(id, move, stones, gameOver) {
            for (let child of this.children) {
                if (child.lastMove == null) {
                    throw new Error('Child node shouldn\'t have a null lastMove');
                }
                if (base_1.movesEqual(child.lastMove, move)) {
                    if (!base_1.stonesEqual(stones, child.stones)) {
                        throw new Error(`Position has child ${base_1.toKgs(move)} with different stones`);
                    }
                    return child;
                }
            }
            let isMainLine = this.isMainLine && this.children.length == 0;
            let child = new Position(id, this, stones, move, base_1.otherColor(this.toPlay), gameOver, isMainLine);
            this.children.push(child);
            return child;
        }
        update(update) {
            const props = ['n', 'q', 'childN', 'childQ'];
            util_1.partialUpdate(update, this, props);
            if (update.variations != null) {
                for (let key in update.variations) {
                    this.variations.set(key, update.variations[key]);
                }
                if ("pv" in update.variations) {
                    let pv = update.variations["pv"];
                    if (pv.length > 0) {
                        this.variations.set(base_1.toKgs(pv[0]), pv);
                    }
                }
            }
        }
    }
    exports.Position = Position;
});
//# sourceMappingURL=position.js.map