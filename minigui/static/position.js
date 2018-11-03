define(["require", "exports", "./base"], function (require, exports, base_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    var Annotation;
    (function (Annotation) {
        let Shape;
        (function (Shape) {
            Shape[Shape["Dot"] = 0] = "Dot";
            Shape[Shape["Triangle"] = 1] = "Triangle";
        })(Shape = Annotation.Shape || (Annotation.Shape = {}));
    })(Annotation || (Annotation = {}));
    exports.Annotation = Annotation;
    class Position {
        constructor(parent, moveNum, stones, q, lastMove, toPlay) {
            this.parent = parent;
            this.moveNum = moveNum;
            this.stones = stones;
            this.q = q;
            this.lastMove = lastMove;
            this.toPlay = toPlay;
            this.search = [];
            this.pv = [];
            this.n = null;
            this.dq = null;
            this.annotations = [];
            this.childQ = null;
            this.children = [];
            if (lastMove != null && lastMove != 'pass' && lastMove != 'resign') {
                this.annotations.push({
                    p: lastMove,
                    shape: Annotation.Shape.Dot,
                    color: '#ef6c02',
                });
            }
        }
        addChild(move, stones, q) {
            for (let child of this.children) {
                if (child.lastMove == null) {
                    throw new Error('Child node shouldn\'t have a null lastMove');
                }
                if (base_1.movesEqual(child.lastMove, move)) {
                    if (!base_1.stonesEqual(stones, child.stones)) {
                        throw new Error(`Position has child ${move} with different stones`);
                    }
                    return child;
                }
            }
            let child = new Position(this, this.moveNum + 1, stones, q, move, base_1.otherColor(this.toPlay));
            this.children.push(child);
            return child;
        }
    }
    exports.Position = Position;
});
//# sourceMappingURL=position.js.map