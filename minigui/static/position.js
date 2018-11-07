define(["require", "exports", "./base"], function (require, exports, base_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    var Annotation;
    (function (Annotation) {
        let Shape;
        (function (Shape) {
            Shape[Shape["Dot"] = 0] = "Dot";
            Shape[Shape["Triangle"] = 1] = "Triangle";
            Shape[Shape["DashedCircle"] = 2] = "DashedCircle";
        })(Shape = Annotation.Shape || (Annotation.Shape = {}));
    })(Annotation || (Annotation = {}));
    exports.Annotation = Annotation;
    class Position {
        constructor(id, parent, stones, q, lastMove, toPlay, isMainline) {
            this.id = id;
            this.parent = parent;
            this.stones = stones;
            this.q = q;
            this.lastMove = lastMove;
            this.toPlay = toPlay;
            this.isMainline = isMainline;
            this.search = [];
            this.pv = [];
            this.n = null;
            this.dq = null;
            this.annotations = [];
            this.childQ = null;
            this.children = [];
            this.moveNum = parent != null ? parent.moveNum + 1 : 0;
            if (lastMove != null && lastMove != 'pass' && lastMove != 'resign') {
                this.annotations.push({
                    p: lastMove,
                    shape: Annotation.Shape.Dot,
                    color: '#ef6c02',
                });
            }
        }
        addChild(id, move, stones, q) {
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
            let isMainline = this.isMainline && this.children.length == 0;
            let child = new Position(id, this, stones, q, move, base_1.otherColor(this.toPlay), isMainline);
            this.children.push(child);
            if (move != 'pass' && move != 'resign') {
                let color;
                this.annotations.push({
                    p: move,
                    shape: Annotation.Shape.DashedCircle,
                    color: this.toPlay == base_1.Color.Black ? '#000' : '#fff',
                });
            }
            return child;
        }
    }
    exports.Position = Position;
});
//# sourceMappingURL=position.js.map