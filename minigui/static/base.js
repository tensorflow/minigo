define(["require", "exports"], function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    const COL_LABELS = 'ABCDEFGHJKLMNOPQRST';
    exports.COL_LABELS = COL_LABELS;
    var Color;
    (function (Color) {
        Color[Color["Empty"] = 0] = "Empty";
        Color[Color["Black"] = 1] = "Black";
        Color[Color["White"] = 2] = "White";
    })(Color || (Color = {}));
    exports.Color = Color;
    function otherColor(color) {
        if (color != Color.White && color != Color.Black) {
            throw new Error(`invalid color ${color}`);
        }
        return color == Color.White ? Color.Black : Color.White;
    }
    exports.otherColor = otherColor;
    function gtpColor(color) {
        if (color != Color.White && color != Color.Black) {
            throw new Error(`invalid color ${color}`);
        }
        return color == Color.Black ? 'b' : 'w';
    }
    exports.gtpColor = gtpColor;
    function stonesEqual(a, b) {
        if (a.length != b.length) {
            throw new Error(`Expected arrays of equal length, got lengths ${a.length} & ${b.length}`);
        }
        for (let i = 0; i < a.length; ++i) {
            if (a[i] != b[i]) {
                return false;
            }
        }
        return true;
    }
    exports.stonesEqual = stonesEqual;
    var BoardSize;
    (function (BoardSize) {
        BoardSize[BoardSize["Nine"] = 9] = "Nine";
        BoardSize[BoardSize["Nineteen"] = 19] = "Nineteen";
    })(BoardSize || (BoardSize = {}));
    exports.BoardSize = BoardSize;
    let N = BoardSize.Nineteen;
    exports.N = N;
    function setBoardSize(size) {
        if (size == BoardSize.Nine || size == BoardSize.Nineteen) {
            exports.N = N = size;
        }
        else {
            throw new Error(`Unsupported board size ${size}`);
        }
    }
    exports.setBoardSize = setBoardSize;
    function moveIsPoint(move) {
        return move != null && move != 'pass' && move != 'resign';
    }
    exports.moveIsPoint = moveIsPoint;
    function toGtp(move) {
        if (move == 'pass' || move == 'resign') {
            return move;
        }
        let row = N - move.row;
        let col = COL_LABELS[move.col];
        return `${col}${row}`;
    }
    exports.toGtp = toGtp;
    function movesEqual(a, b) {
        if (moveIsPoint(a) && moveIsPoint(b)) {
            return a.row == b.row && a.col == b.col;
        }
        else {
            return a == b;
        }
    }
    exports.movesEqual = movesEqual;
});
//# sourceMappingURL=base.js.map