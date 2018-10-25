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
    var BoardSize;
    (function (BoardSize) {
        BoardSize[BoardSize["Nine"] = 9] = "Nine";
        BoardSize[BoardSize["Nineteen"] = 19] = "Nineteen";
    })(BoardSize || (BoardSize = {}));
    exports.BoardSize = BoardSize;
    let N = BoardSize.Nineteen;
    exports.N = N;
    function setBoardSize(size) {
        switch (size) {
        }
        if (size == BoardSize.Nine || size == BoardSize.Nineteen) {
            exports.N = N = size;
        }
        else {
            throw new Error(`Unsupported board size ${size}`);
        }
    }
    exports.setBoardSize = setBoardSize;
});
//# sourceMappingURL=base.js.map