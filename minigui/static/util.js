define(["require", "exports", "./base"], function (require, exports, base_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    function getElement(id) {
        return document.getElementById(id);
    }
    exports.getElement = getElement;
    function querySelector(selector) {
        return document.querySelector(selector);
    }
    exports.querySelector = querySelector;
    function parseGtpColor(color) {
        let c = color[0].toLowerCase();
        return c == 'b' ? base_1.Color.Black : base_1.Color.White;
    }
    exports.parseGtpColor = parseGtpColor;
    function parseGtpMove(gtpCoord, size) {
        if (gtpCoord == 'pass' || gtpCoord == 'resign') {
            return gtpCoord;
        }
        let col = gtpCoord.charCodeAt(0) - 65;
        if (col >= 8) {
            --col;
        }
        let row = size - parseInt(gtpCoord.slice(1), 10);
        return { row: row, col: col };
    }
    exports.parseGtpMove = parseGtpMove;
    function parseMoves(moves, size) {
        let variation = [];
        for (let move of moves) {
            variation.push(parseGtpMove(move, size));
        }
        return variation;
    }
    exports.parseMoves = parseMoves;
    function pixelRatio() {
        return window.devicePixelRatio || 1;
    }
    exports.pixelRatio = pixelRatio;
    function emptyBoard(size) {
        let result = new Array(size * size);
        result.fill(base_1.Color.Empty);
        return result;
    }
    exports.emptyBoard = emptyBoard;
    function partialUpdate(src, dst, propNames) {
        for (let name of propNames) {
            if (name in src) {
                dst[name] = src[name];
            }
        }
        return dst;
    }
    exports.partialUpdate = partialUpdate;
    function toPrettyResult(result) {
        let prettyResult;
        if (result[0] == 'W') {
            prettyResult = 'White wins by ';
        }
        else {
            prettyResult = 'Black wins by ';
        }
        if (result[2] == 'R') {
            prettyResult += 'resignation';
        }
        else {
            prettyResult += result.substr(2) + ' points';
        }
        return prettyResult;
    }
    exports.toPrettyResult = toPrettyResult;
});
//# sourceMappingURL=util.js.map