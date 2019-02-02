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
    function parseColor(color) {
        let c = color[0].toLowerCase();
        return c == 'b' ? base_1.Color.Black : base_1.Color.White;
    }
    exports.parseColor = parseColor;
    function parseMove(gtpCoord) {
        if (gtpCoord == 'pass' || gtpCoord == 'resign') {
            return gtpCoord;
        }
        let col = gtpCoord.charCodeAt(0) - 65;
        if (col >= 8) {
            --col;
        }
        let row = base_1.N - parseInt(gtpCoord.slice(1), 10);
        if (row < 0 || row >= base_1.N || col < 0 || col > base_1.N) {
            throw new Error(`Can't parse "${gtpCoord}" as a GTP coord`);
        }
        return { row: row, col: col };
    }
    exports.parseMove = parseMove;
    function parseMoves(moveStrs) {
        let moves = [];
        for (let str of moveStrs) {
            moves.push(parseMove(str));
        }
        return moves;
    }
    exports.parseMoves = parseMoves;
    function pixelRatio() {
        return window.devicePixelRatio || 1;
    }
    exports.pixelRatio = pixelRatio;
    function partialUpdate(src, dst, propNames) {
        for (let name of propNames) {
            if (src[name] != null) {
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