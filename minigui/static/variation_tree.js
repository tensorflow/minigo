define(["require", "exports", "./util", "./view"], function (require, exports, util_1, view_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    const SPACING = 16;
    const PADDING = 8;
    const RADIUS = 2.5;
    class Node {
        constructor(parent, position, x, y) {
            this.parent = parent;
            this.position = position;
            this.x = x;
            this.y = y;
            this.mainlineDepth = 0;
            this.children = [];
        }
    }
    class VariationTree extends view_1.View {
        constructor(parent) {
            super();
            this.rootPosition = null;
            this.rootNode = null;
            this.hoveredPosition = null;
            this.listeners = [];
            if (typeof (parent) == 'string') {
                parent = util_1.getElement(parent);
            }
            let canvas = document.createElement('canvas');
            this.ctx = canvas.getContext('2d');
            parent.appendChild(canvas);
            parent.addEventListener('mousemove', (e) => {
                let oldActivePosition = this.hoveredPosition;
                this.hoveredPosition = this.hitTest(e.offsetX, e.offsetY);
                if (this.hoveredPosition != oldActivePosition) {
                    if (this.hoveredPosition != null) {
                        canvas.classList.add('pointer');
                    }
                    else {
                        canvas.classList.remove('pointer');
                    }
                }
            });
            parent.addEventListener('click', (e) => {
                if (this.hoveredPosition != null) {
                    let p = this.hoveredPosition;
                    let positions = [];
                    while (p.parent != null) {
                        positions.push(p);
                        p = p.parent;
                    }
                    positions.reverse();
                    for (let listener of this.listeners) {
                        listener(positions);
                    }
                }
            });
        }
        newGame(rootPosition) {
            this.rootPosition = rootPosition;
            this.resizeCanvas(1, 1, 1);
        }
        onClick(cb) {
            this.listeners.push(cb);
        }
        hitTest(x, y) {
            if (this.rootNode == null) {
                return null;
            }
            let pr = util_1.pixelRatio();
            x *= pr;
            y *= pr;
            let threshold = 0.4 * SPACING * pr;
            let traverse = (node) => {
                let dx = x - node.x;
                let dy = y - node.y;
                if (dx * dx + dy * dy < threshold * threshold) {
                    return node.position;
                }
                for (let child of node.children) {
                    let result = traverse(child);
                    if (result != null) {
                        return result;
                    }
                }
                return null;
            };
            return traverse(this.rootNode);
        }
        drawImpl() {
            let ctx = this.ctx;
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            if (this.rootPosition == null) {
                return;
            }
            let pr = util_1.pixelRatio();
            this.rootNode = this.layout(this.rootPosition);
            ctx.fillStyle = '#fff';
            ctx.strokeStyle = '#fff';
            let pad = PADDING * pr + 0.5;
            let space = SPACING * pr;
            let drawEdges = (node) => {
                if (node.children.length == 0) {
                    return;
                }
                if (node.children.length > 1) {
                    let lastChild = node.children[node.children.length - 1];
                    if (lastChild.x - space > node.x) {
                        ctx.moveTo(node.x, node.y);
                        ctx.lineTo(lastChild.x - space, node.y);
                    }
                }
                for (let child of node.children) {
                    let x = child.x;
                    if (child != node.children[0]) {
                        x -= space;
                    }
                    ctx.moveTo(x, node.y);
                    ctx.lineTo(child.x, child.y);
                    drawEdges(child);
                }
            };
            drawEdges(this.rootNode);
            ctx.stroke();
            let r = RADIUS * pr;
            let drawNodes = (node) => {
                ctx.beginPath();
                ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
                ctx.fill();
                for (let child of node.children) {
                    drawNodes(child);
                }
            };
            drawNodes(this.rootNode);
        }
        layout(rootPosition) {
            let rightNode = [];
            let traverse = (node, depth) => {
                for (let i = 0; i < 2 && depth + i < rightNode.length; ++i) {
                    node.x = Math.max(node.x, rightNode[depth + i].x + 1);
                }
                rightNode[depth] = node;
                for (let childPosition of node.position.children) {
                    let childNode = new Node(node, childPosition, node.x, depth + 1);
                    node.children.push(childNode);
                    traverse(childNode, depth + 1);
                }
                if (node.parent && node == node.parent.children[0]) {
                    node.parent.x = Math.max(node.parent.x, node.x);
                }
            };
            let rootNode = new Node(null, rootPosition, 0, 0);
            traverse(rootNode, 0);
            let pr = util_1.pixelRatio();
            let pad = PADDING * pr + 0.5;
            let space = SPACING * pr;
            let requiredWidth = pad * 2;
            let reposition = (node) => {
                node.x = pad + space * node.x;
                node.y = pad + space * node.y;
                requiredWidth = Math.max(requiredWidth, pad + node.x);
                for (let child of node.children) {
                    reposition(child);
                }
            };
            reposition(rootNode);
            let requiredHeight = pad * 2 + rightNode.length * space;
            if (requiredWidth > this.ctx.canvas.width ||
                requiredHeight > this.ctx.canvas.height) {
                this.resizeCanvas(requiredWidth, requiredHeight, pr);
            }
            return rootNode;
        }
        resizeCanvas(width, height, pixelRatio) {
            let canvas = this.ctx.canvas;
            canvas.width = width * pixelRatio;
            canvas.height = height * pixelRatio;
            canvas.style.width = `${width}px`;
            canvas.style.height = `${height}px`;
        }
    }
    exports.VariationTree = VariationTree;
});
//# sourceMappingURL=variation_tree.js.map