define(["require", "exports", "./util", "./view"], function (require, exports, util_1, view_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    const SPACE = 16;
    const PAD = 9;
    const RADIUS = 4;
    class Node {
        constructor(parent, position, x, y) {
            this.parent = parent;
            this.position = position;
            this.x = x;
            this.y = y;
            this.children = [];
        }
    }
    class VariationTree extends view_1.View {
        constructor(parent) {
            super();
            this.rootNode = null;
            this.hoveredNode = null;
            this.activeNode = null;
            this.listeners = [];
            if (typeof (parent) == 'string') {
                parent = util_1.getElement(parent);
            }
            let canvas = document.createElement('canvas');
            this.ctx = canvas.getContext('2d');
            parent.appendChild(canvas);
            parent.addEventListener('mousemove', (e) => {
                let oldNode = this.hoveredNode;
                this.hoveredNode = this.hitTest(e.offsetX, e.offsetY);
                if (this.hoveredNode != oldNode) {
                    if (this.hoveredNode != null) {
                        canvas.classList.add('pointer');
                    }
                    else {
                        canvas.classList.remove('pointer');
                    }
                }
            });
            parent.addEventListener('click', (e) => {
                if (this.hoveredNode != null) {
                    this.activeNode = this.hoveredNode;
                    let p = this.hoveredNode.position;
                    let positions = [];
                    while (p.parent != null) {
                        positions.push(p);
                        p = p.parent;
                    }
                    positions.reverse();
                    for (let listener of this.listeners) {
                        listener(positions);
                    }
                    this.draw();
                }
            });
        }
        newGame(rootPosition) {
            this.rootNode = new Node(null, rootPosition, PAD, PAD);
            this.activeNode = this.rootNode;
            this.resizeCanvas(1, 1, 1);
            this.layout();
            this.draw();
        }
        addChild(parentPosition, childPosition) {
            if (this.rootNode == null) {
                throw new Error('Must start a game before attempting to add children');
            }
            let findParent = (node) => {
                if (node.position == parentPosition) {
                    return node;
                }
                for (let child of node.children) {
                    let node = findParent(child);
                    if (node != null) {
                        return node;
                    }
                }
                return null;
            };
            let parentNode = findParent(this.rootNode);
            if (parentNode == null) {
                throw new Error('Couldn\'t find parent node');
            }
            let childNode = null;
            for (let child of parentNode.children) {
                if (child.position == childPosition) {
                    childNode = child;
                    break;
                }
            }
            if (childNode == null) {
                let x = parentNode.x + SPACE * parentNode.children.length;
                let y = parentNode.y + SPACE;
                childNode = new Node(parentNode, childPosition, x, y);
                parentNode.children.push(childNode);
            }
            if (childNode != this.activeNode) {
                this.activeNode = childNode;
                this.layout();
                this.draw();
            }
        }
        onClick(cb) {
            this.listeners.push(cb);
        }
        hitTest(x, y) {
            if (this.rootNode == null) {
                return null;
            }
            let threshold = 0.4 * SPACE;
            let traverse = (node) => {
                let dx = x - node.x;
                let dy = y - node.y;
                if (dx * dx + dy * dy < threshold * threshold) {
                    return node;
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
            if (this.rootNode == null) {
                return;
            }
            let pr = util_1.pixelRatio();
            ctx.strokeStyle = '#888';
            ctx.lineWidth = pr;
            let drawEdges = (node) => {
                if (node.children.length == 0) {
                    return;
                }
                for (let child of node.children) {
                    let x = child.x;
                    if (child != node.children[0]) {
                        x -= SPACE;
                    }
                    ctx.moveTo(pr * x, pr * node.y);
                    ctx.lineTo(pr * child.x, pr * child.y);
                    drawEdges(child);
                }
                if (node.children.length > 1) {
                    let lastChild = node.children[node.children.length - 1];
                    if (lastChild.x - SPACE > node.x) {
                        ctx.moveTo(pr * node.x, pr * node.y);
                        ctx.lineTo(pr * (lastChild.x - SPACE), pr * node.y);
                    }
                }
            };
            ctx.beginPath();
            drawEdges(this.rootNode);
            ctx.stroke();
            let r = RADIUS * pr;
            if (this.activeNode != null) {
                ctx.fillStyle = '#800';
                ctx.strokeStyle = '#ff0';
                ctx.beginPath();
                ctx.arc(pr * this.activeNode.x, pr * this.activeNode.y, r, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            }
            ctx.fillStyle = '#666';
            ctx.strokeStyle = '#888';
            ctx.beginPath();
            ctx.arc(pr * this.rootNode.x, pr * this.rootNode.y, r, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            let drawNodes = (node, draw) => {
                if (draw && node != this.rootNode && node != this.activeNode) {
                    ctx.beginPath();
                    ctx.arc(pr * node.x, pr * node.y, r, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.stroke();
                }
                for (let child of node.children) {
                    drawNodes(child, !draw);
                }
            };
            ctx.fillStyle = '#000';
            drawNodes(this.rootNode, false);
            ctx.fillStyle = '#fff';
            drawNodes(this.rootNode, true);
        }
        layout() {
            if (this.rootNode == null) {
                return;
            }
            let rightNode = [this.rootNode];
            let requiredWidth = PAD * 2;
            let traverse = (node, depth) => {
                let parent = node.parent;
                node.x = parent.x;
                for (let i = 0; i < 2 && depth + i < rightNode.length; ++i) {
                    node.x = Math.max(node.x, rightNode[depth + i].x + SPACE);
                }
                rightNode[depth] = node;
                for (let child of node.children) {
                    traverse(child, depth + 1);
                }
                if (node == parent.children[0]) {
                    parent.x = Math.max(parent.x, node.x);
                }
                requiredWidth = Math.max(requiredWidth, PAD + node.x);
            };
            for (let child of this.rootNode.children) {
                traverse(child, 1);
            }
            let requiredHeight = PAD * 2 + (rightNode.length - 1) * SPACE;
            let pr = util_1.pixelRatio();
            if (requiredWidth * pr > this.ctx.canvas.width ||
                requiredHeight * pr > this.ctx.canvas.height) {
                this.resizeCanvas(requiredWidth, requiredHeight, pr);
            }
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