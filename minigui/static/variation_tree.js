define(["require", "exports", "./util", "./view"], function (require, exports, util_1, view_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    const SPACE = 19;
    const PAD = 11;
    const RADIUS = 5;
    class Node {
        constructor(parent, position, x, y) {
            this.parent = parent;
            this.position = position;
            this.x = x;
            this.y = y;
            this.children = [];
            if (parent == null) {
                this.isMainline = true;
            }
            else {
                this.isMainline = parent.isMainline && parent.children.length == 0;
                parent.children.push(this);
            }
        }
    }
    class VariationTree extends view_1.View {
        constructor(parent) {
            super();
            this.rootNode = null;
            this.hoveredNode = null;
            this.activeNode = null;
            this.clickListeners = [];
            this.hoverListeners = [];
            this.width = 0;
            this.height = 0;
            this.maxX = 0;
            this.maxY = 0;
            this.scrollX = 0;
            this.scrollY = 0;
            this.mouseX = 0;
            this.mouseY = 0;
            if (typeof (parent) == 'string') {
                parent = util_1.getElement(parent);
            }
            let canvas = document.createElement('canvas');
            this.ctx = canvas.getContext('2d');
            parent.appendChild(canvas);
            this.resizeCanvas();
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
                    for (let listener of this.hoverListeners) {
                        listener(this.hoveredNode ? this.hoveredNode.position : null);
                    }
                }
            });
            parent.addEventListener('mousedown', (e) => {
                this.mouseX = e.screenX;
                this.mouseY = e.screenY;
                let moveHandler = (e) => {
                    this.scrollX += this.mouseX - e.screenX;
                    this.scrollY += this.mouseY - e.screenY;
                    this.scrollX = Math.min(this.scrollX, this.maxX + PAD - this.width);
                    this.scrollY = Math.min(this.scrollY, this.maxY + PAD - this.height);
                    this.scrollX = Math.max(this.scrollX, 0);
                    this.scrollY = Math.max(this.scrollY, 0);
                    this.mouseX = e.screenX;
                    this.mouseY = e.screenY;
                    this.draw();
                    e.preventDefault();
                    return false;
                };
                let upHandler = (e) => {
                    window.removeEventListener('mousemove', moveHandler);
                    window.removeEventListener('mouseup', upHandler);
                };
                window.addEventListener('mousemove', moveHandler);
                window.addEventListener('mouseup', upHandler);
            });
            parent.addEventListener('click', (e) => {
                if (this.hoveredNode != null) {
                    for (let listener of this.clickListeners) {
                        listener(this.hoveredNode.position);
                    }
                }
            });
            window.addEventListener('resize', () => {
                this.resizeCanvas();
                this.scrollIntoView();
                this.draw();
            });
        }
        newGame() {
            this.rootNode = null;
            this.activeNode = null;
            this.layout();
            this.scrollIntoView();
            this.draw();
        }
        setRoot(position) {
            if (this.rootNode != null) {
                throw new Error('Already have a root note, you should call newGame first');
            }
            this.rootNode = new Node(null, position, PAD, PAD);
            this.activeNode = this.rootNode;
        }
        setActive(position) {
            if (this.activeNode != null && this.activeNode.position != position) {
                this.activeNode = this.lookupNode(position);
                this.scrollIntoView();
                this.draw();
            }
        }
        addChild(parentPosition, childPosition) {
            if (this.rootNode == null) {
                throw new Error('Must start a game before attempting to add children');
            }
            let parentNode = this.lookupNode(parentPosition);
            let childNode = null;
            for (let child of parentNode.children) {
                if (child.position == childPosition) {
                    childNode = child;
                    break;
                }
            }
            if (childNode == null) {
                let x = parentNode.x + SPACE;
                let y = parentNode.y + SPACE * parentNode.children.length;
                childNode = new Node(parentNode, childPosition, x, y);
                this.layout();
            }
        }
        onClick(cb) {
            this.clickListeners.push(cb);
        }
        onHover(cb) {
            this.hoverListeners.push(cb);
        }
        lookupNode(position) {
            let impl = (node) => {
                if (node.position == position) {
                    return node;
                }
                for (let child of node.children) {
                    let node = impl(child);
                    if (node != null) {
                        return node;
                    }
                }
                return null;
            };
            let node = this.rootNode != null ? impl(this.rootNode) : null;
            if (node == null) {
                throw new Error('Couldn\'t find node');
            }
            return node;
        }
        hitTest(x, y) {
            if (this.rootNode == null) {
                return null;
            }
            x += this.scrollX;
            y += this.scrollY;
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
            ctx.save();
            ctx.translate(0.5 - pr * this.scrollX, 0.5 - pr * this.scrollY);
            ctx.lineWidth = pr;
            let drawEdges = (node, drawMainline) => {
                if (node.children.length == 0) {
                    return;
                }
                for (let child of node.children) {
                    let y = child.y;
                    if (child != node.children[0]) {
                        y -= SPACE;
                    }
                    if (drawMainline == child.isMainline) {
                        ctx.moveTo(pr * node.x, pr * y);
                        ctx.lineTo(pr * child.x, pr * child.y);
                    }
                    drawEdges(child, drawMainline);
                }
                if (node.children.length > 1 && !drawMainline) {
                    let lastChild = node.children[node.children.length - 1];
                    if (lastChild.y - SPACE > node.y) {
                        ctx.moveTo(pr * node.x, pr * node.y);
                        ctx.lineTo(pr * node.x, pr * (lastChild.y - SPACE));
                    }
                }
            };
            for (let style of ['#fff', '#888']) {
                ctx.beginPath();
                ctx.strokeStyle = style;
                drawEdges(this.rootNode, style == '#fff');
                ctx.stroke();
            }
            let r = RADIUS * pr;
            if (this.activeNode != null) {
                ctx.fillStyle = '#800';
                ctx.strokeStyle = '#ff0';
                ctx.beginPath();
                ctx.arc(pr * this.activeNode.x, pr * this.activeNode.y, r, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            }
            ctx.strokeStyle = '#888';
            if (this.activeNode != this.rootNode) {
                ctx.fillStyle = '#666';
                ctx.beginPath();
                ctx.arc(pr * this.rootNode.x, pr * this.rootNode.y, r, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            }
            let drawNodes = (node, draw) => {
                if (draw && node != this.rootNode && node != this.activeNode) {
                    ctx.moveTo(pr * node.x + r, pr * node.y);
                    ctx.arc(pr * node.x, pr * node.y, r, 0, 2 * Math.PI);
                }
                for (let child of node.children) {
                    drawNodes(child, !draw);
                }
            };
            for (let style of ['#000', '#fff']) {
                ctx.fillStyle = style;
                ctx.beginPath();
                drawNodes(this.rootNode, style != '#000');
                ctx.fill();
                ctx.stroke();
            }
            ctx.restore();
        }
        scrollIntoView() {
            if (this.activeNode == null) {
                return;
            }
            if (this.activeNode.x - this.scrollX > this.width - PAD) {
                this.scrollX = this.activeNode.x - this.width + PAD;
            }
            else if (this.activeNode.x - this.scrollX < PAD) {
                this.scrollX = this.activeNode.x - PAD;
            }
            if (this.activeNode.y - this.scrollY > this.height - PAD) {
                this.scrollY = this.activeNode.y - this.height + PAD;
            }
            else if (this.activeNode.y - this.scrollY < PAD) {
                this.scrollY = this.activeNode.y - PAD;
            }
        }
        layout() {
            this.maxX = 0;
            this.maxY = 0;
            if (this.rootNode == null) {
                return;
            }
            let bottomNode = [this.rootNode];
            let traverse = (node, depth) => {
                let parent = node.parent;
                node.y = parent.y;
                for (let i = 0; i < 2 && depth + i < bottomNode.length; ++i) {
                    node.y = Math.max(node.y, bottomNode[depth + i].y + SPACE);
                }
                bottomNode[depth] = node;
                for (let child of node.children) {
                    traverse(child, depth + 1);
                }
                if (node == parent.children[0]) {
                    parent.y = Math.max(parent.y, node.y);
                }
                this.maxX = Math.max(this.maxX, node.x);
                this.maxY = Math.max(this.maxY, node.y);
            };
            for (let child of this.rootNode.children) {
                traverse(child, 1);
            }
        }
        resizeCanvas() {
            let pr = util_1.pixelRatio();
            let canvas = this.ctx.canvas;
            let parent = canvas.parentElement;
            this.width = parent.offsetWidth;
            this.height = parent.offsetHeight;
            canvas.width = pr * this.width;
            canvas.height = pr * this.height;
            canvas.style.width = `${this.width}px`;
            canvas.style.height = `${this.height}px`;
        }
    }
    exports.VariationTree = VariationTree;
});
//# sourceMappingURL=variation_tree.js.map