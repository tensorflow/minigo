define(["require", "exports", "./util"], function (require, exports, util_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
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
    class FakePosition {
        constructor(name, moveNum = 0) {
            this.name = name;
            this.moveNum = moveNum;
            this.children = [];
            positions[name] = this;
        }
        addChild(name) {
            this.children.push(new FakePosition(name, this.moveNum + 1));
        }
    }
    let positions = {};
    class VariationTree {
        constructor(parent, rootPosition) {
            this.rootPosition = rootPosition;
            if (typeof (parent) == 'string') {
                parent = util_1.getElement(parent);
            }
            let canvas = document.createElement('canvas');
            this.ctx = canvas.getContext('2d');
            parent.appendChild(canvas);
            window.addEventListener('resize', () => {
                this.resizeCanvas();
                this.draw();
            });
            this.resizeCanvas();
            this.draw();
        }
        draw() {
            let { maxDepth, rootNode } = this.layout();
            let impl = (node) => {
                console.log(`draw: (${node.x}, ${node.y})`);
                for (let child of node.children) {
                    impl(child);
                }
            };
            impl(rootNode);
        }
        resizeCanvas() {
        }
        layout() {
            let maxDepth = -1;
            let nodesAtDepth = [];
            let traverse = (node, position, depth) => {
                if (depth > maxDepth) {
                    nodesAtDepth[depth] = [];
                    maxDepth = depth;
                }
                else {
                    for (let i = 0; i < 2 && depth + i < nodesAtDepth.length; ++i) {
                        let nodes = nodesAtDepth[depth + i];
                        node.x = Math.max(node.x, nodes[nodes.length - 1].x + 1);
                    }
                }
                nodesAtDepth[depth].push(node);
                for (let childPosition of position.children) {
                    let childNode = new Node(node, childPosition, node.x, depth + 1);
                    node.children.push(childNode);
                    traverse(childNode, childPosition, depth + 1);
                }
                if (node.parent && node == node.parent.children[0]) {
                    node.parent.x = Math.max(node.parent.x, node.x);
                }
            };
            let rootNode = new Node(null, this.rootPosition, 0, 0);
            traverse(rootNode, this.rootPosition, 0);
            return {
                maxDepth: maxDepth,
                rootNode: rootNode,
            };
        }
    }
    exports.VariationTree = VariationTree;
    new FakePosition('a');
    positions['a'].addChild('b');
    positions['b'].addChild('c');
    positions['c'].addChild('d');
    positions['d'].addChild('e');
    positions['e'].addChild('f');
    positions['f'].addChild('g');
    positions['f'].addChild('h');
    positions['f'].addChild('i');
    positions['i'].addChild('j');
    positions['f'].addChild('k');
    positions['e'].addChild('l');
    positions['l'].addChild('m');
    positions['e'].addChild('n');
    positions['d'].addChild('o');
    positions['a'].addChild('p');
    positions['p'].addChild('q');
    positions['a'].addChild('r');
    positions['r'].addChild('s');
    positions['s'].addChild('t');
    console.log('positions:', positions);
    function testVariationTree() {
        let tree = new VariationTree(document.createElement('div'), positions['a']);
    }
    exports.testVariationTree = testVariationTree;
});
//# sourceMappingURL=variation_tree.js.map