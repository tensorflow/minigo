// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {Nullable} from './base'
import {getElement, pixelRatio} from './util'

interface Position {
  children: Position[];
}

class Node {
  mainlineDepth = 0;
  children: Node[] = [];
  constructor(public parent: Nullable<Node>,
              public position: Position,
              public x: number, public y: number) {}
}

class FakePosition {
  children: FakePosition[] = [];
  constructor(public name: string, public moveNum = 0) {
    positions[name] = this;
  }

  addChild(name: string) {
    this.children.push(new FakePosition(name, this.moveNum + 1));
  }
}

let positions: {[key: string]: FakePosition} = {};

interface LayoutResult {
  maxDepth: number;
  rootNode: Node;
}

class VariationTree {
  private ctx: CanvasRenderingContext2D;

  constructor(parent: HTMLElement | string, private rootPosition: Position) {
    if (typeof(parent) == 'string') {
      parent = getElement(parent);
    }

    let canvas = document.createElement('canvas');
    this.ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
    parent.appendChild(canvas);

    window.addEventListener('resize', () => {
      this.resizeCanvas();
      this.draw();
    });

    this.resizeCanvas();
    this.draw();
  }

  public draw() {
    let {maxDepth, rootNode} = this.layout();

    let impl = (node: Node) => {
      console.log(`draw: (${node.x}, ${node.y})`);
      for (let child of node.children) {
        impl(child);
      }
    };
    impl(rootNode);
  }

  private resizeCanvas() {
  }

  private layout() {
    // Maximum depth of the tree.
    let maxDepth = -1;

    // List of nodes at each depth of the tree, in traversal order.
    let nodesAtDepth: Node[][] = [];

    // We want to lay the nodes in the tree out such that the mainline of each
    // node (the chain of first children of all descendants) is a vertical line.
    // We do this by incrementally building a new tree of layout data in
    // traversal order of the game tree, satisfying the following constraints at
    // each step:
    //  1) A newly inserted node's x coordinate is greater than all other nodes
    //     at the same depth. This is enforced pre-traversal.
    //  2) A newly inserted node's x coordinate is greater or equal to the x
    //     coordinate of its first child. This is enforced post-traversal.
    let traverse = (node: Node, position: Position, depth: number) => {
      if (depth > maxDepth) {
        // First node at this depth. Don't need to check constraint 1.
        nodesAtDepth[depth] = [];
        maxDepth = depth;
      } else {
        // Satisfy constraint 1. In order to make the layout less cramped, we
        // actually check both the current depth and the next depth.
        for (let i = 0; i < 2 && depth + i < nodesAtDepth.length; ++i) {
          let nodes = nodesAtDepth[depth + i];
          node.x = Math.max(node.x, nodes[nodes.length - 1].x + 1);
        }
      }

      // Traverse the tree.
      nodesAtDepth[depth].push(node);
      for (let childPosition of position.children) {
        let childNode = new Node(node, childPosition, node.x, depth + 1);
        node.children.push(childNode);
        traverse(childNode, childPosition, depth + 1);
      }

      // Satisfy constraint 2.
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

export {
  VariationTree,
  testVariationTree,
}
