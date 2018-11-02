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
import {View} from './view'

// Spacing between nodes in the tree.
const SPACING = 16;

// Padding from the edge of the canvas.
const PADDING = 8;

// Raius of the nodes.
const RADIUS = 2.5;

interface Position {
  parent: Nullable<Position>;
  children: Position[];
}

class Node {
  mainlineDepth = 0;
  children: Node[] = [];
  constructor(public parent: Nullable<Node>,
              public position: Position,
              public x: number, public y: number) {}
}

interface LayoutResult {
  maxDepth: number;
  rootNode: Node;
}

type ClickListener = (positions: Position[]) => void;

class VariationTree extends View {
  private rootPosition: Nullable<Position> = null;
  private rootNode: Nullable<Node> = null;
  private ctx: CanvasRenderingContext2D;

  // Which node in the tree the cursor is hovering over, if any.
  // Since we rebuild the Node tree each time, we track hovered node by its
  // position, which is stable.
  private hoveredPosition: Nullable<Position> = null;

  private listeners: ClickListener[] = [];

  constructor(parent: HTMLElement | string) {
    super();

    if (typeof(parent) == 'string') {
      parent = getElement(parent);
    }

    let canvas = document.createElement('canvas');
    this.ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
    parent.appendChild(canvas);

    parent.addEventListener('mousemove', (e) => {
      let oldActivePosition = this.hoveredPosition;
      this.hoveredPosition = this.hitTest(e.offsetX, e.offsetY);
      if (this.hoveredPosition != oldActivePosition) {
        if (this.hoveredPosition != null) {
          canvas.classList.add('pointer');
        } else {
          canvas.classList.remove('pointer');
        }
      }
    });

    parent.addEventListener('click', (e) => {
      if (this.hoveredPosition != null) {
        let p = this.hoveredPosition;
        let positions: Position[] = [];
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

  public newGame(rootPosition: Position) {
    this.rootPosition = rootPosition;
    this.resizeCanvas(1, 1, 1);
  }

  public onClick(cb: ClickListener) {
    this.listeners.push(cb);
  }

  private hitTest(x: number, y: number) {
    if (this.rootNode == null) {
      return null;
    }

    let pr = pixelRatio();

    x *= pr;
    y *= pr;
    let threshold = 0.4 * SPACING * pr;

    let traverse = (node: Node): Nullable<Position> => {
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

  public drawImpl() {
    let ctx = this.ctx;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    if (this.rootPosition == null) {
      return;
    }

    let pr = pixelRatio();

    this.rootNode = this.layout(this.rootPosition);
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = '#fff';

    let pad = PADDING * pr + 0.5;
    let space = SPACING * pr;

    // Recursively draw the edges in the tree.
    let drawEdges = (node: Node) => {
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

    // Recursively draw the nodes in the tree.
    let r = RADIUS * pr;
    let drawNodes = (node: Node) => {
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
      ctx.fill();
      for (let child of node.children) {
        drawNodes(child);
      }
    };
    drawNodes(this.rootNode);
  }

  private layout(rootPosition: Position) {
    // Right-most node at each depth in the tree.
    let rightNode: Node[] = [];

    // We want to lay the nodes in the tree out such that the mainline of each
    // node (the chain of first children of all descendants) is a vertical line.
    // We do this by incrementally building a new tree of layout data in
    // traversal order of the game tree, satisfying the following constraints at
    // each step:
    //  1) A newly inserted node's x coordinate is greater than all other nodes
    //     at the same depth. This is enforced pre-traversal.
    //  2) A newly inserted node's x coordinate is greater or equal to the x
    //     coordinate of its first child. This is enforced post-traversal.
    let traverse = (node: Node, depth: number) => {
      // Satisfy constraint 1. In order to make the layout less cramped, we
      // actually check both the current depth and the next depth.
      for (let i = 0; i < 2 && depth + i < rightNode.length; ++i) {
        node.x = Math.max(node.x, rightNode[depth + i].x + 1);
      }
      rightNode[depth] = node;

      for (let childPosition of node.position.children) {
        let childNode = new Node(node, childPosition, node.x, depth + 1);
        node.children.push(childNode);
        traverse(childNode, depth + 1);
      }

      // Satisfy constraint 2.
      if (node.parent && node == node.parent.children[0]) {
        node.parent.x = Math.max(node.parent.x, node.x);
      }
    };

    let rootNode = new Node(null, rootPosition, 0, 0);
    traverse(rootNode, 0);

    // For clarity, the traverse function above set the node (x, y) positions
    // to lie on the unit grid. We'll transform those coordinates to the actual
    // canvas coordinates now.

    let pr = pixelRatio();
    let pad = PADDING * pr + 0.5;
    let space = SPACING * pr;

    let requiredWidth = pad * 2;
    let reposition = (node: Node) => {
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

  private resizeCanvas(width: number, height: number, pixelRatio: number) {
    let canvas = this.ctx.canvas;
    canvas.width = width * pixelRatio;
    canvas.height = height * pixelRatio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }
}

export {
  VariationTree,
}
