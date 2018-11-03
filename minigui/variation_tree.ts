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
const SPACE = 16;

// Padding from the edge of the canvas.
const PAD = 9;

// Raius of the nodes.
const RADIUS = 4;

interface Position {
  parent: Nullable<Position>;
  children: Position[];
}

class Node {
  children: Node[] = [];
  constructor(public parent: Nullable<Node>, public position: Position,
              public x: number, public y: number) {}
}

interface LayoutResult {
  maxDepth: number;
  rootNode: Node;
}

type ClickListener = (positions: Position[]) => void;

class VariationTree extends View {
  private rootNode: Nullable<Node> = null;
  private ctx: CanvasRenderingContext2D;

  private hoveredNode: Nullable<Node> = null;
  private activeNode: Nullable<Node> = null;

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
      let oldNode = this.hoveredNode;
      this.hoveredNode = this.hitTest(e.offsetX, e.offsetY);
      if (this.hoveredNode != oldNode) {
        if (this.hoveredNode != null) {
          canvas.classList.add('pointer');
        } else {
          canvas.classList.remove('pointer');
        }
      }
    });

    parent.addEventListener('click', (e) => {
      if (this.hoveredNode != null) {
        this.activeNode = this.hoveredNode;
        let p = this.hoveredNode.position;
        let positions: Position[] = [];
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

  public newGame(rootPosition: Position) {
    this.rootNode = new Node(null, rootPosition, PAD, PAD);
    this.activeNode = this.rootNode;
    this.resizeCanvas(1, 1, 1);
    this.layout();
    this.draw();
  }

  public addChild(parentPosition: Position, childPosition: Position) {
    if (this.rootNode == null) {
      throw new Error('Must start a game before attempting to add children');
    }

    // Find the corresponding node for parentPosition.
    let findParent = (node: Node): Nullable<Node> => {
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

    // Add a new child node to the parent if necessary, or reuse the existing
    // node.
    let childNode: Nullable<Node> = null;
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

  public onClick(cb: ClickListener) {
    this.listeners.push(cb);
  }

  private hitTest(x: number, y: number) {
    if (this.rootNode == null) {
      return null;
    }

    let threshold = 0.4 * SPACE;

    let traverse = (node: Node): Nullable<Node> => {
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

  public drawImpl() {
    let ctx = this.ctx;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    if (this.rootNode == null) {
      return;
    }

    let pr = pixelRatio();

    ctx.strokeStyle = '#888';
    ctx.lineWidth = pr;

    // Recursively draw the edges in the tree.
    let drawEdges = (node: Node) => {
      if (node.children.length == 0) {
        return;
      }

      // Draw edges from parent to all children.
      // The first child's edge is vertical.
      // The remiaining childrens' edges slope at 45 degrees.
      for (let child of node.children) {
        let x = child.x;
        if (child != node.children[0]) {
          x -= SPACE;
        }
        ctx.moveTo(pr * x, pr * node.y);
        ctx.lineTo(pr * child.x, pr * child.y);
        drawEdges(child);
      }

      // If a node has two or more children, draw a horizontal line to
      // connect the tops of all of the sloping edges.
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

    // Draw the active node in red.
    let r = RADIUS * pr;
    if (this.activeNode != null) {
      ctx.fillStyle = '#800';
      ctx.strokeStyle = '#ff0';
      ctx.beginPath();
      ctx.arc(pr * this.activeNode.x, pr * this.activeNode.y, r, 0,
              2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }

    // Draw the root node in gray.
    ctx.fillStyle = '#666';
    ctx.strokeStyle = '#888';
    ctx.beginPath();
    ctx.arc(pr * this.rootNode.x, pr * this.rootNode.y, r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Recursively draw the non-root nodes.
    // To avoid repeatedly changing the fill style between black and white,
    // we draw in two passes: first all the black nodes, then all the white
    // ones.
    let drawNodes = (node: Node, draw: boolean) => {
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

  private layout() {
    if (this.rootNode == null) {
      return;
    }

    // Right-most node at each depth in the tree.
    let rightNode: Node[] = [];

    // Space required to draw the canvas.
    let requiredWidth = PAD * 2;

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
      // Tell the compiler that node.parent is guaranteed to be non-null.
      let parent = node.parent as Node;

      // Satisfy constraint 1. In order to make the layout less cramped, we
      // actually check both the current depth and the next depth.
      node.x = parent.x;
      for (let i = 0; i < 2 && depth + i < rightNode.length; ++i) {
        node.x = Math.max(node.x, rightNode[depth + i].x + SPACE);
      }
      rightNode[depth] = node;

      for (let child of node.children) {
        traverse(child, depth + 1);
      }

      // Satisfy constraint 2.
      if (node == parent.children[0]) {
        parent.x = Math.max(parent.x, node.x);
      }
      requiredWidth = Math.max(requiredWidth, PAD + node.x);
    };

    // Traverse the root node's children: the root node doesn't need laying
    // out and this guarantees that all traversed nodes have parents.
    for (let child of this.rootNode.children) {
      traverse(child, 1);
    }

    let requiredHeight = PAD * 2 + (rightNode.length - 1) * SPACE;
    let pr = pixelRatio();
    if (requiredWidth * pr > this.ctx.canvas.width ||
        requiredHeight * pr > this.ctx.canvas.height) {
      this.resizeCanvas(requiredWidth, requiredHeight, pr);
    }
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
