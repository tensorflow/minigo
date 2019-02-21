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
const SPACE = 19;

// Padding from the edge of the canvas.
const PAD = 11;

// Radius of the nodes.
const RADIUS = 5;

interface Position {
  parent: Nullable<Position>;
  children: Position[];
}

class Node {
  isMainline: boolean;
  children: Node[] = [];
  constructor(public parent: Nullable<Node>, public position: Position,
              public x: number, public y: number) {
    if (parent == null) {
      // Root node.
      this.isMainline = true;
    } else {
      this.isMainline = parent.isMainline && parent.children.length == 0;
      parent.children.push(this);
    }
  }
}

interface LayoutResult {
  maxDepth: number;
  rootNode: Node;
}

type ClickListener = (position: Position) => void;
type HoverListener = (position: Nullable<Position>) => void;

class VariationTree extends View {
  private rootNode: Nullable<Node> = null;
  private ctx: CanvasRenderingContext2D;

  private hoveredNode: Nullable<Node> = null;
  private activeNode: Nullable<Node> = null;

  private clickListeners: ClickListener[] = [];
  private hoverListeners: HoverListener[] = [];

  // Size of the canvas (ignoring the device's pixel ratio).
  private width = 0;
  private height = 0;

  // Maximum x and y coordinate of tree nodes.
  private maxX = 0;
  private maxY = 0;

  // Scroll offset for rendering the tree.
  private scrollX = 0;
  private scrollY = 0;

  // It's 2018 and MouseEvent.movementX and MouseEvent.movementY is still not
  // supported by all browsers, so we must calculate the movement deltas
  // ourselves... *sigh*
  private mouseX = 0;
  private mouseY = 0;

  constructor(parent: HTMLElement | string) {
    super();

    if (typeof(parent) == 'string') {
      parent = getElement(parent);
    }

    let canvas = document.createElement('canvas');
    this.ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
    parent.appendChild(canvas);
    this.resizeCanvas();

    parent.addEventListener('mousemove', (e) => {
      let oldNode = this.hoveredNode;
      this.hoveredNode = this.hitTest(e.offsetX, e.offsetY);
      if (this.hoveredNode != oldNode) {
        if (this.hoveredNode != null) {
          canvas.classList.add('pointer');
        } else {
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
      let moveHandler = (e: MouseEvent) => {
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
      let upHandler = (e: MouseEvent) => {
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

  setRoot(position: Position) {
    if (this.rootNode != null){
      throw new Error('Already have a root note, you should call newGame first');
    }
    this.rootNode = new Node(null, position, PAD, PAD);
    this.activeNode = this.rootNode;
  }

  setActive(position: Position) {
    if (this.activeNode != null && this.activeNode.position != position) {
      this.activeNode = this.lookupNode(position);
      this.scrollIntoView();
      this.draw();
    }
  }

  addChild(parentPosition: Position, childPosition: Position) {
    if (this.rootNode == null) {
      throw new Error('Must start a game before attempting to add children');
    }

    // Add a new child node to the parent if necessary, or reuse the existing
    // node.
    let parentNode = this.lookupNode(parentPosition);
    let childNode: Nullable<Node> = null;
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

  onClick(cb: ClickListener) {
    this.clickListeners.push(cb);
  }

  onHover(cb: HoverListener) {
    this.hoverListeners.push(cb);
  }

  private lookupNode(position: Position) {
    let impl = (node: Node): Nullable<Node> => {
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
    }
    let node = this.rootNode != null ? impl(this.rootNode) : null;
    if (node == null) {
      throw new Error('Couldn\'t find node');
    }
    return node;
  }

  private hitTest(x: number, y: number) {
    if (this.rootNode == null) {
      return null;
    }

    x += this.scrollX;
    y += this.scrollY;

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

  // TODO(tommadams): Don't draw parts of the tree that are out of view.
  drawImpl() {
    let ctx = this.ctx;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    if (this.rootNode == null) {
      return;
    }

    let pr = pixelRatio();

    ctx.save();
    ctx.translate(0.5 - pr * this.scrollX, 0.5 - pr * this.scrollY);

    ctx.lineWidth = pr;

    // Recursively draw the edges in the tree.
    let drawEdges = (node: Node, drawMainline: boolean) => {
      if (node.children.length == 0) {
        return;
      }

      // Draw edges from parent to all children.
      // The first child's edge is horizontal.
      // The remaining childrens' edges slope at 45 degrees.
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

      // If a node has two or more children, draw a vertical line to
      // connect the tops of all of the sloping edges.
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
    ctx.strokeStyle = '#888';
    if (this.activeNode != this.rootNode) {
      ctx.fillStyle = '#666';
      ctx.beginPath();
      ctx.arc(pr * this.rootNode.x, pr * this.rootNode.y, r, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }

    // Recursively draw the non-root nodes.
    // To avoid repeatedly changing the fill style between black and white,
    // we draw in two passes: first all the black nodes, then all the white
    // ones.
    let drawNodes = (node: Node, draw: boolean) => {
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

  private scrollIntoView() {
    if (this.activeNode == null) {
      return;
    }
    if (this.activeNode.x - this.scrollX > this.width - PAD) {
      this.scrollX = this.activeNode.x - this.width + PAD;
    } else if (this.activeNode.x - this.scrollX < PAD) {
      this.scrollX = this.activeNode.x - PAD;
    }
    if (this.activeNode.y - this.scrollY > this.height - PAD) {
      this.scrollY = this.activeNode.y - this.height + PAD;
    } else if (this.activeNode.y - this.scrollY < PAD) {
      this.scrollY = this.activeNode.y - PAD;
    }
  }

  private layout() {
    this.maxX = 0;
    this.maxY = 0;
    if (this.rootNode == null) {
      return;
    }

    // Bottom-most node at each depth in the tree.
    let bottomNode: Node[] = [this.rootNode];

    // We want to lay the nodes in the tree out such that the main line of each
    // node (the chain of first children of all descendants) is a horizontal
    // line. We do this by incrementally building a new tree of layout data in
    // traversal order of the game tree, satisfying the following constraints at
    // each step:
    //  1) A newly inserted node's y coordinate is greater than all other nodes
    //     at the same depth. This is enforced pre-traversal.
    //  2) A newly inserted node's y coordinate is greater or equal to the y
    //     coordinate of its first child. This is enforced post-traversal.
    let traverse = (node: Node, depth: number) => {
      // Tell the compiler that node.parent is guaranteed to be non-null.
      let parent = node.parent as Node;

      // Satisfy constraint 1. In order to make the layout less cramped, we
      // actually check both the current depth and the next depth.
      node.y = parent.y;
      for (let i = 0; i < 2 && depth + i < bottomNode.length; ++i) {
        node.y = Math.max(node.y, bottomNode[depth + i].y + SPACE);
      }
      bottomNode[depth] = node;

      for (let child of node.children) {
        traverse(child, depth + 1);
      }

      // Satisfy constraint 2.
      if (node == parent.children[0]) {
        parent.y = Math.max(parent.y, node.y);
      }

      this.maxX = Math.max(this.maxX, node.x);
      this.maxY = Math.max(this.maxY, node.y);
    };

    // Traverse the root node's children: the root node doesn't need laying
    // out and this guarantees that all traversed nodes have parents.
    for (let child of this.rootNode.children) {
      traverse(child, 1);
    }
  }

  private resizeCanvas() {
    let pr = pixelRatio();
    let canvas = this.ctx.canvas;
    let parent = canvas.parentElement as HTMLElement;
    this.width = parent.offsetWidth;
    this.height = parent.offsetHeight;
    canvas.width = pr * this.width;
    canvas.height = pr * this.height;
    canvas.style.width = `${this.width}px`;
    canvas.style.height = `${this.height}px`;
  }
}

export {
  VariationTree,
}
