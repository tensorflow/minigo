# Minigui

A UI for Minigo that runs in your browser.

Minigui runs as a Python web server that you can connect to from your browser
of choice. We recommend Chrome and occasionally test with Firefox.

Like other UIs, Minigui doesn't contain a Go playing engine and in fact doesn't
even understand the rules of Go (it relies on the engine to tell it what stones
are on the board after each move is played).

Minigui communicates to the engine using Go Text Protocol
[GTP](https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html),
but requires some extensions (described later).

We recommend using the C++ Minigo engine as the Go engine (see
[cc/README.md](https://github.com/tensorflow/minigo/tree/master/cc/README.md))
for more information, though most modes will work with the Python engine with
reduced performance. Currently, Minigui's study mode requires the C++ engine.

## Simple Instructions

1. Make sure you have Docker installed

1. Pick a model from [Cloudy Go](http://cloudygo.com). Make note of the Model
   Name and the Board Size.

    ```shell
    export MINIGUI_BOARD_SIZE=9
    export MINIGUI_MODEL=000360-grown-teal
    ```

1. From the root directory, run:

    ```shell
    cluster/minigui/run-local.sh
    ```

1. Navigate to `localhost:5001`

## Advanced Instructions

1. Install the minigo python requirements: `pip install -r requirements.txt` (or
   `pip3 ...` depending how you've set things up).

1. Install TensorFlow (here, we use the CPU install): `pip install "tensorflow>=1.7,<1.8"`

1. Install the **minigui** python requirements: `pip install -r minigui/requirements.txt`

1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/downloads)

1. Pick a model. See [Cloudy Go](http://cloudygo.com) for the available models.

1. Change the variables you want (these are the defaults):

    ```shell
    export MINIGUI_BUCKET_NAME=minigo-pub
    export MINIGUI_GCS_DIR=v15-19x19/models
    export MINIGUI_MODEL=000990-cormorant
    export MINIGUI_MODEL_TMPDIR=/tmp/minigo-models
    export MINIGUI_BOARD_SIZE=19
    ```

1. Run `source minigui-common.sh`

1. Compile the Typescript to JavaScript. (Requires
   [typescript compiler](https://www.typescriptlang.org/#download-links)).
   From the `minigui` directory run: `tsc`

1. Run `./fetch-and-run.sh`

1. open localhost:5001 (or whatever value you used for $MINIGUI\_PORT).

1. The buttons in the upper right that say 'Human' can be toggled to set which
   color Minigo will play.

## Minigui modes

Minigui has various modes of operation.

### Study mode

Study mode allows you to load previously recorded SGF games, explore variations
and explore what the engine thinks of each position.

**Minigui's study mode currently requires the C++ Minigo engine.**

### Vs mode

A mode that plays two models or engines against each other and displays the
variations that each engine is considering in real time.

### Demo mode

The original demo built for GTC.

### Lightweight Demo mode

A lightweight demo mode built as experiment for running Minigo on a Raspberry
Pi.

### Kiosk mode

A self-play only version of the demo mode.

## Technical discussion

A technical discussion of the features that a Go engine must implement to
support Minigui now follows.

### Communicating game state

#### Position messages

The Minigui frontend doesn't understand the rules of Go. Consequently, every
time a move is played or a new game is started, the engine must communicate the
game state to the frontend. This is done by printing a JSON object on a single
line to `stderr` with the prefix `mg-position:`.

The JSON object is made up of the following fields (a `?` denotes an optional
field, other fields are required):

 - `id: string`: an ID that uniquely identifies the current board
   position. Note that using something like the Zobrist of the board stones is
   insufficient for modes like Study that support multiple active variations
   because multiple variations can lead to the same board state.
 - `parentId?: string`: an ID that uniquely identifies the previous board
   position (the parent node in the game tree). This field must be set for
   all board positions except the starting empty board.
 - `moveNum: number`: the current move number.
 - `toPlay: string`: whether it is black or white to play. Must be either `"b"`
   or `"w"`.
 - `stones?: string`: a string of N\*N characters (where N is the board size,
   usually 19) containing all stones concatenated together, where `.` is used
   to represent an empty point, `X` is a black stone and `O` is a white stone.
   The `stones` string should be constructed by working from top to bottom, left
   to right. For a 19\*19 board, this means the first character in the string
   is for A19 and the last is for T1.
 - `gameOver?: boolean`: true or false. If `gameOver` is not set, Minigui
   assumes that the game isn't over.
 - `move?: string`: the move that led to this position, formatted as a GTP
   coordinate. This field must be set for all board positions except the
   starting empty board.
 - `comment?: string`: any comments for this position (perhaps from a loaded
   SGF).
 - `caps?: number[]`: the number of captured stones. Black captures should be
   stored in `caps[0]` and white captures should be stored in `caps[1]`. If
   `caps` is not provided, the number of captures for both players is assumed
   to be 0.

These `mg-position` messages should be printed (and flushed) while processing
`clear_board`, `play` and `genmove` command **before** the command outputs its
result. Additionally, when processing a `loadsgf` message, an `mg-position`
object should be printed after playing each move in the SGF.

Below is a example from Minigo. This example has been formatted nicely to make it
more legible, but the engine **must** print the JSON object to a single line
otherwise Minigui will not be able to parse it.

```
mg-position: {
  "gameOver": false,
  "id": "0x559687366c10",
  "move": "G10",
  "moveNum": 36,
  "parentId": "0x559687364820",
  "stones": "........................................O.O...............XX....X.......X....OXO................XOO................XXO.................XO.................XXOO................OXXO.......................................................................X...............................O......O........X.X.O.......O..X....XOO.........................................",
  "toPlay": "B"
}
```

And the stones in this board position are:

```
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . O . O . . . . . . . . . . . . . .
. X X . . . . X . . . . . . . X . . .
. O X O . . . . . . . . . . . . . . .
. X O O . . . . . . . . . . . . . . .
. X X O . . . . . . . . . . . . . . .
. . X O . . . . . . . . . . . . . . .
. . X X O O . . . . . . . . . . . . .
. . . O X X O . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
. . X . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . O . . .
. . . O . . . . . . . . X . X . O . .
. . . . . O . . X . . . . X O O . . .
. . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . .
```

And here's how an engine should print the position JSON: all one one line with a
`mg-position:` prefix.

```
mg-position: {"gameOver":false,"id":"0x559687366c10","move":"G10","moveNum":36,"parentId":"0x559687364820","stones":"........................................O.O...............XX....X.......X....OXO................XOO................XXO.................XO.................XXOO................OXXO.......................................................................X...............................O......O........X.X.O.......O..X....XOO.........................................","toPlay":"B"}
```

#### Update messages

While an engine is thinking (in response to a `genmove` command or pondering
while it's the other player's turn), an engine can also send incremental
updates for the current position, containing variations considered, expected win
rate, number of visits, etc.

These updates should be printed as a JSON object (again on a single line) with
the prefix `mg-update:`.

 - `id: string`: the position's id.
 - `n?: number`: the number of reads for the position.
 - `q?: number`: estimated winrate for the position from black's perspective in
   the range [-1, 1].
 - `treeStats?: TreeStats`: stats for the current search tree (see below).
 - `variations?: {string: Variation}`: a map from GTP coordinate to a variation
   (see below), e.g. "K10" for the variation beginning on point K10. Some
   Minigui modes also display a live view of the current variation being
   searched. This should be written to `variations` under the key `"live"`.

Where `TreeStats` is:
 - `numNodes: number`: total number of nodes in the search tree.
 - `numLeafNodes: number`: number of leaf nodes in the search tree.
 - `maxDepth: number`: maximum depth of the search tree.

And `Variation` is:
 - `n: number`: number of reads for this variation.
 - `q: number`: expected winrate for the variation from black's perspective in
   the range [-1, 1].
 - `moves: string[]`: the list of moves in this variation as GTP coordinates.
   Note that the first coordinate is present in the `move` list even though it
   is also used as the key in the parent `variations` object.

We recommend only reporting variations that your engine considers interesting.

Here's a simple example from Minigo, again pretty-printed to aid legibility:

```
mg-update: {
  "id": "0x55834bd840f0",
  "n": 23,
  "q": -0.9926629662513733,
  "variations": {
    "B14": {
      "moves": ["B14"],
      "n": 1,
      "q": -0.9774199724197388
    },
    "C14": {
      "moves": ["C14","A13"],
      "n": 2,
      "q":-0.9983565211296082
    },
    "D15": {
      "moves": ["D15","A13"],
      "n": 2,
      "q": -0.9999065399169922
    },
    "H8":{
      "moves": ["H8","G10"],
      "n": 3,
      "q": -0.9687618017196655
    },
    "J4": {
      "moves": ["J4","J5","K5"],
      "n": 4,
      "q": -0.9998496770858765
    },
  }
}
```

### Synchronizing stdout and stderr

All `mg-position` and `mg-update` generated in response to a GTP command must be
printed and flushed to `stderr` because GTP mandates what can and cannot be
written to `stdout`.

When processing a GTP command, the engine must ensure that both `stdout` and
`stderr` stay in sync by printing and flushing the string `__GTP_CMD_DONE__` to
`stderr` before it writes the GTP response to `stdout`, e.g.:

```
stdin: genmove b
stderr: mg-update: {...}
stderr: mg-update: {...}
stderr: mg-update: {...}
stderr: mg-position: {...}
stderr: __GTP_CMD_DONE__
stdout: = D4
```

### GTP extensions

 - **echo**
   - *arguments*: `string*` - text to output
   - *effects*: none
   - *output*: `string*` - the command arguments
   - *fails*: never
   - *comments*: Simply outputs the arguments as given. The `echo` command is
     used as part of the handshaking protocol when establishing a connection
     with the backend.

 - **select\_position**
   - *arguments*: `string id` - the id of a position.
   - *effects*: The game is reset to the position that has the given id.
   - *output*: none
   - *fails*: If the id doesn't correspond to a position played in the game or
     one of its variations.
   - *comments*: Required only for Minigui's study mode.
