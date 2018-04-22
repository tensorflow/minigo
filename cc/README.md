# Minigo C++ implementation

This directory contains a in-developent C++ port of Minigo.

## Set up

The C++ Minigo port uses the [Bazel](https://bazel.build/) build system.
It depends on the Tensorflow C++ libraries, but we have not yet set up
Bazel WORKSPACE and BUILD rules to automatically download and configure
Tensorflow so (for now at least) you must perform a manual step to build the
library:

```shell
./cc/configure_tensorflow.sh
```

This will automatically perform the first steps of
[Installing Tensorflow from Sources](https://www.tensorflow.org/install/install_sources)
but instead of installing the Tensorflow package, it extracts the generated C++
headers into the `cc/tensorflow` subdirectory of the repo. The script then
builds the required Tensorflow shared libraries and copies them to the same
directory. The tensorflow cc\_library build target in cc/BUILD pulls these
header & library files together into a format the Bazel understands how to link
against.

## Building

The C++ Minigo implementation requires that the board size be defined at compile
time, using the MINIGO\_BOARD\_SIZE preprocessor define. This allows us to
significantly reduce the number of heap allocations performed. However, it does
mean that you must specify the board size when compiling by invoking Bazel with
either `--define=board_size=9` or `--define=board_size=19`.
This defines MINIGO\_BOARD\_SIZE when compiling the Minigo sources, but none of
their dependencies, which avoids unnecessarily recompiling Minigo's
dependencies when switching board size.

Pass `--define=board_size=9` when invoking Bazel to build Minigo for a 9x9
board.

Pass `--define=board_size=19` when invoking Bazel to build Minigo for a 19x19
board.

## Running the unit tests

Minigo's C++ unit tests operate on both 9x9 and 19x19, and some tests are only
enabled for a particular board size. Consequently, you must run the tests
multiple times:

```shell
bazel test --define=board_size=9 cc/...  &&  bazel test --define=board_size=19 cc/...
```

## Running self play:

The C++ Minigo binary requires the model to be provided in GraphDef binary
proto format.

To run Minigo with a 9x9 model:

```shell
bazel build --config=minigo9 -c opt cc:main
bazel-bin/cc/main --model=$MODEL_PATH
```

To run Minigo with a 19x19 model:

```shell
bazel build --config=minigo9 -c opt cc:main
bazel-bin/cc/main --model=$MODEL_PATH
```

The Minigo binary has a lot of command line arguments that configure its
behavior, run it with --helpshort to see the full list.

```shell
bazel-bin/cc/main --helpshort
```

## Design

The general structure of the C++ code tries to follow the Python code where
appropriate, however a lot of the implementation details are different for
performance reasons. In particular, the logic that handles the board state
strives for a small memory footprint and eschews data structures more
sophisticated than a simple array in an effort to minimize memory allocations
and maximize cache locality of data. In doing so, the implementation is kept
fairly simple (there is no need for a LibertyTracker) and at the time of writing
performance of the Position code was more than 450x that of its Python
counterpart.

## Style guide

The C++ code follows
[Google's C++ style guide](https://github.com/google/styleguide)
and we use cpplint to delint.
