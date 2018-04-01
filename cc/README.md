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
time, using the MINIGO\_BOARD\_SIZE preprocessor define. To support this, the
Bazel build rules provide two configurations: minigo9 and minigo19.

Pass --config=minigo9 when invoking bazel to build Minigo for a 9x9 board.
Pass --config=minigo19 when invoking bazel to build Minigo for a 19x19 board.

## Running the unit tests

```shell
bazel test --config=minigo9 cc/...
bazel test --config=minigo19 cc/...
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
