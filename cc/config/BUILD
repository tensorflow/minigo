# Config settings that are used by the build rules defined in
# cc/config/minigo.bzl.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache License 2.0

# Build condition label that matches when C++ Minigo is being built for a 9x9
# board. By default, C++ Minigo is built for a 19x19 board.
config_setting(
    name = "minigo9",
    define_values = {"board_size": "9"},
)
