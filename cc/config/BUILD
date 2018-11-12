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

# Build condition labels that configure which inference engines are enabled.
# Additionally, enable_tf is also required in order for the following
# functionality, which is provided by TensorFlow:
#  - Google Cloud Storage access.
#  - Writing TensorFlow examples at the end of self play.
# enable_tf is not required to play Minigo over GTP.

config_setting(
    name = "enable_tf",
    define_values = {"tf": "1"},
)

config_setting(
    name = "enable_lite",
    define_values = {"lite": "1"},
)

config_setting(
    name = "enable_trt",
    define_values = {"trt": "1"},
)

config_setting(
    name = "enable_bt",
    define_values = {"bt": "1"},
)

config_setting(
    name = "enable_tpu",
    define_values = {"tpu": "1"},
)
