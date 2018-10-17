# Defines the preprocessor macro MINIGO_BOARD_SIZE=9 for all minigo_cc_*
# build targets when bazel build is invoked with --define=board_size=9.
# Defines the preprocessor macro MINIGO_BOARD_SIZE=19 for all minigo_cc_*
# build targets by default.

def _board_size_copts():
    return select({
        "//cc/config:minigo9": ["-DMINIGO_BOARD_SIZE=9"],
        "//conditions:default": ["-DMINIGO_BOARD_SIZE=19"],
    })

# Generates a cc_binary target that defines MINIGO_BOARD_SIZE.
def minigo_cc_binary(name, copts = [], **kwargs):
    native.cc_binary(
        name = name,
        copts = _board_size_copts() + copts,
        **kwargs
    )

# Generates a cc_library target that defines MINIGO_BOARD_SIZE.
def minigo_cc_library(name, copts = [], **kwargs):
    native.cc_library(
        name = name,
        copts = _board_size_copts() + copts,
        **kwargs
    )

# Generates a cc_test target that defines MINIGO_BOARD_SIZE.
def minigo_cc_test(name, size = "small", copts = [], **kwargs):
    native.cc_test(
        name = name,
        size = size,
        copts = _board_size_copts() + copts,
        **kwargs
    )

# Generates a cc_test target that defines MINIGO_BOARD_SIZE when bazel test is
# invoked with --define=board_size=9.
# Generates an empty test stub if the board_size is not defined or set to 19.
# this should be used when writing unit tests that require a 9x9 board.
def minigo_cc_test_9_only(name, srcs, size = "small", deps = [], copts = [], **kwargs):
    native.cc_test(
        name = name,
        size = size,
        srcs = select({
            "//cc/config:minigo9": srcs,
            "//conditions:default": [],
        }),
        deps = select({
            "//cc/config:minigo9": deps,
            "//conditions:default": ["@com_google_googletest//:gtest_main"],
        }),
        copts = _board_size_copts() + copts,
        **kwargs
    )

# Generates a cc_test target that defines MINIGO_BOARD_SIZE when bazel test is
# invoked with --define=board_size=19.
# Generates an empty test stub if the board_size is not defined or set to 9.
# This should be used when writing unit tests that require a 19x19 board.
def minigo_cc_test_19_only(name, srcs, size = "small", deps = [], copts = [], **kwargs):
    native.cc_test(
        name = name,
        size = size,
        srcs = select({
            "//cc/config:minigo9": [],
            "//conditions:default": srcs,
        }),
        deps = select({
            "//cc/config:minigo9": ["@com_google_googletest//:gtest_main"],
            "//conditions:default": deps,
        }),
        copts = _board_size_copts() + copts,
        **kwargs
    )
