# Generates a cc_test target that runs when bazel is invoked with
# --config=minigo9.
# For --config=minigo19 builds, a valid but empty cc_test target is generated.
#
# This should be used instead of cc_test when writing unit tests that require
# a 9x9 board.
def minigo_9_test(name, srcs, size="small", deps=[], **kwargs):
  native.cc_test(
      name = name,
      size = size,
      srcs = select({
          "//cc/config:minigo9": srcs,
          "//conditions:default": [],
      }),
      deps = select({
          "//cc/config:minigo9": deps + ["@com_google_googletest//:gtest_main"],
          "//conditions:default": ["@com_google_googletest//:gtest_main"],
      }),
      **kwargs)

# Generates a cc_test target that runs when bazel is invoked with
# --config=minigo19.
# For --config=minigo9 builds, a valid but empty cc_test target is generated.
#
# This should be used instead of cc_test when writing unit tests that require
# a 19x19 board.
def minigo_19_test(name, srcs, size="small", deps=[], **kwargs):
  native.cc_test(
      name = name,
      size = size,
      srcs = select({
          "//cc/config:minigo19": srcs,
          "//conditions:default": [],
      }),
      deps = select({
          "//cc/config:minigo19": deps + ["@com_google_googletest//:gtest_main"],
          "//conditions:default": ["@com_google_googletest//:gtest_main"],
      }),
      **kwargs)
