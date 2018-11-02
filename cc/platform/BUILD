package(default_visibility = [
    "//cc:__subpackages__",
])

licenses(["notice"])  # Apache License 2.0

cc_library(
    name = "utils",
    srcs = select({
        "@bazel_tools//src/conditions:darwin": ["utils_osx.cc"],
        "//conditions:default": ["utils_linux.cc"],
    }),
    hdrs = [
        "utils.h"
    ],
    deps = [
        "//cc:check",
    ],
)

