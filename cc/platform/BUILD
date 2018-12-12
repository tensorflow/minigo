package(default_visibility = [
    "//cc:__subpackages__",
])

licenses(["notice"])  # Apache License 2.0

cc_library(
    name = "platform",
    srcs = select({
        "@bazel_tools//src/conditions:darwin": ["utils_osx.cc"],
        "@bazel_tools//src/conditions:windows": ["utils_windows.cc"],
        "//conditions:default": ["utils_linux.cc"],
    }),
    hdrs = [
        "utils.h",
    ],
    deps = [
        "//cc:logging",
    ],
)
