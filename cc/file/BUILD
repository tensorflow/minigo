package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache License 2.0


cc_library(
    name = "path",
    srcs = ["path.cc"],
    hdrs = ["path.h"],
    deps = [
        "//cc/platform:utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "utils_tf",
    srcs = ["utils_tf.cc"],
    hdrs = ["utils.h"],
    deps = [
        ":path",
        "//cc:tensorflow",
        "//cc/platform:utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "utils_no_tf",
    srcs = select({
        "@bazel_tools//src/conditions:windows": ["utils_windows.cc"],
        # TODO(tommadams): rename utils_posix.cc
        "//conditions:default": ["utils_stdio.cc"],
    }),
    hdrs = ["utils.h"],
    deps = [
        ":path",
        "//cc/platform:utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "file",
    deps = [":path"] + select({
        "//cc/config:enable_tf": [":utils_tf"],
        "//conditions:default": [":utils_no_tf"],
    }),
)
