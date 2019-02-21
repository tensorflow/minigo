package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache License 2.0

cc_library(
    name = "path",
    srcs = ["path.cc"],
    hdrs = ["path.h"],
    deps = [
        "//cc/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "utils_tf",
    srcs = ["utils_tf.cc"],
    hdrs = ["utils.h"],
    tags = ["manual"],
    deps = [
        ":path",
        "//cc:logging",
        "//cc:tensorflow",
        "//cc/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "utils_no_tf",
    srcs = select({
        "@bazel_tools//src/conditions:windows": ["utils_windows.cc"],
        "//conditions:default": ["utils_posix.cc"],
    }),
    hdrs = ["utils.h"],
    deps = [
        ":path",
        "//cc:logging",
        "//cc/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "file",
    hdrs = [
        "path.h",
        "utils.h",
    ],
    deps = [
        ":path",
        "//cc/platform",
        "@com_google_absl//absl/strings",
    ] + select({
        "//cc/config:enable_tf": [":utils_tf"],
        "//conditions:default": [":utils_no_tf"],
    }),
)

cc_test(
    name = "path_test",
    srcs = ["path_test.cc"],
    deps = [
        ":file",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":file",
        "//cc:base",
        "//cc:logging",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)
