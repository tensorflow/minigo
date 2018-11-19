package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache License 2.0

cc_library(
    name = "file",
    srcs = [
        "path.cc",
        "utils_shared.cc",
    ] + select({
        "//cc/config:enable_tf": ["utils_tf.cc"],
        "//conditions:default": ["utils_stdio.cc"],
    }),
    hdrs = [
        "path.h",
        "utils.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
    ] + select({
        "//cc/config:enable_tf": ["//cc:tensorflow"],
        "//conditions:default": [],
    }),
)
