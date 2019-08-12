package(default_visibility = [
    ":__pkg__",
    "//cc:__pkg__",
    "//cc/dual_net:__pkg__",
])

licenses(["notice"])  # Apache License 2.0

load(
    "//cc/config:minigo.bzl",
    "minigo_cc_library",
)

minigo_cc_library(
    name = "model",
    srcs = ["model.cc"],
    hdrs = ["model.h"],
    deps = [
        "//cc:base",
        "//cc:position",
        "@com_google_absl//absl/types:span",
    ],
)
