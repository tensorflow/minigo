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
        "//cc:inline_vector",
        "//cc:position",
    ],
)

minigo_cc_library(
    name = "buffered_model",
    srcs = ["buffered_model.cc"],
    hdrs = ["buffered_model.h"],
    deps = [
        ":model",
        "//cc:thread_safe_queue",
    ],
)
