package(default_visibility = [
    ":__pkg__",
    "//cc:__pkg__",
    "//cc/dual_net:__pkg__",
])

licenses(["notice"])  # Apache License 2.0

load(
    "//cc/config:minigo.bzl",
    "minigo_cc_library",
    "minigo_cc_test",
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
    name = "batching_model",
    srcs = ["batching_model.cc"],
    hdrs = ["batching_model.h"],
    deps = [
        "//cc:base",
        "//cc:logging",
        "//cc/model",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@wtf",
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

minigo_cc_library(
    name = "inference_cache",
    srcs = ["inference_cache.cc"],
    hdrs = ["inference_cache.h"],
    deps = [
        ":model",
        "//cc:base",
        "//cc:logging",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

minigo_cc_library(
    name = "reloading_model",
    srcs = ["reloading_model.cc"],
    hdrs = ["reloading_model.h"],
    deps = [
        ":model",
        "//cc:base",
        "//cc:logging",
        "//cc:position",
        "//cc/file",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

minigo_cc_test(
    name = "batching_model_test",
    size = "small",
    srcs = ["batching_model_test.cc"],
    deps = [
        ":batching_model",
        ":buffered_model",
        ":model",
        "//cc/tensorflow",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_test(
    name = "inference_cache_test",
    srcs = ["inference_cache_test.cc"],
    deps = [
        ":inference_cache",
        "//cc:random",
        "//cc:test_utils",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

minigo_cc_test(
    name = "reloading_model_test",
    size = "small",
    srcs = ["reloading_model_test.cc"],
    deps = [
        ":reloading_model",
        "//cc:logging",
        "//cc/file",
        "@com_google_googletest//:gtest_main",
    ],
)
