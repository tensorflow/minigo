package(default_visibility = [
    ":__pkg__",
    "//cc:__pkg__",
    "//cc/dual_net:__pkg__",
])

licenses(["notice"])  # Apache License 2.0

load(
    "//cc/config:minigo.bzl",
    "minigo_cc_binary",
    "minigo_cc_library",
    "minigo_cc_test",
    "minigo_engine_copts",
    "minigo_engine_deps",
)

minigo_cc_library(
    name = "loader",
    srcs = ["loader.cc"],
    hdrs = ["loader.h"],
    copts = minigo_engine_copts,
    deps = [
        ":model",
        ":factory",
        "//cc:logging",
        "//cc:json",
        "//cc/file",
        "//cc/dual_net:random_dual_net",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ] + minigo_engine_deps,
)

minigo_cc_library(
    name = "factory",
    srcs = ["factory.cc"],
    hdrs = ["factory.h"],
    deps = [
        ":model",
        "//cc:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

minigo_cc_library(
    name = "model",
    srcs = [
        "features.cc",
        "features_internal.h",
        "model.cc",
        "types.cc",
    ],
    hdrs = [
        "features.h",
        "model.h",
        "types.h",
    ],
    deps = [
        "//cc:base",
        "//cc:inline_vector",
        "//cc:logging",
        "//cc:position",
        "//cc:symmetries",
        "//cc:tiny_set",
        "//cc/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

minigo_cc_library(
    name = "batching_model",
    srcs = ["batching_model.cc"],
    hdrs = ["batching_model.h"],
    deps = [
        ":buffered_model",
        "//cc:base",
        "//cc:logging",
        "//cc/model",
        "//cc/model:factory",
        "//cc/model:loader",
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
        "//cc/async:thread_safe_queue",
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
        "//cc:position",
        "//cc:symmetries",
        "//cc:zobrist",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

minigo_cc_test(
    name = "types_test",
    srcs = ["types_test.cc"],
    deps = [
        ":model",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_test(
    name = "features_test",
    srcs = ["features_test.cc"],
    deps = [
        ":model",
        "//cc:position",
        "//cc:random",
        "//cc:symmetries",
        "//cc:test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_test(
    name = "inference_cache_test",
    srcs = ["inference_cache_test.cc"],
    deps = [
        ":inference_cache",
        "//cc:random",
        "//cc:symmetries",
        "//cc:test_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

minigo_cc_binary(
    name = "features_benchmark",
    srcs = ["features_benchmark.cc"],
    deps = [
        ":model",
        "//cc:base",
        "//cc:init",
        "//cc:logging",
        "//cc:position",
        "//cc:random",
        "//cc:symmetries",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)
