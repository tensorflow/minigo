package(default_visibility = [
    ":__pkg__",
    "//cc:__subpackages__",
])

licenses(["notice"])  # Apache License 2.0

load(
    "//cc/config:minigo.bzl",
    "minigo_cc_binary",
    "minigo_cc_library",
    "minigo_cc_test",
    "minigo_cc_test_19_only",
    "minigo_cc_test_9_only",
    "minigo_engine_copts",
)

minigo_cc_library(
    name = "fake_dual_net",
    srcs = ["fake_dual_net.cc"],
    hdrs = ["fake_dual_net.h"],
    deps = [
        "//cc:base",
        "//cc:logging",
        "//cc/model",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

minigo_cc_library(
    name = "random_dual_net",
    srcs = ["random_dual_net.cc"],
    hdrs = ["random_dual_net.h"],
    deps = [
        "//cc:base",
        "//cc:logging",
        "//cc:random",
        "//cc/model",
        "//cc/model:factory",
        "//cc/platform",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

minigo_cc_library(
    name = "tf_dual_net",
    srcs = ["tf_dual_net.cc"],
    hdrs = ["tf_dual_net.h"],
    tags = ["manual"],
    deps = [
        "//cc:base",
        "//cc:logging",
        "//cc/file:path",
        "//cc/model",
        "//cc/model:factory",
        "//cc/tensorflow",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@wtf",
    ],
)

minigo_cc_library(
    name = "lite_dual_net",
    srcs = ["lite_dual_net.cc"],
    hdrs = ["lite_dual_net.h"],
    tags = ["manual"],
    deps = [
        "//cc:base",
        "//cc:logging",
        "//cc/file:path",
        "//cc/file",
        "//cc/model",
        "//cc/model:factory",
        "//cc/platform",
        "//cc/tensorflow:tf_lite",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

minigo_cc_library(
    name = "tpu_dual_net",
    srcs = ["tpu_dual_net.cc"],
    hdrs = ["tpu_dual_net.h"],
    tags = ["manual"],
    deps = [
        "//cc:base",
        "//cc:logging",
        "//cc:random",
        "//cc/file:path",
        "//cc/model",
        "//cc/model:factory",
        "//cc/tensorflow",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@wtf",
    ],
)

minigo_cc_test_9_only(
    name = "dual_net_test",
    size = "small",
    srcs = ["dual_net_test.cc"],
    copts = minigo_engine_copts,
    data = glob(["test_*.minigo"]),
    deps = [
        "//cc:logging",
        "//cc:position",
        "//cc:random",
        "//cc:symmetries",
        "//cc:test_utils",
        "//cc/model",
        "//cc/model:factory",
        "//cc/model:loader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
