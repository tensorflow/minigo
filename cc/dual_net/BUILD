package(default_visibility = [
    ":__pkg__",
    "//cc:__pkg__",
])

licenses(["notice"])  # Apache License 2.0

load(
    "//cc/config:minigo.bzl",
    "minigo_cc_binary",
    "minigo_cc_library",
    "minigo_cc_test",
    "minigo_cc_test_19_only",
    "minigo_cc_test_9_only",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_tensorflow//:build_defs.bzl", "check")
load("@local_config_tensorrt//:build_defs.bzl", "if_tensorrt")

minigo_cc_library(
    name = "dual_net",
    srcs = ["dual_net.cc"],
    hdrs = ["dual_net.h"],
    deps = [
        "//cc:base",
        "//cc:position",
        "@com_google_absl//absl/types:span",
    ],
)

factory_engine_copts = select({
    "//cc/config:enable_tf": ["-DMG_ENABLE_TF_DUAL_NET"],
    "//conditions:default": [],
}) + select({
    "//cc/config:enable_lite": ["-DMG_ENABLE_LITE_DUAL_NET"],
    "//conditions:default": [],
}) + select({
    "//cc/config:enable_tpu": ["-DMG_ENABLE_TPU_DUAL_NET"],
    "//conditions:default": [],
}) + select({
    "//cc/config:enable_trt": ["-DMG_ENABLE_TRT_DUAL_NET"],
    "//conditions:default": [],
})

factory_engine_deps = select({
    "//cc/config:enable_tf": [":tf_dual_net"],
    "//conditions:default": [],
}) + select({
    "//cc/config:enable_lite": [":lite_dual_net"],
    "//conditions:default": [],
}) + select({
    "//cc/config:enable_tpu": [":tpu_dual_net"],
    "//conditions:default": [],
}) + select({
    "//cc/config:enable_trt": [":trt_dual_net"],
    "//conditions:default": [],
})

minigo_cc_library(
    name = "factory",
    srcs = ["factory.cc"],
    hdrs = ["factory.h"],
    copts = factory_engine_copts,
    deps = [
        ":dual_net",
        ":fake_dual_net",
        "//cc:base",
        "//cc:check",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ] + factory_engine_deps,
)

minigo_cc_library(
    name = "fake_dual_net",
    srcs = ["fake_dual_net.cc"],
    hdrs = ["fake_dual_net.h"],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:check",
    ],
)

minigo_cc_library(
    name = "tf_dual_net",
    srcs = ["tf_dual_net.cc"],
    hdrs = ["tf_dual_net.h"],
    tags = ["manual"],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:check",
        "//cc:thread_safe_queue",
        "//cc/tensorflow",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

minigo_cc_library(
    name = "lite_dual_net",
    srcs = ["lite_dual_net.cc"],
    hdrs = ["lite_dual_net.h"],
    tags = ["manual"],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:check",
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
        ":dual_net",
        "//cc:base",
        "//cc:check",
        "//cc:thread_safe_queue",
        "//cc/tensorflow",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

check(
    name = "trt_configured",
    condition = if_tensorrt(True, False),
    message = "Cannot build with --define=tf=1 when TensorFlow was not configured for TensorRT.",
    visibility = ["//visibility:private"],
)

minigo_cc_library(
    name = "trt_dual_net",
    srcs = ["trt_dual_net.cc"],
    hdrs = ["trt_dual_net.h"],
    data = [":trt_configured"],
    tags = ["manual"],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:check",
        "//cc:thread_safe_queue",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_tensorrt//:nv_infer",
        "@local_config_tensorrt//:tensorrt_headers",
        "@local_config_tensorrt_parsers//:nvparsers",
    ],
)

minigo_cc_library(
    name = "batching_dual_net",
    srcs = ["batching_dual_net.cc"],
    hdrs = [
        "batching_dual_net.h",
    ],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

minigo_cc_test_9_only(
    name = "dual_net_test",
    size = "small",
    srcs = ["dual_net_test.cc"],
    copts = factory_engine_copts,
    data = glob(["test_model.*"]),
    deps = [
        ":dual_net",
        ":factory",
        "//cc:position",
        "//cc:test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)
