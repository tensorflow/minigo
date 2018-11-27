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
    copts = select({
        "//cc/config:enable_gpu": ["-DMINIGO_ENABLE_GPU=1"],
        "//conditions:default": [],
    }),
    tags = ["manual"],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:check",
        "//cc:tensorflow",
        "//cc:thread_safe_queue",
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
        "//cc:tf_lite",
        "//cc/platform",
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
        "//cc:tensorflow",
        "//cc:thread_safe_queue",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

minigo_cc_library(
    name = "trt_dual_net",
    srcs = ["trt_dual_net.cc"],
    hdrs = ["trt_dual_net.h"],
    tags = ["manual"],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:check",
        "//cc:cuda",
        "//cc:tensorrt",
        "//cc:thread_safe_queue",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
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
