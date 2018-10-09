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
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/types:span",
    ],
)

factory_engine_copts = select({
    "//cc/config:enable_remote": ["-DMG_ENABLE_REMOTE_DUAL_NET"],
    "//conditions:default": [],
}) + select({
    "//cc/config:enable_tf": ["-DMG_ENABLE_TF_DUAL_NET"],
    "//conditions:default": [],
}) + select({
    "//cc/config:enable_lite": ["-DMG_ENABLE_LITE_DUAL_NET"],
    "//conditions:default": [],
})

factory_engine_deps = select({
    "//cc/config:enable_remote": [":inference_server"],
    "//conditions:default": [],
}) + select({
    "//cc/config:enable_tf": [":tf_dual_net"],
    "//conditions:default": [],
}) + select({
    "//cc/config:enable_lite": [":lite_dual_net"],
    "//conditions:default": [],
})

minigo_cc_library(
    name = "factory",
    srcs = ["factory.cc"],
    hdrs = ["factory.h"],
    copts = factory_engine_copts,
    deps = [
        ":dual_net",
        ":batching_dual_net",
        "//cc:base",
        "//cc:check",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ] + factory_engine_deps,
)

minigo_cc_library(
    name = "fake_net",
    testonly = 1,
    srcs = ["fake_net.cc"],
    hdrs = ["fake_net.h"],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:check",
    ],
)

# TODO(tommadams): rename inference_server to remote_dual_net
minigo_cc_library(
    name = "inference_server",
    srcs = ["inference_server.cc"],
    hdrs = ["inference_server.h"],
    tags = ["manual"],
    deps = [
        ":dual_net",
        "//cc:thread_safe_queue",
        "//proto:inference_service_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
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
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

minigo_cc_library(
    name = "batching_dual_net",
    srcs = ["batching_dual_net.cc"],
    hdrs = [
        "batching_dual_net.h",
        "factory.h",
    ],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

minigo_cc_test_9_only(
    name = "dual_net_test",
    size = "small",
    srcs = ["dual_net_test.cc"],
    deps = [
        ":dual_net",
        "//cc:position",
        "//cc:test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_test_9_only(
    name = "inference_server_test",
    srcs = ["inference_server_test.cc"],
    deps = [
        ":fake_net",
        ":inference_server",
        "//cc:random",
        "//proto:inference_service_proto",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
