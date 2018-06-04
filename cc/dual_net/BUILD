package(default_visibility = [
    ":__pkg__",
    "//cc:__pkg__",
])

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
        "//cc:random",
        "//cc:symmetries",
        "@com_google_absl//absl/types:span",
    ],
)

minigo_cc_library(
    name = "fake_net",
    testonly = 1,
    srcs = ["fake_net.cc"],
    hdrs = ["fake_net.h"],
    deps = [
        ":dual_net",
        "//cc:base",
    ],
)

minigo_cc_library(
    name = "inference_server",
    srcs = ["inference_server.cc"],
    hdrs = ["inference_server.h"],
    deps = [
        ":dual_net",
        "//cc:thread_safe_queue",
        "//proto:inference_service_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_grpc//:grpc++",
    ],
)

# TODO(tommadams): rename tf_dual_net to tf_net
minigo_cc_library(
    name = "tf_dual_net",
    srcs = ["tf_dual_net.cc"],
    hdrs = ["tf_dual_net.h"],
    deps = [
        ":dual_net",
        "//cc:base",
        "//cc:position",
        "//cc:random",
        "//cc:symmetries",
        "//cc:tensorflow",
        "@com_google_absl//absl/types:span",
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

minigo_cc_test(
    name = "inference_server_test",
    srcs = ["inference_server_test.cc"],
    deps = [
        ":fake_net",
        ":inference_server",
        "//proto:inference_service_proto",
        "@com_google_googletest//:gtest_main",
    ],
)
