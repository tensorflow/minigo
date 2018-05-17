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
