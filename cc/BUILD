package(default_visibility = [
    ":__pkg__",
])

load(
    "//cc/config:minigo.bzl",
    "minigo_cc_binary",
    "minigo_cc_library",
    "minigo_cc_test",
    "minigo_cc_test_19_only",
    "minigo_cc_test_9_only",
)

cc_library(
    name = "tensorflow",
    srcs = [
        "tensorflow/libtensorflow_cc.so",
        "tensorflow/libtensorflow_framework.so",
    ],
    hdrs = glob(
        ["tensorflow/**/*"],
        exclude = ["tensorflow/*.so"],
    ),
    includes = [
        "tensorflow",
        "tensorflow/external/nsync/public",
    ],
)

minigo_cc_library(
    name = "base",
    srcs = [
        "color.cc",
        "coord.cc",
    ],
    hdrs = [
        "algorithm.h",
        "color.h",
        "constants.h",
        "coord.h",
        "move.h",
        "stone.h",
    ],
    deps = [
        ":check",
        "@com_google_absl//absl/strings",
    ],
)

minigo_cc_library(
    name = "check",
    srcs = [
        "check.cc",
    ],
    hdrs = [
        "check.h",
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/debugging:stacktrace",
        "@com_google_absl//absl/debugging:symbolize",
    ],
)

minigo_cc_library(
    name = "dual_net",
    srcs = ["dual_net.cc"],
    hdrs = ["dual_net.h"],
    deps = [
        ":base",
        ":position",
        ":random",
        ":symmetries",
        ":tensorflow",
    ],
)

minigo_cc_library(
    name = "gtp_player",
    srcs = ["gtp_player.cc"],
    hdrs = ["gtp_player.h"],
    deps = [
        ":dual_net",
        ":mcts",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

minigo_cc_library(
    name = "init",
    srcs = ["init.cc"],
    hdrs = ["init.h"],
    deps = [
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/debugging:symbolize",
    ],
)

minigo_cc_library(
    name = "inline_vector",
    hdrs = ["inline_vector.h"],
    deps = [
        ":check",
    ],
)

minigo_cc_library(
    name = "mcts",
    srcs = [
        "mcts_node.cc",
        "mcts_player.cc",
    ],
    hdrs = [
        "mcts_node.h",
        "mcts_player.h",
    ],
    deps = [
        ":base",
        ":dual_net",
        ":position",
        ":random",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

minigo_cc_library(
    name = "position",
    srcs = [
        "group.cc",
        "position.cc",
    ],
    hdrs = [
        "group.h",
        "position.h",
    ],
    deps = [
        ":base",
        ":inline_vector",
        ":tiny_set",
    ],
)

minigo_cc_library(
    name = "random",
    srcs = ["random.cc"],
    hdrs = ["random.h"],
    deps = [
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

minigo_cc_library(
    name = "sgf",
    srcs = ["sgf.cc"],
    hdrs = ["sgf.h"],
    deps = [
        ":base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

minigo_cc_library(
    name = "symmetries",
    hdrs = ["symmetries.h"],
    deps = [
        ":check",
    ],
)

minigo_cc_library(
    name = "tf_utils",
    srcs = ["tf_utils.cc"],
    hdrs = ["tf_utils.h"],
    deps = [
        ":base",
        ":dual_net",
        ":tensorflow",
        "@com_google_absl//absl/types:span",
    ],
)

minigo_cc_library(
    name = "tiny_set",
    hdrs = ["tiny_set.h"],
)

minigo_cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        ":base",
        ":mcts",
        ":position",
        "@com_google_absl//absl/strings",
    ],
)

minigo_cc_test(
    name = "coord_test",
    size = "small",
    srcs = ["coord_test.cc"],
    deps = [
        ":base",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_test_9_only(
    name = "dual_net_test",
    size = "small",
    srcs = ["dual_net_test.cc"],
    deps = [
        ":dual_net",
        ":position",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_test_9_only(
    name = "mcts_node_test",
    size = "small",
    srcs = ["mcts_node_test.cc"],
    deps = [
        ":mcts",
        ":position",
        ":random",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_test_9_only(
    name = "mcts_player_test",
    size = "small",
    srcs = ["mcts_player_test.cc"],
    deps = [
        ":base",
        ":dual_net",
        ":mcts",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_test_9_only(
    name = "position_test",
    size = "small",
    srcs = ["position_test.cc"],
    deps = [
        ":base",
        ":position",
        ":test_utils",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_test(
    name = "random_test",
    size = "small",
    srcs = ["random_test.cc"],
    deps = [
        ":random",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_test_19_only(
    name = "sgf_test",
    size = "small",
    srcs = ["sgf_test.cc"],
    deps = [
        ":sgf",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_test(
    name = "symmetries_test",
    size = "small",
    srcs = ["symmetries_test.cc"],
    deps = [
        ":symmetries",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":base",
        ":dual_net",
        ":gtp_player",
        ":init",
        ":mcts",
        ":random",
        ":sgf",
        ":tf_utils",
        "//cc/file",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

minigo_cc_binary(
    name = "position_benchmark",
    srcs = ["position_benchmark.cc"],
    deps = [
        ":base",
        ":position",
        "@com_google_benchmark//:benchmark",
    ],
)
