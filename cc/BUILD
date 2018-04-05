load(":config/minigo_test.bzl", "minigo_9_test", "minigo_19_test")

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

cc_library(
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
        "stone.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
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

cc_library(
    name = "inline_vector",
    hdrs = ["inline_vector.h"],
)

cc_library(
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

cc_library(
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

cc_library(
    name = "random",
    srcs = ["random.cc"],
    hdrs = ["random.h"],
    deps = [
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "symmetries",
    hdrs = ["symmetries.h"],
)

cc_library(
    name = "tiny_set",
    hdrs = ["tiny_set.h"],
)

cc_library(
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

cc_library(
    name = "test_utils_9",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [":minigo_9"],
)

cc_library(
    name = "test_utils_19",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [":minigo_19"],
)

cc_test(
    name = "coord_test",
    size = "small",
    srcs = ["coord_test.cc"],
    deps = [
        ":base",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_9_test(
    name = "dual_net_test",
    size = "small",
    srcs = ["dual_net_test.cc"],
    deps = [
        ":dual_net",
        ":position",
        ":test_utils",
    ],
)

minigo_9_test(
    name = "mcts_node_test",
    size = "small",
    srcs = ["mcts_node_test.cc"],
    deps = [
        ":mcts",
        ":position",
        ":random",
        ":test_utils",
    ],
)

minigo_9_test(
    name = "mcts_player_test",
    size = "small",
    srcs = ["mcts_player_test.cc"],
    deps = [
        ":base",
        ":dual_net",
        ":mcts",
        ":test_utils",
    ],
)

minigo_9_test(
    name = "position_test",
    size = "small",
    srcs = ["position_test.cc"],
    deps = [
        ":base",
        ":position",
        ":test_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "random_test",
    size = "small",
    srcs = ["random_test.cc"],
    deps = [
        ":random",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "symmetries_test",
    size = "small",
    srcs = ["symmetries_test.cc"],
    deps = [
        ":symmetries",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "tf_test",
    srcs = ["tf_test.cc"],
    deps = [
        ":base",
        ":dual_net",
        ":mcts",
        "@com_github_gflags_gflags//:gflags",
    ],
)

cc_binary(
    name = "position_benchmark",
    srcs = ["position_benchmark.cc"],
    deps = [
        ":base",
        ":position",
        "@com_google_benchmark//:benchmark",
    ],
)
