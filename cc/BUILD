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

minigo_srcs = [
    "color.cc",
    "coord.cc",
    "dual_net.cc",
    "group.cc",
    "mcts_node.cc",
    "mcts_player.cc",
    "position.cc",
    "random.cc",
]

minigo_hdrs = [
    "algorithm.h",
    "color.h",
    "constants.h",
    "coord.h",
    "dual_net.h",
    "group.h",
    "inline_vector.h",
    "mcts_node.h",
    "mcts_player.h",
    "position.h",
    "random.h",
    "stone.h",
    "symmetries.h",
    "tiny_set.h",
]

minigo_deps = [
    ":tensorflow",
    "@com_google_absl//absl/base",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/time",
    "@com_google_absl//absl/types:span",
]

cc_library(
    name = "minigo_9",
    srcs = minigo_srcs,
    hdrs = minigo_hdrs,
    defines = ["MINIGO_BOARD_SIZE=9"],
    deps = minigo_deps,
)

cc_library(
    name = "minigo_19",
    srcs = minigo_srcs,
    hdrs = minigo_hdrs,
    defines = ["MINIGO_BOARD_SIZE=19"],
    deps = minigo_deps,
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
    name = "coord_19_test",
    size = "small",
    srcs = ["coord_test.cc"],
    deps = [
        ":minigo_19",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dual_net_test",
    size = "small",
    srcs = ["dual_net_test.cc"],
    deps = [
        ":minigo_9",
        ":test_utils_9",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "coord_9_test",
    size = "small",
    srcs = ["coord_test.cc"],
    deps = [
        ":minigo_9",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "mcts_node_test",
    size = "small",
    srcs = ["mcts_node_test.cc"],
    deps = [
        ":minigo_9",
        ":test_utils_9",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "mcts_player_test",
    size = "small",
    srcs = ["mcts_player_test.cc"],
    deps = [
        ":minigo_9",
        ":test_utils_9",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "position_test",
    size = "small",
    srcs = ["position_test.cc"],
    deps = [
        ":minigo_9",
        ":test_utils_9",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "random_test",
    size = "small",
    srcs = ["random_test.cc"],
    deps = [
        ":minigo_9",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "symmetries_test",
    size = "small",
    srcs = ["symmetries_test.cc"],
    deps = [
        # TODO(tommadams): Break minigo into smaller libraries so small tests
        # like this don't have to transitively depend on stuff like Tensorflow.
        ":minigo_9",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "tf_test",
    srcs = ["tf_test.cc"],
    deps = [
        ":minigo_9",
        ":tensorflow",
        "@com_github_gflags_gflags//:gflags",
    ],
)

cc_binary(
    name = "position_benchmark",
    srcs = ["position_benchmark.cc"],
    deps = [
        ":minigo_19",
        "@com_google_benchmark//:benchmark",
    ],
)
