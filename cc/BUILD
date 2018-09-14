package(default_visibility = [
    ":__subpackages__",
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

cc_library(
    name = "tensorflow",
    srcs = [
        "tensorflow/libtensorflow_cc.so",
        "tensorflow/libtensorflow_framework.so",
    ],
    hdrs = glob(
        ["tensorflow/**/*"],
        exclude = [
            "**/*.a",
            "**/*.so",
            "tensorflow/flatbuffers/**",
            "tensorflow/tensorflow/contrib/lite/**",
        ],
    ),
    includes = [
        "tensorflow",
        "tensorflow/external/nsync/public",
    ],
)

cc_library(
    name = "tf_lite",
    srcs = [
        "tensorflow/libtensorflow_lite.a",
    ],
    hdrs = glob(
        [
            "tensorflow/flatbuffers/*.h",
            "tensorflow/tensorflow/contrib/lite/**",
        ],
        exclude = ["**/*.so"],
    ),
    includes = ["tensorflow"],
    linkopts = ["-ldl"],
)

minigo_cc_library(
    name = "base",
    srcs = [
        "color.cc",
        "coord.cc",
        "group.cc",
    ],
    hdrs = [
        "algorithm.h",
        "color.h",
        "constants.h",
        "coord.h",
        "group.h",
        "move.h",
        "stone.h",
    ],
    deps = [
        ":check",
        ":inline_vector",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/debugging:stacktrace",
        "@com_google_absl//absl/debugging:symbolize",
    ],
)

minigo_cc_library(
    name = "gtp_player",
    srcs = ["gtp_player.cc"],
    hdrs = ["gtp_player.h"],
    deps = [
        ":base",
        ":mcts",
        ":sgf",
        "//cc/dual_net",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
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
        ":check",
        ":position",
        ":random",
        ":symmetries",
        "//cc/dual_net",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

minigo_cc_library(
    name = "position",
    srcs = [
        "position.cc",
    ],
    hdrs = [
        "position.h",
    ],
    deps = [
        ":base",
        ":check",
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
        ":check",
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

cc_library(
    name = "thread_safe_queue",
    hdrs = ["thread_safe_queue.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

minigo_cc_library(
    name = "tf_utils",
    srcs = select({
        "//cc/config:enable_tf": ["tf_utils.cc"],
        "//conditions:default": ["tf_utils_dummy.cc"],
    }),
    hdrs = ["tf_utils.h"],
    deps = [
        ":base",
        ":check",
        ":mcts",
        "//cc/file",
        "//cc/dual_net",
    ] + select({
        "//cc/config:enable_tf": ["//cc:tensorflow"],
        "//conditions:default": [],
    }),
)

minigo_cc_library(
    name = "tiny_set",
    hdrs = ["tiny_set.h"],
    deps = [
        ":inline_vector",
    ],
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
        ":mcts",
        ":test_utils",
        "//cc/dual_net:fake_net",
        "@com_google_absl//absl/memory",
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
        ":random",
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
        ":base",
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

minigo_cc_test(
    name = "thread_safe_queue_test",
    size = "small",
    srcs = ["thread_safe_queue_test.cc"],
    deps = [
        ":thread_safe_queue",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":base",
        ":check",
        ":gtp_player",
        ":init",
        ":mcts",
        ":random",
        ":sgf",
        ":tf_utils",
        "//cc/dual_net:factory",
        "//cc/file",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
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
