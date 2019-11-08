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

minigo_cc_library(
    name = "base",
    srcs = [
        "algorithm.cc",
        "color.cc",
        "coord.cc",
        "group.cc",
        "move.cc",
    ],
    hdrs = [
        "algorithm.h",
        "color.h",
        "constants.h",
        "coord.h",
        "group.h",
        "move.h",
        "padded_array.h",
        "stone.h",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:windows": ["ws2_32.lib"],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":inline_vector",
        ":logging",
        "//cc/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

minigo_cc_library(
    name = "game",
    srcs = ["game.cc"],
    hdrs = ["game.h"],
    deps = [
        ":base",
        ":position",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

minigo_cc_library(
    name = "game_utils",
    srcs = ["game_utils.cc"],
    hdrs = ["game_utils.h"],
    deps = [
        ":base",
        ":game",
        ":logging",
        ":sgf",
        "//cc/file",
        "//cc/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

minigo_cc_library(
    name = "gtp_client",
    srcs = ["gtp_client.cc"],
    hdrs = ["gtp_client.h"],
    deps = [
        ":base",
        ":logging",
        ":mcts",
        ":sgf",
        ":thread_safe_queue",
        "//cc/file",
        "//cc/model",
        "@com_google_absl//absl/container:flat_hash_map",
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
        "@com_google_absl//absl/debugging:failure_signal_handler",
        "@com_google_absl//absl/debugging:symbolize",
    ],
)

minigo_cc_library(
    name = "inline_vector",
    hdrs = ["inline_vector.h"],
    deps = [
        ":logging",
        "//cc/platform",
    ],
)

minigo_cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/debugging:stacktrace",
        "@com_google_absl//absl/debugging:symbolize",
        "@com_google_absl//absl/synchronization",
    ],
)

minigo_cc_library(
    name = "minigui_gtp_client",
    srcs = ["minigui_gtp_client.cc"],
    hdrs = ["minigui_gtp_client.h"],
    deps = [
        ":base",
        ":gtp_client",
        ":json",
        ":logging",
        ":mcts",
        ":sgf",
        ":thread",
        ":thread_safe_queue",
        "//cc/file",
        "//cc/model:batching_model",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "json",
    hdrs = [
        "json.h",
        "@com_github_nlohmann_json_single_header//file",
    ],
)

minigo_cc_library(
    name = "mcts",
    srcs = [
        "mcts_player.cc",
        "mcts_tree.cc",
    ],
    hdrs = [
        "mcts_player.h",
        "mcts_tree.h",
    ],
    deps = [
        ":base",
        ":game",
        ":inline_vector",
        ":logging",
        ":padded_array",
        ":position",
        ":random",
        ":symmetries",
        ":zobrist",
        "//cc/model",
        "//cc/model:inference_cache",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
    ],
)

minigo_cc_library(
    name = "padded_array",
    hdrs = ["padded_array.h"],
)

minigo_cc_library(
    name = "position",
    srcs = ["position.cc"],
    hdrs = ["position.h"],
    deps = [
        ":base",
        ":inline_vector",
        ":logging",
        ":tiny_set",
        ":zobrist",
        "@com_google_absl//absl/strings:str_format",
    ],
)

minigo_cc_library(
    name = "random",
    srcs = ["random.cc"],
    hdrs = ["random.h"],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

minigo_cc_library(
    name = "sgf",
    srcs = ["sgf.cc"],
    hdrs = ["sgf.h"],
    deps = [
        ":base",
        ":logging",
        "//cc/platform",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

minigo_cc_library(
    name = "symmetries",
    srcs = ["symmetries.cc"],
    hdrs = ["symmetries.h"],
    deps = [
        ":base",
        ":logging",
    ],
)

cc_library(
    name = "thread",
    srcs = ["thread.cc"],
    hdrs = ["thread.h"],
    deps = [
        ":logging",
        "@com_google_absl//absl/synchronization",
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
           }) +
           select({
               "//cc/config:enable_bt": ["tf_bt_utils.cc"],
               "//conditions:default": ["tf_bt_utils_dummy.cc"],
           }),
    hdrs = ["tf_utils.h"],
    deps = [
               ":base",
               ":logging",
               ":game",
               "//cc/file",
               "//cc/model",
               "@com_google_absl//absl/strings:str_format",
           ] + select({
               "//cc/config:enable_bt": ["@com_github_googlecloudplatform_google_cloud_cpp//google/cloud/bigtable:bigtable_client"],
               "//conditions:default": [],
           }) +
           select({
               "//cc/config:enable_tf": ["//cc/tensorflow"],
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
    name = "zobrist",
    srcs = ["zobrist.cc"],
    hdrs = ["zobrist.h"],
    deps = [
        ":base",
        ":random",
    ],
)

minigo_cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        ":base",
        ":logging",
        ":position",
        ":random",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "algorithm_test",
    size = "small",
    srcs = ["algorithm_test.cc"],
    deps = [
        ":base",
        ":random",
        "@com_google_googletest//:gtest_main",
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
    name = "mcts_tree_test",
    size = "small",
    srcs = ["mcts_tree_test.cc"],
    deps = [
        ":mcts",
        ":position",
        ":random",
        ":test_utils",
        ":zobrist",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

minigo_cc_test_9_only(
    name = "mcts_player_test",
    size = "small",
    srcs = ["mcts_player_test.cc"],
    deps = [
        ":base",
        ":mcts",
        ":position",
        ":test_utils",
        ":zobrist",
        "//cc/dual_net:fake_dual_net",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
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
    name = "pass_alive_test",
    size = "small",
    srcs = ["pass_alive_test.cc"],
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
    linkopts = ["-lm"],
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
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_test(
    name = "thread_safe_queue_test",
    size = "small",
    srcs = ["thread_safe_queue_test.cc"],
    deps = [
        ":logging",
        ":thread_safe_queue",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

minigo_cc_binary(
    name = "concurrent_selfplay",
    srcs = ["concurrent_selfplay.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":game",
        ":game_utils",
        ":init",
        ":logging",
        ":mcts",
        ":random",
        ":tf_utils",
        ":thread",
        ":thread_safe_queue",
        ":zobrist",
        "//cc/async:poll_thread",
        "//cc/async:sharded_executor",
        "//cc/dual_net:factory",
        "//cc/file",
        "//cc/file:directory_watcher",
        "//cc/model:inference_cache",
        "//cc/model:reloading_model",
        "//cc/platform",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@wtf",
    ],
)

minigo_cc_binary(
    name = "eval",
    srcs = ["eval.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":game",
        ":game_utils",
        ":init",
        ":logging",
        ":mcts",
        ":random",
        ":tf_utils",
        ":zobrist",
        "//cc/dual_net:factory",
        "//cc/file",
        "//cc/model",
        "//cc/model:batching_model",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

minigo_cc_binary(
    name = "gtp",
    srcs = ["gtp.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":gtp_client",
        ":init",
        ":minigui_gtp_client",
        ":zobrist",
        "//cc/dual_net:factory",
        "//cc/file",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

minigo_cc_binary(
    name = "replay_games",
    srcs = ["replay_games.cc"],
    deps = [
        ":base",
        ":init",
        ":logging",
        ":position",
        ":sgf",
        ":thread",
        ":thread_safe_queue",
        "//cc/file",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/memory",
    ],
)

minigo_cc_binary(
    name = "selfplay",
    srcs = ["selfplay.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":game",
        ":game_utils",
        ":init",
        ":logging",
        ":mcts",
        ":random",
        ":tf_utils",
        ":zobrist",
        "//cc/dual_net:factory",
        "//cc/file",
        "//cc/model:batching_model",
        "//cc/model:inference_cache",
        "//cc/model:reloading_model",
        "//cc/platform",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@wtf",
    ],
)

cc_binary(
    name = "sample_records",
    srcs = ["sample_records.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":init",
        ":logging",
        ":random",
        ":thread",
        "//cc/tensorflow",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

minigo_cc_binary(
    name = "simple_example",
    srcs = ["simple_example.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":init",
        ":logging",
        ":mcts",
        ":random",
        ":zobrist",
        "//cc:game",
        "//cc/dual_net:factory",
        "//cc/platform",
        "@com_github_gflags_gflags//:gflags",
    ],
)

minigo_cc_binary(
    name = "tfrzz_to_cbt",
    srcs = ["tfrzz_to_cbt.cc"],
    tags = ["manual"],
    deps = [
        ":base",
        ":init",
        ":logging",
        ":tf_utils",
        "//cc/file",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

minigo_cc_binary(
    name = "tpu_test",
    srcs = ["tpu_test.cc"],
    tags = ["manual"],
    deps = [
        "//cc/tensorflow",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)
