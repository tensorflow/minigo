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
    name = "tensorflow_dynamic",
    srcs = [
        "tensorflow/libtensorflow_framework.so",
    ] + select({
        "//cc/config:enable_tpu": ["tensorflow/libgrpc_runtime.so"],
        "//conditions:default": ["tensorflow/libtensorflow_cc.so"],
    }),
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
    tags = ["manual"],
)

# TODO(tommadams): Set this up for windows compilation.
cc_library(
    name = "tensorflow_static",
    srcs = [],
    hdrs = [],
    tags = ["manual"],
)

cc_library(
    name = "tensorflow",
    tags = ["manual"],
    deps = select({
        "@bazel_tools//src/conditions:windows": ["tensorflow_static"],
        "//conditions:default": ["tensorflow_dynamic"],
    }),
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
    tags = ["manual"],
)

cc_library(
    name = "cuda",
    tags = ["manual"],
    deps = select({
        "//cc/config:enable_gpu": [
            "@local_config_cuda//:cuda_headers",
            "@local_config_cuda//:cudart",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "tensorrt",
    tags = ["manual"],
    deps = [
        "@local_config_tensorrt//:nv_infer",
        "@local_config_tensorrt//:nvparsers",
        "@local_config_tensorrt//:tensorrt_headers",
    ],
)

minigo_cc_library(
    name = "base",
    srcs = [
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
        "//cc:thread_safe_queue",
        "//cc/dual_net",
        "@com_github_nlohmann_json//:json",
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
        ":game",
        ":logging",
        ":position",
        ":random",
        ":symmetries",
        ":zobrist",
        "//cc/dual_net",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
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
        ":logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

minigo_cc_library(
    name = "symmetries",
    hdrs = ["symmetries.h"],
    deps = [
        ":logging",
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
               "//cc/dual_net",
               "@com_google_absl//absl/strings:str_format",
           ] + select({
               "//cc/config:enable_bt": ["@com_github_googlecloudplatform_google_cloud_cpp//google/cloud/bigtable:bigtable_client"],
               "//conditions:default": [],
           }) +
           select({
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
    name = "gtp",
    srcs = ["gtp.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":gtp_player",
        ":init",
        ":zobrist",
        "//cc/dual_net:factory",
        "//cc/file",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/strings",
    ],
)

minigo_cc_binary(
    name = "main",
    srcs = ["main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":init",
        ":logging",
        ":mcts",
        ":random",
        ":sgf",
        ":tf_utils",
        ":zobrist",
        "//cc/dual_net:batching_dual_net",
        "//cc/dual_net:factory",
        "//cc/dual_net:reloading_dual_net",
        "//cc/file",
        "//cc/platform",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

minigo_cc_binary(
    name = "puzzle",
    srcs = ["puzzle.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":init",
        ":logging",
        ":mcts",
        ":sgf",
        ":zobrist",
        "//cc/dual_net:batching_dual_net",
        "//cc/dual_net:factory",
        "//cc/file",
        "//cc/platform",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
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
        ":tensorflow",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
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
