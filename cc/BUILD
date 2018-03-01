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
    "position.cc",
]

minigo_hdrs = [
    "color.h",
    "constants.h",
    "coord.h",
    "dual_net.h",
    "group.h",
    "inline_vector.h",
    "position.h",
    "stone.h",
    "tiny_set.h",
]

minigo_deps = [
    ":tensorflow",
    "@com_google_absl//absl/base",
    "@com_google_absl//absl/strings",
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
    name = "coord_9_test",
    size = "small",
    srcs = ["coord_test.cc"],
    deps = [
        ":minigo_9",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "position_test",
    size = "small",
    srcs = ["position_test.cc"],
    deps = [
        ":minigo_9",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "tf_test",
    srcs = ["tf_test.cc"],
    deps = [
        ":minigo_19",
        ":tensorflow",
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

cc_binary(
    name = "play_game",
    srcs = ["play_game.cc"],
    deps = [
        ":minigo_19",
    ],
)
