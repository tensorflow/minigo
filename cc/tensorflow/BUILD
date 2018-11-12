load("@org_tensorflow//tensorflow:tensorflow.bzl", "transitive_hdrs")
load("@local_config_tensorflow//:build_defs.bzl", "check", "if_tf_configured", "if_tf_prebuilt")

cc_library(
    name = "tensorflow",
    srcs = if_tf_prebuilt(
        [
            "@local_config_tensorflow//:lib/libtensorflow_framework.so",
            ":local_runtime",
        ],
        ["@org_tensorflow//tensorflow:libtensorflow_framework.so"],
    ),
    data = [":tf_configured"],
    linkopts = ["-lm"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = if_tf_prebuilt(
        ["@local_config_tensorflow//:headers"],
        [":bazel_runtime"],
    ),
)

cc_library(
    name = "tf_lite",
    srcs = if_tf_prebuilt(
        ["@local_config_tensorflow//:lib/libtensorflow_lite.so"],
        [],
    ),
    data = [":tf_configured"],
    linkopts = ["-ldl"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = [
        "@flatbuffers//:runtime_cc",
    ] + if_tf_prebuilt(
        ["@local_config_tensorflow//:headers"],
        [
            "@org_tensorflow//tensorflow/contrib/lite:framework",
            "@org_tensorflow//tensorflow/contrib/lite/kernels:builtin_ops",
        ],
    ),
)

# Run this target to build TensorFlow and save shared objects and headers.
sh_binary(
    name = "build",
    srcs = ["build.sh"],
    data = [
        ":libgrpc_runtime.so",
        ":libtensorflow_lite.so",
        ":tensorflow_hdrs",
        ":tf_configured",
        "@org_tensorflow//tensorflow:libtensorflow_cc.so",
        "@org_tensorflow//tensorflow:libtensorflow_framework.so",
        "@org_tensorflow//third_party/eigen3:eigen_third_party_header_files",
    ],
    visibility = ["//visibility:public"],
)

alias(
    name = "local_runtime",
    actual = select({
        "//cc/config:enable_tpu": "@local_config_tensorflow//:lib/libgrpc_runtime.so",
        "//conditions:default": "@local_config_tensorflow//:lib/libtensorflow_cc.so",
    }),
)

alias(
    name = "bazel_runtime",
    actual = select({
        "//cc/config:enable_tpu": "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_runtime",
        "//conditions:default": "@org_tensorflow//tensorflow/core:tensorflow",
    }),
)

check(
    name = "tf_configured",
    condition = if_tf_configured(True, False),
    message = "TensorFlow has not been configured yet, run cc/configure_tensorflow.sh.",
)

cc_binary(
    name = "libgrpc_runtime.so",
    linkshared = True,
    linkstatic = True,
    deps = [
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_runtime",
    ],
)

cc_binary(
    name = "libtensorflow_lite.so",
    linkshared = True,
    linkstatic = True,
    deps = [
        "@org_tensorflow//tensorflow/contrib/lite:framework",
        "@org_tensorflow//tensorflow/contrib/lite/kernels:builtin_ops",
    ],
)

transitive_hdrs(
    name = "tensorflow_hdrs",
    deps = [
        ":libtensorflow_lite.so",
        "@eigen_archive//:eigen",
        "@org_tensorflow//tensorflow:libtensorflow.so",
    ],
)
