load("@org_tensorflow//tensorflow:tensorflow.bzl", "transitive_hdrs")

cc_library(
    name = "tensorflow",
    srcs = [
        "lib/libtensorflow_framework.so",
    ] + select({
        "//cc/config:enable_tpu": ["lib/libgrpc_runtime.so"],
        "//conditions:default": ["lib/libtensorflow_cc.so"],
    }),
    hdrs = glob(["include/tensorflow/**/*"]),
    includes = [
        "include",
        "include/tensorflow/external/nsync/public",
        "include/third_party/eigen3/",
    ],
    linkopts = ["-lm"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tf_lite",
    srcs = ["lib/libtensorflow_lite.so"],
    hdrs = glob(["include/tensorflow/lite/**"]),
    includes = ["include"],
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
    deps = ["@flatbuffers//:runtime_cc"],
)

# The following targets are used by cc/configure_tensorflow.sh to build the
# files required by the cc_library targets above.

# Compiling TensorFlow is handled by the :build rule's dependencies.
# The copy_outputs.sh script just copies the generated libraries and headers
# into the correct location under //cc/tensorflow/.
sh_binary(
    name = "build",
    srcs = ["copy_outputs.sh"],
    data = [
        ":libgrpc_runtime.so",
        ":libtensorflow_lite.so",
        ":tensorflow_hdrs",
        "@org_tensorflow//tensorflow:libtensorflow_cc.so",
        "@org_tensorflow//tensorflow:libtensorflow_framework.so",
        "@org_tensorflow//tensorflow/lite/toco",
        "@org_tensorflow//third_party/eigen3:eigen_third_party_header_files",
    ],
    visibility = ["//visibility:public"],
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
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

transitive_hdrs(
    name = "tensorflow_hdrs",
    deps = [
        ":libtensorflow_lite.so",
        "@eigen_archive//:eigen",
        "@org_tensorflow//tensorflow:libtensorflow.so",
        "@protobuf_archive//:protobuf_headers",
    ],
)
