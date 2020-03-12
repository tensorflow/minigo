load("@org_tensorflow//tensorflow:tensorflow.bzl", "transitive_hdrs")

cc_library(
    name = "tensorflow",
    srcs = select({
        "@org_tensorflow//tensorflow:macos": ["lib/libtensorflow_framework.1.dylib"],
        "//conditions:default": ["lib/libtensorflow_framework.so.1"],
    }) + select({
        "//cc/config:enable_tpu": ["lib/libgrpc_runtime.so"],
        "@org_tensorflow//tensorflow:macos": ["lib/libtensorflow_cc.1.dylib"],
        "//conditions:default": ["lib/libtensorflow_cc.so.1"],
    }) + glob(["lib/libtensorrt_ops.so"]),  # glob because this is optional
    hdrs = glob(["include/tensorflow/**/*"]),
    includes = [
        "include",
        "include/tensorflow/external/nsync/public",
        "include/third_party/eigen3/",
    ],
    linkopts = ["-lm"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/types:optional",
    ],
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
        "@org_tensorflow//tensorflow:tensorflow_cc",
        "@org_tensorflow//tensorflow:tensorflow_framework",
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
        "@com_google_protobuf//:protobuf_headers",
        "@eigen_archive//:eigen",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/c:c_api_experimental",
        "@org_tensorflow//tensorflow/c/eager:c_api",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_session",
    ],
)
