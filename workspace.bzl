load("//cc:tensorflow_configure.bzl", "tensorflow_configure")
load("//cc:tensorrt_parsers_configure.bzl", "tensorrt_parsers_configure")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

def minigo_workspace():
    http_archive(
        name = "com_google_benchmark",
        build_file = "@//cc:benchmark.BUILD",
        strip_prefix = "benchmark-1.3.0",
        urls = ["https://github.com/google/benchmark/archive/v1.3.0.zip"],
    )

    tf_workspace()

    tensorflow_configure(name = "local_config_tensorflow")

    # TensorFlow does not expose TensorRT parsers.
    tensorrt_parsers_configure(name = "local_config_tensorrt_parsers")
