load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

### http_archive(
###     name = "com_google_protobuf",
###     strip_prefix = "protobuf-3.6.1",
###     url = "https://github.com/google/protobuf/archive/v3.6.1.tar.gz",
### )

### http_archive(
###     name = "com_github_gflags_gflags",
###     strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
###     urls = ["https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip"],
### )

### http_archive(
###     name = "com_google_absl",
###     strip_prefix = "abseil-cpp-389ec3f906f018661a5308458d623d01f96d7b23",
###     urls = [
###         "https://mirror.bazel.build/github.com/abseil/abseil-cpp/archive/389ec3f906f018661a5308458d623d01f96d7b23.tar.gz",
###         "https://github.com/abseil/abseil-cpp/archive/389ec3f906f018661a5308458d623d01f96d7b23.tar.gz",
###     ],
### )

### http_archive(
###     name = "com_github_googlecloudplatform_google_cloud_cpp",
###     strip_prefix = "google-cloud-cpp-0.4.0",
###     url = "https://github.com/GoogleCloudPlatform/google-cloud-cpp/archive/v0.4.0.zip",
### )

### new_http_archive(
###     name = "com_google_benchmark",
###     build_file = "cc/benchmark.BUILD",
###     strip_prefix = "benchmark-1.3.0",
###     urls = ["https://github.com/google/benchmark/archive/v1.3.0.zip"],
### )

### new_http_archive(
###     name = "com_github_nlohmann_json",
###     build_file = "cc/json.BUILD",
###     strip_prefix = "json-3.2.0",
###     urls = ["https://github.com/nlohmann/json/archive/v3.2.0.zip"],
### )

### http_archive(
###     name = "com_google_benchmark",
###     build_file = "@//cc:benchmark.BUILD",
###     strip_prefix = "benchmark-1.3.0",
###     urls = ["https://github.com/google/benchmark/archive/v1.3.0.zip"],
### )

# Must match io_bazel_rules_closure from tensorflow/WORKSPACE of the correct
# TF version.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "a38539c5b5c358548e75b44141b4ab637bba7c4dc02b46b1f62a96d6433f56ae",
    strip_prefix = "rules_closure-dbb96841cc0a5fb2664c37822803b06dab20c7d1",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",  # 2018-04-13
    ],
)

http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-1.13.1",
    urls = ["https://github.com/tensorflow/tensorflow/archive/v1.13.1.zip"],
)

### http_archive(
###     name = "com_google_googletest",
###     strip_prefix = "googletest-master",
###     urls = ["https://github.com/google/googletest/archive/master.zip"],
### )

### load("@com_github_googlecloudplatform_google_cloud_cpp//bazel:google_cloud_cpp_deps.bzl", "google_cloud_cpp_deps")

### google_cloud_cpp_deps()

# Have to manually call the corresponding function for gRPC:
#   https://github.com/bazelbuild/bazel/issues/1550
### load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
###
### grpc_deps()

### load("//cc:cuda_configure.bzl", "cuda_configure")
### load("//cc:tensorrt_configure.bzl", "tensorrt_configure")
### cuda_configure(name = "local_config_cuda")
### tensorrt_configure(name = "local_config_tensorrt")

load("//cc:tensorflow_configure.bzl", "tensorflow_configure")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace()

tensorflow_configure(name = "local_config_tensorflow")
