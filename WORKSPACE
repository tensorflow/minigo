http_archive(
    name = "com_google_protobuf",
    strip_prefix = "protobuf-3.6.1",
    url = "https://github.com/google/protobuf/archive/v3.6.1.tar.gz",
)

http_archive(
    name = "com_github_gflags_gflags",
    strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    urls = ["https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip"],
)

http_archive(
    name = "com_google_absl",
    strip_prefix = "abseil-cpp-666fc1266bccfd8e6eaaa084e7b42580bb8eb199",
    urls = [
        "http://mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/666fc1266bccfd8e6eaaa084e7b42580bb8eb199.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/666fc1266bccfd8e6eaaa084e7b42580bb8eb199.tar.gz",
    ],
)

http_archive(
    name = "com_github_googlecloudplatform_google_cloud_cpp",
    strip_prefix = "google-cloud-cpp-0.4.0",
    url = "https://github.com/GoogleCloudPlatform/google-cloud-cpp/archive/v0.4.0.zip",
)

new_http_archive(
    name = "com_google_benchmark",
    build_file = "cc/benchmark.BUILD",
    strip_prefix = "benchmark-1.3.0",
    urls = ["https://github.com/google/benchmark/archive/v1.3.0.zip"],
)

new_http_archive(
    name = "com_github_nlohmann_json",
    build_file = "cc/json.BUILD",
    strip_prefix = "json-3.2.0",
    urls = ["https://github.com/nlohmann/json/archive/v3.2.0.zip"],
)

http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-master",
    urls = ["https://github.com/google/googletest/archive/master.zip"],
)

load("@com_github_googlecloudplatform_google_cloud_cpp//bazel:google_cloud_cpp_deps.bzl", "google_cloud_cpp_deps")

google_cloud_cpp_deps()

# Have to manually call the corresponding function for gRPC:
#   https://github.com/bazelbuild/bazel/issues/1550
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("//cc:cuda_configure.bzl", "cuda_configure")
load("//cc:tensorrt_configure.bzl", "tensorrt_configure")

cuda_configure(name = "local_config_cuda")

tensorrt_configure(name = "local_config_tensorrt")
