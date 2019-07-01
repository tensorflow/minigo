load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# This must be kept up to date with the io_bazel_rules_closure archive from
# tensorflow/WORKSPACE.
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

http_archive(
    name = "com_github_nlohmann_json",
    build_file = "//cc:json.BUILD",
    sha256 = "2ef2fe6f1a615ad97beb39f91ef5e319d776f6ba0af91570003276e6ffb1c47c",
    strip_prefix = "json-3.2.0",
    urls = ["https://github.com/nlohmann/json/archive/v3.2.0.zip"],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace()

load("//cc:cuda_configure.bzl", "cuda_configure")
load("//cc:tensorrt_configure.bzl", "tensorrt_configure")

cuda_configure(name = "local_config_cuda")

tensorrt_configure(name = "local_config_tensorrt")
