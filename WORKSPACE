workspace(name = "minigo")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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
    patches = ["//cc/tensorflow:tensorflow.patch"],
    sha256 = "dfee0f57366a6fab16a103d3a6d190c327f01f9a12651e45a128051eaf612f20",
    strip_prefix = "tensorflow-1.11.0",
    urls = ["https://github.com/tensorflow/tensorflow/archive/v1.11.0.zip"],
)

load("//:workspace.bzl", "minigo_workspace")

minigo_workspace()
