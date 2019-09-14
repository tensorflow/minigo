load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

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

# This should also be kept up to date with the version used by Tensorflow.
http_file(
    name = "com_github_nlohmann_json_single_header",
    urls = [
        "https://github.com/nlohmann/json/releases/download/v3.4.0/json.hpp",
    ],
)

http_archive(
    name = "org_tensorflow",
    sha256 = "902a6d90bb69549fe241377210aa459773459820da1333b67dcfdef37836f25f",
    strip_prefix = "tensorflow-1.13.1",
    urls = ["https://github.com/tensorflow/tensorflow/archive/v1.13.1.zip"],
)

http_archive(
    name = "wtf",
    build_file = "//cc:wtf.BUILD",
    sha256 = "e9434641b5923df85d1fe0082030ce2ac8aad9d95676682aa072ad88421a2bc1",
    strip_prefix = "tracing-framework-495ced98de99a5895e484b2e09771edb42d3c7ab",
    urls = ["https://github.com/google/tracing-framework/archive/495ced98de99a5895e484b2e09771edb42d3c7ab.zip"],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace()
