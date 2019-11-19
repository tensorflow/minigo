load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

# These must be kept up to date with the rules from tensorflow/WORKSPACE.
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz"],
)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# These must be kept up to date with the rules from tensorflow/WORKSPACE.

# This should also be kept up to date with the version used by Tensorflow.
http_file(
    name = "com_github_nlohmann_json_single_header",
    sha256 = "63da6d1f22b2a7bb9e4ff7d6b255cf691a161ff49532dcc45d398a53e295835f",
    urls = [
        "https://github.com/nlohmann/json/releases/download/v3.4.0/json.hpp",
    ],
)

http_archive(
    name = "org_tensorflow",
    sha256 = "76abfd5045d1474500754566edd54ce4c386a1fbccf22a3a91d6832c6b7e90ad",
    strip_prefix = "tensorflow-1.15.0",
    urls = ["https://github.com/tensorflow/tensorflow/archive/v1.15.0.zip"],
)

http_archive(
    name = "wtf",
    build_file = "//cc:wtf.BUILD",
    sha256 = "1837833cd159060f8bd6f6dd87edf854ed3135d07a6937b7e14b0efe70580d74",
    strip_prefix = "tracing-framework-fb639271fa3d56ed1372a792d74d257d4e0c235c",
    urls = ["https://github.com/google/tracing-framework/archive/fb639271fa3d56ed1372a792d74d257d4e0c235c.zip"],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace()
