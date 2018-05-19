http_archive(
    name = "com_github_gflags_gflags",
    strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    urls = ["https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip"],
)

http_archive(
    name = "com_google_absl",
    strip_prefix = "abseil-cpp-a7e522daf1ec9cda69b356472f662142dd0c1215",
    urls = ["https://github.com/abseil/abseil-cpp/archive/a7e522daf1ec9cda69b356472f662142dd0c1215.zip"],
)

new_http_archive(
    name = "com_google_benchmark",
    build_file = "cc/benchmark.BUILD",
    strip_prefix = "benchmark-1.3.0",
    urls = ["https://github.com/google/benchmark/archive/v1.3.0.zip"],
)

http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-master",
    urls = ["https://github.com/google/googletest/archive/master.zip"],
)

http_archive(
    name = "com_googlesource_code_cctz",
    strip_prefix = "cctz-2.2",
    urls = ["https://github.com/google/cctz/archive/v2.2.zip"],
)

http_archive(
    name = "org_pubref_rules_protobuf",
    strip_prefix = "rules_protobuf-0.8.2",
    urls = ["https://github.com/pubref/rules_protobuf/archive/v0.8.2.zip"],
)

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cpp_proto_repositories")

cpp_proto_repositories()
