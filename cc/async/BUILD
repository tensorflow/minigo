package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache License 2.0

cc_library(
    name = "poll_thread",
    hdrs = ["poll_thread.h"],
    srcs = ["poll_thread.cc"],
    deps = [
        "//cc:thread",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "semaphore",
    hdrs = ["semaphore.h"],
    deps = ["@com_google_absl//absl/synchronization"],
)

cc_library(
    name = "sharded_executor",
    srcs = ["sharded_executor.cc"],
    hdrs = ["sharded_executor.h"],
    deps = [
        ":semaphore",
        "//cc:thread",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@wtf",
    ],
)
