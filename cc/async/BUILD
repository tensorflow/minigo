package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache License 2.0

cc_library(
    name = "poll_thread",
    hdrs = ["poll_thread.h"],
    srcs = ["poll_thread.cc"],
    deps = [
        ":thread",
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
        ":thread",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@wtf",
    ],
)

cc_library(
    name = "thread",
    srcs = ["thread.cc"],
    hdrs = ["thread.h"],
    deps = [
        "//cc:logging",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "thread_safe_queue",
    hdrs = ["thread_safe_queue.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "thread_safe_queue_test",
    size = "small",
    srcs = ["thread_safe_queue_test.cc"],
    deps = [
        ":thread_safe_queue",
        "//cc:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)
