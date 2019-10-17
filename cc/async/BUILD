package(default_visibility = [
    ":__pkg__",
    "//cc:__pkg__",
])

licenses(["notice"])  # Apache License 2.0

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
        "@com_google_absl//absl/synchronization",
        "@wtf",
    ],
)
