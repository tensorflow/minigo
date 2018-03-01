cc_library(
    name = "benchmark",
    srcs = glob(
        ["src/*.cc"],
        exclude = [
            "src/re_posix.cc",
            "src/gnuregex.cc",
        ],
    ),
    hdrs = glob(
        [
            "src/*.h",
            "include/benchmark/*.h",
        ],
        exclude = [
            "src/re_posix.h",
            "src/gnuregex.h",
        ],
    ),
    copts = [
        "-DHAVE_STD_REGEX",
    ],
    includes = [
        "include",
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)
