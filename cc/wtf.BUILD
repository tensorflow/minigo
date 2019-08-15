cc_library(
    name = "wtf",
    srcs = [
        "bindings/cpp/buffer.cc",
        "bindings/cpp/event.cc",
        "bindings/cpp/platform.cc",
        "bindings/cpp/runtime.cc",
    ] + glob(["bindings/cpp/include/wtf/platform/*.h"]),
    hdrs = [
	"bindings/cpp/include/wtf/argtypes.h",
	"bindings/cpp/include/wtf/config.h",
	"bindings/cpp/include/wtf/event.h",
	"bindings/cpp/include/wtf/macros.h",
	"bindings/cpp/include/wtf/platform.h",
	"bindings/cpp/include/wtf/runtime.h",
        "bindings/cpp/include/wtf/buffer.h",
    ],
    copts = [
        "-O3",
    ],
    includes = [
        "bindings/cpp/include/",
    ],
    visibility = ["//visibility:public"],
)
