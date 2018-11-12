"""Build rule generator for TensorRT parsers."""

# TensorFlow already exposes TensorRT, but not the parsers that are part of it.
# This rule therefore exposes just the parsers.

def _get_env_var(repository_ctx, name, default):
    if name in repository_ctx.os.environ:
        return repository_ctx.os.environ[name]
    return default

def _impl(repository_ctx):
    tensorrt_lib_path = _get_env_var(repository_ctx, "TENSORRT_INSTALL_PATH", "/usr/lib/x86_64-linux-gnu")
    tensorrt_include_path = _get_env_var(repository_ctx, "TENSORRT_INCLUDE_PATH", tensorrt_lib_path.replace("lib", "include"))

    repository_ctx.symlink(tensorrt_include_path, "tensorrt/include")
    repository_ctx.symlink(tensorrt_lib_path, "tensorrt/lib")

    repository_ctx.file("BUILD", """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nvparsers",
    srcs = glob([
          "tensorrt/include/Nv*.h",
          "tensorrt/lib/libnvparsers.so*",
        ]),
    hdrs = [
          "tensorrt/include/NvOnnxParser.h",
          "tensorrt/include/NvUffParser.h",
        ],
    # include_prefix = "tensorrt",
    # strip_include_prefix = "tensorrt/include",
)
""")

tensorrt_parsers_configure = repository_rule(
    implementation = _impl,
    environ = ["TENSORRT_INSTALL_PATH", "TENSORRT_INCLUDE_PATH"],
)
