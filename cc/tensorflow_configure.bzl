"""Build rule generator to provide prebuilt TensorFlow binaries."""

_MG_PREBUILT_TF_PATH = "MG_PREBUILT_TF_PATH"

def _get_child(path, child):
    for ch in child.split("/"):
        path = path.get_child(ch)
        if not path.exists:
            fail("'%s' not found" % path)
    return path


def _impl(repository_ctx):
    root = repository_ctx.path(Label("@//:BUILD")).dirname

    configured = root.get_child("tensorflow.bazelrc").exists and \
                 root.get_child("tf_configure.bazelrc").exists

    prebuilt = _MG_PREBUILT_TF_PATH in repository_ctx.os.environ

    if prebuilt:
        path = repository_ctx.os.environ[_MG_PREBUILT_TF_PATH]
        if path.startswith("/"):
            path = repository_ctx.path(path)
        else:
            root = repository_ctx.path(Label("@//:BUILD")).dirname
            path = _get_child(root, path)

        repository_ctx.symlink(_get_child(path, "lib"), "lib")
        repository_ctx.symlink(_get_child(path, "include"), "include")

        _get_child(path, "lib/libgrpc_runtime.so")
        _get_child(path, "lib/libtensorflow_lite.so")
        _get_child(path, "lib/libtensorflow_cc.so")
        _get_child(path, "lib/libtensorflow_framework.so")

    repository_ctx.file("build_defs.bzl", """
def _check_impl(ctx):
    if not ctx.attr.condition:
        fail(ctx.attr.message)

check = rule(
    implementation = _check_impl,
    attrs = {
        "condition": attr.bool(),
        "message": attr.string(),
    },
)

def if_tf_configured(if_true, if_false):
    return if_true if %s else if_false

def if_tf_prebuilt(if_true, if_false):
    return if_true if %s else if_false
""" % (configured, prebuilt))

    repository_ctx.file("BUILD", """
exports_files(glob(["lib/*.so"]))

cc_library(
    name = "eigen_hdrs",
    hdrs = glob(["include/third_party/eigen3/**"]),
    strip_include_prefix = "include/third_party/eigen3",
)

cc_library(
    name = "headers",
    hdrs = glob(["include/**"]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = [
        ":eigen_hdrs",
        "@protobuf_archive//:protobuf_headers",
    ],
)
""")

tensorflow_configure = repository_rule(
    implementation = _impl,
    local = True,
    environ = [_MG_PREBUILT_TF_PATH],
)
