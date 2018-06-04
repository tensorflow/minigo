package(default_visibility = [
    ":__pkg__",
    "//cc:__pkg__",
    "//cc/dual_net:__pkg__",
])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

cc_proto_library(
    name = "inference_service_proto",
    protos = ["inference_service.proto"],
    with_grpc = True,
)

py_proto_library(
    name = "inference_service_py_pb2",
    protos = ["inference_service.proto"],
    with_grpc = True,
)
