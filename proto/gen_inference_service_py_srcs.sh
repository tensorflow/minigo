# This script generates the proto and gRPC Python runtime for
# inference_service.proto.

# Switch to the repo root directory.
pushd $(dirname $0)/..

bazel build proto:inference_service_py_pb2

cp -f bazel-genfiles/proto/inference_service_pb2.py \
      bazel-genfiles/proto/inference_service_pb2_grpc.py \
      bazel-genfiles/proto/inference_service_py_pb2.pb.descriptor_set \
      proto

popd
