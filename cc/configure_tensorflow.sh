#!/bin/bash

set -e

_DEFAULT_CUDA_VERSION=10
_DEFAULT_CUDNN_VERSION=7

# Run the TensorFlow configuration script, setting reasonable values for most
# of the options.
echo "Configuring tensorflow"

# //cc/tensorflow:build needs to know whether the user wants TensorRT support,
# so it can build extra libraries.
if [ -z "${TF_NEED_TENSORRT}" ]; then
  read -p "Enable TensorRT support? [y/N]: " yn
  case $yn in
      [Yy]* ) export TF_NEED_TENSORRT=1;;
      * ) export TF_NEED_TENSORRT=0;;
  esac
fi

# //org_tensorflow//:configure script must be run from the TensorFlow repository
# root. Build the script in order to pull the repository contents from GitHub.
# The `bazel fetch` and `bazel sync` commands that are usually used to fetch
# external Bazel dependencies don't work correctly on the TensorFlow repository.
bazel --bazelrc=/dev/null build @org_tensorflow//:configure

pushd bazel-minigo/external/org_tensorflow

CC_OPT_FLAGS="${CC_OPT_FLAGS:--march=native}" \
CUDA_TOOLKIT_PATH=${CUDA_TOOLKIT_PATH:-/usr/local/cuda} \
CUDNN_INSTALL_PATH=${CUDNN_INSTALL_PATH:-/usr/local/cuda} \
TF_NEED_JEMALLOC=${TF_NEED_JEMALLOC:-1} \
TF_NEED_GCP=${TF_NEED_GCP:-1} \
TF_CUDA_VERSION=${TF_CUDA_VERSION:-$_DEFAULT_CUDA_VERSION} \
TF_CUDNN_VERSION=${TF_CUDNN_VERSION:-$_DEFAULT_CUDNN_VERSION} \
TF_NEED_HDFS=${TF_NEED_HDFS:-0} \
TF_ENABLE_XLA=${TF_ENABLE_XLA:-1} \
TF_NEED_S3=${TF_NEED_S3:-0} \
TF_NEED_KAFKA=${TF_NEED_KAFKA:-0} \
TF_NEED_CUDA=${TF_NEED_CUDA:-1} \
TF_NEED_GDR=${TF_NEED_GDR:-0} \
TF_NEED_VERBS=${TF_NEED_VERBS:-0} \
TF_NEED_OPENCL_SYCL=${TF_NEED_OPENCL_SYCL:-0} \
TF_CUDA_CLANG=${TF_CUDA_CLANG:-0} \
TF_NEED_ROCM=${TF_NEED_ROCM:-0} \
TF_NEED_MPI=${TF_NEED_MPI:-0} \
TF_SET_ANDROID_WORKSPACE=${TF_SET_ANDROID_WORKSPACE:-0} \
bazel --bazelrc=/dev/null run @org_tensorflow//:configure

# Copy from the TensorFlow output_base.
output_base=$(bazel info output_base)
popd

# Copy to the Minigo workspace.
workspace=$(bazel info workspace)

# Copy TensorFlow's bazelrc files to workspace.
cp ${output_base}/external/org_tensorflow/.bazelrc ${workspace}/tensorflow.bazelrc
cp ${output_base}/external/org_tensorflow/.tf_configure.bazelrc ${workspace}/tf_configure.bazelrc

echo "Building tensorflow package"
bazel run -c opt \
  --copt=-Wno-comment \
  --copt=-Wno-deprecated-declarations \
  --copt=-Wno-ignored-attributes \
  --copt=-Wno-maybe-uninitialized \
  --copt=-Wno-sign-compare \
  --define=need_trt="$TF_NEED_TENSORRT" \
  //cc/tensorflow:build -- ${workspace}/cc/tensorflow
