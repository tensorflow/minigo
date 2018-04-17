#!/bin/bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dst_dir="${script_dir}/tensorflow"
tmp_dir="$(mktemp -d -p /tmp minigo_tf.XXXXXX)"
version_tag="v1.6.0-rc1"

echo "Cloning tensorflow to ${tmp_dir}"
git clone https://github.com/tensorflow/tensorflow "${tmp_dir}"

pushd "${tmp_dir}"

echo "Checking out ${version_tag}"
git checkout "tags/${version_tag}"

# Run the TensorFlow configuration script, setting reasonable values for most
# of the options.
echo "Configuring tensorflow"
TF_NEED_JEMALLOC=1 \
TF_NEED_GCP=1 \
TF_NEED_HDFS=0 \
TF_NEED_S3=0 \
TF_NEED_KAFKA=0 \
TF_NEED_GDR=0 \
TF_NEED_VERBS=0 \
TF_NEED_OPENCL_SYCL=0 \
TF_CUDA_CLANG=0 \
TF_NEED_TENSORRT=0 \
TF_NEED_MPI=0 \
TF_SET_ANDROID_WORKSPACE=0 \
./configure

echo "Building tensorflow package"
bazel build -c opt --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

echo "Unpacking tensorflow package"
wheel unpack /tmp/tensorflow_pkg/tensorflow-*.whl

echo "Copying tensor flow headers to ${dst_dir}"
cp -r tensorflow-*/tensorflow-*.data/purelib/tensorflow/include "${dst_dir}"

echo "Building tensorflow libraries"
bazel build -c opt --config=opt //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so

echo "Copying tensorflow libraries and headers to ${dst_dir}"
cp bazel-bin/tensorflow/libtensorflow_*.so "${dst_dir}"

echo "Deleting tmp dir ${tmp_dir}"
rm -rf "${tmp_dir}"
popd
