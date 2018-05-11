#!/bin/bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dst_dir="${script_dir}/tensorflow"
tmp_dir="/tmp/minigo_tf"
rm -rfd ${tmp_dir}
mkdir -p ${tmp_dir}
version_tag="v1.6.0-rc0"

echo "Cloning tensorflow to ${tmp_dir}"
git clone https://github.com/tensorflow/tensorflow "${tmp_dir}"

pushd "${tmp_dir}"

echo "Checking out ${version_tag}"
git checkout "tags/${version_tag}"

# Run the TensorFlow configuration script, setting reasonable values for most
# of the options.
echo "Configuring tensorflow"
CC_OPT_FLAGS="-march=ivybridge" \
TF_NEED_JEMALLOC=1 \
TF_NEED_GCP=1 \
TF_NEED_HDFS=0 \
TF_NEED_S3=0 \
TF_NEED_KAFKA=0 \
TF_NEED_CUDA=1 \
TF_NEED_GDR=0 \
TF_NEED_VERBS=0 \
TF_NEED_OPENCL_SYCL=0 \
TF_CUDA_CLANG=0 \
TF_NEED_TENSORRT=0 \
TF_NEED_MPI=0 \
TF_SET_ANDROID_WORKSPACE=0 \
./configure

echo "Building tensorflow package"
bazel build --copt=-march=ivybridge -c opt --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

echo "Tensorflow built-ish"
echo "Unpacking tensorflow package..."
unzip -q /tmp/tensorflow_pkg/tensorflow-*.whl -d ${tmp_dir}

echo "Copying tensor flow headers to ${dst_dir}"
cp -r ${tmp_dir}/tensorflow-*.data/purelib/tensorflow/include "${dst_dir}"

echo "Building tensorflow libraries"
bazel build --copt=-march=ivybridge -c opt --config=opt //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so

echo "Copying tensorflow libraries to ${dst_dir}"
cp bazel-bin/tensorflow/libtensorflow_*.so "${dst_dir}"

echo "Building toco"
bazel build -c opt --config=opt --copt=-march=ivybridge //tensorflow/contrib/lite/toco:toco
cp bazel-bin/tensorflow/contrib/lite/toco/toco "${dst_dir}"

popd
echo "Deleting tmp dir ${tmp_dir}"
rm -rf "${tmp_dir}"
